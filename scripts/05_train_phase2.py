"""
Step 5: Phase 2 Training - Noise Robustness.
Continues fine-tuning the Phase 1 model on augmented (noisy) data.

Pure HuggingFace + PEFT — no Unsloth dependency.
Loads Phase 1 LoRA weights and continues training with lower LR.

Usage:
    python scripts/05_train_phase2.py \
        --model_dir ./checkpoints/phase1/final \
        --data_dir ./data/saudi_augmented \
        --eval_dir ./data/saudi_clean \
        --output_dir ./checkpoints/phase2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Patch Gemma3n MLP to skip _gaussian_topk on GB10 (sm_121)
def _patch_gemma3n_mlp():
    try:
        from transformers.models.gemma3n import modeling_gemma3n as _m
        orig_forward = _m.Gemma3nTextMLP.forward
        def _patched_forward(self, x):
            gate = self.gate_proj(x)
            gate = self.act_fn(gate)
            up = self.up_proj(x)
            return self.down_proj(gate * up)
        _m.Gemma3nTextMLP.forward = _patched_forward
        print("  [patch] Gemma3n MLP gaussian_topk disabled for GB10 compatibility")
    except Exception as e:
        print(f"  [patch] Warning: could not patch Gemma3n MLP: {e}")

_patch_gemma3n_mlp()

from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    Gemma3nForConditionalGeneration,
    TrainerCallback,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from utils.arabic_normalizer import normalize_arabic_for_eval


# ──────────────────────────────────────────────────────────────
# Model loading — resume from Phase 1 LoRA
# ──────────────────────────────────────────────────────────────

def load_phase1_model(model_dir, base_model_name, max_seq_length, load_in_4bit):
    """
    Load Phase 1 fine-tuned model (base + LoRA adapters).
    The LoRA adapters are loaded in trainable mode for continued training.
    """
    print(f"Loading Phase 1 model from: {model_dir}")
    print(f"  Base model: {base_model_name}")

    processor = AutoProcessor.from_pretrained(model_dir)

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load base model
    base_model = Gemma3nForConditionalGeneration.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    base_model.config.use_cache = False

    # Load LoRA adapters from Phase 1 (trainable, not merged)
    model = PeftModel.from_pretrained(
        base_model,
        model_dir,
        is_trainable=True,
    )

    model.print_trainable_parameters()
    return model, processor


# ──────────────────────────────────────────────────────────────
# Data formatting + collation (same as Phase 1)
# ──────────────────────────────────────────────────────────────

def format_for_training(example):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": example["audio"]["array"]},
                {"type": "text", "text": "Please transcribe this audio."}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["transcript"]}]
        }
    ]
    return {
        "messages": messages,
        "audio_array": example["audio"]["array"],
        "transcript": example["transcript"],
    }


def create_collate_fn(processor):
    """Create collate function with proper label masking."""
    response_marker = "<start_of_turn>model\n"
    marker_ids = processor.tokenizer.encode(response_marker, add_special_tokens=False)

    def find_subsequence(seq, subseq):
        n, m = len(seq), len(subseq)
        for i in range(n - m + 1):
            if seq[i:i + m] == subseq:
                return i
        return -1

    def collate_fn(examples):
        texts = []
        audios = []
        for ex in examples:
            msgs = ex["messages"]
            text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            ).strip()
            texts.append(text)
            audios.append(np.array(ex["audio_array"], dtype=np.float32))

        batch = processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        for i in range(len(texts)):
            input_ids_list = batch["input_ids"][i].tolist()
            idx = find_subsequence(input_ids_list, marker_ids)
            if idx >= 0:
                mask_end = idx + len(marker_ids)
                labels[i, :mask_end] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


# ──────────────────────────────────────────────────────────────
# WER callback
# ──────────────────────────────────────────────────────────────

class WERLoggingCallback(TrainerCallback):
    def __init__(self, model, processor, eval_data, log_n=50):
        self.model = model
        self.processor = processor
        n = min(log_n, len(eval_data))
        self.eval_subset = eval_data.select(range(n))
        self.wer_history = []

    def on_evaluate(self, args, state, control, **kwargs):
        try:
            from jiwer import wer as compute_wer
        except ImportError:
            return

        self.model.eval()
        device = next(self.model.parameters()).device

        predictions, references = [], []
        for example in self.eval_subset:
            audio = np.array(example["audio_array"], dtype=np.float32)
            ref = example["transcript"]
            msgs = [
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Please transcribe this audio."}
                ]}
            ]
            try:
                inputs = self.processor.apply_chat_template(
                    msgs, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs, max_new_tokens=256, do_sample=False,
                    )
                text = self.processor.decode(
                    out[0][input_len:], skip_special_tokens=True
                ).strip()
                predictions.append(normalize_arabic_for_eval(text))
                references.append(normalize_arabic_for_eval(ref))
            except Exception:
                pass

        if predictions:
            step_wer = compute_wer(references, predictions) * 100
            self.wer_history.append({
                "step": state.global_step, "wer": round(step_wer, 2)
            })
            print(
                f"\n[WER @ step {state.global_step}] WER = {step_wer:.2f}%  "
                f"(on {len(predictions)} samples)"
            )

        self.model.train()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Phase 1 model
    model, processor = load_phase1_model(
        args.model_dir, args.base_model_name,
        args.max_seq_length, args.load_in_4bit,
    )

    # 2. Load augmented training data + clean eval data
    print(f"\nLoading augmented data from {args.data_dir}...")
    train_data = load_from_disk(str(Path(args.data_dir) / "train_augmented"))
    print(f"Augmented train samples: {len(train_data)}")

    print(f"Loading eval data from {args.eval_dir}...")
    eval_data = load_from_disk(str(Path(args.eval_dir) / "eval"))
    print(f"Eval samples: {len(eval_data)}")

    if args.max_train_samples:
        train_data = train_data.select(
            range(min(args.max_train_samples, len(train_data)))
        )

    if args.max_eval_samples:
        eval_data = eval_data.select(
            range(min(args.max_eval_samples, len(eval_data)))
        )

    # 3. Format data
    print("\nFormatting data...")
    train_data = train_data.map(
        format_for_training, remove_columns=train_data.column_names
    )
    eval_data = eval_data.map(
        format_for_training, remove_columns=eval_data.column_names
    )

    # 4. Training config (lower LR for Phase 2)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Lower learning rate for Phase 2 (preserve Phase 1 knowledge)
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        num_train_epochs=args.num_epochs,

        gradient_checkpointing=False,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        optim="adamw_torch",

        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="none",

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        eval_strategy="steps",
        eval_steps=args.save_steps,

        weight_decay=0.01,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        seed=42,

        # Required for custom collate with SFTTrainer
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # 5. Create trainer
    collate_fn = create_collate_fn(processor)
    wer_callback = WERLoggingCallback(model, processor, eval_data, log_n=50)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collate_fn,
        callbacks=[
            wer_callback,
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # 6. Train
    print("\n" + "=" * 60)
    print("Starting Phase 2: Noise Robustness Training")
    print("=" * 60)
    print(f"  Base: Phase 1 model from {args.model_dir}")
    print(f"  Learning rate: {args.learning_rate} (lower than Phase 1)")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Stack: Pure HuggingFace + PEFT (no Unsloth)")
    print("=" * 60 + "\n")

    trainer.train()

    # 7. Save
    print("\nSaving Phase 2 model...")
    final_dir = str(output_dir / "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    # Save WER history
    if wer_callback.wer_history:
        wer_log_path = output_dir / "wer_history.json"
        with open(wer_log_path, "w") as f:
            json.dump(wer_callback.wer_history, f, indent=2)
        print(f"\nWER history saved to: {wer_log_path}")
        print("WER progress:")
        for entry in wer_callback.wer_history:
            print(f"  step {entry['step']:>6}: WER = {entry['wer']:.2f}%")

    print(f"\nPhase 2 complete! Model saved to: {output_dir / 'final'}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate: python scripts/04_evaluate.py --model_dir {output_dir / 'final'}")
    print(f"  2. Merge:    python scripts/06_merge_and_export.py --model_dir {output_dir / 'final'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Noise robustness training")

    parser.add_argument("--model_dir", type=str, default="./checkpoints/phase1/final")
    parser.add_argument("--base_model_name", type=str,
                        default="oddadmix/MasriSwitch-Gemma3n-Transcriber-v1",
                        help="Base model name (needed to reload architecture for PEFT)")
    parser.add_argument("--data_dir", type=str, default="./data/saudi_augmented")
    parser.add_argument("--eval_dir", type=str, default="./data/saudi_clean")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase2")

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=500)

    args = parser.parse_args()
    main(args)
