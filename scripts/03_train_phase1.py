"""
Step 3: Phase 1 Training - Saudi Dialect Adaptation.
Fine-tunes MasriSwitch on clean Saudi dialect data.

Pure HuggingFace + PEFT — no Unsloth dependency.

Usage:
    python scripts/03_train_phase1.py \
        --data_dir ./data/saudi_clean \
        --output_dir ./checkpoints/phase1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    Gemma3nForConditionalGeneration,
    TrainerCallback,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import get_peft_model, LoraConfig, TaskType

sys.path.insert(0, str(Path(__file__).parent))
from segment import apply_duration_filter
from utils.arabic_normalizer import normalize_arabic_for_eval


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

def load_model(args):
    """Load model with pure HuggingFace + PEFT (no Unsloth)."""
    print(f"Loading model: {args.model_name}")
    print(f"  4-bit quantization: {args.load_in_4bit}")
    print(f"  Max sequence length: {args.max_seq_length}")

    # Processor — always from the base model
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Quantization config
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Gemma3nForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # Freeze base model
    model.config.use_cache = False

    # LoRA config matching the architecture diagram:
    # - Attention + FFN (core language understanding)
    # - Audio-specific modules excluded from modules_to_save because
    #   they are quantized (int4) — cannot require_grad on quantized tensors.
    #   Instead we target them via LoRA target_modules.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            # Attention
            "q_proj", "k_proj", "v_proj", "o_proj",
            # FFN
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


# ──────────────────────────────────────────────────────────────
# Data formatting
# ──────────────────────────────────────────────────────────────

def format_for_training(example):
    """Format example into Gemma3n chat template with audio."""
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
            "content": [
                {"type": "text", "text": example["transcript"]}
            ]
        }
    ]
    return {
        "messages": messages,
        "audio_array": example["audio"]["array"],
        "transcript": example["transcript"],
    }


def create_collate_fn(processor):
    """Create a collate function with proper label masking."""

    # Pre-compute the response marker token IDs for reliable label masking
    response_marker = "<start_of_turn>model\n"
    marker_ids = processor.tokenizer.encode(response_marker, add_special_tokens=False)

    def find_subsequence(seq, subseq):
        """Find the start index of subseq in seq. Returns -1 if not found."""
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

        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask everything BEFORE the assistant response for each sample.
        # This is the key label masking step — only train on the transcription.
        for i in range(len(texts)):
            input_ids_list = batch["input_ids"][i].tolist()
            idx = find_subsequence(input_ids_list, marker_ids)
            if idx >= 0:
                # Mask everything up to and including the response marker
                mask_end = idx + len(marker_ids)
                labels[i, :mask_end] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


# ──────────────────────────────────────────────────────────────
# WER logging callback
# ──────────────────────────────────────────────────────────────

class WERLoggingCallback(TrainerCallback):
    """Compute WER on a small eval subset at each evaluation step."""

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

    # 1. Load model
    model, processor = load_model(args)

    # 2. Load data
    print(f"\nLoading training data from {args.data_dir}...")
    train_data = load_from_disk(str(Path(args.data_dir) / "train"))
    eval_data = load_from_disk(str(Path(args.data_dir) / "eval"))

    print(f"Train samples (before filter): {len(train_data)}")
    print(f"Eval samples (before filter):  {len(eval_data)}")

    # Duration filter: 5-30s
    print("\nApplying duration filter (5-30s)...")
    train_data = apply_duration_filter(train_data, min_sec=5.0, max_sec=30.0)
    eval_data = apply_duration_filter(eval_data, min_sec=5.0, max_sec=30.0)

    print(f"Train samples (after filter): {len(train_data)}")
    print(f"Eval samples (after filter):  {len(eval_data)}")

    if args.max_train_samples:
        train_data = train_data.select(
            range(min(args.max_train_samples, len(train_data)))
        )
        print(f"Limited to {len(train_data)} training samples")

    if args.max_eval_samples:
        eval_data = eval_data.select(
            range(min(args.max_eval_samples, len(eval_data)))
        )

    # 3. Format data
    print("\nFormatting data for training...")
    train_data = train_data.map(
        format_for_training, remove_columns=train_data.column_names
    )
    eval_data = eval_data.map(
        format_for_training, remove_columns=eval_data.column_names
    )

    # 4. Configure training
    print("\nConfiguring training...")
    training_args = SFTConfig(
        output_dir=str(output_dir),

        # Batch settings
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Learning rate
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # Duration
        num_train_epochs=args.num_epochs,

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        optim="adamw_8bit",

        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="none",

        # Saving — save best checkpoint
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Evaluation
        eval_strategy="steps",
        eval_steps=args.save_steps,

        # Misc
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
        max_seq_length=args.max_seq_length,
        callbacks=[
            wer_callback,
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # 6. Train
    print("\n" + "=" * 60)
    print("Starting Phase 1: Saudi Dialect Adaptation")
    print("=" * 60)
    print(f"  Model: {args.model_name}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Stack: Pure HuggingFace + PEFT (no Unsloth)")
    print("=" * 60 + "\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 7. Save final model + WER history
    print("\nSaving final model...")
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

    print(f"\nPhase 1 training complete!")
    print(f"Model saved to: {output_dir / 'final'}")
    print(f"\nNext step: Run scripts/04_evaluate.py to evaluate the model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Saudi dialect fine-tuning")

    # Model
    parser.add_argument("--model_name", type=str,
                        default="oddadmix/MasriSwitch-Gemma3n-Transcriber-v1")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Data
    parser.add_argument("--data_dir", type=str, default="./data/saudi_clean")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=500)

    # Training
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    main(args)
