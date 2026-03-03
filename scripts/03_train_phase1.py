"""
Step 3: Phase 1 Training - Saudi Dialect Adaptation.
Fine-tunes MasriSwitch on clean Saudi dialect data.

Usage:
    python scripts/03_train_phase1.py \
        --data_dir ./data/saudi_clean \
        --output_dir ./checkpoints/phase1
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, EarlyStoppingCallback

# Fix torch compilation cache issue on some GPUs
torch._dynamo.config.cache_size_limit = 32

sys.path.insert(0, str(Path(__file__).parent))
from segment import apply_duration_filter
from utils.arabic_normalizer import normalize_arabic_for_eval


def load_model(args):
    """Load model with Unsloth for memory-efficient training."""
    from unsloth import FastModel

    print(f"Loading model: {args.model_name}")
    print(f"  4-bit quantization: {args.load_in_4bit}")
    print(f"  Max sequence length: {args.max_seq_length}")

    model, processor = FastModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    # Apply LoRA
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            # Attention
            "q_proj", "k_proj", "v_proj", "o_proj",
            # FFN
            "gate_proj", "up_proj", "down_proj",
            # Audio-specific (CRITICAL for STT fine-tuning)
            "post", "linear_start", "linear_end",
            "embedding_projection",
        ],
        bias="none",
    )

    model.print_trainable_parameters()
    return model, processor


def format_for_training(example):
    """Format example into Gemma3n chat template with audio."""
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an assistant that transcribes speech accurately."
                }
            ]
        },
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
    return {"messages": messages}


def create_collate_fn(processor):
    """Create a collate function for batching audio + text data."""
    def collate_fn(examples):
        texts = []
        audios = []

        for example in examples:
            text = processor.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            ).strip()
            texts.append(text)

            # Extract audio from the user message
            for msg in example["messages"]:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content.get("type") == "audio":
                            audios.append(content["audio"])

        batch = processor(
            text=texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )

        # Create labels (mask padding and audio tokens)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if hasattr(processor.tokenizer, "audio_token_id") and processor.tokenizer.audio_token_id is not None:
            labels[labels == processor.tokenizer.audio_token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load model
    # -------------------------------------------------------------------------
    model, processor = load_model(args)

    # -------------------------------------------------------------------------
    # 2. Load data
    # -------------------------------------------------------------------------
    print(f"\nLoading training data from {args.data_dir}...")
    train_data = load_from_disk(str(Path(args.data_dir) / "train"))
    eval_data = load_from_disk(str(Path(args.data_dir) / "eval"))

    print(f"Train samples (before filter): {len(train_data)}")
    print(f"Eval samples (before filter):  {len(eval_data)}")

    # Apply duration filter: keep only 5-30s samples
    # This is CRITICAL — short clips (<3s) cause poor gradient signal,
    # and clips >30s overflow the model's audio context.
    print("\nApplying duration filter (5-30s)...")
    train_data = apply_duration_filter(train_data, min_sec=5.0, max_sec=30.0)
    eval_data  = apply_duration_filter(eval_data,  min_sec=5.0, max_sec=30.0)

    print(f"Train samples (after filter): {len(train_data)}")
    print(f"Eval samples (after filter):  {len(eval_data)}")

    # Limit samples if specified (for quick testing)
    if args.max_train_samples:
        train_data = train_data.select(range(min(args.max_train_samples, len(train_data))))
        print(f"Limited to {len(train_data)} training samples")

    if args.max_eval_samples:
        eval_data = eval_data.select(range(min(args.max_eval_samples, len(eval_data))))

    # -------------------------------------------------------------------------
    # 3. Format data
    # -------------------------------------------------------------------------
    print("\nFormatting data for training...")
    train_data = train_data.map(format_for_training, remove_columns=train_data.column_names)
    eval_data = eval_data.map(format_for_training, remove_columns=eval_data.column_names)

    # -------------------------------------------------------------------------
    # 4. Configure training
    # -------------------------------------------------------------------------
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

        # Sequence
        max_seq_length=args.max_seq_length,

        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="none",

        # Saving — save best checkpoint for early stopping
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

    # -------------------------------------------------------------------------
    # 5. WER logging callback
    # Logs WER on a small eval subset at each checkpoint step.
    # Gives a real-world signal beyond eval_loss.
    # -------------------------------------------------------------------------
    class WERLoggingCallback(TrainerCallback):
        def __init__(self, model, processor, eval_data, log_n=50):
            self.model = model
            self.processor = processor
            # Fixed sample of 50 for fast WER check
            n = min(log_n, len(eval_data))
            self.eval_subset = eval_data.select(range(n))
            self.wer_history = []

        def on_evaluate(self, args, state, control, **kwargs):
            from jiwer import wer as compute_wer
            self.model.eval()
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype

            predictions, references = [], []
            for example in self.eval_subset:
                audio_array = np.array(example["audio"]["array"], dtype=np.float32)
                reference = example["transcript"]
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are an assistant that transcribes speech accurately."}]},
                    {"role": "user", "content": [
                        {"type": "audio", "audio": audio_array},
                        {"type": "text", "text": "Please transcribe this audio."}
                    ]},
                ]
                try:
                    inputs = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors="pt",
                    )
                    moved = {}
                    for k, v in inputs.items():
                        if hasattr(v, "to"):
                            v = v.to(device)
                            if v.is_floating_point():
                                v = v.to(dtype)
                        moved[k] = v
                    input_len = moved["input_ids"].shape[-1]
                    with torch.inference_mode():
                        out = self.model.generate(**moved, max_new_tokens=256, do_sample=False)
                    text = self.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
                    predictions.append(normalize_arabic_for_eval(text))
                    references.append(normalize_arabic_for_eval(reference))
                except Exception:
                    pass

            if predictions:
                step_wer = compute_wer(references, predictions) * 100
                self.wer_history.append({"step": state.global_step, "wer": round(step_wer, 2)})
                print(f"\n[WER @ step {state.global_step}] WER = {step_wer:.2f}%  "
                      f"(on {len(predictions)} samples)")

            self.model.train()

    # -------------------------------------------------------------------------
    # 5b. Create trainer
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 5b. Mask labels to only train on assistant response (CRITICAL)
    # Without this, the model also trains on the user prompt and system message,
    # which wastes capacity and can hurt transcription quality.
    # -------------------------------------------------------------------------
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # -------------------------------------------------------------------------
    # 6. Train!
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting Phase 1: Saudi Dialect Adaptation")
    print("=" * 60)
    print(f"  Model: {args.model_name}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print("=" * 60 + "\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # -------------------------------------------------------------------------
    # 7. Save final model + WER history
    # -------------------------------------------------------------------------
    print("\nSaving final model...")
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))

    # Save WER history to disk
    if wer_callback.wer_history:
        import json
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
    parser.add_argument("--lora_dropout", type=float, default=0.05)

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
