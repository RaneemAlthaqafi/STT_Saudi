"""
Step 5: Phase 2 Training - Noise Robustness.
Continues fine-tuning the Phase 1 model on augmented (noisy) data.

Usage:
    python scripts/05_train_phase2.py \
        --model_dir ./checkpoints/phase1/final \
        --data_dir ./data/saudi_augmented \
        --eval_dir ./data/saudi_clean \
        --output_dir ./checkpoints/phase2
"""

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig

torch._dynamo.config.cache_size_limit = 32


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load Phase 1 model (already has Saudi dialect knowledge)
    # -------------------------------------------------------------------------
    from unsloth import FastModel

    print(f"Loading Phase 1 model from: {args.model_dir}")
    model, processor = FastModel.from_pretrained(
        model_name=args.model_dir,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # -------------------------------------------------------------------------
    # 2. Load augmented training data + clean eval data
    # -------------------------------------------------------------------------
    print(f"\nLoading augmented data from {args.data_dir}...")
    train_data = load_from_disk(str(Path(args.data_dir) / "train_augmented"))
    print(f"Augmented train samples: {len(train_data)}")

    print(f"Loading eval data from {args.eval_dir}...")
    eval_data = load_from_disk(str(Path(args.eval_dir) / "eval"))
    print(f"Eval samples: {len(eval_data)}")

    if args.max_train_samples:
        train_data = train_data.select(range(min(args.max_train_samples, len(train_data))))

    if args.max_eval_samples:
        eval_data = eval_data.select(range(min(args.max_eval_samples, len(eval_data))))

    # -------------------------------------------------------------------------
    # 3. Format data
    # -------------------------------------------------------------------------
    def format_for_training(example):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an assistant that transcribes speech accurately."}]
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
                "content": [{"type": "text", "text": example["transcript"]}]
            }
        ]
        return {"messages": messages}

    print("\nFormatting data...")
    train_data = train_data.map(format_for_training, remove_columns=train_data.column_names)
    eval_data = eval_data.map(format_for_training, remove_columns=eval_data.column_names)

    # -------------------------------------------------------------------------
    # 4. Collate function
    # -------------------------------------------------------------------------
    def collate_fn(examples):
        texts = []
        audios = []
        for example in examples:
            text = processor.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            ).strip()
            texts.append(text)
            for msg in example["messages"]:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content.get("type") == "audio":
                            audios.append(content["audio"])

        batch = processor(text=texts, audio=audios, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if hasattr(processor.tokenizer, "audio_token_id") and processor.tokenizer.audio_token_id is not None:
            labels[labels == processor.tokenizer.audio_token_id] = -100
        batch["labels"] = labels
        return batch

    # -------------------------------------------------------------------------
    # 5. Training config (lower LR for Phase 2)
    # -------------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Lower learning rate for Phase 2 (preserving Phase 1 knowledge)
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        num_train_epochs=args.num_epochs,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        optim="adamw_8bit",

        max_seq_length=args.max_seq_length,

        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="none",

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,

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

    # -------------------------------------------------------------------------
    # 6. Train Phase 2
    # -------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collate_fn,
    )

    # -------------------------------------------------------------------------
    # 6b. Mask labels to only train on assistant response (CRITICAL)
    # Without this, the model also trains on the user prompt and system message,
    # which wastes capacity and can hurt transcription quality.
    # -------------------------------------------------------------------------
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print("\n" + "=" * 60)
    print("Starting Phase 2: Noise Robustness Training")
    print("=" * 60)
    print(f"  Base: Phase 1 model from {args.model_dir}")
    print(f"  Learning rate: {args.learning_rate} (lower than Phase 1)")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Train samples: {len(train_data)}")
    print("=" * 60 + "\n")

    trainer.train()

    # -------------------------------------------------------------------------
    # 7. Save
    # -------------------------------------------------------------------------
    print("\nSaving Phase 2 model...")
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))

    print(f"\nPhase 2 complete! Model saved to: {output_dir / 'final'}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate: python scripts/04_evaluate.py --model_dir {output_dir / 'final'}")
    print(f"  2. Merge:    python scripts/06_merge_and_export.py --model_dir {output_dir / 'final'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Noise robustness training")

    parser.add_argument("--model_dir", type=str, default="./checkpoints/phase1/final")
    parser.add_argument("--data_dir", type=str, default="./data/saudi_augmented")
    parser.add_argument("--eval_dir", type=str, default="./data/saudi_clean")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/phase2")

    parser.add_argument("--max_seq_length", type=int, default=1024)
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
