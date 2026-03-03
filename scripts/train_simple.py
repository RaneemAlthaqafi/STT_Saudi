"""
Simple Phase 1 training — no Unsloth, pure HuggingFace + PEFT.
Avoids all Unsloth/AltUp/dynamo compilation bugs.

Usage:
    python scripts/train_simple.py \
        --data_dir ./data/saudi_clean \
        --output_dir ./checkpoints/phase1
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.arabic_normalizer import normalize_arabic_for_eval
from segment import apply_duration_filter


def load_model(args):
    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # LoRA config — no modules_to_save to avoid quantized tensor issue
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # 0 avoids unsloth warning, required for compile
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


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
    return {"messages": messages, "audio_array": example["audio"]["array"], "transcript": example["transcript"]}


def create_collate_fn(processor):
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

        # Mask everything before the assistant response
        for i, text in enumerate(texts):
            # Find where assistant response starts
            response_marker = "<start_of_turn>model\n"
            if response_marker in text:
                # Tokenize up to response marker to find mask boundary
                prefix = text[:text.rfind(response_marker) + len(response_marker)]
                prefix_ids = processor.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                prefix_len = len(prefix_ids)
                labels[i, :prefix_len] = -100

        batch["labels"] = labels
        return batch
    return collate_fn


class WERCallback(TrainerCallback):
    def __init__(self, model, processor, eval_data, log_n=30):
        self.model = model
        self.processor = processor
        n = min(log_n, len(eval_data))
        self.eval_subset = eval_data.select(range(n))
        self.wer_history = []

    def on_evaluate(self, args, state, control, **kwargs):
        from jiwer import wer as compute_wer
        self.model.eval()
        device = next(self.model.parameters()).device
        predictions, references = [], []

        for example in self.eval_subset:
            audio = np.array(example["audio"]["array"], dtype=np.float32)
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
                    out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
                text = self.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
                predictions.append(normalize_arabic_for_eval(text))
                references.append(normalize_arabic_for_eval(ref))
            except Exception as e:
                pass

        if predictions:
            step_wer = compute_wer(references, predictions) * 100
            self.wer_history.append({"step": state.global_step, "wer": round(step_wer, 2)})
            print(f"\n[WER @ step {state.global_step}] WER = {step_wer:.2f}% (on {len(predictions)} samples)")

        self.model.train()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args)

    print(f"\nLoading data from {args.data_dir}...")
    train_data = load_from_disk(str(Path(args.data_dir) / "train"))
    eval_data = load_from_disk(str(Path(args.data_dir) / "eval"))

    print(f"Train: {len(train_data)}  Eval: {len(eval_data)}")

    train_data = apply_duration_filter(train_data, min_sec=3.0, max_sec=30.0)
    eval_data = apply_duration_filter(eval_data, min_sec=3.0, max_sec=30.0)

    print(f"After filter — Train: {len(train_data)}  Eval: {len(eval_data)}")

    if args.max_train_samples:
        train_data = train_data.select(range(min(args.max_train_samples, len(train_data))))

    print("Formatting data...")
    train_data = train_data.map(format_for_training, remove_columns=train_data.column_names)
    eval_data = eval_data.map(format_for_training, remove_columns=eval_data.column_names)

    collate_fn = create_collate_fn(processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        optim="adamw_torch",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=0.01,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=42,
        report_to="none",
    )

    wer_cb = WERCallback(model, processor, eval_data, log_n=30)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collate_fn,
        callbacks=[wer_cb, EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n" + "=" * 60)
    print("Starting Phase 1 Training (pure HF, no Unsloth)")
    print(f"  Train: {len(train_data)} samples")
    print(f"  LR: {args.learning_rate}  Epochs: {args.num_epochs}")
    print("=" * 60 + "\n")

    trainer.train()

    print("\nSaving model...")
    trainer.save_model(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))

    if wer_cb.wer_history:
        with open(output_dir / "wer_history.json", "w") as f:
            json.dump(wer_cb.wer_history, f, indent=2)
        print("\nWER progress:")
        for e in wer_cb.wer_history:
            print(f"  step {e['step']:>5}: {e['wer']:.2f}%")

    print(f"\nDone! Model saved to {output_dir / 'final'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="oddadmix/MasriSwitch-Gemma3n-Transcriber-v1")
    parser.add_argument("--data_dir", default="./data/saudi_clean")
    parser.add_argument("--output_dir", default="./checkpoints/phase1")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_train_samples", type=int, default=None)
    args = parser.parse_args()
    main(args)
