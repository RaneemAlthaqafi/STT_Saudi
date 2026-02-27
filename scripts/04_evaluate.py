"""
Step 4: Evaluate the fine-tuned model on Saudi dialect test data.
Computes WER, CER, MER, WIL, WIP, RTF, and optional SNR-stratified breakdown.

Uses shared normalization and metrics from scripts/utils/.

Usage:
    python scripts/04_evaluate.py \
        --model_dir ./checkpoints/phase1/final \
        --data_dir ./data/saudi_clean \
        --max_samples 200

    # With SNR-stratified breakdown:
    python scripts/04_evaluate.py \
        --model_dir ./checkpoints/phase1/final \
        --data_dir ./data/saudi_clean \
        --snr_stratified
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.arabic_normalizer import normalize_arabic_for_eval
from utils.metrics import (
    compute_all_metrics,
    compute_per_sample_metrics,
    RTFTracker,
    estimate_snr,
    compute_snr_stratified_metrics,
    print_results,
)


def load_model_for_eval(model_dir, use_4bit=True):
    """Load fine-tuned model for evaluation."""
    from unsloth import FastModel

    print(f"Loading model from {model_dir}...")

    model, _ = FastModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=1024,
        load_in_4bit=use_4bit,
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.eval()

    # Use Google's processor (Unsloth's saved version may have bugs)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")

    return model, processor


def transcribe_single(model, processor, audio_array, sr=16000, max_new_tokens=256):
    """Transcribe a single audio sample."""
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are an assistant that transcribes speech accurately."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": "Please transcribe this audio."}
            ]
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    # Sync GPU before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - t0

    gen_tokens = output[0][input_len:]
    text = processor.decode(gen_tokens, skip_special_tokens=True)
    return text.strip(), inference_time


def main(args):
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_dir).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────
    model, processor = load_model_for_eval(args.model_dir)

    # ── 2. Load eval data ─────────────────────────────────────
    print(f"\nLoading eval data from {args.data_dir}...")
    eval_data = load_from_disk(str(Path(args.data_dir) / "eval"))
    print(f"Total eval samples: {len(eval_data)}")

    if args.max_samples and args.max_samples < len(eval_data):
        eval_data = eval_data.select(range(args.max_samples))
        print(f"Evaluating on {len(eval_data)} samples")

    # ── 3. Warmup (1 sample, not counted) ─────────────────────
    print("\nWarmup inference...")
    warmup_audio = np.array(eval_data[0]["audio"]["array"], dtype=np.float32)
    transcribe_single(model, processor, warmup_audio)

    # ── 4. Run evaluation ─────────────────────────────────────
    print("\nRunning evaluation...")
    predictions = []
    references = []
    rtf_tracker = RTFTracker()
    snr_values = []
    results_detail = []

    for i, example in enumerate(tqdm(eval_data, desc="Evaluating")):
        audio_array = np.array(example["audio"]["array"], dtype=np.float32)
        sr = example["audio"]["sampling_rate"]
        reference = example["transcript"]
        duration_sec = len(audio_array) / sr

        try:
            prediction, inference_time = transcribe_single(model, processor, audio_array, sr)
        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            prediction = ""
            inference_time = 0.0

        # Normalize for evaluation (OALL standard)
        pred_norm = normalize_arabic_for_eval(prediction)
        ref_norm = normalize_arabic_for_eval(reference)

        predictions.append(pred_norm)
        references.append(ref_norm)

        # Track RTF
        rtf_tracker.record(duration_sec, inference_time)

        # Estimate SNR if needed
        if args.snr_stratified:
            snr_db = estimate_snr(audio_array, sr)
            snr_values.append(snr_db)

        # Per-sample metrics
        sample_metrics = compute_per_sample_metrics(ref_norm, pred_norm)

        results_detail.append({
            "index": i,
            "reference": reference,
            "prediction": prediction,
            "ref_normalized": ref_norm,
            "pred_normalized": pred_norm,
            "duration_seconds": round(duration_sec, 2),
            "inference_time": round(inference_time, 4),
            "wer": sample_metrics.get("wer"),
            "cer": sample_metrics.get("cer"),
            "snr_db": round(snr_values[-1], 1) if snr_values else None,
        })

        # Print periodic samples
        if i < 5 or (i + 1) % 50 == 0:
            print(f"\n--- Sample {i} ---")
            print(f"  REF: {reference}")
            print(f"  PRD: {prediction}")
            if sample_metrics.get("wer") is not None:
                print(f"  WER: {sample_metrics['wer']*100:.1f}%  CER: {sample_metrics['cer']*100:.1f}%")

    # ── 5. Compute overall metrics ────────────────────────────
    overall = compute_all_metrics(references, predictions)
    rtf = rtf_tracker.summary()
    snr_results = None
    if args.snr_stratified and snr_values:
        snr_results = compute_snr_stratified_metrics(references, predictions, snr_values)

    # Print results
    print_results(overall, rtf, snr_results)

    # ── 6. Save results ───────────────────────────────────────
    results_summary = {
        "model_dir": args.model_dir,
        "data_dir": args.data_dir,
        "num_samples": len(predictions),
        "overall": overall,
        "rtf": rtf,
    }
    if snr_results:
        results_summary["snr_stratified"] = snr_results

    results_file = output_dir / "eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    detail_file = output_dir / "eval_detail.json"
    with open(detail_file, "w", encoding="utf-8") as f:
        json.dump(results_detail, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:")
    print(f"  Summary: {results_file}")
    print(f"  Details: {detail_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Saudi STT model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/saudi_clean",
                        help="Path to dataset directory with eval split")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for evaluation results")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max samples to evaluate")
    parser.add_argument("--snr_stratified", action="store_true",
                        help="Compute SNR-stratified WER/CER breakdown")
    args = parser.parse_args()
    main(args)
