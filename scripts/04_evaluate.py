"""
Step 4: Evaluate the fine-tuned model on Saudi dialect test data.
Computes WER, CER, MER, WIL, WIP, RTF, and optional SNR-stratified breakdown.

Two fixed test sets:
  - Saudi Clean:  standard eval split from saudi_clean dataset
  - Saudi Noisy:  bodycam-simulated noisy version of the same samples

Outputs a comparison table: Baseline → Phase 1 → Phase 2

Supports both pure HF+PEFT models and Unsloth models.

Usage:
    # Evaluate Phase 1
    python scripts/04_evaluate.py \
        --model_dir ./checkpoints/phase1/final \
        --data_dir ./data/saudi_clean \
        --phase phase1

    # Evaluate Phase 2 (both clean + noisy test sets)
    python scripts/04_evaluate.py \
        --model_dir ./checkpoints/phase2/final \
        --data_dir ./data/saudi_clean \
        --phase phase2 \
        --snr_stratified

    # Print comparison table across all phases
    python scripts/04_evaluate.py --compare_results ./eval_results/
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


def _patch_gemma3n_mlp():
    """Patch Gemma3n gaussian_topk to avoid CUDA JIT erfinv on GB10."""
    try:
        import transformers.models.gemma3n.modeling_gemma3n as _m

        def _patched_gaussian_topk(self, x):
            target_sparsity_tensor = torch.tensor(
                self.target_sparsity, device="cpu", dtype=torch.float32
            )
            std_multiplier = torch.distributions.Normal(
                loc=torch.tensor(0.0), scale=torch.tensor(1.0)
            ).icdf(target_sparsity_tensor)
            std_multiplier = std_multiplier.to(device=x.device, dtype=x.dtype)
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            threshold = mean + std_multiplier * std
            return torch.where(x > threshold, x, torch.zeros_like(x))

        _m.Gemma3nTextMLP._gaussian_topk = _patched_gaussian_topk
        print("  [patch] Gemma3n gaussian_topk patched for GB10 evaluation")
    except Exception as e:
        print(f"  [patch] Warning: could not patch Gemma3n MLP: {e}")


_patch_gemma3n_mlp()


# ──────────────────────────────────────────────────────────────
# Bodycam noise simulation for fixed noisy test set
# ──────────────────────────────────────────────────────────────

def _apply_bodycam_noise(audio_array: np.ndarray, sr: int = 16000, snr_db: float = 10.0) -> np.ndarray:
    """Apply street noise at SNR 10dB + bodycam effects for noisy test set."""
    # Gaussian street-like noise
    signal_power = np.mean(audio_array ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio_array)).astype(np.float32)
    noisy = np.clip(audio_array + noise, -1.0, 1.0)

    # Bodycam quantization (8-bit)
    levels = 256
    noisy = np.round(noisy * levels) / levels

    # Gain variation
    gain = np.random.uniform(0.6, 1.2)
    noisy = np.clip(noisy * gain, -1.0, 1.0)

    return noisy.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Model loading — supports both PEFT and Unsloth
# ──────────────────────────────────────────────────────────────

def load_model_for_eval(model_dir, base_model_name=None, use_4bit=False):
    """
    Load fine-tuned model for evaluation.
    Tries pure HF+PEFT first, falls back to Unsloth.
    """
    model_dir = str(model_dir)

    # Check if this is a PEFT model (has adapter_config.json)
    adapter_config_path = Path(model_dir) / "adapter_config.json"
    is_peft = adapter_config_path.exists()

    if is_peft:
        print(f"Loading PEFT model from {model_dir}...")
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration, BitsAndBytesConfig
        from peft import PeftModel

        # Determine base model name
        if base_model_name is None:
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get(
                "base_model_name_or_path",
                "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1"
            )

        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base_model = Gemma3nForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )

        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()

        # Use the processor saved with the model, fall back to Google's
        try:
            processor = AutoProcessor.from_pretrained(model_dir)
        except Exception:
            processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")

        return model, processor

    else:
        # Fallback: try Unsloth
        print(f"Loading Unsloth model from {model_dir}...")
        from unsloth import FastModel
        from transformers import AutoProcessor

        model, _ = FastModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=1024,
            load_in_4bit=use_4bit,
            dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        model.eval()

        # Use Google's processor (Unsloth's saved version may have bugs)
        processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")
        return model, processor


def transcribe_single(model, processor, audio_array, sr=16000, max_new_tokens=256):
    """Transcribe a single audio sample."""
    messages = [
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


def print_comparison_table(results_dir: str):
    """
    Load all eval_results.json files from a directory and print
    a comparison table: Baseline → Phase 1 → Phase 2.
    """
    results_dir = Path(results_dir)
    phase_order = ["baseline", "phase1", "phase2"]
    rows = {}

    for phase in phase_order:
        candidates = list(results_dir.glob(f"*{phase}*eval_results.json"))
        if not candidates:
            candidates = list(results_dir.glob(f"{phase}/eval_results.json"))
        if candidates:
            with open(candidates[0]) as f:
                data = json.load(f)
            rows[phase] = data.get("overall", {})

    if not rows:
        print("No eval_results.json files found in", results_dir)
        return

    header = f"{'Phase':<12} {'WER':>8} {'CER':>8} {'Sub%':>8} {'Del%':>8} {'Ins%':>8}"
    print("\n" + "=" * 60)
    print("  EVALUATION COMPARISON TABLE")
    print("=" * 60)
    print(header)
    print("-" * 60)

    for phase in phase_order:
        if phase not in rows:
            continue
        r = rows[phase]
        wer = r.get("wer", 0) * 100
        cer = r.get("cer", 0) * 100
        sub = r.get("word_S", 0) / max(r.get("word_N", 1), 1) * 100
        del_ = r.get("word_D", 0) / max(r.get("word_N", 1), 1) * 100
        ins = r.get("word_I", 0) / max(r.get("word_N", 1), 1) * 100
        print(f"{phase.capitalize():<12} {wer:>7.1f}% {cer:>7.1f}% {sub:>7.1f}% {del_:>7.1f}% {ins:>7.1f}%")

    print("=" * 60)

    if "baseline" in rows and "phase2" in rows:
        b_wer = rows["baseline"].get("wer", 0) * 100
        p2_wer = rows["phase2"].get("wer", 0) * 100
        rel_improvement = (b_wer - p2_wer) / b_wer * 100 if b_wer > 0 else 0
        print(f"\nRelative WER improvement (Baseline → Phase 2): {rel_improvement:.1f}%")
        print(f"Target: <20% WER  |  Current Phase 2: {p2_wer:.1f}%")
    print()


def main(args):
    # Special mode: just print comparison table
    if args.compare_results:
        print_comparison_table(args.compare_results)
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_dir).parent / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    model, processor = load_model_for_eval(
        args.model_dir, base_model_name=args.base_model_name
    )

    # 2. Load eval data — TWO FIXED TEST SETS
    print(f"\nLoading eval data from {args.data_dir}...")
    eval_data = load_from_disk(str(Path(args.data_dir) / "eval"))
    print(f"Total eval samples: {len(eval_data)}")

    if args.max_samples and args.max_samples < len(eval_data):
        eval_data = eval_data.select(range(args.max_samples))
        print(f"Evaluating on {len(eval_data)} samples")

    # Fixed noisy test set (same seed every time for fair comparison)
    print(f"\nCreating fixed noisy test set (bodycam simulation, SNR 10dB, seed=42)...")
    rng_state = np.random.get_state()
    np.random.seed(42)

    noisy_eval_audio = []
    for ex in eval_data:
        audio_arr = np.array(ex["audio"]["array"], dtype=np.float32)
        noisy = _apply_bodycam_noise(audio_arr, ex["audio"]["sampling_rate"], snr_db=10.0)
        noisy_eval_audio.append(noisy)
    np.random.set_state(rng_state)
    print(f"Noisy test set ready: {len(noisy_eval_audio)} samples")

    # 3. Warmup (1 sample, not counted)
    print("\nWarmup inference...")
    warmup_audio = np.array(eval_data[0]["audio"]["array"], dtype=np.float32)
    transcribe_single(model, processor, warmup_audio)

    # 4. Run evaluation on CLEAN test set
    print("\nRunning evaluation (clean)...")
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

        pred_norm = normalize_arabic_for_eval(prediction)
        ref_norm = normalize_arabic_for_eval(reference)

        predictions.append(pred_norm)
        references.append(ref_norm)
        rtf_tracker.record(duration_sec, inference_time)

        if args.snr_stratified:
            snr_db = estimate_snr(audio_array, sr)
            snr_values.append(snr_db)

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

        if i < 5 or (i + 1) % 50 == 0:
            print(f"\n--- Sample {i} ---")
            print(f"  REF: {reference}")
            print(f"  PRD: {prediction}")
            if sample_metrics.get("wer") is not None:
                print(f"  WER: {sample_metrics['wer']*100:.1f}%  CER: {sample_metrics['cer']*100:.1f}%")

    # 5. Compute clean metrics
    overall_clean = compute_all_metrics(references, predictions)
    rtf = rtf_tracker.summary()
    snr_results = None
    if args.snr_stratified and snr_values:
        snr_results = compute_snr_stratified_metrics(references, predictions, snr_values)

    print("\n" + "=" * 50)
    print("  CLEAN TEST SET RESULTS")
    print("=" * 50)
    print_results(overall_clean, rtf, snr_results)

    # 6. Evaluate on NOISY test set
    print("\n" + "=" * 50)
    print("  NOISY TEST SET (bodycam simulation, SNR 10dB)")
    print("=" * 50)

    predictions_noisy = []
    references_noisy = []
    rtf_tracker_noisy = RTFTracker()

    for i, example in enumerate(tqdm(eval_data, desc="Evaluating (noisy)")):
        audio_array = noisy_eval_audio[i]
        sr = example["audio"]["sampling_rate"]
        reference = example["transcript"]
        duration_sec = len(audio_array) / sr

        try:
            prediction, inference_time = transcribe_single(model, processor, audio_array, sr)
        except Exception as e:
            prediction = ""
            inference_time = 0.0

        predictions_noisy.append(normalize_arabic_for_eval(prediction))
        references_noisy.append(normalize_arabic_for_eval(reference))
        rtf_tracker_noisy.record(duration_sec, inference_time)

    overall_noisy = compute_all_metrics(references_noisy, predictions_noisy)
    rtf_noisy = rtf_tracker_noisy.summary()
    print_results(overall_noisy, rtf_noisy, None)

    clean_wer = overall_clean.get("wer", 0) * 100
    noisy_wer = overall_noisy.get("wer", 0) * 100
    print(f"\n  Clean WER: {clean_wer:.2f}%")
    print(f"  Noisy WER: {noisy_wer:.2f}%")
    print(f"  WER degradation under noise: +{noisy_wer - clean_wer:.2f}%")

    # 7. Save results
    results_summary = {
        "model_dir": args.model_dir,
        "data_dir": args.data_dir,
        "phase": args.phase,
        "num_samples": len(predictions),
        "overall": overall_clean,
        "overall_noisy": overall_noisy,
        "rtf": rtf,
        "rtf_noisy": rtf_noisy,
    }
    if snr_results:
        results_summary["snr_stratified"] = snr_results

    results_file = output_dir / f"{args.phase}_eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    detail_file = output_dir / f"{args.phase}_eval_detail.json"
    with open(detail_file, "w", encoding="utf-8") as f:
        json.dump(results_detail, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:")
    print(f"  Summary: {results_file}")
    print(f"  Details: {detail_file}")
    print(f"\nTo compare all phases:")
    print(f"  python scripts/04_evaluate.py --compare_results {output_dir.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Saudi STT model")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--base_model_name", type=str, default=None,
                        help="Base model name (auto-detected from adapter_config.json if not set)")
    parser.add_argument("--data_dir", type=str, default="./data/saudi_clean",
                        help="Path to dataset directory with eval split")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory for evaluation results")
    parser.add_argument("--phase", type=str, default="phase1",
                        choices=["baseline", "phase1", "phase2"],
                        help="Which phase this evaluation is for (used in filenames)")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max samples to evaluate")
    parser.add_argument("--snr_stratified", action="store_true",
                        help="Compute SNR-stratified WER/CER breakdown")
    parser.add_argument("--compare_results", type=str, default=None,
                        help="Print comparison table from a directory of eval_results.json files")
    args = parser.parse_args()
    main(args)
