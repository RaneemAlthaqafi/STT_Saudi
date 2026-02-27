"""
Step 0: Benchmark Arabic STT models on SADA Saudi dialect test data.

Tests 10 models across 3 architecture families (Whisper, CTC, Generative)
with SNR-stratified and per-dialect breakdown.

Run this BEFORE fine-tuning to establish baselines.
Run again AFTER fine-tuning to measure improvement.

Usage:
    python scripts/00_benchmark.py --max_samples 200
    python scripts/00_benchmark.py --models whisper-v3 mms gemma --max_samples 500
    python scripts/00_benchmark.py --all --max_samples 100

Models tested:
    Whisper family:   whisper-large-v3, whisper-large-v3-turbo, Byne/arabic
    CTC family:       wav2vec2-xlsr-arabic, mms-1b-all, saudi-wav2vec2
    Generative:       MasriSwitch-Gemma3n
    Faster-Whisper:   faster-whisper-large-v3 (production speed)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.arabic_normalizer import normalize_arabic_for_eval
from utils.metrics import (
    compute_all_metrics, compute_per_sample_metrics,
    RTFTracker, estimate_snr, get_snr_bin,
    compute_snr_stratified_metrics, print_results,
)

# ──────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # Whisper family
    "whisper-v3": {
        "id": "openai/whisper-large-v3",
        "type": "whisper",
        "desc": "Whisper Large V3 (1.5B, universal baseline)",
    },
    "whisper-v3-turbo": {
        "id": "openai/whisper-large-v3-turbo",
        "type": "whisper",
        "desc": "Whisper Large V3 Turbo (809M, speed baseline)",
    },
    "whisper-arabic": {
        "id": "Byne/whisper-large-v3-arabic",
        "type": "whisper",
        "desc": "Byne Arabic Whisper (best Arabic fine-tune, 9.38% WER reported)",
    },
    "whisper-codeswitching": {
        "id": "MohamedRashad/Arabic-Whisper-CodeSwitching-Edition",
        "type": "whisper",
        "desc": "Arabic-English code-switching Whisper",
    },
    # CTC family
    "wav2vec2-arabic": {
        "id": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        "type": "ctc",
        "desc": "Wav2Vec2 XLSR-53 Arabic (top CTC baseline)",
    },
    "mms": {
        "id": "facebook/mms-1b-all",
        "type": "mms",
        "desc": "MMS 1B (CTC, 1162 languages, dialect adapters)",
    },
    "wav2vec2-saudi": {
        "id": "salmujaiwel/wav2vec2-large-xls-r-300m-arabic-saudi-colab",
        "type": "ctc",
        "desc": "XLS-R 300M Saudi Arabic (only Saudi-specific CTC)",
    },
    # Generative
    "gemma": {
        "id": "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1",
        "type": "gemma",
        "desc": "MasriSwitch Gemma3n (our base model)",
    },
    # Faster-Whisper (production speed)
    "faster-whisper": {
        "id": "Systran/faster-whisper-large-v3",
        "type": "faster-whisper",
        "desc": "Faster-Whisper Large V3 (CTranslate2, production speed)",
    },
}

DEFAULT_MODELS = ["whisper-v3", "whisper-v3-turbo", "whisper-arabic",
                  "wav2vec2-arabic", "mms", "gemma"]


# ──────────────────────────────────────────────────────────────
# Model loaders & transcribers
# ──────────────────────────────────────────────────────────────

def load_whisper(model_id):
    from transformers import pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return pipe


def transcribe_whisper(pipe, audio_array, sr=16000):
    result = pipe(
        {"raw": audio_array, "sampling_rate": sr},
        generate_kwargs={"language": "arabic", "task": "transcribe"},
    )
    return result["text"]


def load_ctc(model_id):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, processor


def transcribe_ctc(model_and_proc, audio_array, sr=16000):
    model, processor = model_and_proc
    inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])


def load_mms(model_id):
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    processor.tokenizer.set_target_lang("ara")
    model.load_adapter("ara")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, processor


def transcribe_mms(model_and_proc, audio_array, sr=16000):
    return transcribe_ctc(model_and_proc, audio_array, sr)


def load_gemma(model_id):
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration
    processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    return model, processor


def transcribe_gemma(model_and_proc, audio_array, sr=16000):
    model, processor = model_and_proc
    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": "You are an assistant that transcribes speech accurately."}
        ]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_array},
            {"type": "text", "text": "Please transcribe this audio."}
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return processor.decode(output[0][input_len:], skip_special_tokens=True).strip()


def load_faster_whisper(model_id):
    from faster_whisper import WhisperModel
    compute = "float16" if torch.cuda.is_available() else "int8"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_id, device=device, compute_type=compute)
    return model


def transcribe_faster_whisper(model, audio_array, sr=16000):
    segments, _ = model.transcribe(audio_array, language="ar")
    return " ".join(seg.text for seg in segments)


LOADERS = {
    "whisper": (load_whisper, transcribe_whisper),
    "ctc": (load_ctc, transcribe_ctc),
    "mms": (load_mms, transcribe_mms),
    "gemma": (load_gemma, transcribe_gemma),
    "faster-whisper": (load_faster_whisper, transcribe_faster_whisper),
}


# ──────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────

def load_test_data(max_samples, dialect_filter=None):
    """Load SADA test split, optionally filtered by dialect."""
    print("Loading SADA test data...")
    test = load_dataset("MohamedRashad/SADA22", split="test")
    print(f"  Total test samples: {len(test)}")

    if dialect_filter:
        test = test.filter(
            lambda x: str(x.get("speaker_dialect", "")).lower() in dialect_filter
        )
        print(f"  After dialect filter ({dialect_filter}): {len(test)}")

    # Filter by duration (2-30s)
    def valid_duration(x):
        dur = len(x["audio"]["array"]) / x["audio"]["sampling_rate"]
        return 2.0 <= dur <= 30.0
    test = test.filter(valid_duration)
    print(f"  After duration filter (2-30s): {len(test)}")

    # Filter by text quality
    import re
    def valid_text(x):
        text = x.get("cleaned_text", x.get("text", ""))
        return len(re.findall(r'[\u0600-\u06FF]', text)) >= 2
    test = test.filter(valid_text)
    print(f"  After text quality filter: {len(test)}")

    if max_samples and max_samples < len(test):
        test = test.shuffle(seed=42).select(range(max_samples))
        print(f"  Sampled: {max_samples}")

    return test


def benchmark_model(model_key, test_data, args):
    """Benchmark a single model on test data."""
    info = MODEL_REGISTRY[model_key]
    model_id = info["id"]
    model_type = info["type"]
    loader, transcriber = LOADERS[model_type]

    print(f"\n{'='*60}")
    print(f"  {model_key}: {info['desc']}")
    print(f"  Loading {model_id}...")
    print(f"{'='*60}")

    try:
        model_obj = loader(model_id)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    # Warmup (2 dummy inferences)
    dummy = np.zeros(16000, dtype=np.float32)
    for _ in range(2):
        try:
            transcriber(model_obj, dummy, 16000)
        except Exception:
            pass

    rtf_tracker = RTFTracker()
    references, hypotheses, snr_values, dialects = [], [], [], []

    for i, example in enumerate(tqdm(test_data, desc=f"  {model_key}")):
        audio = np.array(example["audio"]["array"], dtype=np.float32)
        sr = example["audio"]["sampling_rate"]
        ref = example.get("cleaned_text", example.get("text", ""))
        duration = len(audio) / sr

        snr_values.append(estimate_snr(audio, sr))
        dialects.append(str(example.get("speaker_dialect", "unknown")).lower())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        try:
            hyp = transcriber(model_obj, audio, sr)
        except Exception as e:
            if i < 3:
                print(f"    Error sample {i}: {e}")
            hyp = ""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        rtf_tracker.record(duration, t1 - t0)
        references.append(normalize_arabic_for_eval(ref))
        hypotheses.append(normalize_arabic_for_eval(hyp))

        if i < 3:
            print(f"    REF: {ref[:80]}")
            print(f"    HYP: {hyp[:80]}")
            print()

    # Compute metrics
    overall = compute_all_metrics(references, hypotheses)
    rtf = rtf_tracker.summary()
    snr_strat = compute_snr_stratified_metrics(references, hypotheses, snr_values)

    # Per-dialect breakdown
    dialect_groups = {}
    for ref, hyp, d in zip(references, hypotheses, dialects):
        if d not in dialect_groups:
            dialect_groups[d] = {"refs": [], "hyps": []}
        dialect_groups[d]["refs"].append(ref)
        dialect_groups[d]["hyps"].append(hyp)

    dialect_metrics = {}
    for d, data in dialect_groups.items():
        if data["refs"]:
            dm = compute_all_metrics(data["refs"], data["hyps"])
            dialect_metrics[d] = {"wer": dm["wer"], "cer": dm["cer"], "count": len(data["refs"])}

    # Print
    print_results(overall, rtf, snr_strat)
    if dialect_metrics:
        print(f"\n  Per-dialect:")
        for d, m in sorted(dialect_metrics.items()):
            print(f"    {d:<15s}  WER: {m['wer']*100:>6.2f}%  CER: {m['cer']*100:>6.2f}%  (n={m['count']})")

    # Clean up GPU memory
    del model_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "model_id": model_id,
        "model_type": model_type,
        "desc": info["desc"],
        "overall": {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in overall.items()},
        "rtf": rtf,
        "snr_stratified": snr_strat,
        "per_dialect": dialect_metrics,
    }


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to benchmark
    if args.all:
        model_keys = list(MODEL_REGISTRY.keys())
    elif args.models:
        model_keys = args.models
    else:
        model_keys = DEFAULT_MODELS

    # Validate model keys
    for k in model_keys:
        if k not in MODEL_REGISTRY:
            print(f"Unknown model: {k}. Available: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    # Dialect filter
    dialect_filter = None
    if args.saudi_only:
        dialect_filter = ["najidi", "hijazi", "khaliji"]

    # Load test data
    test_data = load_test_data(args.max_samples, dialect_filter)

    # Run benchmarks
    print(f"\nBenchmarking {len(model_keys)} models on {len(test_data)} samples")
    results = {}
    for model_key in model_keys:
        result = benchmark_model(model_key, test_data, args)
        if result:
            results[model_key] = result

    # Print comparison table
    print("\n\n" + "=" * 90)
    print("                          BENCHMARK COMPARISON")
    print("=" * 90)
    print(f"  {'Model':<30s} {'WER':>8s} {'CER':>8s} {'RTF':>8s} {'Speed':>8s} {'Type':>12s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

    sorted_results = sorted(results.values(), key=lambda x: x["overall"].get("wer", 999))
    for r in sorted_results:
        wer_str = f"{r['overall']['wer']*100:.2f}%" if r['overall'].get('wer') is not None else "N/A"
        cer_str = f"{r['overall']['cer']*100:.2f}%" if r['overall'].get('cer') is not None else "N/A"
        rtf_str = f"{r['rtf']['overall_rtf']:.3f}"
        spd_str = f"{r['rtf']['speedup']}x"
        print(f"  {r['model_key']:<30s} {wer_str:>8s} {cer_str:>8s} {rtf_str:>8s} {spd_str:>8s} {r['model_type']:>12s}")

    print("=" * 90)

    # Save results
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {results_file}")

    # Recommend best model
    if sorted_results:
        best = sorted_results[0]
        print(f"\nBest model by WER: {best['model_key']} ({best['overall']['wer']*100:.2f}% WER)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Arabic STT models on SADA Saudi test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available models: {', '.join(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to benchmark (default: top 6)")
    parser.add_argument("--all", action="store_true",
                        help="Benchmark ALL models")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max test samples per model")
    parser.add_argument("--saudi_only", action="store_true", default=True,
                        help="Filter for Saudi dialects only")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")

    args = parser.parse_args()
    main(args)
