"""
Test MasriSwitch-Gemma3n-Transcriber-v1 on public Saudi Arabic data.

Evaluates the model on YOUR use case:
  - Saudi dialect (Najdi, Hijazi, Khaliji)
  - Noisy environments (SNR-stratified)
  - Code-switching (Arabic + English)
  - Various audio durations

Generates a full report with metrics, examples, and comparisons.

Usage on RunPod:
    # Quick test (50 samples, ~10 min)
    python scripts/test_masriswitch.py --max_samples 50

    # Full test (500 samples, ~1 hour)
    python scripts/test_masriswitch.py --max_samples 500

    # Test on your own audio files
    python scripts/test_masriswitch.py --audio_dir /workspace/my_audio

    # Compare with Whisper baseline
    python scripts/test_masriswitch.py --max_samples 100 --compare_whisper
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.arabic_normalizer import normalize_arabic_for_eval
from utils.metrics import (
    compute_all_metrics, compute_per_sample_metrics,
    RTFTracker, estimate_snr, get_snr_bin,
    compute_snr_stratified_metrics, print_results,
)


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

def load_masriswitch():
    """Load MasriSwitch-Gemma3n with Google's processor (known bug fix)."""
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    print("Loading MasriSwitch-Gemma3n-Transcriber-v1...")
    print("  Processor: google/gemma-3n-E4B-it (required — Unsloth bug)")
    print("  Model: oddadmix/MasriSwitch-Gemma3n-Transcriber-v1")

    processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")
    model = Gemma3nForConditionalGeneration.from_pretrained(
        "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    # Report memory
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory: {mem_gb:.1f} GB")

    print("  Model loaded!\n")
    return model, processor


def load_whisper_baseline():
    """Load Whisper-large-v3 for comparison."""
    from transformers import pipeline

    print("Loading Whisper-large-v3 baseline...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    print("  Whisper loaded!\n")
    return pipe


def transcribe_masriswitch(model, processor, audio_array, sr=16000):
    """Transcribe audio with MasriSwitch-Gemma3n."""
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


def transcribe_whisper(pipe, audio_array, sr=16000):
    """Transcribe audio with Whisper."""
    result = pipe(
        {"raw": audio_array, "sampling_rate": sr},
        generate_kwargs={"language": "arabic", "task": "transcribe"},
    )
    return result["text"]


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_sada_test(max_samples, saudi_only=True):
    """Load SADA22 test split with Saudi dialect filter."""
    import re
    from datasets import load_dataset

    print("Loading SADA22 test data from HuggingFace...")
    ds = load_dataset("MohamedRashad/SADA22", split="test")
    print(f"  Total test: {len(ds)}")

    if saudi_only:
        dialects = ["najidi", "hijazi", "khaliji"]
        ds = ds.filter(lambda x: str(x.get("speaker_dialect", "")).lower() in dialects)
        print(f"  Saudi only ({', '.join(dialects)}): {len(ds)}")

    # Duration filter: 2-30s
    def valid_duration(x):
        dur = len(x["audio"]["array"]) / x["audio"]["sampling_rate"]
        return 2.0 <= dur <= 30.0
    ds = ds.filter(valid_duration)
    print(f"  After duration filter (2-30s): {len(ds)}")

    # Text quality filter
    def valid_text(x):
        text = x.get("cleaned_text", x.get("text", ""))
        return len(re.findall(r'[\u0600-\u06FF]', text)) >= 2
    ds = ds.filter(valid_text)
    print(f"  After text quality filter: {len(ds)}")

    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))
        print(f"  Sampled: {max_samples}")

    print()
    return ds


def load_local_audio(audio_dir):
    """Load audio files from local directory."""
    import soundfile as sf

    extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"]
    files = []
    for ext in extensions:
        files.extend(sorted(glob.glob(os.path.join(audio_dir, ext))))
        files.extend(sorted(glob.glob(os.path.join(audio_dir, "**", ext), recursive=True)))

    # Deduplicate
    files = sorted(set(files))
    print(f"Found {len(files)} audio files in {audio_dir}\n")

    samples = []
    for fpath in files:
        try:
            audio, sr = sf.read(fpath)
            # Mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            # Resample to 16kHz
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            audio = audio.astype(np.float32)
            # Truncate to 30s max
            if len(audio) > 30 * 16000:
                audio = audio[:30 * 16000]

            samples.append({
                "file": os.path.basename(fpath),
                "audio": {"array": audio, "sampling_rate": sr},
                "text": "",  # No reference — just transcribe
                "duration": len(audio) / sr,
            })
        except Exception as e:
            print(f"  Skip {os.path.basename(fpath)}: {e}")

    return samples


# ──────────────────────────────────────────────────────────────
# Main test pipeline
# ──────────────────────────────────────────────────────────────

def test_on_sada(model, processor, test_data, args):
    """Run full evaluation on SADA test data."""
    print("=" * 70)
    print("  TESTING MasriSwitch-Gemma3n on SADA Saudi Data")
    print("=" * 70)

    # Warmup
    print("  Warming up (2 dummy inferences)...")
    dummy = np.zeros(16000, dtype=np.float32)
    for _ in range(2):
        try:
            transcribe_masriswitch(model, processor, dummy)
        except Exception:
            pass

    rtf_tracker = RTFTracker()
    references, hypotheses = [], []
    snr_values, dialects, durations = [], [], []
    examples = []

    for i, example in enumerate(tqdm(test_data, desc="  MasriSwitch")):
        audio = np.array(example["audio"]["array"], dtype=np.float32)
        sr = example["audio"]["sampling_rate"]
        ref = example.get("cleaned_text", example.get("text", ""))
        duration = len(audio) / sr
        dialect = str(example.get("speaker_dialect", "unknown")).lower()
        snr = estimate_snr(audio, sr)

        snr_values.append(snr)
        dialects.append(dialect)
        durations.append(duration)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        try:
            hyp = transcribe_masriswitch(model, processor, audio, sr)
        except Exception as e:
            if i < 5:
                print(f"    Error sample {i}: {e}")
            hyp = ""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        rtf_tracker.record(duration, t1 - t0)
        ref_norm = normalize_arabic_for_eval(ref)
        hyp_norm = normalize_arabic_for_eval(hyp)
        references.append(ref_norm)
        hypotheses.append(hyp_norm)

        sample_metrics = compute_per_sample_metrics(ref_norm, hyp_norm)

        # Save examples (first 10 + worst 10 will be selected later)
        examples.append({
            "index": i,
            "ref": ref,
            "hyp": hyp,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "wer": sample_metrics.get("wer", 1.0),
            "cer": sample_metrics.get("cer", 1.0),
            "snr": round(snr, 1),
            "dialect": dialect,
            "duration": round(duration, 1),
            "inference_time": round(t1 - t0, 2),
        })

        # Print first 3
        if i < 3:
            print(f"\n    [{i}] dialect={dialect}  snr={snr:.0f}dB  dur={duration:.1f}s")
            print(f"    REF: {ref[:100]}")
            print(f"    HYP: {hyp[:100]}")
            print(f"    WER: {sample_metrics.get('wer', 0)*100:.1f}%")

    # ── Compute overall metrics ──
    overall = compute_all_metrics(references, hypotheses)
    rtf = rtf_tracker.summary()
    snr_strat = compute_snr_stratified_metrics(references, hypotheses, snr_values)

    # ── Per-dialect breakdown ──
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
            dialect_metrics[d] = {
                "wer": dm["wer"], "cer": dm["cer"], "count": len(data["refs"])
            }

    # ── Per-duration breakdown ──
    dur_bins = {"short (2-5s)": (2, 5), "medium (5-15s)": (5, 15), "long (15-30s)": (15, 30)}
    dur_metrics = {}
    for label, (lo, hi) in dur_bins.items():
        idxs = [i for i, d in enumerate(durations) if lo <= d < hi]
        if idxs:
            dur_refs = [references[i] for i in idxs]
            dur_hyps = [hypotheses[i] for i in idxs]
            dm = compute_all_metrics(dur_refs, dur_hyps)
            dur_metrics[label] = {"wer": dm["wer"], "cer": dm["cer"], "count": len(idxs)}

    # ── Print results ──
    print_results(overall, rtf, snr_strat)

    print(f"\n  Per-dialect:")
    for d, m in sorted(dialect_metrics.items()):
        print(f"    {d:<15s}  WER: {m['wer']*100:>6.2f}%  CER: {m['cer']*100:>6.2f}%  (n={m['count']})")

    print(f"\n  Per-duration:")
    for label, m in dur_metrics.items():
        print(f"    {label:<20s}  WER: {m['wer']*100:>6.2f}%  CER: {m['cer']*100:>6.2f}%  (n={m['count']})")

    # ── Best & Worst examples ──
    examples_sorted = sorted(examples, key=lambda x: x["wer"])
    best_5 = examples_sorted[:5]
    worst_5 = examples_sorted[-5:]

    print(f"\n  {'─'*60}")
    print(f"  BEST 5 transcriptions (lowest WER):")
    for ex in best_5:
        print(f"    WER={ex['wer']*100:>5.1f}%  snr={ex['snr']:>4.0f}dB  {ex['dialect']}")
        print(f"      REF: {ex['ref'][:80]}")
        print(f"      HYP: {ex['hyp'][:80]}")

    print(f"\n  WORST 5 transcriptions (highest WER):")
    for ex in worst_5:
        print(f"    WER={ex['wer']*100:>5.1f}%  snr={ex['snr']:>4.0f}dB  {ex['dialect']}")
        print(f"      REF: {ex['ref'][:80]}")
        print(f"      HYP: {ex['hyp'][:80]}")

    return {
        "model": "MasriSwitch-Gemma3n-Transcriber-v1",
        "overall": {k: round(v, 4) if isinstance(v, float) else v for k, v in overall.items()},
        "rtf": rtf,
        "snr_stratified": snr_strat,
        "per_dialect": dialect_metrics,
        "per_duration": dur_metrics,
        "num_samples": len(references),
        "examples_best5": best_5,
        "examples_worst5": worst_5,
    }


def test_on_local(model, processor, samples, args):
    """Transcribe local audio files (no reference text)."""
    print("=" * 70)
    print("  TRANSCRIBING LOCAL FILES with MasriSwitch-Gemma3n")
    print("=" * 70)

    results = []
    for sample in tqdm(samples, desc="  Transcribing"):
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]

        t0 = time.perf_counter()
        try:
            hyp = transcribe_masriswitch(model, processor, audio, sr)
        except Exception as e:
            hyp = f"[ERROR: {e}]"
        t1 = time.perf_counter()

        snr = estimate_snr(audio, sr)

        result = {
            "file": sample["file"],
            "transcription": hyp,
            "duration_sec": round(sample["duration"], 1),
            "snr_db": round(snr, 1),
            "inference_sec": round(t1 - t0, 2),
        }
        results.append(result)
        print(f"  {sample['file']}: {hyp[:100]}")

    return results


def compare_with_whisper(test_data, masri_results, args):
    """Compare MasriSwitch vs Whisper on same data."""
    print("\n" + "=" * 70)
    print("  COMPARING: MasriSwitch-Gemma3n vs Whisper-large-v3")
    print("=" * 70)

    pipe = load_whisper_baseline()

    # Warmup
    dummy = np.zeros(16000, dtype=np.float32)
    for _ in range(2):
        try:
            transcribe_whisper(pipe, dummy)
        except Exception:
            pass

    rtf_tracker = RTFTracker()
    references, hypotheses = [], []
    snr_values = []

    for example in tqdm(test_data, desc="  Whisper"):
        audio = np.array(example["audio"]["array"], dtype=np.float32)
        sr = example["audio"]["sampling_rate"]
        ref = example.get("cleaned_text", example.get("text", ""))
        duration = len(audio) / sr
        snr = estimate_snr(audio, sr)
        snr_values.append(snr)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        try:
            hyp = transcribe_whisper(pipe, audio, sr)
        except Exception:
            hyp = ""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        rtf_tracker.record(duration, t1 - t0)
        references.append(normalize_arabic_for_eval(ref))
        hypotheses.append(normalize_arabic_for_eval(hyp))

    whisper_overall = compute_all_metrics(references, hypotheses)
    whisper_rtf = rtf_tracker.summary()
    whisper_snr = compute_snr_stratified_metrics(references, hypotheses, snr_values)

    # ── Print comparison ──
    m = masri_results["overall"]
    w = whisper_overall

    print(f"\n  {'─'*60}")
    print(f"  {'Metric':<20s} {'MasriSwitch':>15s} {'Whisper-v3':>15s} {'Winner':>12s}")
    print(f"  {'─'*20} {'─'*15} {'─'*15} {'─'*12}")

    for metric in ["wer", "cer", "mer"]:
        mv = m.get(metric, 0)
        wv = w.get(metric, 0)
        winner = "MasriSwitch" if mv <= wv else "Whisper"
        diff = (wv - mv) * 100
        sign = "+" if diff >= 0 else ""
        print(f"  {metric.upper():<20s} {mv*100:>14.2f}% {wv*100:>14.2f}% {winner:>12s} ({sign}{diff:.1f}%)")

    mr = masri_results["rtf"]
    wr = whisper_rtf
    print(f"  {'RTF':<20s} {mr['overall_rtf']:>15.3f} {wr['overall_rtf']:>15.3f}")
    print(f"  {'Speed':<20s} {mr['speedup']:>14.1f}x {wr['speedup']:>14.1f}x")

    # SNR comparison
    print(f"\n  SNR-stratified WER comparison:")
    print(f"  {'SNR Bin':<25s} {'MasriSwitch':>12s} {'Whisper':>12s} {'Diff':>8s}")
    print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*8}")
    for label in masri_results["snr_stratified"]:
        ms = masri_results["snr_stratified"].get(label, {})
        ws = whisper_snr.get(label, {})
        mw = ms.get("wer")
        ww = ws.get("wer")
        if mw is not None and ww is not None:
            diff = (ww - mw) * 100
            print(f"  {label:<25s} {mw*100:>11.2f}% {ww*100:>11.2f}% {diff:>+7.1f}%")

    print(f"  {'─'*60}\n")

    # Cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "whisper_overall": {k: round(v, 4) if isinstance(v, float) else v for k, v in w.items()},
        "whisper_rtf": wr,
        "whisper_snr": whisper_snr,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    # ── Load model ──
    model, processor = load_masriswitch()

    # ── Mode 1: Test on local audio files ──
    if args.audio_dir:
        samples = load_local_audio(args.audio_dir)
        if not samples:
            print("No audio files found!")
            return

        results = test_on_local(model, processor, samples, args)
        report["mode"] = "local_audio"
        report["transcriptions"] = results

        # Save
        out_file = output_dir / f"local_transcriptions_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {out_file}")
        return

    # ── Mode 2: Test on SADA public data ──
    test_data = load_sada_test(args.max_samples, saudi_only=not args.all_dialects)
    masri_results = test_on_sada(model, processor, test_data, args)
    report["mode"] = "sada_benchmark"
    report["masriswitch"] = masri_results

    # ── Optional: Compare with Whisper ──
    if args.compare_whisper:
        # Free MasriSwitch memory for Whisper
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        whisper_results = compare_with_whisper(test_data, masri_results, args)
        report["whisper_comparison"] = whisper_results

    # ── Save full report ──
    out_file = output_dir / f"masriswitch_test_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull report saved to: {out_file}")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("                    TEST SUMMARY")
    print("=" * 70)
    m = masri_results["overall"]
    print(f"  Model:     MasriSwitch-Gemma3n-Transcriber-v1")
    print(f"  Samples:   {masri_results['num_samples']}")
    print(f"  GPU:       {report['gpu']}")
    print(f"  WER:       {m['wer']*100:.2f}%")
    print(f"  CER:       {m['cer']*100:.2f}%")
    print(f"  RTF:       {masri_results['rtf']['overall_rtf']:.3f} ({masri_results['rtf']['speedup']}x real-time)")
    print(f"  Report:    {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test MasriSwitch-Gemma3n on Saudi Arabic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test on 50 SADA samples
  python scripts/test_masriswitch.py --max_samples 50

  # Full test with Whisper comparison
  python scripts/test_masriswitch.py --max_samples 200 --compare_whisper

  # Test on your own audio files
  python scripts/test_masriswitch.py --audio_dir /workspace/my_audio
        """,
    )
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Max SADA test samples (default: 100)")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory with local audio files to transcribe")
    parser.add_argument("--compare_whisper", action="store_true",
                        help="Also run Whisper-v3 for comparison")
    parser.add_argument("--all_dialects", action="store_true",
                        help="Include all dialects, not just Saudi")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                        help="Output directory for results")

    args = parser.parse_args()
    main(args)
