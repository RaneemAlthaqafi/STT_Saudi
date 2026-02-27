"""
ASR evaluation metrics: WER, CER, MER, WIL, WIP, RTF, SNR-stratified.

All metrics follow standard definitions:
  WER = (S + D + I) / N           — Word Error Rate (primary)
  CER = (Sc + Dc + Ic) / Nc      — Character Error Rate
  MER = (S + D + I) / (N + I)    — Match Error Rate (bounded [0,1])
  WIL = 1 - (C/N)(C/P)           — Word Information Lost
  WIP = (C/N)(C/P)               — Word Information Preserved
  RTF = inference_time / audio_duration  — Real-Time Factor

Dependencies: pip install jiwer>=3.0
"""

import time
from collections import defaultdict

import numpy as np

try:
    import jiwer
    from jiwer import process_words, process_characters
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


# ──────────────────────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────────────────────

def compute_all_metrics(references: list, hypotheses: list) -> dict:
    """Compute all standard ASR metrics (micro-averaged across corpus)."""
    if not HAS_JIWER:
        return {"error": "jiwer not installed (pip install jiwer>=3.0)"}

    valid = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not valid:
        return {"wer": 1.0, "cer": 1.0, "mer": 1.0, "wil": 1.0, "wip": 0.0,
                "num_valid": 0, "num_skipped": len(references)}

    refs, hyps = zip(*valid)
    refs, hyps = list(refs), list(hyps)

    w = process_words(refs, hyps)
    c = process_characters(refs, hyps)

    S, D, I, C = w.substitutions, w.deletions, w.insertions, w.hits
    N = S + D + C
    P = S + I + C  # hypothesis total

    cS, cD, cI, cC = c.substitutions, c.deletions, c.insertions, c.hits
    cN = cS + cD + cC

    wer_val = (S + D + I) / N if N > 0 else 0.0
    cer_val = (cS + cD + cI) / cN if cN > 0 else 0.0
    mer_val = (S + D + I) / (N + I) if (N + I) > 0 else 0.0
    wil_val = 1.0 - (C / N) * (C / P) if N > 0 and P > 0 else 1.0
    wip_val = 1.0 - wil_val

    return {
        "wer": wer_val, "cer": cer_val, "mer": mer_val,
        "wil": wil_val, "wip": wip_val,
        "word_S": S, "word_D": D, "word_I": I, "word_C": C, "word_N": N,
        "char_S": cS, "char_D": cD, "char_I": cI, "char_C": cC, "char_N": cN,
        "num_valid": len(valid), "num_skipped": len(references) - len(valid),
    }


def compute_per_sample_metrics(reference: str, hypothesis: str) -> dict:
    """Compute WER and CER for a single utterance."""
    if not HAS_JIWER:
        return {"wer": None, "cer": None}
    if not reference.strip():
        return {"wer": 1.0 if hypothesis.strip() else 0.0,
                "cer": 1.0 if hypothesis.strip() else 0.0}

    return {
        "wer": jiwer.wer(reference, hypothesis),
        "cer": jiwer.cer(reference, hypothesis),
    }


# ──────────────────────────────────────────────────────────────
# RTF (Real-Time Factor)
# ──────────────────────────────────────────────────────────────

class RTFTracker:
    """
    Track Real-Time Factor: RTF = processing_time / audio_duration.
    RTF < 1.0 means faster than real-time.

    Usage:
        tracker = RTFTracker()
        # ... inference ...
        tracker.record(audio_duration_sec, inference_time_sec)
        print(tracker.summary())
    """

    def __init__(self):
        self.total_audio = 0.0
        self.total_inference = 0.0
        self.per_sample = []

    def record(self, audio_sec: float, inference_sec: float):
        self.total_audio += audio_sec
        self.total_inference += inference_sec
        if audio_sec > 0:
            self.per_sample.append(inference_sec / audio_sec)

    @property
    def overall_rtf(self):
        return self.total_inference / self.total_audio if self.total_audio > 0 else 0.0

    def summary(self) -> dict:
        arr = np.array(self.per_sample) if self.per_sample else np.array([0.0])
        return {
            "overall_rtf": round(self.overall_rtf, 4),
            "mean_rtf": round(float(arr.mean()), 4),
            "median_rtf": round(float(np.median(arr)), 4),
            "std_rtf": round(float(arr.std()), 4),
            "total_audio_sec": round(self.total_audio, 2),
            "total_inference_sec": round(self.total_inference, 2),
            "speedup": round(1.0 / self.overall_rtf if self.overall_rtf > 0 else 0.0, 2),
            "num_samples": len(self.per_sample),
        }


# ──────────────────────────────────────────────────────────────
# SNR estimation & stratified evaluation
# ──────────────────────────────────────────────────────────────

SNR_BINS = {
    "clean (>=30dB)":     (30, float('inf')),
    "light (20-30dB)":    (20, 30),
    "moderate (10-20dB)": (10, 20),
    "heavy (0-10dB)":     (0, 10),
    "very_noisy (<0dB)":  (float('-inf'), 0),
}


def estimate_snr(audio_array: np.ndarray, sr: int = 16000,
                 frame_len: int = 2048, hop: int = 512) -> float:
    """
    Estimate SNR using energy-based VAD.
    Frames below 20th percentile energy = noise, above = speech.
    """
    n_frames = 1 + (len(audio_array) - frame_len) // hop
    if n_frames <= 0:
        return 30.0

    energies = np.array([
        np.sqrt(np.mean(audio_array[i * hop:i * hop + frame_len] ** 2))
        for i in range(n_frames)
    ])

    threshold = np.percentile(energies, 20)
    speech = energies[energies > threshold]
    noise = energies[energies <= threshold]

    if len(noise) == 0 or np.mean(noise ** 2) < 1e-12:
        return 40.0
    if len(speech) == 0:
        return 0.0

    snr = 10 * np.log10(np.mean(speech ** 2) / (np.mean(noise ** 2) + 1e-10))
    return float(np.clip(snr, -10, 60))


def get_snr_bin(snr_db: float) -> str:
    for label, (lo, hi) in SNR_BINS.items():
        if lo <= snr_db < hi:
            return label
    return "unknown"


def compute_snr_stratified_metrics(references, hypotheses, snr_values):
    """Compute WER/CER broken down by SNR bin."""
    if not HAS_JIWER:
        return {"error": "jiwer not installed"}

    bins = defaultdict(lambda: {"refs": [], "hyps": [], "count": 0})
    for ref, hyp, snr in zip(references, hypotheses, snr_values):
        if not ref.strip():
            continue
        label = get_snr_bin(snr)
        bins[label]["refs"].append(ref)
        bins[label]["hyps"].append(hyp)
        bins[label]["count"] += 1

    results = {}
    for label in SNR_BINS:
        if label in bins and bins[label]["count"] > 0:
            results[label] = {
                "wer": round(jiwer.wer(bins[label]["refs"], bins[label]["hyps"]), 4),
                "cer": round(jiwer.cer(bins[label]["refs"], bins[label]["hyps"]), 4),
                "count": bins[label]["count"],
            }
        else:
            results[label] = {"wer": None, "cer": None, "count": 0}
    return results


def print_results(overall: dict, rtf: dict = None, snr: dict = None):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print("                       EVALUATION RESULTS")
    print("=" * 70)

    if overall.get("wer") is not None:
        print(f"\n  WER  (Word Error Rate):            {overall['wer']*100:.2f}%")
        print(f"  CER  (Character Error Rate):       {overall['cer']*100:.2f}%")
        print(f"  MER  (Match Error Rate):           {overall['mer']*100:.2f}%")
        print(f"  WIL  (Word Information Lost):      {overall['wil']*100:.2f}%")
        print(f"  WIP  (Word Information Preserved): {overall['wip']*100:.2f}%")
        print(f"\n  Word alignment: C={overall['word_C']}  S={overall['word_S']}  "
              f"D={overall['word_D']}  I={overall['word_I']}  N={overall['word_N']}")
        print(f"  Char alignment: C={overall['char_C']}  S={overall['char_S']}  "
              f"D={overall['char_D']}  I={overall['char_I']}  N={overall['char_N']}")
        print(f"  Samples: {overall['num_valid']} valid, {overall['num_skipped']} skipped")

    if rtf:
        print(f"\n  RTF:     {rtf['overall_rtf']:.4f}  ({rtf['speedup']}x real-time)")
        print(f"  Median:  {rtf['median_rtf']:.4f}  Std: {rtf['std_rtf']:.4f}")

    if snr:
        print(f"\n  {'SNR Bin':<25s} {'WER':>8s} {'CER':>8s} {'Count':>6s}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*6}")
        for label, m in snr.items():
            if m["wer"] is not None:
                print(f"  {label:<25s} {m['wer']*100:>7.2f}% {m['cer']*100:>7.2f}% {m['count']:>6d}")
            else:
                print(f"  {label:<25s} {'N/A':>8s} {'N/A':>8s} {m['count']:>6d}")

    print("=" * 70)
