"""
Step 2: Noise augmentation for Saudi STT training data.
Creates augmented versions of clean data for noise robustness.

Research-backed parameters:
  - MUSAN noise at SNR 5-15 dB  (NVIDIA NeMo standard, SADA paper)
  - Speed perturbation: 0.9 / 1.0 / 1.1  (Povey 2015, industry standard)
  - 50:50 clean-to-noisy ratio  (multi-condition training standard)

Usage:
    python scripts/02_augment_data.py \
        --input_dir ./data/saudi_clean/train \
        --output_dir ./data/saudi_augmented \
        --noise_dir ./data/musan/noise
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk, Dataset, concatenate_datasets
from tqdm import tqdm

try:
    from audiomentations import (
        Compose, AddGaussianSNR, TimeStretch,
        Gain, BandPassFilter, AddBackgroundNoise,
    )
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False
    print("WARNING: audiomentations not installed. Using basic numpy augmentation.")


# ──────────────────────────────────────────────────────────────
# Basic fallback augmentation (when audiomentations is missing)
# ──────────────────────────────────────────────────────────────

def basic_add_noise(audio_array, snr_db=10.0):
    """Add Gaussian noise at a target SNR."""
    signal_power = np.mean(audio_array ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-10)), len(audio_array))
    return (audio_array + noise).astype(np.float32)


def basic_speed_perturb(audio_array, factor):
    """Speed perturbation via linear interpolation."""
    indices = np.arange(0, len(audio_array), factor)
    indices = indices[indices < len(audio_array) - 1].astype(int)
    return audio_array[indices]


# ──────────────────────────────────────────────────────────────
# Audiomentations pipelines (research-backed)
# ──────────────────────────────────────────────────────────────

def create_augmentation_pipeline(noise_dir=None):
    """Create a single research-backed augmentation pipeline.

    SNR 5-15 dB: the standard range from NVIDIA NeMo configs and the
    SADA ASR paper.  This is the sweet spot — noisy enough to teach
    robustness, clean enough that the transcript is still valid.
    """
    if not HAS_AUDIOMENTATIONS:
        return None

    transforms = [
        # Core: additive noise at research-backed SNR range
        AddGaussianSNR(min_snr_db=5.0, max_snr_db=15.0, p=0.7),
        # Mild gain variation (microphone distance / volume)
        Gain(min_gain_db=-4, max_gain_db=4, p=0.3),
        # Telephone-band simulation (occasional)
        BandPassFilter(min_center_freq=200, max_center_freq=3500, p=0.15),
    ]

    # Real noise from MUSAN (preferred over synthetic Gaussian)
    if noise_dir and Path(noise_dir).is_dir():
        transforms.insert(0, AddBackgroundNoise(
            sounds_path=str(noise_dir),
            min_snr_db=5.0,
            max_snr_db=15.0,
            p=0.6,
        ))

    return Compose(transforms)


# ──────────────────────────────────────────────────────────────
# Speed perturbation (applied separately, changes duration)
# ──────────────────────────────────────────────────────────────

SPEED_FACTORS = [0.9, 1.0, 1.1]  # Povey 2015 standard


def apply_speed_perturbation(audio_array, sr):
    """Randomly apply one of the standard speed factors."""
    factor = np.random.choice(SPEED_FACTORS)
    if factor == 1.0:
        return audio_array

    if HAS_AUDIOMENTATIONS:
        ts = TimeStretch(min_rate=factor, max_rate=factor, p=1.0)
        return ts(samples=audio_array, sample_rate=sr)
    else:
        return basic_speed_perturb(audio_array, factor)


# ──────────────────────────────────────────────────────────────
# Main augmentation loop
# ──────────────────────────────────────────────────────────────

def augment_dataset(dataset, pipeline, args):
    """Create augmented copies.  Returns list of dicts."""
    augmented = []

    for example in tqdm(dataset, desc="Augmenting"):
        audio = example["audio"]
        audio_array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

        # Decide whether to augment this sample
        if np.random.random() >= args.augment_ratio:
            continue

        # 1. Speed perturbation (30% chance)
        if np.random.random() < args.speed_perturb_prob:
            audio_array = apply_speed_perturbation(audio_array, sr)

        # 2. Noise / distortion pipeline
        if pipeline is not None:
            audio_array = pipeline(samples=audio_array, sample_rate=sr)
        else:
            snr = np.random.uniform(5.0, 15.0)
            audio_array = basic_add_noise(audio_array, snr_db=snr)

        audio_array = np.clip(audio_array, -1.0, 1.0)

        augmented.append({
            "audio": {"array": audio_array, "sampling_rate": sr},
            "transcript": example["transcript"],
            "duration_seconds": example.get("duration_seconds", 0),
            "augmentation": "noisy",
        })

    return augmented


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load clean data ────────────────────────────────────
    print("Loading clean training data...")
    train_data = load_from_disk(args.input_dir)
    print(f"Clean samples: {len(train_data)}")

    # ── 2. Build pipeline ─────────────────────────────────────
    print("\nSetting up augmentation...")
    pipeline = create_augmentation_pipeline(args.noise_dir)

    if HAS_AUDIOMENTATIONS:
        print("  Using audiomentations library")
    else:
        print("  Using basic numpy augmentation (pip install audiomentations for better results)")

    if args.noise_dir and Path(args.noise_dir).is_dir():
        print(f"  Real noise from: {args.noise_dir}")
    else:
        print("  No real noise directory — using synthetic Gaussian noise")

    # ── 3. Augment ────────────────────────────────────────────
    print(f"\nAugmentation settings:")
    print(f"  Augment ratio:       {args.augment_ratio}  (fraction of samples to augment)")
    print(f"  Speed perturb prob:  {args.speed_perturb_prob}")
    print(f"  SNR range:           5-15 dB (research standard)")
    print(f"  Target ratio:        ~50:50 clean:noisy")

    augmented_samples = augment_dataset(train_data, pipeline, args)
    print(f"\nAugmented samples created: {len(augmented_samples)}")

    # ── 4. Combine clean + augmented ──────────────────────────
    if augmented_samples:
        aug_dataset = Dataset.from_list(augmented_samples)

        # Tag originals as clean
        original = train_data.map(lambda x: {"augmentation": "clean"})

        common_cols = ["audio", "transcript", "duration_seconds", "augmentation"]
        original_sub = original.select_columns(
            [c for c in common_cols if c in original.column_names]
        )
        aug_sub = aug_dataset.select_columns(
            [c for c in common_cols if c in aug_dataset.column_names]
        )

        combined = concatenate_datasets([original_sub, aug_sub]).shuffle(seed=42)
    else:
        combined = train_data

    print(f"\nFinal dataset: {len(combined)} samples")

    # Distribution
    if "augmentation" in combined.column_names:
        counts = {}
        for t in combined["augmentation"]:
            counts[t] = counts.get(t, 0) + 1
        print("Distribution:")
        for t, c in sorted(counts.items()):
            pct = c / len(combined) * 100
            print(f"  {t}: {c} ({pct:.0f}%)")

    # ── 5. Save ───────────────────────────────────────────────
    print(f"\nSaving to {output_dir}...")
    combined.save_to_disk(str(output_dir / "train_augmented"))

    metadata = {
        "original_samples": len(train_data),
        "augmented_samples": len(augmented_samples),
        "total_samples": len(combined),
        "augment_ratio": args.augment_ratio,
        "speed_perturb_prob": args.speed_perturb_prob,
        "snr_range_db": "5-15",
        "speed_factors": SPEED_FACTORS,
        "used_audiomentations": HAS_AUDIOMENTATIONS,
        "noise_dir": args.noise_dir,
    }
    with open(output_dir / "augmentation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"  Data: {output_dir / 'train_augmented'}")
    print(f"  Meta: {output_dir / 'augmentation_metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment Saudi STT data with noise")
    parser.add_argument("--input_dir", type=str, default="./data/saudi_clean/train")
    parser.add_argument("--output_dir", type=str, default="./data/saudi_augmented")
    parser.add_argument("--noise_dir", type=str, default=None,
                        help="Path to noise files (e.g., ./data/musan/noise)")
    parser.add_argument("--augment_ratio", type=float, default=1.0,
                        help="Fraction of samples to create noisy copies for (1.0 = all → 50:50 ratio)")
    parser.add_argument("--speed_perturb_prob", type=float, default=0.3,
                        help="Probability of applying speed perturbation per sample")
    args = parser.parse_args()
    main(args)
