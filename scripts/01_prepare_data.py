"""
Step 1: Download and prepare Saudi dialect data from SADA dataset.
Uses STREAMING mode to avoid downloading the full 40GB+ dataset.
Filters for Saudi dialects, optimal audio length, and prepares text.

Usage:
    python scripts/01_prepare_data.py --output_dir ./data/saudi_clean
"""

import argparse
import os
import re
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset, Audio, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.arabic_normalizer import normalize_arabic_for_training


# CORRECT field values in SADA22 (verified from dataset)
SAUDI_DIALECT_KEYWORDS = ["najidi", "hijazi", "khaliji"]


def is_saudi_dialect(dialect_str):
    """Check if dialect string matches Saudi dialects."""
    dialect = str(dialect_str).lower()
    return any(kw in dialect for kw in SAUDI_DIALECT_KEYWORDS)


def is_valid_text(text):
    """Check transcript has at least 2 Arabic characters."""
    arabic_chars = re.findall(r'[\u0600-\u06FF]', str(text))
    return len(arabic_chars) >= 2


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Stream SADA dataset (avoids downloading all 40GB+)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Streaming SADA dataset (no full download needed)...")
    print("=" * 60)

    ds = load_dataset("MohamedRashad/SADA22", split="train", streaming=True)

    # -------------------------------------------------------------------------
    # 2. Stream through data, filtering as we go
    # -------------------------------------------------------------------------
    samples = []
    dialect_counts = {}
    skipped_dialect = 0
    skipped_duration = 0
    skipped_text = 0
    total_seen = 0

    print(f"\nFiltering: Saudi dialects, {args.min_duration}-{args.max_duration}s duration, text quality")
    if args.max_samples:
        print(f"Max samples to collect: {args.max_samples}")
    print()

    for example in tqdm(ds, desc="Streaming SADA22"):
        total_seen += 1

        # Count dialects
        dialect = example.get("speaker_dialect", "Unknown")
        dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1

        # Filter 1: Saudi dialect
        if not is_saudi_dialect(dialect):
            skipped_dialect += 1
            continue

        # Filter 2: Audio duration (2-30s)
        audio = example.get("audio")
        if audio is None:
            continue
        audio_array = audio["array"]
        sr = audio["sampling_rate"]
        duration = len(audio_array) / sr
        if not (args.min_duration <= duration <= args.max_duration):
            skipped_duration += 1
            continue

        # Filter 3: Text quality
        text = example.get("cleaned_text", example.get("text", ""))
        if not is_valid_text(text):
            skipped_text += 1
            continue

        # Prepare transcript
        transcript = normalize_arabic_for_training(text)

        # Store sample
        samples.append({
            "audio": audio,
            "transcript": transcript,
            "speaker_dialect": dialect,
            "duration_seconds": round(duration, 2),
        })

        # Progress update every 5000 samples
        if len(samples) % 5000 == 0:
            print(f"  Collected {len(samples)} samples so far (seen {total_seen} total)...")

        # Stop if we have enough
        if args.max_samples and len(samples) >= args.max_samples:
            print(f"\n  Reached max_samples limit ({args.max_samples})")
            break

    print(f"\n{'=' * 60}")
    print(f"Streaming complete!")
    print(f"  Total seen: {total_seen}")
    print(f"  Skipped (wrong dialect): {skipped_dialect}")
    print(f"  Skipped (duration): {skipped_duration}")
    print(f"  Skipped (text quality): {skipped_text}")
    print(f"  Collected: {len(samples)}")
    print(f"{'=' * 60}")

    if len(samples) == 0:
        print("\nERROR: No samples collected! Check dialect filter.")
        print("Dialect distribution:")
        for d, c in sorted(dialect_counts.items(), key=lambda x: -x[1]):
            print(f"  {d}: {c}")
        print("\nTry with --use_all_sada flag to use all dialects.")
        sys.exit(1)

    # If too few Saudi samples, offer to use all
    if len(samples) < 5000 and not args.use_all_sada:
        print(f"\nWARNING: Only {len(samples)} Saudi samples found.")
        print("Dialect distribution:")
        for d, c in sorted(dialect_counts.items(), key=lambda x: -x[1]):
            print(f"  {d}: {c}")
        print("\nConsider re-running with --use_all_sada to use all dialects.")

    # -------------------------------------------------------------------------
    # 3. Convert to HuggingFace Dataset
    # -------------------------------------------------------------------------
    print("\nConverting to HuggingFace Dataset...")
    ds_dict = {key: [s[key] for s in samples] for key in samples[0].keys()}
    saudi_data = Dataset.from_dict(ds_dict)
    saudi_data = saudi_data.cast_column("audio", Audio(sampling_rate=16000))

    # -------------------------------------------------------------------------
    # 4. Compute stats
    # -------------------------------------------------------------------------
    total_seconds = sum(saudi_data["duration_seconds"])
    total_hours = total_seconds / 3600
    avg_duration = total_seconds / len(saudi_data)

    print(f"\nTotal audio duration: {total_hours:.1f} hours")
    print(f"Average sample duration: {avg_duration:.1f} seconds")

    # Saudi dialect breakdown
    saudi_dialect_counts = {}
    for s in samples:
        d = s["speaker_dialect"]
        saudi_dialect_counts[d] = saudi_dialect_counts.get(d, 0) + 1
    print("\nSaudi dialect breakdown:")
    for d, c in sorted(saudi_dialect_counts.items(), key=lambda x: -x[1]):
        print(f"  {d}: {c} ({c/len(samples)*100:.1f}%)")

    # -------------------------------------------------------------------------
    # 5. Split into train/eval
    # -------------------------------------------------------------------------
    eval_size = min(args.eval_samples, int(len(saudi_data) * 0.05))
    dataset_split = saudi_data.train_test_split(test_size=eval_size, seed=42)
    train_data = dataset_split["train"]
    eval_data = dataset_split["test"]

    print(f"\nTrain samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")

    # -------------------------------------------------------------------------
    # 6. Save to disk
    # -------------------------------------------------------------------------
    print(f"\nSaving to {output_dir}...")
    train_data.save_to_disk(str(output_dir / "train"))
    eval_data.save_to_disk(str(output_dir / "eval"))

    # Save metadata
    metadata = {
        "total_samples": len(saudi_data),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "total_hours": round(total_hours, 2),
        "avg_duration_seconds": round(avg_duration, 2),
        "dialect_counts_full": dialect_counts,
        "saudi_dialect_counts": saudi_dialect_counts,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "source": "MohamedRashad/SADA22",
        "streaming": True,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nDone! Dataset saved successfully.")
    print(f"  Train: {output_dir / 'train'}")
    print(f"  Eval:  {output_dir / 'eval'}")
    print(f"  Meta:  {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Saudi dialect STT data")
    parser.add_argument("--output_dir", type=str, default="./data/saudi_clean")
    parser.add_argument("--min_duration", type=float, default=2.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--eval_samples", type=int, default=1000)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to collect (None = all matching)")
    parser.add_argument("--use_all_sada", action="store_true",
                        help="Use all SADA data if Saudi dialect filter is too restrictive")
    args = parser.parse_args()
    main(args)
