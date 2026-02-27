"""
Step 1: Download and prepare Saudi dialect data from SADA dataset.
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


def filter_by_length(example, min_duration=2.0, max_duration=30.0):
    """Filter audio samples by duration."""
    audio = example["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    return min_duration <= duration <= max_duration


def filter_by_text_quality(example):
    """Filter out samples with very short or empty transcripts."""
    text = example.get("cleaned_text", example.get("text", ""))
    # Must have at least 2 Arabic characters
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    return len(arabic_chars) >= 2


def compute_duration(example):
    """Add duration field to example."""
    audio = example["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    return {"duration_seconds": round(duration, 2)}


def prepare_transcript(example):
    """Prepare transcript from SADA's cleaned_text field."""
    text = example.get("cleaned_text", example.get("text", ""))
    text = normalize_arabic_for_training(text)
    return {"transcript": text}


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load SADA dataset
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Loading SADA dataset (this may take a while on first run)...")
    print("=" * 60)

    sada = load_dataset("MohamedRashad/SADA22", split="train")
    print(f"Total SADA training samples: {len(sada)}")

    # -------------------------------------------------------------------------
    # 2. Check available dialects
    # -------------------------------------------------------------------------
    print("\nDialect distribution:")
    dialect_counts = {}
    for example in tqdm(sada, desc="Counting dialects"):
        dialect = example.get("speaker_dialect", "Unknown")
        dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1

    for dialect, count in sorted(dialect_counts.items(), key=lambda x: -x[1]):
        print(f"  {dialect}: {count}")

    # -------------------------------------------------------------------------
    # 3. Filter for Saudi dialects
    # -------------------------------------------------------------------------
    # CORRECT field values in SADA22 (verified from dataset)
    saudi_dialect_keywords = [
        "najidi", "hijazi", "khaliji",  # actual SADA22 field values
    ]

    def is_saudi_dialect(example):
        dialect = str(example.get("speaker_dialect", "")).lower()
        return any(kw in dialect for kw in saudi_dialect_keywords)

    print("\nFiltering for Saudi dialects...")
    saudi_data = sada.filter(is_saudi_dialect)
    print(f"Saudi dialect samples: {len(saudi_data)}")

    # If Saudi-specific filter is too restrictive, fall back to all data
    if len(saudi_data) < 5000:
        print(f"\nWARNING: Only {len(saudi_data)} Saudi samples found.")
        print("Consider using all SADA data (it's all Saudi-sourced content).")
        if args.use_all_sada:
            saudi_data = sada
            print(f"Using all SADA data: {len(saudi_data)} samples")

    # -------------------------------------------------------------------------
    # 4. Filter by audio duration (2-30 seconds)
    # -------------------------------------------------------------------------
    print("\nFiltering by audio duration (2-30 seconds)...")
    saudi_data = saudi_data.filter(
        lambda x: filter_by_length(x, args.min_duration, args.max_duration)
    )
    print(f"After duration filter: {len(saudi_data)}")

    # -------------------------------------------------------------------------
    # 5. Filter by text quality
    # -------------------------------------------------------------------------
    print("\nFiltering by text quality...")
    saudi_data = saudi_data.filter(filter_by_text_quality)
    print(f"After text quality filter: {len(saudi_data)}")

    # -------------------------------------------------------------------------
    # 6. Prepare transcripts
    # -------------------------------------------------------------------------
    print("\nPreparing transcripts...")
    saudi_data = saudi_data.map(prepare_transcript)
    saudi_data = saudi_data.map(compute_duration)

    # -------------------------------------------------------------------------
    # 7. Compute total duration
    # -------------------------------------------------------------------------
    total_seconds = sum(saudi_data["duration_seconds"])
    total_hours = total_seconds / 3600
    print(f"\nTotal audio duration: {total_hours:.1f} hours")
    print(f"Average sample duration: {total_seconds / len(saudi_data):.1f} seconds")

    # -------------------------------------------------------------------------
    # 8. Split into train/eval
    # -------------------------------------------------------------------------
    dataset_split = saudi_data.train_test_split(
        test_size=min(args.eval_samples, int(len(saudi_data) * 0.05)),
        seed=42,
    )
    train_data = dataset_split["train"]
    eval_data = dataset_split["test"]

    print(f"\nTrain samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")

    # -------------------------------------------------------------------------
    # 9. Save to disk
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
        "dialect_counts": dialect_counts,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "source": "MohamedRashad/SADA22",
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
    parser.add_argument("--use_all_sada", action="store_true",
                        help="Use all SADA data if Saudi dialect filter is too restrictive")
    args = parser.parse_args()
    main(args)
