"""
Segmentation layer for Saudi STT.

Mandatory before training AND inference:
  - Chunk audio into 15-25 second windows
  - 1-2 second overlap to avoid word cut-off at boundaries
  - Skip chunks shorter than 3 seconds
  - Merge transcripts after decoding

Usage (standalone):
    from scripts.segment import segment_audio, merge_transcripts

Usage (CLI):
    python scripts/segment.py --input audio.wav --output_dir ./chunks
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────
# Core segmentation
# ──────────────────────────────────────────────────────────────

def segment_audio(
    audio_array: np.ndarray,
    sr: int = 16000,
    chunk_sec: float = 20.0,
    overlap_sec: float = 1.5,
    min_chunk_sec: float = 3.0,
) -> List[Dict]:
    """
    Split audio array into overlapping chunks for STT processing.

    Args:
        audio_array:   1D float32 array, any length
        sr:            sample rate (default 16000)
        chunk_sec:     target chunk length in seconds (default 20s)
        overlap_sec:   overlap between chunks in seconds (default 1.5s)
        min_chunk_sec: skip chunks shorter than this (default 3s)

    Returns:
        List of dicts with keys:
          - audio:      np.ndarray (the chunk)
          - start_sec:  float (start time in original)
          - end_sec:    float (end time in original)
          - index:      int
    """
    total_samples = len(audio_array)
    total_sec = total_samples / sr

    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step_samples = chunk_samples - overlap_samples
    min_samples = int(min_chunk_sec * sr)

    # Short audio: return as single chunk if long enough
    if total_samples <= chunk_samples:
        if total_samples >= min_samples:
            return [{
                "audio": audio_array.astype(np.float32),
                "start_sec": 0.0,
                "end_sec": round(total_sec, 3),
                "index": 0,
            }]
        else:
            return []  # Too short — skip

    chunks = []
    start = 0
    idx = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio_array[start:end]

        if len(chunk) >= min_samples:
            chunks.append({
                "audio": chunk.astype(np.float32),
                "start_sec": round(start / sr, 3),
                "end_sec": round(end / sr, 3),
                "index": idx,
            })
            idx += 1

        start += step_samples

    return chunks


def merge_transcripts(chunks: List[Dict]) -> str:
    """
    Merge transcript strings from decoded chunks.

    Simple concatenation with space. Overlap regions may produce
    repeated words at boundaries — this is acceptable for current
    phase. A deduplication pass can be added later.

    Args:
        chunks: list of dicts with 'transcript' key

    Returns:
        Single merged transcript string
    """
    parts = [c["transcript"].strip() for c in chunks if c.get("transcript", "").strip()]
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────
# Dataset-level filter — used in training data prep
# ──────────────────────────────────────────────────────────────

def filter_by_duration(
    audio_array: np.ndarray,
    sr: int,
    min_sec: float = 5.0,
    max_sec: float = 30.0,
) -> bool:
    """
    Return True if audio duration is within [min_sec, max_sec].
    Used to filter training samples before loading into trainer.
    """
    duration = len(audio_array) / sr
    return min_sec <= duration <= max_sec


def apply_duration_filter(dataset, min_sec: float = 5.0, max_sec: float = 30.0):
    """
    Filter a HuggingFace Dataset to only keep samples within duration range.
    Skips samples shorter than min_sec (bodycam short clips problem).
    """
    def _check(example):
        audio = example["audio"]
        arr = np.array(audio["array"])
        sr = audio["sampling_rate"]
        return filter_by_duration(arr, sr, min_sec, max_sec)

    before = len(dataset)
    dataset = dataset.filter(_check)
    after = len(dataset)
    print(f"Duration filter ({min_sec}-{max_sec}s): {before} → {after} samples "
          f"(removed {before - after})")
    return dataset


# ──────────────────────────────────────────────────────────────
# CLI: split a wav file into chunks and save
# ──────────────────────────────────────────────────────────────

def main(args):
    import soundfile as sf

    audio_array, sr = sf.read(args.input, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # stereo → mono

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        sr = 16000

    chunks = segment_audio(
        audio_array,
        sr=sr,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        min_chunk_sec=args.min_chunk_sec,
    )

    print(f"Input:  {args.input}")
    print(f"Length: {len(audio_array)/sr:.1f}s")
    print(f"Chunks: {len(chunks)} × ~{args.chunk_sec}s (overlap {args.overlap_sec}s)")

    if not chunks:
        print("ERROR: No valid chunks produced. Audio may be too short.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.input).stem
    for chunk in chunks:
        fname = output_dir / f"{stem}_chunk{chunk['index']:03d}_{chunk['start_sec']:.1f}-{chunk['end_sec']:.1f}.wav"
        sf.write(str(fname), chunk["audio"], sr)

    print(f"\nSaved {len(chunks)} chunks to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment audio into overlapping chunks")
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output_dir", default="./chunks", help="Output directory")
    parser.add_argument("--chunk_sec", type=float, default=20.0, help="Chunk length in seconds")
    parser.add_argument("--overlap_sec", type=float, default=1.5, help="Overlap in seconds")
    parser.add_argument("--min_chunk_sec", type=float, default=3.0, help="Minimum chunk length")
    args = parser.parse_args()
    main(args)
