"""
Utility: Collect Saudi dialect audio from YouTube.
Downloads audio, segments it, and prepares for transcription.

Usage:
    python scripts/collect_youtube.py \
        --urls_file ./data/youtube_urls.txt \
        --output_dir ./data/youtube_raw \
        --segment_length 20
"""

import argparse
import os
import subprocess
import json
from pathlib import Path

import numpy as np

try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa/soundfile not installed. Install: pip install librosa soundfile")


def download_audio(url, output_dir, max_duration=None):
    """Download audio from YouTube URL using yt-dlp."""
    output_template = str(Path(output_dir) / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",                          # Extract audio
        "--audio-format", "wav",        # WAV format
        "--audio-quality", "0",         # Best quality
        "--no-playlist",                # Single video only
        "-o", output_template,
    ]

    if max_duration:
        cmd.extend(["--match-filter", f"duration < {max_duration}"])

    cmd.append(url)

    print(f"  Downloading: {url}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:200]}")
            return None
        # Find the downloaded file
        for f in Path(output_dir).glob("*.wav"):
            if f.stat().st_mtime > (Path(output_dir).stat().st_mtime - 60):
                return str(f)
    except subprocess.TimeoutExpired:
        print("  ERROR: Download timed out")
    except FileNotFoundError:
        print("  ERROR: yt-dlp not found. Install: pip install yt-dlp")

    return None


def segment_audio(audio_path, output_dir, segment_length=20, overlap=2, sr=16000):
    """Split long audio into segments of specified length."""
    if not HAS_LIBROSA:
        print("  Cannot segment without librosa. Install: pip install librosa soundfile")
        return []

    audio, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    total_duration = len(audio) / sr

    segments = []
    step = segment_length - overlap
    start = 0

    stem = Path(audio_path).stem
    seg_dir = Path(output_dir) / stem
    seg_dir.mkdir(parents=True, exist_ok=True)

    while start < total_duration:
        end = min(start + segment_length, total_duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        segment = audio[start_sample:end_sample]

        # Skip very short segments
        if len(segment) / sr < 2.0:
            break

        # Skip silence
        if np.max(np.abs(segment)) < 0.01:
            start += step
            continue

        seg_path = seg_dir / f"seg_{int(start):06d}_{int(end):06d}.wav"
        sf.write(str(seg_path), segment, sr)
        segments.append({
            "path": str(seg_path),
            "start": start,
            "end": end,
            "duration": end - start,
            "source": audio_path,
        })

        start += step

    return segments


def main(args):
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    segments_dir = output_dir / "segments"
    raw_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Load URLs
    urls = []
    if args.urls_file and os.path.exists(args.urls_file):
        with open(args.urls_file) as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    elif args.url:
        urls = [args.url]
    else:
        print("No URLs provided. Create a file with YouTube URLs (one per line):")
        print(f"  {args.urls_file or './data/youtube_urls.txt'}")
        print("\nExample URLs for Saudi dialect content:")
        print("  # Saudi podcasts/shows - add URLs here")
        print("  # فنجان (Finjan podcast)")
        print("  # ثمانية (Thmanyah)")
        print("  # Saudi news/interviews")
        return

    print(f"Processing {len(urls)} URLs...")

    all_segments = []
    for i, url in enumerate(urls):
        print(f"\n[{i+1}/{len(urls)}] {url}")

        # Download
        audio_path = download_audio(url, str(raw_dir), args.max_video_duration)
        if not audio_path:
            continue

        # Segment
        segments = segment_audio(
            audio_path, str(segments_dir),
            segment_length=args.segment_length,
            overlap=args.overlap,
        )
        all_segments.extend(segments)
        print(f"  Created {len(segments)} segments")

    # Save manifest
    manifest_path = output_dir / "segments_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)

    total_duration = sum(s["duration"] for s in all_segments)
    print(f"\n{'='*60}")
    print(f"Total segments: {len(all_segments)}")
    print(f"Total duration: {total_duration/3600:.1f} hours")
    print(f"Manifest saved: {manifest_path}")
    print(f"\nNext step: Transcribe segments with scripts/transcribe_and_validate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Saudi dialect audio from YouTube")
    parser.add_argument("--url", type=str, default=None, help="Single YouTube URL")
    parser.add_argument("--urls_file", type=str, default="./data/youtube_urls.txt")
    parser.add_argument("--output_dir", type=str, default="./data/youtube_raw")
    parser.add_argument("--segment_length", type=int, default=20, help="Segment length in seconds")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap between segments")
    parser.add_argument("--max_video_duration", type=int, default=7200, help="Max video duration (seconds)")
    args = parser.parse_args()
    main(args)
