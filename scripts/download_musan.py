"""
Utility: Download the MUSAN noise dataset for noise augmentation.
MUSAN contains music, speech, and noise recordings.

Usage:
    python scripts/download_musan.py --output_dir ./data/musan
"""

import argparse
import os
import subprocess
import tarfile
from pathlib import Path


MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / "musan.tar.gz"

    # Check if already downloaded
    if (output_dir / "musan" / "noise").exists():
        print("MUSAN already downloaded and extracted!")
        count_files(output_dir / "musan")
        return

    # Download
    print(f"Downloading MUSAN dataset (~1.7GB)...")
    print(f"URL: {MUSAN_URL}")
    print(f"Saving to: {tar_path}")

    try:
        subprocess.run(
            ["wget", "-c", MUSAN_URL, "-O", str(tar_path)],
            check=True,
        )
    except FileNotFoundError:
        # wget not available, try curl
        try:
            subprocess.run(
                ["curl", "-L", "-C", "-", MUSAN_URL, "-o", str(tar_path)],
                check=True,
            )
        except FileNotFoundError:
            print("ERROR: Neither wget nor curl found.")
            print(f"Please download manually from: {MUSAN_URL}")
            print(f"Save to: {tar_path}")
            return

    # Extract
    print("\nExtracting...")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(path=str(output_dir))

    # Clean up
    if args.remove_tar:
        os.remove(str(tar_path))
        print("Removed tar.gz file")

    count_files(output_dir / "musan")
    print(f"\nMUSAN dataset ready at: {output_dir / 'musan'}")
    print(f"Use noise files at: {output_dir / 'musan' / 'noise'}")


def count_files(musan_dir):
    """Count files in MUSAN subdirectories."""
    for subdir in ["music", "noise", "speech"]:
        path = musan_dir / subdir
        if path.exists():
            count = sum(1 for _ in path.rglob("*.wav"))
            print(f"  {subdir}: {count} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MUSAN noise dataset")
    parser.add_argument("--output_dir", type=str, default="./data/musan")
    parser.add_argument("--remove_tar", action="store_true", help="Remove tar.gz after extraction")
    args = parser.parse_args()
    main(args)
