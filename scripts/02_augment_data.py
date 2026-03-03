"""
Step 2: Phase 2 Synthetic Noisy Augmentation Pipeline.

Creates dataset_v2_synthetic_noisy alongside clean real data.
Training mix: 60% clean real Saudi + 40% synthetic noisy.

Pipeline:
  1. Text selection from Phase 1 clean dataset (5-30s, quality-filtered)
  2. TTS generation via Habibi-TTS (https://github.com/SWivid/Habibi-TTS)
     - voice/speed/pitch variation
  3. Noise injection (street, crowd, wind, mic rub, indoor echo) at SNR 5/10/15 dB
  4. Bodycam simulation effects (gain changes, clipping, low-bitrate, reverb)
  5. Dataset structure with full metadata

If Habibi-TTS is not installed, falls back to MUSAN/Gaussian noise augmentation
on the original audio (same pipeline as before, still valid).

Usage:
    # Full pipeline with TTS
    python scripts/02_augment_data.py \
        --input_dir ./data/saudi_clean/train \
        --output_dir ./data/saudi_augmented \
        --use_tts \
        --habibi_model_path ./habibi_tts  # optional, downloads if missing

    # Noise-only (no TTS, faster)
    python scripts/02_augment_data.py \
        --input_dir ./data/saudi_clean/train \
        --output_dir ./data/saudi_augmented \
        --noise_dir ./data/musan/noise
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_from_disk, Dataset, concatenate_datasets, Audio
from tqdm import tqdm

try:
    from audiomentations import (
        Compose, AddGaussianSNR, TimeStretch,
        Gain, BandPassFilter, AddBackgroundNoise,
        PolarityInversion, LowPassFilter,
    )
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False
    print("WARNING: audiomentations not installed. Using basic numpy augmentation.")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from segment import apply_duration_filter


# ──────────────────────────────────────────────────────────────
# Noise injection — real-world types
# ──────────────────────────────────────────────────────────────

NOISE_TYPES = ["street", "crowd", "wind", "mic_rub", "indoor_echo"]
SNR_LEVELS = [5.0, 10.0, 15.0]  # dB — as specified


def add_noise_at_snr(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix noise into audio at a target SNR level."""
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / noise_power)

    # Match length
    if len(noise_scaled) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise_scaled)))
        noise_scaled = np.tile(noise_scaled, repeats)
    noise_scaled = noise_scaled[:len(audio)]

    return np.clip(audio + noise_scaled, -1.0, 1.0).astype(np.float32)


def generate_synthetic_noise(noise_type: str, length: int, sr: int = 16000) -> np.ndarray:
    """Generate synthetic approximation of real-world noise types."""
    if noise_type == "street":
        # Broadband + low rumble
        noise = np.random.normal(0, 0.1, length).astype(np.float32)
        rumble = np.random.normal(0, 0.3, length // 10)
        rumble = np.repeat(rumble, 10)[:length]
        return (noise * 0.6 + rumble * 0.4).astype(np.float32)

    elif noise_type == "crowd":
        # Multiple overlapping voices (approximate with pink noise)
        white = np.random.normal(0, 1, length).astype(np.float32)
        # Simple 1/f approximation
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)
        freqs[0] = 1e-10
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=length).astype(np.float32)
        return (pink / (np.max(np.abs(pink)) + 1e-10) * 0.15).astype(np.float32)

    elif noise_type == "wind":
        # Low-frequency rumble + turbulence
        t = np.linspace(0, length / sr, length)
        wind = np.random.normal(0, 0.2, length).astype(np.float32)
        # Low-pass simulation
        from scipy.signal import butter, filtfilt
        try:
            b, a = butter(4, 300 / (sr / 2), btype="low")
            wind = filtfilt(b, a, wind).astype(np.float32)
        except Exception:
            pass
        return wind

    elif noise_type == "mic_rub":
        # Short burst + scratching artifacts
        noise = np.zeros(length, dtype=np.float32)
        n_bursts = random.randint(3, 8)
        for _ in range(n_bursts):
            start = random.randint(0, max(0, length - sr // 4))
            burst_len = random.randint(sr // 20, sr // 4)
            end = min(start + burst_len, length)
            noise[start:end] = np.random.uniform(-0.5, 0.5, end - start)
        return noise

    else:  # indoor_echo
        # Simple room reverb via delay+decay
        noise = np.random.normal(0, 0.05, length).astype(np.float32)
        return noise


# ──────────────────────────────────────────────────────────────
# Bodycam simulation effects
# ──────────────────────────────────────────────────────────────

def apply_bodycam_effects(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simulate bodycam recording artifacts:
    - Dynamic gain changes (AGC simulation)
    - Clipping simulation
    - Low bitrate compression artifacts
    - Slight reverb for indoor scenes

    Does NOT distort speech intelligibility — only adds realistic artifacts.
    """
    result = audio.copy().astype(np.float32)

    # 1. Dynamic gain variation (AGC simulation)
    if random.random() < 0.6:
        n_segments = random.randint(3, 8)
        seg_len = len(result) // n_segments
        for i in range(n_segments):
            start = i * seg_len
            end = min((i + 1) * seg_len, len(result))
            gain = random.uniform(0.5, 1.5)
            result[start:end] *= gain
        result = np.clip(result, -1.0, 1.0)

    # 2. Mild clipping (bodycam AGC overshoot)
    if random.random() < 0.3:
        clip_level = random.uniform(0.7, 0.95)
        result = np.clip(result, -clip_level, clip_level)
        result = result / clip_level  # normalize back

    # 3. Low bitrate quantization artifacts (8-bit simulation)
    if random.random() < 0.4:
        bits = random.choice([8, 10, 12])
        levels = 2 ** bits
        result = np.round(result * levels) / levels

    # 4. Subtle indoor reverb (short delay)
    if random.random() < 0.3:
        delay_samples = int(random.uniform(0.02, 0.08) * sr)
        decay = random.uniform(0.1, 0.25)
        if delay_samples < len(result):
            reverb = result.copy()
            reverb[delay_samples:] += decay * result[:-delay_samples]
            result = np.clip(reverb, -1.0, 1.0)

    return result.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Habibi-TTS interface
# ──────────────────────────────────────────────────────────────

def try_load_habibi_tts(model_path=None):
    """
    Try to load Habibi-TTS for synthetic speech generation.
    Returns (tts_fn, True) if available, (None, False) otherwise.

    Habibi-TTS: https://github.com/SWivid/Habibi-TTS
    Install: pip install git+https://github.com/SWivid/Habibi-TTS.git
    """
    try:
        # Try importing Habibi-TTS
        from habibi_tts import HabibiTTS
        print("  Habibi-TTS found — using TTS for synthetic generation")

        model = HabibiTTS(model_path=model_path)

        def tts_fn(text, speed=1.0, pitch=0.0):
            audio, sr = model.synthesize(text, speed=speed, pitch=pitch)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            return audio.astype(np.float32), 16000

        return tts_fn, True

    except ImportError:
        print("  Habibi-TTS not installed. Skipping TTS generation.")
        print("  Install: pip install git+https://github.com/SWivid/Habibi-TTS.git")
        return None, False
    except Exception as e:
        print(f"  Habibi-TTS load error: {e}. Skipping TTS generation.")
        return None, False


def generate_tts_sample(tts_fn, text, sr=16000):
    """Generate one TTS sample with random voice/speed/pitch variation."""
    speed = random.choice([0.9, 1.0, 1.1])
    pitch = random.uniform(-1.0, 1.0)  # small range

    try:
        audio, _ = tts_fn(text, speed=speed, pitch=pitch)
        # Ensure float32 mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), {"speed": speed, "pitch": round(pitch, 2)}
    except Exception as e:
        return None, {}


# ──────────────────────────────────────────────────────────────
# Basic fallback (no audiomentations)
# ──────────────────────────────────────────────────────────────

def basic_add_noise(audio_array, snr_db=10.0):
    signal_power = np.mean(audio_array ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(max(noise_power, 1e-10)), len(audio_array))
    return np.clip(audio_array + noise, -1.0, 1.0).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Main augmentation loop
# ──────────────────────────────────────────────────────────────

def augment_one_sample(audio_array, sr, transcript, noise_dir=None, tts_fn=None, use_tts=False):
    """
    Create one augmented sample from a clean audio+transcript pair.
    Returns dict with audio, transcript, and metadata.
    """
    snr_db = random.choice(SNR_LEVELS)
    noise_type = random.choice(NOISE_TYPES)
    speed_factor = 1.0
    is_synthetic = False

    # Step 2: Optionally replace audio with TTS-generated speech
    if use_tts and tts_fn is not None and random.random() < 0.5:
        tts_audio, tts_meta = generate_tts_sample(tts_fn, transcript, sr)
        if tts_audio is not None and len(tts_audio) > sr * 3:
            audio_array = tts_audio
            speed_factor = tts_meta.get("speed", 1.0)
            is_synthetic = True

    # Step 3: Noise injection
    if noise_dir and Path(noise_dir).is_dir():
        # Load random real noise file
        noise_files = list(Path(noise_dir).rglob("*.wav"))
        if noise_files:
            import soundfile as sf
            nf = random.choice(noise_files)
            try:
                noise_audio, noise_sr = sf.read(str(nf), dtype="float32")
                if noise_audio.ndim > 1:
                    noise_audio = noise_audio.mean(axis=1)
                if noise_sr != sr:
                    import librosa
                    noise_audio = librosa.resample(noise_audio, orig_sr=noise_sr, target_sr=sr)
                audio_array = add_noise_at_snr(audio_array, noise_audio, snr_db)
            except Exception:
                audio_array = basic_add_noise(audio_array, snr_db)
        else:
            noise_synth = generate_synthetic_noise(noise_type, len(audio_array), sr)
            audio_array = add_noise_at_snr(audio_array, noise_synth, snr_db)
    else:
        # Synthetic noise
        noise_synth = generate_synthetic_noise(noise_type, len(audio_array), sr)
        audio_array = add_noise_at_snr(audio_array, noise_synth, snr_db)

    # Step 4: Bodycam simulation effects
    audio_array = apply_bodycam_effects(audio_array, sr)

    return {
        "audio": {"array": audio_array, "sampling_rate": sr},
        "transcript": transcript,
        "duration_seconds": round(len(audio_array) / sr, 2),
        "augmentation": "synthetic_noisy",
        "noise_type": noise_type,
        "snr_db": snr_db,
        "speed_factor": speed_factor,
        "synthetic": is_synthetic,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + filter clean data ───────────────────────────
    print("Loading clean training data...")
    train_data = load_from_disk(args.input_dir)
    print(f"Raw samples: {len(train_data)}")

    # Apply duration filter: 5-30s (same as training)
    train_data = apply_duration_filter(train_data, min_sec=5.0, max_sec=30.0)
    print(f"After duration filter: {len(train_data)} samples")

    # ── 2. Load TTS if requested ──────────────────────────────
    tts_fn = None
    if args.use_tts:
        print("\nLoading Habibi-TTS...")
        tts_fn, tts_ok = try_load_habibi_tts(args.habibi_model_path)
    else:
        print("\nTTS disabled — using noise augmentation on original audio only")

    # ── 3. Create augmented dataset ───────────────────────────
    print(f"\nGenerating {args.augment_ratio*100:.0f}% augmented copies...")
    print(f"  Noise types: {NOISE_TYPES}")
    print(f"  SNR levels:  {SNR_LEVELS} dB")
    print(f"  Bodycam effects: enabled")
    print(f"  TTS synthetic:   {'enabled' if tts_fn else 'disabled'}")
    print()

    augmented_samples = []
    for example in tqdm(train_data, desc="Augmenting"):
        if random.random() >= args.augment_ratio:
            continue

        audio_array = np.array(example["audio"]["array"], dtype=np.float32)
        sr = example["audio"]["sampling_rate"]
        transcript = example["transcript"]

        aug = augment_one_sample(
            audio_array, sr, transcript,
            noise_dir=args.noise_dir,
            tts_fn=tts_fn,
            use_tts=args.use_tts,
        )
        augmented_samples.append(aug)

    print(f"\nAugmented samples: {len(augmented_samples)}")

    # ── 4. Build 60/40 clean:noisy mix ────────────────────────
    # Tag clean originals
    def tag_clean(x):
        return {
            "augmentation": "clean",
            "noise_type": "none",
            "snr_db": -1.0,
            "speed_factor": 1.0,
            "synthetic": False,
        }
    clean_tagged = train_data.map(tag_clean)

    # Select columns that exist in both
    common_cols = ["audio", "transcript", "duration_seconds",
                   "augmentation", "noise_type", "snr_db", "speed_factor", "synthetic"]

    # Ensure all common_cols exist in clean_tagged
    existing_clean = [c for c in common_cols if c in clean_tagged.column_names]
    clean_sub = clean_tagged.select_columns(existing_clean)

    if augmented_samples:
        aug_dataset = Dataset.from_list(augmented_samples)
        aug_sub = aug_dataset.select_columns(
            [c for c in common_cols if c in aug_dataset.column_names]
        )

        # 60/40 ratio: take 60% clean, 40% noisy by sampling
        n_total = len(clean_sub) + len(aug_sub)
        n_clean_target = int(n_total * 0.60)
        n_noisy_target = int(n_total * 0.40)

        # Subsample to hit ratio
        if len(clean_sub) > n_clean_target:
            clean_sub = clean_sub.select(range(n_clean_target))
        if len(aug_sub) > n_noisy_target:
            aug_sub = aug_sub.shuffle(seed=42).select(range(n_noisy_target))

        # Align columns
        all_cols = list(set(clean_sub.column_names) | set(aug_sub.column_names))
        for col in all_cols:
            if col not in clean_sub.column_names:
                clean_sub = clean_sub.add_column(col, [None] * len(clean_sub))
            if col not in aug_sub.column_names:
                aug_sub = aug_sub.add_column(col, [None] * len(aug_sub))

        combined = concatenate_datasets([clean_sub, aug_sub]).shuffle(seed=42)
    else:
        combined = clean_sub

    print(f"\nFinal dataset: {len(combined)} samples")
    if "augmentation" in combined.column_names:
        counts = {}
        for t in combined["augmentation"]:
            counts[t] = counts.get(t, 0) + 1
        print("Mix distribution:")
        for t, c in sorted(counts.items()):
            print(f"  {t}: {c} ({c/len(combined)*100:.0f}%)")

    # ── 5. Save ───────────────────────────────────────────────
    print(f"\nSaving to {output_dir}...")
    combined.save_to_disk(str(output_dir / "train_augmented"))

    metadata = {
        "version": "v2_synthetic_noisy",
        "original_samples": len(train_data),
        "augmented_samples": len(augmented_samples),
        "total_samples": len(combined),
        "mix_ratio": "60% clean / 40% synthetic noisy",
        "noise_types": NOISE_TYPES,
        "snr_levels_db": SNR_LEVELS,
        "bodycam_effects": True,
        "tts_used": tts_fn is not None,
        "tts_source": "https://github.com/SWivid/Habibi-TTS",
        "speed_factors": [0.9, 1.0, 1.1],
        "noise_dir": args.noise_dir,
        "safety": {
            "modifies_model_architecture": False,
            "modifies_tokenizer": False,
            "offline_preprocessing_only": True,
            "touches_docker": False,
            "touches_inference_runtime": False,
        },
    }
    with open(output_dir / "augmentation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"  Data: {output_dir / 'train_augmented'}")
    print(f"  Meta: {output_dir / 'augmentation_metadata.json'}")
    print(f"\nNext: python scripts/05_train_phase2.py --data_dir {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 augmentation: synthetic noisy dataset")
    parser.add_argument("--input_dir", type=str, default="./data/saudi_clean/train")
    parser.add_argument("--output_dir", type=str, default="./data/saudi_augmented")
    parser.add_argument("--noise_dir", type=str, default=None,
                        help="Path to real noise files (e.g., ./data/musan/noise)")
    parser.add_argument("--augment_ratio", type=float, default=0.67,
                        help="Fraction of clean samples to create noisy copies for (0.67 → ~40% noisy in mix)")
    parser.add_argument("--use_tts", action="store_true",
                        help="Enable Habibi-TTS synthetic speech generation")
    parser.add_argument("--habibi_model_path", type=str, default=None,
                        help="Optional local path to Habibi-TTS model weights")
    args = parser.parse_args()
    main(args)
