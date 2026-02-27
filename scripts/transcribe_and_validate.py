"""
Utility: Auto-transcribe audio segments using MasriSwitch or Whisper,
then prepare for manual validation.

Usage:
    python scripts/transcribe_and_validate.py \
        --segments_manifest ./data/youtube_raw/segments_manifest.json \
        --output_dir ./data/youtube_transcribed \
        --model whisper  # or masriswitch
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def load_whisper_model(model_size="large-v3"):
    """Load Whisper model for initial transcription."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print(f"Loading Whisper {model_size}...")
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{model_size}",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, processor, "whisper"


def load_masriswitch_model():
    """Load MasriSwitch for transcription."""
    from transformers import AutoProcessor, Gemma3nForConditionalGeneration

    model_id = "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1"
    print(f"Loading {model_id}...")

    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, "masriswitch"


def transcribe_whisper(model, processor, audio_array, sr=16000):
    """Transcribe using Whisper."""
    input_features = processor(
        audio_array, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(model.device, dtype=torch.float16)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="arabic", task="transcribe"
    )

    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=256,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def transcribe_masriswitch(model, processor, audio_array, sr=16000):
    """Transcribe using MasriSwitch."""
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
        tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    return processor.decode(output[0][input_len:], skip_special_tokens=True).strip()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segments manifest
    with open(args.segments_manifest) as f:
        segments = json.load(f)

    print(f"Total segments to transcribe: {len(segments)}")

    # Load model
    if args.model == "whisper":
        model, processor, model_type = load_whisper_model(args.whisper_size)
    else:
        model, processor, model_type = load_masriswitch_model()

    # Transcribe all segments
    results = []
    for i, seg in enumerate(tqdm(segments, desc="Transcribing")):
        audio_path = seg["path"]

        try:
            if HAS_LIBROSA:
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            else:
                import soundfile as sf
                audio, sr = sf.read(audio_path)
                if sr != 16000:
                    print(f"  WARNING: Sample rate {sr} != 16000 for {audio_path}")

            audio = np.array(audio, dtype=np.float32)

            if model_type == "whisper":
                transcript = transcribe_whisper(model, processor, audio, sr=16000)
            else:
                transcript = transcribe_masriswitch(model, processor, audio, sr=16000)

        except Exception as e:
            print(f"\n  Error on {audio_path}: {e}")
            transcript = "[ERROR]"

        results.append({
            "audio_path": audio_path,
            "transcript": transcript,
            "duration": seg.get("duration", 0),
            "source": seg.get("source", ""),
            "needs_review": True,  # Flag for manual review
        })

    # Save transcriptions
    output_file = output_dir / "transcriptions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also save as TSV for easy editing in spreadsheet
    tsv_file = output_dir / "transcriptions.tsv"
    with open(tsv_file, "w", encoding="utf-8") as f:
        f.write("audio_path\ttranscript\tduration\tneeds_correction\n")
        for r in results:
            f.write(f"{r['audio_path']}\t{r['transcript']}\t{r['duration']}\t\n")

    total_duration = sum(r["duration"] for r in results)
    print(f"\n{'='*60}")
    print(f"Transcribed {len(results)} segments ({total_duration/3600:.1f} hours)")
    print(f"Results saved to:")
    print(f"  JSON: {output_file}")
    print(f"  TSV:  {tsv_file} (open in Excel/Google Sheets for manual review)")
    print(f"\nIMPORTANT: Review and correct transcripts before using for training!")
    print(f"Open the TSV file, correct errors, and fill the 'needs_correction' column.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-transcribe audio segments")
    parser.add_argument("--segments_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data/youtube_transcribed")
    parser.add_argument("--model", type=str, default="whisper",
                        choices=["whisper", "masriswitch"])
    parser.add_argument("--whisper_size", type=str, default="large-v3")
    args = parser.parse_args()
    main(args)
