# Saudi Arabic STT Fine-Tuning Guide
## Fine-tuning MasriSwitch-Gemma3n-Transcriber-v1 for Saudi Dialect + Noise Robustness

---

## Table of Contents
1. [Overview & Strategy](#1-overview--strategy)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Data Sources](#3-data-sources)
4. [Workflow: Benchmark → Train → Evaluate → Iterate](#4-workflow)
5. [Step-by-Step Execution](#5-step-by-step-execution)
6. [Key Technical Details](#6-key-technical-details)
7. [Inference & Deployment](#7-inference--deployment)
8. [Tips & Common Pitfalls](#8-tips--common-pitfalls)
9. [References](#9-references)

---

## 1. Overview & Strategy

### Base Model
- **Model**: `oddadmix/MasriSwitch-Gemma3n-Transcriber-v1`
- **Architecture**: `google/gemma-3n-E4B` (8B params, BF16)
- **Already fine-tuned on**: Egyptian Arabic + English code-switching
- **Audio specs**: 16kHz mono, up to 30 seconds, 6.25 tokens/sec
- **License**: Apache 2.0

### Strategy: Benchmark First, Then Fine-Tune

```
Step 0: Benchmark 10+ Arabic STT models on Saudi noisy data
        → Pick the best model as your base

Step 1: Prepare SADA Saudi dialect data (filter, normalize, split)

Step 2: Augment with noise (SNR 5-15 dB, speed perturbation 0.9/1.0/1.1)

Step 3: Phase 1 — Saudi dialect adaptation (clean data, LR 2e-5, 2 epochs)

Step 4: Evaluate → compare against pre-training benchmark

Step 5: Phase 2 — Noise robustness (augmented data, LR 1e-5, 1 epoch)

Step 6: Final evaluation → merge & export for production
```

### Why This Approach Works
1. **Benchmarking first** ensures you start with the strongest base model
2. **Progressive training** (clean → noisy) is more stable than mixing everything
3. **Research-backed augmentation** (SNR 5-15 dB) matches real-world conditions
4. **Proper label masking** (`train_on_responses_only`) focuses learning on transcription

---

## 2. Hardware Requirements

### Minimum (QLoRA 4-bit with Unsloth)
| Component | Requirement |
|-----------|-------------|
| GPU VRAM | 16GB+ (RTX 4060 Ti 16GB, RTX 3090, A100) |
| RAM | 32GB+ |
| Storage | 200GB+ (for datasets) |

### Recommended Options
| Option | GPU | VRAM | Cost |
|--------|-----|------|------|
| **Local** | RTX 3090/4090 | 24GB | One-time |
| **Colab Pro+** | A100 40GB | 40GB | ~$50/mo |
| **RunPod** | A100 80GB | 80GB | ~$1.5/hr |
| **Lambda** | A100 80GB | 80GB | ~$1.1/hr |

### VRAM Estimates (Unsloth QLoRA 4-bit)
- Text-only: ~10GB
- With audio: ~14-18GB
- Gradient checkpointing saves ~30%
- Batch size 1 + gradient accumulation 4: most memory efficient

---

## 3. Data Sources

### Primary: SADA Dataset (667 hours)
- **HuggingFace**: `MohamedRashad/SADA22`
- **Size**: 667+ hours, 253,166 samples
- **Source**: 57 Saudi Broadcasting Authority TV shows
- **Dialects**: `najidi`, `hijazi`, `khaliji` + MSA
  - **IMPORTANT**: These are the actual field values in the dataset (lowercase, `najidi` not `najdi`)
- **Audio quality**: ~33% clean, ~33% noisy, ~33% with music
- **Key fields**: `audio`, `cleaned_text`, `speaker_dialect`, `speaker_gender`
- **License**: CC BY-NC-SA 4.0

### Noise Source: MUSAN
- **URL**: https://www.openslr.org/17/
- **Size**: ~109 hours (music, speech, noise)
- **Download**: `python scripts/download_musan.py --output_dir ./data/musan`

### Additional Datasets
| Dataset | Size | Notes |
|---------|------|-------|
| `MohamedRashad/arabic-english-code-switching` | 12K samples | Code-switching data |
| `mozilla-foundation/common_voice_17_0` | 300h+ | Mixed Arabic dialects |
| `google/fleurs` | ~12h | MSA baseline |

---

## 4. Workflow

### Step 0: Benchmark (scripts/00_benchmark.py)
Benchmarks 10+ models on SADA Saudi test split:

| # | Model | Type |
|---|-------|------|
| 1 | `openai/whisper-large-v3` | Seq2Seq |
| 2 | `openai/whisper-large-v3-turbo` | Seq2Seq |
| 3 | `Byne/whisper-large-v3-arabic` | Seq2Seq (Arabic-specific) |
| 4 | `MohamedRashad/Arabic-Whisper-CodeSwitching-Edition` | Seq2Seq |
| 5 | `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` | CTC |
| 6 | `facebook/mms-1b-all` | CTC |
| 7 | `salmujaiwel/wav2vec2-large-xls-r-300m-arabic-saudi-colab` | CTC (Saudi) |
| 8 | `oddadmix/MasriSwitch-Gemma3n-Transcriber-v1` | Generative |
| 9 | `Systran/faster-whisper-large-v3` | Seq2Seq (CTranslate2) |

**Metrics per model**: WER, CER, MER, WIL, WIP, RTF, SNR-stratified breakdown

```bash
python scripts/00_benchmark.py --max_samples 500 --snr_stratified
```

### Steps 1-6: Train & Evaluate
```bash
# 1. Prepare data
python scripts/01_prepare_data.py --output_dir ./data/saudi_clean

# 2. Augment (optional: add --noise_dir for MUSAN noise)
python scripts/02_augment_data.py \
    --input_dir ./data/saudi_clean/train \
    --output_dir ./data/saudi_augmented \
    --noise_dir ./data/musan/noise

# 3. Phase 1: Saudi dialect adaptation
python scripts/03_train_phase1.py \
    --data_dir ./data/saudi_clean \
    --output_dir ./checkpoints/phase1

# 4. Evaluate Phase 1
python scripts/04_evaluate.py \
    --model_dir ./checkpoints/phase1/final \
    --data_dir ./data/saudi_clean \
    --snr_stratified

# 5. Phase 2: Noise robustness
python scripts/05_train_phase2.py \
    --model_dir ./checkpoints/phase1/final \
    --data_dir ./data/saudi_augmented \
    --output_dir ./checkpoints/phase2

# 6. Final evaluation + merge
python scripts/04_evaluate.py \
    --model_dir ./checkpoints/phase2/final \
    --snr_stratified

python scripts/06_merge_and_export.py \
    --model_dir ./checkpoints/phase2/final
```

---

## 5. Step-by-Step Execution

### Quick Start (Colab/Cloud)
Use the notebook: `notebooks/Saudi_STT_FineTune.ipynb`

It includes everything: data prep, augmentation, Phase 1+2 training, evaluation, and export — all in one notebook with proper label masking.

### Full Pipeline (Local/Server)
1. Install dependencies: `pip install -r requirements.txt`
2. Run scripts in order: `00_benchmark.py` → `01_prepare_data.py` → `02_augment_data.py` → `03_train_phase1.py` → `04_evaluate.py` → `05_train_phase2.py` → `04_evaluate.py` → `06_merge_and_export.py`

### Collecting Your Own Data
1. Add YouTube URLs to `data/youtube_urls.txt`
2. Run `python scripts/collect_youtube.py --urls_file data/youtube_urls.txt`
3. Run `python scripts/transcribe_and_validate.py --segments_manifest ...`
4. Review and correct transcripts in the TSV file

---

## 6. Key Technical Details

### Label Masking (CRITICAL)
The most important fix over basic tutorials:

```python
# WRONG: only masks padding/audio tokens, still trains on user prompt
labels[labels == pad_id] = -100

# CORRECT: mask everything except assistant response
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)
```

### SFTConfig Requirements
When using a custom collate function with SFTTrainer, you **must** include:
```python
SFTConfig(
    ...,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)
```

### LoRA Configuration
Audio-specific modules are CRITICAL for STT fine-tuning:
```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",     # Attention
    "gate_proj", "up_proj", "down_proj",          # FFN
    "post", "linear_start", "linear_end",         # Audio-specific
    "embedding_projection",
]
modules_to_save=["lm_head", "embed_tokens", "embed_audio"]
```

### SADA Dialect Filtering
```python
# CORRECT field values (verified from dataset):
saudi_keywords = ["najidi", "hijazi", "khaliji"]  # NOT "Najdi"/"Hijazi"/"Khaliji"
```

### Arabic Text Normalization
Two levels provided in `scripts/utils/arabic_normalizer.py`:
- **Training** (conservative): diacritics, alef variants, alef maqsura, tatweel
- **Evaluation** (OALL standard): + punctuation removal, lowercase, digit normalization

### Augmentation Parameters (Research-Backed)
| Parameter | Value | Source |
|-----------|-------|--------|
| MUSAN SNR range | 5-15 dB | NVIDIA NeMo, SADA paper |
| Speed perturbation | 0.9, 1.0, 1.1 | Povey 2015 |
| Clean:Noisy ratio | 50:50 | Multi-condition training |
| Duration filter | 2-30 seconds | SADA paper (<2s = >100% CER) |

### Training Hyperparameters
| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Learning rate | 2e-5 | 1e-5 |
| Epochs | 2 | 1 |
| Warmup ratio | 0.1 | 0.05 |
| Batch size | 1 | 1 |
| Gradient accumulation | 4 | 4 |
| Optimizer | adamw_8bit | adamw_8bit |
| Expected initial loss | 6-7 (normal for multimodal) | — |

---

## 7. Inference & Deployment

### Load Fine-Tuned Model
```python
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
import torch

# IMPORTANT: Use Google's processor, not Unsloth's saved version
processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")

model = Gemma3nForConditionalGeneration.from_pretrained(
    "./saudi_stt_merged",
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

def transcribe(audio_array, sr=16000):
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
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return processor.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
```

### Merge LoRA for Production
```bash
python scripts/06_merge_and_export.py --model_dir ./checkpoints/phase2/final
```

---

## 8. Tips & Common Pitfalls

### Do
- Benchmark multiple models before choosing a base
- Use `train_on_responses_only` for proper label masking
- Include `embed_audio` in `modules_to_save`
- Use `cleaned_text` field from SADA (not `text`)
- Use Google's processor for inference
- Monitor eval loss — stop if it increases (overfitting)
- Keep 30-50% of training data clean (don't over-augment)

### Don't
- Don't normalize taa marbouta (OALL standard preserves it)
- Don't use batch size > 1 without enough VRAM (OOM on audio)
- Don't skip warmup steps
- Don't use Unsloth's saved processor for inference
- Don't train on noisy data only — progressive (clean → noisy) works better
- Don't use SNR below 5 dB for augmentation (transcript becomes invalid)

### Expected Metrics
| Condition | WER Target | CER Target |
|-----------|-----------|-----------|
| Clean Saudi speech | < 20% | < 10% |
| Noisy Saudi speech (5-15 dB) | < 35% | < 18% |
| Code-switching | < 25% | < 12% |

---

## 9. References

- **SADA Dataset**: https://huggingface.co/datasets/MohamedRashad/SADA22
- **SADA ASR Paper**: https://arxiv.org/abs/2508.12968
- **MasriSwitch Model**: https://huggingface.co/oddadmix/MasriSwitch-Gemma3n-Transcriber-v1
- **Gemma3n Fine-tuning Tutorial**: https://debuggercafe.com/fine-tuning-gemma-3n-for-speech-transcription/
- **Unsloth**: https://github.com/unslothai/unsloth
- **OALL Arabic ASR Leaderboard**: https://huggingface.co/spaces/OALL/Open-Arabic-ASR-Leaderboard
- **MUSAN Noise Dataset**: https://www.openslr.org/17/
- **Audiomentations**: https://github.com/iver56/audiomentations
- **Speed Perturbation**: Povey et al., 2015
