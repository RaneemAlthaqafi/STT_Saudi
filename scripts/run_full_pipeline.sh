#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  Saudi STT Full Fine-Tuning Pipeline
#  Run on RunPod A100 80GB or A40 48GB
# ══════════════════════════════════════════════════════════════
set -e

echo "══════════════════════════════════════════════════════════"
echo "  Saudi STT Fine-Tuning Pipeline"
echo "  Model: MasriSwitch-Gemma3n-Transcriber-v1"
echo "  Data:  SADA22 Saudi Dialects (667h)"
echo "══════════════════════════════════════════════════════════"

# ──────────────────────────────────────────────────────────────
# STEP 0: Environment setup
# ──────────────────────────────────────────────────────────────
echo ""
echo "[STEP 0/7] Installing dependencies..."
# Install unsloth first (has its own pinned deps)
pip install -q "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
# Then force transformers 4.53 (Gemma3n audio support) + remaining deps
pip install -q "transformers==4.53.0" "datasets==3.5.0" \
    accelerate bitsandbytes peft trl \
    librosa soundfile torchaudio audiomentations \
    jiwer gradio timm

# Login to HuggingFace (for gated Gemma model)
if [ ! -f ~/.cache/huggingface/token ]; then
    echo ""
    echo ">>> HuggingFace login required for Gemma model access"
    python -c "from huggingface_hub import login; login()"
fi

cd /workspace/STT_Saudi

# ──────────────────────────────────────────────────────────────
# STEP 1: Prepare Saudi dialect data
# ──────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "[STEP 1/7] Preparing Saudi dialect data from SADA22..."
echo "══════════════════════════════════════════════════════════"

if [ ! -d "./data/saudi_clean/train" ]; then
    python scripts/01_prepare_data.py \
        --output_dir ./data/saudi_clean \
        --min_duration 2.0 \
        --max_duration 30.0 \
        --eval_samples 1000
    echo "  ✓ Data prepared!"
else
    echo "  ✓ Data already exists, skipping."
fi

# ──────────────────────────────────────────────────────────────
# STEP 2: Download MUSAN noise + augment data
# ──────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "[STEP 2/7] Augmenting data with noise..."
echo "══════════════════════════════════════════════════════════"

if [ ! -d "./data/saudi_augmented" ]; then
    # Download MUSAN noise dataset (optional but recommended)
    if [ ! -d "./data/musan" ]; then
        echo "  Downloading MUSAN noise dataset..."
        python scripts/download_musan.py --output_dir ./data/musan || echo "  MUSAN download failed, will use synthetic noise"
    fi

    python scripts/02_augment_data.py \
        --input_dir ./data/saudi_clean/train \
        --output_dir ./data/saudi_augmented \
        --noise_dir ./data/musan/noise \
        --augment_ratio 1.0 \
        --speed_perturb_prob 0.3
    echo "  ✓ Augmentation complete!"
else
    echo "  ✓ Augmented data already exists, skipping."
fi

# ──────────────────────────────────────────────────────────────
# STEP 3: Phase 1 — Saudi dialect adaptation (clean data)
# ──────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "[STEP 3/7] Phase 1: Fine-tuning on clean Saudi data..."
echo "  LR: 2e-5 | Epochs: 2 | Strategy: Dialect adaptation"
echo "══════════════════════════════════════════════════════════"

if [ ! -d "./checkpoints/phase1/final" ]; then
    python scripts/03_train_phase1.py \
        --data_dir ./data/saudi_clean \
        --output_dir ./checkpoints/phase1 \
        --learning_rate 2e-5 \
        --num_epochs 2 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --lora_r 16 \
        --lora_alpha 32 \
        --max_seq_length 1024 \
        --save_steps 500 \
        --load_in_4bit
    echo "  ✓ Phase 1 training complete!"
else
    echo "  ✓ Phase 1 checkpoint exists, skipping."
fi

# ──────────────────────────────────────────────────────────────
# STEP 4: Evaluate Phase 1
# ──────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "[STEP 4/7] Evaluating Phase 1 model..."
echo "══════════════════════════════════════════════════════════"

python scripts/04_evaluate.py \
    --model_dir ./checkpoints/phase1/final \
    --data_dir ./data/saudi_clean \
    --max_samples 200 \
    --snr_stratified
echo "  ✓ Phase 1 evaluation complete!"

# ──────────────────────────────────────────────────────────────
# STEP 5: Phase 2 — Noise robustness (augmented data)
# ──────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "[STEP 5/7] Phase 2: Fine-tuning on noisy data..."
echo "  LR: 1e-5 | Epochs: 1 | Strategy: Noise robustness"
echo "══════════════════════════════════════════════════════════"

if [ ! -d "./checkpoints/phase2/final" ]; then
    python scripts/05_train_phase2.py \
        --model_dir ./checkpoints/phase1/final \
        --data_dir ./data/saudi_augmented \
        --eval_dir ./data/saudi_clean \
        --output_dir ./checkpoints/phase2 \
        --learning_rate 1e-5 \
        --num_epochs 1 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --max_seq_length 1024 \
        --save_steps 500
    echo "  ✓ Phase 2 training complete!"
else
    echo "  ✓ Phase 2 checkpoint exists, skipping."
fi

# ──────────────────────────────────────────────────────────────
# STEP 6: Final evaluation
# ──────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "[STEP 6/7] Final evaluation (Phase 2 model)..."
echo "══════════════════════════════════════════════════════════"

python scripts/04_evaluate.py \
    --model_dir ./checkpoints/phase2/final \
    --data_dir ./data/saudi_clean \
    --max_samples 200 \
    --snr_stratified
echo "  ✓ Final evaluation complete!"

# ──────────────────────────────────────────────────────────────
# STEP 7: Merge LoRA & export final model
# ──────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "[STEP 7/7] Merging LoRA adapters & exporting model..."
echo "══════════════════════════════════════════════════════════"

python scripts/06_merge_and_export.py \
    --model_dir ./checkpoints/phase2/final \
    --output_dir ./model_final \
    --save_method merged_16bit
echo "  ✓ Model exported to ./model_final!"

# ──────────────────────────────────────────────────────────────
# DONE!
# ──────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo "  ✅ PIPELINE COMPLETE!"
echo ""
echo "  Checkpoints:"
echo "    Phase 1: ./checkpoints/phase1/final"
echo "    Phase 2: ./checkpoints/phase2/final"
echo "    Final:   ./model_final"
echo ""
echo "  To launch demo:"
echo "    python app_gradio.py"
echo ""
echo "  To push to HuggingFace:"
echo "    python scripts/06_merge_and_export.py \\"
echo "      --model_dir ./checkpoints/phase2/final \\"
echo "      --push_to_hub --hub_model_id YOUR_ID"
echo "══════════════════════════════════════════════════════════"
