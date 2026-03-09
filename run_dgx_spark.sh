#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Saudi STT Pipeline — NVIDIA DGX Spark Setup & Run
#  Ubuntu OS, clean machine (no libraries pre-installed)
#
#  DGX Spark specs:
#    - NVIDIA Grace Blackwell GB10 chip
#    - 128 GB unified memory (CPU + GPU shared)
#    - Ubuntu 22.04+
#
#  Usage:
#    chmod +x run_dgx_spark.sh
#    ./run_dgx_spark.sh
#
#  Or run individual steps:
#    ./run_dgx_spark.sh setup      # Only install dependencies
#    ./run_dgx_spark.sh prepare    # Only prepare data
#    ./run_dgx_spark.sh augment    # Only augment data
#    ./run_dgx_spark.sh phase1     # Only train Phase 1
#    ./run_dgx_spark.sh eval1      # Only evaluate Phase 1
#    ./run_dgx_spark.sh phase2     # Only train Phase 2
#    ./run_dgx_spark.sh eval2      # Only evaluate Phase 2
#    ./run_dgx_spark.sh merge      # Only merge + export
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
DATA_DIR="${PROJECT_DIR}/data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
EVAL_DIR="${PROJECT_DIR}/eval_results"
FINAL_MODEL_DIR="${PROJECT_DIR}/model_final"

# Training params (tuned for DGX Spark 128GB unified memory)
BATCH_SIZE=1
GRAD_ACCUM=8         # effective batch = 8
MAX_SEQ_LENGTH=1024
PHASE1_LR="2e-5"
PHASE2_LR="1e-5"
PHASE1_EPOCHS=2
PHASE2_EPOCHS=1
LORA_R=16
LORA_ALPHA=32
SAVE_STEPS=500
MAX_SAMPLES=""        # empty = use all, set to e.g. "5000" for quick test

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_step() { echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"; echo -e "${GREEN}  $1${NC}"; echo -e "${BLUE}════════════════════════════════════════════════════════${NC}\n"; }
log_warn() { echo -e "${YELLOW}  WARNING: $1${NC}"; }
log_error() { echo -e "${RED}  ERROR: $1${NC}"; exit 1; }


# ═══════════════════════════════════════════════════════════════
#  STEP 0: System Setup
# ═══════════════════════════════════════════════════════════════

do_setup() {
    log_step "STEP 0: System Setup (DGX Spark — Ubuntu)"

    # 0a. System packages
    echo "Installing system packages..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv python3-dev \
        git git-lfs \
        ffmpeg libsndfile1 libsndfile1-dev \
        build-essential cmake \
        2>/dev/null

    git lfs install 2>/dev/null || true

    # 0b. Create virtual environment
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    source "${VENV_DIR}/bin/activate"

    # 0c. Upgrade pip
    pip install --upgrade pip setuptools wheel 2>&1 | tail -1

    # 0d. Install PyTorch with CUDA (DGX Spark has Blackwell GPU)
    echo "Installing PyTorch with CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

    # 0e. Install project dependencies
    echo "Installing project dependencies..."
    pip install -r "${PROJECT_DIR}/requirements.txt" 2>&1 | tail -5

    # 0f. Verify GPU
    echo ""
    echo "Verifying GPU..."
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'GPU Memory:      {mem_gb:.1f} GB')
    print(f'BF16 support:    {torch.cuda.is_bf16_supported()}')
else:
    print('WARNING: No CUDA GPU found! Training will be extremely slow.')
print(f'Transformers:    {__import__(\"transformers\").__version__}')
print(f'PEFT:            {__import__(\"peft\").__version__}')
print(f'TRL:             {__import__(\"trl\").__version__}')
print(f'BitsAndBytes:    {__import__(\"bitsandbytes\").__version__}')
"

    echo ""
    echo -e "${GREEN}Setup complete!${NC}"
}


# ═══════════════════════════════════════════════════════════════
#  STEP 1: Prepare Data
# ═══════════════════════════════════════════════════════════════

do_prepare() {
    log_step "STEP 1: Prepare Saudi Data (SADA22 — streaming mode)"
    source "${VENV_DIR}/bin/activate"

    SAMPLE_ARG=""
    if [ -n "$MAX_SAMPLES" ]; then
        SAMPLE_ARG="--max_samples ${MAX_SAMPLES}"
    fi

    python3 "${PROJECT_DIR}/scripts/01_prepare_data.py" \
        --output_dir "${DATA_DIR}/saudi_clean" \
        --min_duration 2.0 \
        --max_duration 30.0 \
        --eval_samples 1000 \
        $SAMPLE_ARG

    echo -e "\n${GREEN}Data preparation complete!${NC}"
    echo "  Train: ${DATA_DIR}/saudi_clean/train"
    echo "  Eval:  ${DATA_DIR}/saudi_clean/eval"
}


# ═══════════════════════════════════════════════════════════════
#  STEP 2: Augment Data
# ═══════════════════════════════════════════════════════════════

do_augment() {
    log_step "STEP 2: Augment with Noise (bodycam simulation)"
    source "${VENV_DIR}/bin/activate"

    python3 "${PROJECT_DIR}/scripts/02_augment_data.py" \
        --input_dir "${DATA_DIR}/saudi_clean/train" \
        --output_dir "${DATA_DIR}/saudi_augmented" \
        --augment_ratio 0.67

    echo -e "\n${GREEN}Augmentation complete!${NC}"
    echo "  Output: ${DATA_DIR}/saudi_augmented/train_augmented"
}


# ═══════════════════════════════════════════════════════════════
#  STEP 3: Phase 1 — Saudi Dialect Adaptation
# ═══════════════════════════════════════════════════════════════

do_phase1() {
    log_step "STEP 3: Phase 1 — Saudi Dialect Adaptation"
    source "${VENV_DIR}/bin/activate"

    SAMPLE_ARG=""
    if [ -n "$MAX_SAMPLES" ]; then
        SAMPLE_ARG="--max_train_samples ${MAX_SAMPLES}"
    fi

    python3 "${PROJECT_DIR}/scripts/03_train_phase1.py" \
        --data_dir "${DATA_DIR}/saudi_clean" \
        --output_dir "${CHECKPOINT_DIR}/phase1" \
        --batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${PHASE1_LR} \
        --num_epochs ${PHASE1_EPOCHS} \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --save_steps ${SAVE_STEPS} \
        --max_eval_samples 500 \
        $SAMPLE_ARG

    echo -e "\n${GREEN}Phase 1 training complete!${NC}"
    echo "  Model: ${CHECKPOINT_DIR}/phase1/final"
}


# ═══════════════════════════════════════════════════════════════
#  STEP 4: Evaluate Phase 1
# ═══════════════════════════════════════════════════════════════

do_eval1() {
    log_step "STEP 4: Evaluate Phase 1"
    source "${VENV_DIR}/bin/activate"

    python3 "${PROJECT_DIR}/scripts/04_evaluate.py" \
        --model_dir "${CHECKPOINT_DIR}/phase1/final" \
        --data_dir "${DATA_DIR}/saudi_clean" \
        --output_dir "${EVAL_DIR}" \
        --phase phase1 \
        --max_samples 200 \
        --snr_stratified

    echo -e "\n${GREEN}Phase 1 evaluation complete!${NC}"
    echo "  Results: ${EVAL_DIR}/phase1_eval_results.json"
}


# ═══════════════════════════════════════════════════════════════
#  STEP 5: Phase 2 — Noise Robustness
# ═══════════════════════════════════════════════════════════════

do_phase2() {
    log_step "STEP 5: Phase 2 — Noise Robustness"
    source "${VENV_DIR}/bin/activate"

    SAMPLE_ARG=""
    if [ -n "$MAX_SAMPLES" ]; then
        SAMPLE_ARG="--max_train_samples ${MAX_SAMPLES}"
    fi

    python3 "${PROJECT_DIR}/scripts/05_train_phase2.py" \
        --model_dir "${CHECKPOINT_DIR}/phase1/final" \
        --data_dir "${DATA_DIR}/saudi_augmented" \
        --eval_dir "${DATA_DIR}/saudi_clean" \
        --output_dir "${CHECKPOINT_DIR}/phase2" \
        --batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${PHASE2_LR} \
        --num_epochs ${PHASE2_EPOCHS} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --save_steps ${SAVE_STEPS} \
        --max_eval_samples 500 \
        $SAMPLE_ARG

    echo -e "\n${GREEN}Phase 2 training complete!${NC}"
    echo "  Model: ${CHECKPOINT_DIR}/phase2/final"
}


# ═══════════════════════════════════════════════════════════════
#  STEP 6: Evaluate Phase 2
# ═══════════════════════════════════════════════════════════════

do_eval2() {
    log_step "STEP 6: Evaluate Phase 2 (Final)"
    source "${VENV_DIR}/bin/activate"

    python3 "${PROJECT_DIR}/scripts/04_evaluate.py" \
        --model_dir "${CHECKPOINT_DIR}/phase2/final" \
        --data_dir "${DATA_DIR}/saudi_clean" \
        --output_dir "${EVAL_DIR}" \
        --phase phase2 \
        --max_samples 200 \
        --snr_stratified

    # Print comparison table
    python3 "${PROJECT_DIR}/scripts/04_evaluate.py" \
        --compare_results "${EVAL_DIR}"

    echo -e "\n${GREEN}Phase 2 evaluation complete!${NC}"
}


# ═══════════════════════════════════════════════════════════════
#  STEP 7: Merge & Export
# ═══════════════════════════════════════════════════════════════

do_merge() {
    log_step "STEP 7: Merge LoRA & Export Production Model"
    source "${VENV_DIR}/bin/activate"

    python3 "${PROJECT_DIR}/scripts/06_merge_and_export.py" \
        --model_dir "${CHECKPOINT_DIR}/phase2/final" \
        --output_dir "${FINAL_MODEL_DIR}" \
        --save_method merged_16bit

    echo -e "\n${GREEN}Model exported!${NC}"
    echo "  Production model: ${FINAL_MODEL_DIR}"
    echo ""
    echo "  To use for inference:"
    echo "    from transformers import AutoProcessor, Gemma3nForConditionalGeneration"
    echo "    processor = AutoProcessor.from_pretrained('${FINAL_MODEL_DIR}')"
    echo "    model = Gemma3nForConditionalGeneration.from_pretrained('${FINAL_MODEL_DIR}')"
}


# ═══════════════════════════════════════════════════════════════
#  Main — run all or specific step
# ═══════════════════════════════════════════════════════════════

STEP="${1:-all}"

case "$STEP" in
    setup)   do_setup ;;
    prepare) do_prepare ;;
    augment) do_augment ;;
    phase1)  do_phase1 ;;
    eval1)   do_eval1 ;;
    phase2)  do_phase2 ;;
    eval2)   do_eval2 ;;
    merge)   do_merge ;;
    all)
        do_setup
        do_prepare
        do_augment
        do_phase1
        do_eval1
        do_phase2
        do_eval2
        do_merge

        log_step "ALL DONE!"
        echo "  Final model:  ${FINAL_MODEL_DIR}"
        echo "  Eval results: ${EVAL_DIR}"
        echo ""
        echo "  WER comparison:"
        source "${VENV_DIR}/bin/activate"
        python3 "${PROJECT_DIR}/scripts/04_evaluate.py" --compare_results "${EVAL_DIR}"
        ;;
    *)
        echo "Usage: $0 {setup|prepare|augment|phase1|eval1|phase2|eval2|merge|all}"
        exit 1
        ;;
esac
