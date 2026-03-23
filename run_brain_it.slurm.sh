#!/usr/bin/env bash
# ============================================================================
# Brain-IT Stage 2 — fMRI-to-Image Reconstruction  (Isambard-AI / Cray Shasta)
# ============================================================================
#
# Usage:
#   sbatch run_brain_it.slurm.sh --phase v2c
#   sbatch run_brain_it.slurm.sh --phase lowlevel   [--resume]
#   sbatch run_brain_it.slurm.sh --phase semantic1  [--resume]
#   sbatch run_brain_it.slurm.sh --phase semantic2  [--resume]
#   sbatch run_brain_it.slurm.sh --phase reconstruct
#   sbatch run_brain_it.slurm.sh --phase evaluate
#
# Each phase corresponds to a step in the Brain-IT pipeline:
#   v2c         — V2C mapping: extract encoder embeddings, fit GMM (128 clusters)
#   lowlevel    — Train LowLevelBIT (VGG feature prediction), ~12h
#   semantic1   — Train SemanticBIT stage 1 (CLIP alignment), ~8h
#   semantic2   — Train SemanticBIT stage 2 (joint diffusion), ~20h
#   reconstruct — Run full reconstruction on test set
#   evaluate    — Compute PixCorr/SSIM/Alex/Incep/CLIP/Eff/SwAV metrics
#
# Resume interrupted training:
#   sbatch run_brain_it.slurm.sh --phase lowlevel --resume
#
# ============================================================================
#SBATCH --job-name=brain-it
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=.logs/%x-%j.out
#SBATCH --error=.logs/%x-%j.err
#SBATCH --signal=USR1@300          # send SIGUSR1 300 s before the time limit

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
CONDA_ENV="brain_encoder"
BRAIN_IT_DIR="${PROJECTDIR}/brain/brain_it"
DATA_ROOT="${PROJECTDIR}/brain/algonauts_prepared_data"
ENCODER_CKPT="${PROJECTDIR}/brain/checkpoints/brain_encoder_20260318/best_model.pt"
V2C_DIR="${BRAIN_IT_DIR}/v2c"
SUBJECTS="subj01 subj02 subj03 subj04 subj05 subj06 subj07"

# Output directories (dated at submission time, not job start)
DATE_TAG=$(date +%Y%m%d)
LOWLEVEL_DIR="${PROJECTDIR}/brain/checkpoints/bit_lowlevel_${DATE_TAG}"
SEMANTIC1_DIR="${PROJECTDIR}/brain/checkpoints/bit_semantic1_${DATE_TAG}"
SEMANTIC2_DIR="${PROJECTDIR}/brain/checkpoints/bit_semantic2_${DATE_TAG}"
RECON_DIR="${BRAIN_IT_DIR}/reconstructions"
RESULTS_DIR="${BRAIN_IT_DIR}/results"
MINDEYE2_CKPT="${PROJECTDIR}/brain/checkpoints/mindeye2"
COCO_FMRI_DIR="${PROJECTDIR}/brain/coco_fmri"  # optional; skip if missing

echo "=== $(date) === Job ${SLURM_JOB_ID} on $(hostname) ==="

# ---------------------------------------------------------------------------
# Load CUDA
# ---------------------------------------------------------------------------
module load cuda/12.6 2>/dev/null \
  || module load cudatoolkit/24.11_12.6 2>/dev/null \
  || true
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
nvidia-smi --list-gpus

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${BRAIN_IT_DIR}:${PYTHONPATH:-}"

# Cache dirs
export HF_HOME="${PROJECTDIR}/.cache/hf"
export HF_DATASETS_CACHE="${PROJECTDIR}/.cache/hf/datasets"
export TORCH_HOME="${PROJECTDIR}/.cache/torch"
export TRITON_CACHE_DIR="${SCRATCHDIR}/triton_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}"
mkdir -p .logs

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-brain-it}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-brain-it-$(date +%Y%m%d)}"

# GPU setup
GPU_IDS=$(nvidia-smi --list-gpus | awk '{print NR-1}' | tr '\n' ',' | sed 's/,$//')
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}  (${GPUS} GPU(s))"

# Sanity-check
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
d = torch.cuda.get_device_properties(0)
print(f'CUDA OK: {d.name}  {d.total_memory/1024**3:.0f} GB  bf16={torch.cuda.is_bf16_supported()}')
"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
PHASE=""
RESUME_FLAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)    PHASE="$2";      shift 2 ;;
        --resume)   RESUME_FLAG="--resume"; shift ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${PHASE}" ]]; then
    echo "ERROR: --phase is required."
    echo "  Options: v2c | lowlevel | semantic1 | semantic2 | reconstruct | evaluate"
    exit 1
fi

echo ""
echo "Phase:     ${PHASE}"
echo "Resume:    ${RESUME_FLAG:-no}"
echo "Extra:     ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
echo ""

# ---------------------------------------------------------------------------
# Signal handler: forward SIGUSR1 to Python subprocess
# ---------------------------------------------------------------------------
PY_PID=""

forward_signal() {
    echo "[slurm] Forwarding SIGUSR1 to Python (pid ${PY_PID})..."
    kill -USR1 "${PY_PID}" 2>/dev/null || true
}

trap 'forward_signal' USR1
trap 'forward_signal' TERM

# ---------------------------------------------------------------------------
# Phase: V2C mapping
# ---------------------------------------------------------------------------
run_v2c() {
    echo "=== Phase: V2C Mapping ==="
    mkdir -p "${V2C_DIR}"

    python "${BRAIN_IT_DIR}/v2c_mapping.py" \
        --encoder_checkpoint "${ENCODER_CKPT}" \
        --output_dir "${V2C_DIR}" \
        --n_clusters 128 \
        --subsample 100000 &
    PY_PID=$!
    wait "${PY_PID}"
    echo "V2C mapping complete → ${V2C_DIR}"
}

# ---------------------------------------------------------------------------
# Phase: Train LowLevelBIT
# ---------------------------------------------------------------------------
run_lowlevel() {
    echo "=== Phase: Train LowLevelBIT ==="
    mkdir -p "${LOWLEVEL_DIR}"

    # Find most recent lowlevel dir if resuming
    if [[ -n "${RESUME_FLAG}" ]]; then
        LATEST_LL=$(ls -dt "${PROJECTDIR}/brain/checkpoints/bit_lowlevel_"* 2>/dev/null | head -1 || echo "")
        if [[ -n "${LATEST_LL}" ]]; then
            LOWLEVEL_DIR="${LATEST_LL}"
            echo "Resuming from: ${LOWLEVEL_DIR}"
        fi
    fi

    # Optional COCO fMRI argument
    COCO_ARG=""
    if [[ -d "${COCO_FMRI_DIR}" ]]; then
        COCO_ARG="--coco_fmri_dir ${COCO_FMRI_DIR}"
    fi

    python "${BRAIN_IT_DIR}/train_lowlevel.py" \
        --data_root "${DATA_ROOT}" \
        --v2c_dir "${V2C_DIR}" \
        --subjects ${SUBJECTS} \
        --output_dir "${LOWLEVEL_DIR}" \
        --epochs 60 \
        --batch_size 64 \
        --voxels_per_image 15000 \
        --lr 5e-4 \
        --weight_decay 1e-2 \
        --warmup_epochs 15 \
        --use_amp \
        --num_workers 8 \
        --save_every 5 \
        --log_every_steps 20 \
        --wandb \
        ${COCO_ARG} \
        ${RESUME_FLAG} \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
    PY_PID=$!
    wait "${PY_PID}"
}

# ---------------------------------------------------------------------------
# Phase: Train SemanticBIT Stage 1 (CLIP alignment)
# ---------------------------------------------------------------------------
run_semantic1() {
    echo "=== Phase: Train SemanticBIT Stage 1 ==="
    mkdir -p "${SEMANTIC1_DIR}"

    if [[ -n "${RESUME_FLAG}" ]]; then
        LATEST_S1=$(ls -dt "${PROJECTDIR}/brain/checkpoints/bit_semantic1_"* 2>/dev/null | head -1 || echo "")
        if [[ -n "${LATEST_S1}" ]]; then
            SEMANTIC1_DIR="${LATEST_S1}"
            echo "Resuming from: ${SEMANTIC1_DIR}"
        fi
    fi

    COCO_ARG=""
    if [[ -d "${COCO_FMRI_DIR}" ]]; then
        COCO_ARG="--coco_fmri_dir ${COCO_FMRI_DIR}"
    fi

    python "${BRAIN_IT_DIR}/train_semantic.py" \
        --stage 1 \
        --data_root "${DATA_ROOT}" \
        --v2c_dir "${V2C_DIR}" \
        --subjects ${SUBJECTS} \
        --output_dir "${SEMANTIC1_DIR}" \
        --epochs 60 \
        --batch_size 128 \
        --voxels_per_image 15000 \
        --lr 5e-4 \
        --warmup_epochs 15 \
        --use_amp \
        --num_workers 8 \
        --save_every 5 \
        --log_every_steps 20 \
        --wandb \
        ${COCO_ARG} \
        ${RESUME_FLAG} \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
    PY_PID=$!
    wait "${PY_PID}"
}

# ---------------------------------------------------------------------------
# Phase: Train SemanticBIT Stage 2 (joint with diffusion)
# ---------------------------------------------------------------------------
run_semantic2() {
    echo "=== Phase: Train SemanticBIT Stage 2 ==="
    mkdir -p "${SEMANTIC2_DIR}"

    # Find stage 1 best model
    S1_CKPT="${SEMANTIC1_DIR}/best_model.pt"
    if [[ ! -f "${S1_CKPT}" ]]; then
        # Try to find the most recent semantic1 run
        LATEST_S1=$(ls -dt "${PROJECTDIR}/brain/checkpoints/bit_semantic1_"* 2>/dev/null | head -1 || echo "")
        if [[ -n "${LATEST_S1}" ]]; then
            S1_CKPT="${LATEST_S1}/best_model.pt"
        fi
    fi

    if [[ ! -f "${S1_CKPT}" ]]; then
        echo "ERROR: Stage 1 checkpoint not found. Run --phase semantic1 first."
        echo "Expected: ${S1_CKPT}"
        exit 1
    fi
    echo "Stage 1 checkpoint: ${S1_CKPT}"

    if [[ -n "${RESUME_FLAG}" ]]; then
        LATEST_S2=$(ls -dt "${PROJECTDIR}/brain/checkpoints/bit_semantic2_"* 2>/dev/null | head -1 || echo "")
        if [[ -n "${LATEST_S2}" ]]; then
            SEMANTIC2_DIR="${LATEST_S2}"
            echo "Resuming from: ${SEMANTIC2_DIR}"
        fi
    fi

    COCO_ARG=""
    if [[ -d "${COCO_FMRI_DIR}" ]]; then
        COCO_ARG="--coco_fmri_dir ${COCO_FMRI_DIR}"
    fi

    # Stage 2: smaller batch, more accumulation steps to match paper's effective batch 64
    python "${BRAIN_IT_DIR}/train_semantic.py" \
        --stage 2 \
        --data_root "${DATA_ROOT}" \
        --v2c_dir "${V2C_DIR}" \
        --subjects ${SUBJECTS} \
        --stage1_checkpoint "${S1_CKPT}" \
        --mindeye2_checkpoint "${MINDEYE2_CKPT}" \
        --output_dir "${SEMANTIC2_DIR}" \
        --epochs 10 \
        --batch_size 8 \
        --grad_accum_steps 8 \
        --voxels_per_image 15000 \
        --lr 1e-5 \
        --use_amp \
        --num_workers 4 \
        --save_every 2 \
        --log_every_steps 10 \
        --wandb \
        ${COCO_ARG} \
        ${RESUME_FLAG} \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
    PY_PID=$!
    wait "${PY_PID}"
}

# ---------------------------------------------------------------------------
# Phase: Reconstruct test images
# ---------------------------------------------------------------------------
run_reconstruct() {
    echo "=== Phase: Reconstruct Test Images ==="
    mkdir -p "${RECON_DIR}"

    # Find most recent lowlevel and semantic checkpoints
    LL_CKPT="${LOWLEVEL_DIR}/best_model.pt"
    if [[ ! -f "${LL_CKPT}" ]]; then
        LATEST_LL=$(ls -dt "${PROJECTDIR}/brain/checkpoints/bit_lowlevel_"* 2>/dev/null | head -1 || echo "")
        [[ -n "${LATEST_LL}" ]] && LL_CKPT="${LATEST_LL}/best_model.pt"
    fi

    SEM_CKPT="${SEMANTIC2_DIR}/bit_latest.pt"
    if [[ ! -f "${SEM_CKPT}" ]]; then
        # Fall back to stage 1 for testing without stage 2
        LATEST_S1=$(ls -dt "${PROJECTDIR}/brain/checkpoints/bit_semantic1_"* 2>/dev/null | head -1 || echo "")
        [[ -n "${LATEST_S1}" ]] && SEM_CKPT="${LATEST_S1}/best_model.pt"
    fi

    UNET_CKPT="${SEMANTIC2_DIR}/unet_latest.pt"
    UNET_ARG=""
    [[ -f "${UNET_CKPT}" ]] && UNET_ARG="--unet_checkpoint ${UNET_CKPT}"

    echo "LowLevel checkpoint:  ${LL_CKPT}"
    echo "Semantic checkpoint:  ${SEM_CKPT}"
    echo "UNet checkpoint:      ${UNET_ARG:-none}"

    python "${BRAIN_IT_DIR}/reconstruct.py" \
        --lowlevel_checkpoint "${LL_CKPT}" \
        --semantic_checkpoint "${SEM_CKPT}" \
        --mindeye2_checkpoint "${MINDEYE2_CKPT}" \
        --v2c_dir "${V2C_DIR}" \
        --data_root "${DATA_ROOT}" \
        --subjects subj01 subj02 subj05 \
        --output_dir "${RECON_DIR}" \
        --voxels_per_image 15000 \
        --diffusion_steps 38 \
        --init_step 14 \
        --output_size 256 \
        ${UNET_ARG} \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
    PY_PID=$!
    wait "${PY_PID}"
}

# ---------------------------------------------------------------------------
# Phase: Evaluate
# ---------------------------------------------------------------------------
run_evaluate() {
    echo "=== Phase: Evaluate ==="
    mkdir -p "${RESULTS_DIR}"

    python "${BRAIN_IT_DIR}/evaluate.py" \
        --recon_dir "${RECON_DIR}" \
        --gt_dir "${DATA_ROOT}" \
        --subjects subj01 subj02 subj05 \
        --output_dir "${RESULTS_DIR}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &
    PY_PID=$!
    wait "${PY_PID}"

    echo ""
    echo "Results saved to: ${RESULTS_DIR}/results.json"
    cat "${RESULTS_DIR}/results.json" || true
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${PHASE}" in
    v2c)        run_v2c ;;
    lowlevel)   run_lowlevel ;;
    semantic1)  run_semantic1 ;;
    semantic2)  run_semantic2 ;;
    reconstruct) run_reconstruct ;;
    evaluate)   run_evaluate ;;
    *)
        echo "ERROR: Unknown phase '${PHASE}'"
        echo "  Options: v2c | lowlevel | semantic1 | semantic2 | reconstruct | evaluate"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo ""
echo "=== $(date) === Phase '${PHASE}' exited with code ${EXIT_CODE} ==="
exit "${EXIT_CODE}"
