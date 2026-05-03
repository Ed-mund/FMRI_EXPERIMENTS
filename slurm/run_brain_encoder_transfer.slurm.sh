#!/usr/bin/env bash
# ============================================================================
# Universal Brain Encoder — subj08 Transfer Learning Evaluation
# ============================================================================
#
# Runs three conditions for subj08 and logs each to W&B:
#
#   scratch_100  — train subj08 from scratch, 100% data  (baseline)
#   transfer_20  — pretrained backbone frozen, 20% of subj08 data
#   transfer_100 — pretrained backbone frozen, 100% of subj08 data
#
# Submit manually after the 7-subject run finishes:
#   sbatch run_brain_encoder_transfer.slurm.sh
#
# Or chain automatically as a SLURM dependency (replace JOB_ID):
#   sbatch --dependency=afterok:JOB_ID run_brain_encoder_transfer.slurm.sh
#
# The script auto-detects the most recent best_model.pt under
#   ${PROJECTDIR}/brain/checkpoints/brain_encoder_*/best_model.pt
# Override by setting PRETRAINED_CKPT explicitly below.
# ============================================================================
#SBATCH --job-name=brain-transfer
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=.logs/%x-%j.out
#SBATCH --error=.logs/%x-%j.err
#SBATCH --signal=USR1@300

set -euo pipefail

CONDA_ENV="brain_encoder"
BRAIN_DIR="${PROJECTDIR}/brain/universal_brain_encoder"
DATA_ROOT="${PROJECTDIR}/brain/algonauts_prepared_data"
TRANSFER_BASE="${PROJECTDIR}/brain/checkpoints/transfer_subj08_$(date +%Y%m%d)"

# Auto-detect latest best_model.pt (override by setting PRETRAINED_CKPT)
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
if [[ -z "${PRETRAINED_CKPT}" ]]; then
    PRETRAINED_CKPT=$(ls -t "${PROJECTDIR}"/brain/checkpoints/brain_encoder_*/best_model.pt 2>/dev/null | head -1 || true)
fi

echo "=== $(date) === Job ${SLURM_JOB_ID} on $(hostname) ==="

# ---- Modules & env ----
module load cuda/12.6 2>/dev/null \
  || module load cudatoolkit/24.11_12.6 2>/dev/null \
  || true
echo "CUDA_HOME: ${CUDA_HOME:-not set}"

nvidia-smi --list-gpus

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${BRAIN_DIR}:${PYTHONPATH:-}"

# ---- Cache dirs ----
export HF_HOME="${PROJECTDIR}/.cache/hf"
export HF_DATASETS_CACHE="${PROJECTDIR}/.cache/hf/datasets"
export TORCH_HOME="${PROJECTDIR}/.cache/torch"
export TRITON_CACHE_DIR="${SCRATCHDIR}/triton_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}"
mkdir -p .logs

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- W&B ----
export WANDB_PROJECT="${WANDB_PROJECT:-universal-brain-encoder}"
export WANDB_RUN_GROUP="subj08-transfer"

# ---- GPU setup ----
GPU_IDS=$(nvidia-smi --list-gpus | awk '{print NR-1}' | tr '\n' ',' | sed 's/,$//')
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
d = torch.cuda.get_device_properties(0)
print(f'CUDA OK: {d.name}  {d.total_memory/1024**3:.0f} GB')
"

if [[ -z "${PRETRAINED_CKPT}" ]]; then
    echo "ERROR: No pretrained checkpoint found. Run the 7-subject training first."
    echo "  Or set PRETRAINED_CKPT=/path/to/best_model.pt and resubmit."
    exit 1
fi

echo ""
echo "Pretrained checkpoint: ${PRETRAINED_CKPT}"
echo "Transfer output base:  ${TRANSFER_BASE}"
echo ""

# ---- Signal forwarding (preemption) ----
_run_condition() {
    local name="$1"; shift
    local out_dir="${TRANSFER_BASE}/${name}"
    mkdir -p "${out_dir}"
    echo ">>> $(date) — Starting condition: ${name}"
    python "${BRAIN_DIR}/train.py" "$@" \
        --data_root "${DATA_ROOT}" \
        --subjects subj08 \
        --use_amp \
        --eval_every 5 \
        --save_every 10 \
        --save_total_limit 2 \
        --num_workers 8 \
        --wandb \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_group   "${WANDB_RUN_GROUP}" \
        --wandb_run_name "${name}" \
        --output_dir "${out_dir}"
    echo ">>> $(date) — Finished condition: ${name}"
    echo ""
}

# ============================================================
# Condition 1: scratch_100 — train subj08 from scratch
#   Baseline: how well can a single subject do without transfer?
# ============================================================
_run_condition "scratch_100" \
    --epochs 30 \
    --batch_size 32 \
    --voxels_per_image 5000 \
    --lr 1e-3 \
    --loss_alpha 0.1 \
    --embedding_dim 256 \
    --projection_dim 128 \
    --lora_rank 16

# ============================================================
# Condition 2: transfer_20 — pretrained backbone, 20% data
#   Few-shot: can the universal model work with little data?
# ============================================================
_run_condition "transfer_20pct" \
    --transfer_from "${PRETRAINED_CKPT}" \
    --freeze_shared \
    --data_fraction 0.20 \
    --epochs 20 \
    --batch_size 32 \
    --voxels_per_image 5000 \
    --lr 1e-3 \
    --loss_alpha 0.1 \
    --embedding_dim 256 \
    --projection_dim 128 \
    --lora_rank 16

# ============================================================
# Condition 3: transfer_100 — pretrained backbone, 100% data
#   Full transfer: best expected performance with transfer
# ============================================================
_run_condition "transfer_100pct" \
    --transfer_from "${PRETRAINED_CKPT}" \
    --freeze_shared \
    --data_fraction 1.0 \
    --epochs 20 \
    --batch_size 32 \
    --voxels_per_image 5000 \
    --lr 1e-3 \
    --loss_alpha 0.1 \
    --embedding_dim 256 \
    --projection_dim 128 \
    --lora_rank 16

echo "=== $(date) === All transfer conditions complete ==="
echo ""
echo "Results in: ${TRANSFER_BASE}"
echo "W&B group:  ${WANDB_RUN_GROUP}  (project: ${WANDB_PROJECT})"
