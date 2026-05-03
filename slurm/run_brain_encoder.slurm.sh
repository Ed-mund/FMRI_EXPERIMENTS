#!/usr/bin/env bash
# ============================================================================
# Universal Brain Encoder — NSD / Algonauts 2023  (Isambard-AI / Cray Shasta)
# ============================================================================
#
# Full 8-subject training (all 30 epochs):
#   sbatch run_brain_encoder.slurm.sh
#
# Resume after hitting the time limit (auto-detects latest checkpoint):
#   sbatch run_brain_encoder.slurm.sh --resume
#
# Resume from a specific checkpoint:
#   sbatch run_brain_encoder.slurm.sh --resume checkpoints/brain_encoder_YYYYMMDD/checkpoint_epoch10.pt
#
# Smoke test (CPU-only / no DINO download needed, 2 steps, no AMP):
#   sbatch --time=00:10:00 run_brain_encoder.slurm.sh --smoke-test
#
# Single subject quick run (for debugging on GPU):
#   sbatch run_brain_encoder.slurm.sh --subjects subj01 --epochs 2
#
# Notes:
#   - SLURM sends SIGUSR1 to the job 5 min before the time limit.
#     train.py catches it, saves a checkpoint, and exits cleanly.
#     Resubmit with --resume to continue.
#   - save_total_limit=3 keeps only the 3 most recent epoch checkpoints.
# ============================================================================
#SBATCH --job-name=brain-encoder
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=.logs/%x-%j.out
#SBATCH --error=.logs/%x-%j.err
#SBATCH --signal=USR1@300          # send SIGUSR1 300 s before the time limit

set -euo pipefail

CONDA_ENV="brain_encoder"
BRAIN_DIR="${PROJECTDIR}/brain/universal_brain_encoder"
DATA_ROOT="${PROJECTDIR}/brain/algonauts_prepared_data"
OUTPUT_DIR="${PROJECTDIR}/brain/checkpoints/brain_encoder_$(date +%Y%m%d)"

echo "=== $(date) === Job ${SLURM_JOB_ID} on $(hostname) ==="

# ---- Load CUDA module (Isambard-AI / Cray Shasta) ----
module load cuda/12.6 2>/dev/null \
  || module load cudatoolkit/24.11_12.6 2>/dev/null \
  || true
echo "CUDA_HOME: ${CUDA_HOME:-not set}"

nvidia-smi --list-gpus

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${BRAIN_DIR}:${PYTHONPATH:-}"

# ---- Cache dirs — keep everything off $HOME (quota limited) ----
export HF_HOME="${PROJECTDIR}/.cache/hf"
export HF_DATASETS_CACHE="${PROJECTDIR}/.cache/hf/datasets"
export TORCH_HOME="${PROJECTDIR}/.cache/torch"
export TRITON_CACHE_DIR="${SCRATCHDIR}/triton_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}"
mkdir -p .logs "${OUTPUT_DIR}"

# ---- PyTorch memory ----
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- W&B ----
export WANDB_PROJECT="${WANDB_PROJECT:-universal-brain-encoder}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-nsd-algonauts}"

# ---- GPU setup (Isambard-AI: SLURM does not set CUDA_VISIBLE_DEVICES) ----
GPU_IDS=$(nvidia-smi --list-gpus | awk '{print NR-1}' | tr '\n' ',' | sed 's/,$//')
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
GPUS=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}  (${GPUS} GPU(s))"

# Sanity-check CUDA before burning the allocation
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check PyTorch build'
d = torch.cuda.get_device_properties(0)
print(f'CUDA OK: {d.name}  {d.total_memory/1024**3:.0f} GB  bf16={torch.cuda.is_bf16_supported()}')
"

echo "Brain encoder dir: ${BRAIN_DIR}"
echo "Data root:         ${DATA_ROOT}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Arguments:         $*"
echo ""

# ---- Parse --smoke-test flag before forwarding remaining args ----
SMOKE_TEST=0
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "--smoke-test" ]]; then
        SMOKE_TEST=1
    else
        EXTRA_ARGS+=("${arg}")
    fi
done

if [[ "${SMOKE_TEST}" -eq 1 ]]; then
    echo "=== SMOKE TEST MODE ==="
    python "${BRAIN_DIR}/train.py" \
        --data_root "${DATA_ROOT}" \
        --subjects subj01 \
        --epochs 1 \
        --batch_size 4 \
        --voxels_per_image 100 \
        --num_workers 0 \
        --eval_every 1 \
        --save_every 1 \
        --wandb \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_group   "${WANDB_RUN_GROUP}" \
        --wandb_run_name "smoke-test-${SLURM_JOB_ID}" \
        --output_dir "${OUTPUT_DIR}/smoke_test" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    echo "=== Smoke test passed ==="
    exit 0
fi

# ---- Full / resumed training ----
python "${BRAIN_DIR}/train.py" \
    --data_root "${DATA_ROOT}" \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 \
    --epochs 30 \
    --batch_size 32 \
    --voxels_per_image 5000 \
    --lr 1e-3 \
    --loss_alpha 0.1 \
    --embedding_dim 256 \
    --projection_dim 128 \
    --lora_rank 16 \
    --use_amp \
    --eval_every 5 \
    --save_every 5 \
    --save_total_limit 3 \
    --num_workers 8 \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_group   "${WANDB_RUN_GROUP}" \
    --output_dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" &

PY_PID=$!

# Forward USR1 (preemption warning) and TERM (hard kill) to Python
trap "echo '[slurm] Forwarding USR1 to Python (pid ${PY_PID})…'; kill -USR1 ${PY_PID}" USR1
trap "echo '[slurm] SIGTERM — forwarding to Python…'; kill -USR1 ${PY_PID}" TERM

wait "${PY_PID}"
EXIT_CODE=$?

echo "=== $(date) === Python exited with code ${EXIT_CODE} ==="

if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "Training finished (or cleanly checkpointed)."
    LATEST=$(ls -t "${OUTPUT_DIR}"/checkpoint_epoch*.pt 2>/dev/null | head -1 || true)
    if [ -n "${LATEST}" ]; then
        echo ""
        echo "To continue, resubmit with:"
        echo "  sbatch run_brain_encoder.slurm.sh --resume"
    fi
fi

exit "${EXIT_CODE}"
