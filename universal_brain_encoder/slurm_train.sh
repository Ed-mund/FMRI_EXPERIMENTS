#!/bin/bash
#SBATCH --job-name=brain_encoder
#SBATCH --partition=gpu           # Adjust to your Isambard partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              # Single GH200 GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/brain_encoder_%j.out
#SBATCH --error=logs/brain_encoder_%j.err

# ---- Adjust these paths ----
DATA_ROOT="/path/to/algonauts_2023"
OUTPUT_DIR="checkpoints/universal_8subj_$(date +%Y%m%d)"
CONDA_ENV="brain_encoder"
# ----------------------------

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Load modules (adjust for Isambard)
module load cuda/12.1
module load anaconda3

# Activate environment
conda activate $CONDA_ENV

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "================"

# Multi-subject training (all 8 NSD subjects)
python train.py \
    --data_root "$DATA_ROOT" \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
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
    --save_every 10 \
    --num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Training complete. Output in $OUTPUT_DIR"
