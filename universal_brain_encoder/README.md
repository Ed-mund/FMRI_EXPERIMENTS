# Universal Brain Encoder

Implementation of "The Wisdom of a Crowd of Brains: A Universal Brain Encoder" (Beliy et al., 2025).

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n brain_encoder python=3.11 -y
conda activate brain_encoder

# PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install numpy scipy nibabel h5py pillow tqdm
```

### 2. Data Preparation

The easiest path is using the **Algonauts 2023** preprocessed format of NSD, which gives you
~40k pre-selected visual voxels with Z-scored betas per subject.

**Option A: Algonauts 2023 format (recommended)**

Download from: https://naturalscenesdataset.org/ or the Algonauts challenge page.

Expected structure:
```
algonauts_2023/
├── subj01/
│   ├── training_split/
│   │   ├── training_fmri/
│   │   │   ├── lh_training_fmri.npy   # shape: (N_train, N_voxels_lh)
│   │   │   └── rh_training_fmri.npy   # shape: (N_train, N_voxels_rh)
│   │   └── training_images/
│   │       ├── train-0000_nsd-00001.png
│   │       └── ...
│   └── test_split/
│       └── test_images/
├── subj02/
│   └── ...
...
└── subj08/
```

**Option B: Raw NSD betas**

If you downloaded the full NSD, you'll need:
- `nsddata/ppdata/subj{XX}/func1pt8mm/betas_fithrf_GLMdenoise_RR/` (beta weights)
- `nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5` (73k COCO images)
- `nsddata/experiments/nsd/nsd_expdesign.mat` (trial design)

Use `NSDRawDataset` in `dataset.py` for this path. You'll also need to create a voxel
selection mask (e.g., from the Algonauts ROI definitions or by SNR thresholding).

### 3. Training

**Single subject (baseline):**
```bash
python train.py \
    --data_root /path/to/algonauts_2023 \
    --subjects subj01 \
    --epochs 30 \
    --batch_size 32 \
    --output_dir checkpoints/single_subj01
```

**Multi-subject "Crowd of Brains" (main result):**
```bash
python train.py \
    --data_root /path/to/algonauts_2023 \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --epochs 30 \
    --batch_size 32 \
    --output_dir checkpoints/universal_8subj
```

**Transfer learning to new subject:**
```bash
python train.py \
    --data_root /path/to/algonauts_2023 \
    --subjects subj01 \
    --transfer_from checkpoints/universal_8subj/best_model.pt \
    --freeze_shared \
    --epochs 20 \
    --output_dir checkpoints/transfer_subj01
```

### 4. SLURM on Isambard

```bash
sbatch slurm_train.sh
```

## Architecture Overview

```
Input: Image (224x224) + Voxel Index
                │
    ┌───────────┴───────────┐
    ▼                       ▼
(a) DINO-v2 ViT-L/14    (b) Voxel Embedding
    + LoRA (Wo only)        (256-dim vector)
    Layers: 1,6,12,18,24     │
    Output: (L=5, P=256, C=128)
    │                       │
    └───────────┬───────────┘
                ▼
    (c) Cross-Attention Block
    ├── Spatial Attention  (which patches?)
    ├── Per-layer MLPs     (transform features)
    └── Functional Attention (which feature levels?)
                │
                ▼
    Predicted fMRI activation (scalar)
```

## Key Design Choices

- **Voxel-centric**: Each voxel gets its own 256-dim embedding. All other weights shared.
- **Multi-scale features**: 5 DINO layers from low-level to semantic.
- **LoRA on Wo only**: Efficient DINO adaptation for data-limited regime.
- **Loss**: 0.1*MSE - 0.9*cosine (cosine similarity dominates).
- **Training**: 32 images/batch, 5000 random voxels sampled per image.

## Hardware Requirements

- **Minimum**: Single GPU with 24GB VRAM (RTX 3090 / RTX 4090)
- **Recommended**: 48GB+ GPU (A6000 / RTX 8000 / A100)
- **Paper used**: Single Quadro RTX 8000, ~1 day for 8 NSD subjects
- **Your DGX Spark**: Should handle this comfortably with 120GB unified memory
- **Isambard GH200**: Great fit, use SLURM script below

## Files

- `model.py` - Full architecture (DINO+LoRA, voxel embeddings, cross-attention)
- `dataset.py` - NSD data loaders (Algonauts format + raw NSD)
- `train.py` - Training loop with multi-subject support & transfer learning
- `slurm_train.sh` - SLURM submission script for Isambard
