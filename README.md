# FMRI_EXPERIMENTS

Research code for fMRI-based brain decoding and image reconstruction, running on Isambard-AI (GH200 cluster).

## Projects

### 1. `universal_brain_encoder/`
A universal cross-subject brain encoder based on the hypernetwork architecture from:
> *Beliy et al. — "The Wisdom of a Crowd of Brains: A Universal Brain Encoder" (2025)*

Trains a shared encoder across all NSD subjects using anatomy-conditioned hypernetworks. Produces per-voxel embeddings used downstream by Brain-IT.

### 2. `brain_it/`
fMRI-to-image reconstruction pipeline based on:
> *Beliy et al. — "Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer" (ICLR 2026)*

Takes the universal brain encoder's voxel embeddings and reconstructs perceived images via:
- **BrainTokenizer** — graph-attention over voxel clusters → 128 Brain Tokens
- **LowLevelBIT** — predicts VGG-16 features → coarse image via Deep Image Prior
- **SemanticBIT** — predicts 256 OpenCLIP ViT-bigG/14 spatial tokens
- **Stage 2** — joint diffusion fine-tuning with MindEye2 unCLIP SDXL → 256×256 image

## Pipeline

```
[universal_brain_encoder] → best_model.pt
         │
         ▼
[brain_it/v2c_mapping.py]    # GMM clustering of voxel embeddings
         │
         ├──► [train_lowlevel.py]   # VGG feature prediction (~12h)
         │
         └──► [train_semantic.py]   # CLIP token prediction + diffusion (~28h)
                       │
                       ▼
              [reconstruct.py]      # DIP init + SDXL denoising → image
                       │
                       ▼
              [evaluate.py]         # PixCorr, SSIM, Alex, CLIP, etc.
```

## Setup

```bash
conda activate brain_encoder
pip install open-clip-torch scikit-learn diffusers accelerate torchmetrics clip
```

## Running (Isambard-AI / Slurm)

```bash
# Stage 1: Universal brain encoder (see universal_brain_encoder/README.md)

# Stage 2: Brain-IT pipeline
sbatch run_brain_it.slurm.sh --phase v2c
sbatch --time=14:00:00 run_brain_it.slurm.sh --phase lowlevel
sbatch --time=10:00:00 run_brain_it.slurm.sh --phase semantic1
sbatch --time=24:00:00 run_brain_it.slurm.sh --phase semantic2
sbatch --time=06:00:00 run_brain_it.slurm.sh --phase reconstruct
sbatch --time=01:00:00 run_brain_it.slurm.sh --phase evaluate
```

## Data

- **NSD (Natural Scenes Dataset)**: 7 subjects, ~66K fMRI-image pairs
- **Algonauts 2023**: prepared data at `/projects/b6ac/brain/algonauts_prepared_data/`
- Checkpoints and weights are excluded from this repo (too large)

## Hardware

Trained on NVIDIA GH200 120GB (single GPU) on the Isambard-AI supercomputer.
