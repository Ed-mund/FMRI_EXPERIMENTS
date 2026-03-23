# Brain-IT: Image Reconstruction from fMRI

Implementation of "Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer" (Beliy et al., ICLR 2026), built on top of the Universal Brain Encoder (Stage 1).

## Prerequisites

- **Stage 1 complete**: `best_model.pt` at `/projects/b6ac/brain/checkpoints/brain_encoder_20260318/`
- **Data**: Algonauts 2023 prepared data at `/projects/b6ac/brain/algonauts_prepared_data/`
- **MindEye2 weights** (for semantic stage 2 + reconstruction):  
  Download from HuggingFace: `huggingface.co/datasets/pscotti/mindeyev2`  
  Place at: `/projects/b6ac/brain/checkpoints/mindeye2/`

### Install dependencies

```bash
conda activate brain_encoder
pip install open-clip-torch scikit-learn diffusers accelerate torchmetrics clip
```

## Pipeline Overview

```
best_model.pt (encoder voxel embeddings)
    │
    ▼
[v2c_mapping.py] → GMM (128 clusters) + per-voxel cluster assignments
    │
    ├──► [train_lowlevel.py] → LowLevelBIT (predicts VGG features)
    │        │
    │        ▼
    │    [DIPInverter] → coarse 112→256px image at inference
    │
    └──► [train_semantic.py] → SemanticBIT (predicts 256 CLIP tokens)
             │
             ▼
         [Stage 2] + MindEye2 diffusion model (joint training)
             │
             ▼
         [reconstruct.py] → DIP init + 38-step SDXL denoising → 256×256 image
             │
             ▼
         [evaluate.py] → 8 metrics (PixCorr, SSIM, Alex2/5, Incep, CLIP, Eff, SwAV)
```

## Execution Order

### Step 1: V2C Mapping (~5 minutes)
```bash
sbatch run_brain_it.slurm.sh --phase v2c
```
Outputs to `brain_it/v2c/`: GMM, per-subject cluster assignments, voxel embeddings.

### Step 2: (Optional) Pre-compute COCO fMRI predictions (~1-2h)
Pre-compute synthetic fMRI responses for ~120K unlabeled COCO images using the frozen encoder. This enriches training data significantly.
```bash
# Run predict_coco_fmri.py (create manually or skip for now)
# If skipped, Brain-IT trains on NSD data only (~70K samples, still reasonable)
```

### Step 3: Train Low-Level BIT (~12h)
```bash
sbatch --time=14:00:00 run_brain_it.slurm.sh --phase lowlevel
# Resume if interrupted:
sbatch run_brain_it.slurm.sh --phase lowlevel --resume
```

### Step 4: Train Semantic BIT Stage 1 (~8h)
```bash
sbatch --time=10:00:00 run_brain_it.slurm.sh --phase semantic1
# Resume:
sbatch run_brain_it.slurm.sh --phase semantic1 --resume
```

### Step 5: Train Semantic BIT Stage 2 (~20h on 1×GH200)
Requires MindEye2 diffusion weights. Joint training of SemanticBIT + UNet.
```bash
sbatch --time=24:00:00 run_brain_it.slurm.sh --phase semantic2
# Resume:
sbatch run_brain_it.slurm.sh --phase semantic2 --resume
```

### Step 6: Reconstruct Test Images
```bash
sbatch --time=06:00:00 run_brain_it.slurm.sh --phase reconstruct
```
Outputs to `brain_it/reconstructions/{subj01,subj02,subj05}/`.

### Step 7: Evaluate
```bash
sbatch --time=01:00:00 run_brain_it.slurm.sh --phase evaluate
```
Results saved to `brain_it/results/results.json`.

## Module Reference

| File | Description |
|------|-------------|
| `v2c_mapping.py` | Extract encoder embeddings → GMM → cluster assignments |
| `bit_model.py` | BrainTokenizer, CrossTransformer, LowLevelBIT, SemanticBIT |
| `vgg_features.py` | VGG-16+BN extractor, tokenisation, preprocessing |
| `clip_features.py` | OpenCLIP ViT-bigG/14 spatial token extractor |
| `dip.py` | U-Net Deep Image Prior, inference-time optimiser |
| `dataset.py` | NSD + COCO dataset loaders, multi-subject collate |
| `train_lowlevel.py` | Train LowLevelBIT (InfoNCE, 60 epochs) |
| `train_semantic.py` | Train SemanticBIT stage 1 (L2) + stage 2 (diffusion) |
| `reconstruct.py` | Full inference: DIP → diffusion → 256px image |
| `evaluate.py` | 8 standard metrics, per-subject + aggregate |

## Architecture Details

### BrainTokenizer
- Input: fMRI activations (sampled 15K voxels) × per-voxel 512-dim embeddings
- Single-head graph attention: cluster embeddings (Q) attend over modulated voxel activations (K,V)
- Restriction: each cluster only attends to voxels assigned to it via V2C mapping
- Output: 128 Brain Tokens × 512-dim

### CrossTransformer
- 1 initial cross-attention: learnable query tokens ← Brain Tokens
- 5 × (self-attention + cross-attention), 8 heads each
- Final linear projection to output feature dimension

### Low-level Branch
- Predicts VGG-16+BN features from 5 layers (total ~7K tokens at training time)
- Training: InfoNCE loss per layer, independent contrastive objectives
- Inference: predict all tokens → DIP inversion → coarse image

### Semantic Branch
- Predicts 256 OpenCLIP ViT-bigG/14 spatial tokens (1280-dim each)
- Stage 1: L2 alignment loss
- Stage 2: joint training with MindEye2 unCLIP SDXL (diffusion MSE loss)
- Inference: CLIP tokens condition 38-step SDXL denoising from DIP-initialised latent

## Expected Metrics (from paper Table 1)

| Metric | Brain-IT (paper) |
|--------|-----------------|
| PixCorr | 0.351 |
| SSIM | 0.382 |
| Alex(2) | 0.871 |
| Alex(5) | 0.906 |
| Incep | 0.911 |
| CLIP | 0.867 |
| Eff | 0.829 |
| SwAV | 0.512 |

*Averaged over subj01, subj02, subj05. Our results may differ slightly due to training on 7 subjects (vs. 8 in the paper) and single-GPU training.*

## Notes on Deviations from Paper

1. **GPU**: Paper used 4×H100 for semantic stage 2. We use 1×GH200 (120GB).  
   Compensated by `--batch_size 8 --grad_accum_steps 8` (effective batch = 64).

2. **Subjects**: We have 7 subjects (subj01–07); paper used all 8 NSD subjects.  
   Impact should be minimal since encoders are cross-subject.

3. **COCO augmentation**: Optional. Skip step 2 if COCO data not available;  
   training on NSD alone (~70K samples) should give close results.
