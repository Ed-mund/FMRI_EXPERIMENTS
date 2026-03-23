# Zero-Shot Brain Encoding via Anatomy-Conditioned Voxel Embedding Prediction

## TL;DR

We extend the Universal Brain Encoder (Beliy et al., 2025) to predict fMRI
responses for a **completely new subject with zero fMRI data** — using only their
structural/anatomical MRI. We do this by training a hypernetwork that predicts
voxel embeddings from anatomical features, eliminating the need for any
functional scanning of the new subject.

---

## 1. Background & Motivation

### The Universal Brain Encoder (what already exists)

The "Wisdom of a Crowd of Brains" paper (Beliy et al., 2025 — `2406.12179v3`)
introduced a Universal Image-to-fMRI Brain Encoder with this architecture:

```
Input: Image (224×224) + Voxel Index
         │                    │
         ▼                    ▼
(a) DINO-v2 ViT-L/14    (b) Voxel Embedding
    + LoRA (Wo only)         (256-dim learned vector)
    Layers 1,6,12,18,24      │
    → (L=5, P=256, C=128)    │
         │                    │
         └────────┬───────────┘
                  ▼
(c) Cross-Attention Block
    ├── Spatial Attention   (which image patches matter?)
    ├── Per-layer MLPs      (transform features)
    └── Functional Attention (which feature levels matter?)
                  │
                  ▼
    Predicted fMRI activation (scalar)
```

**Key insight:** All network weights are shared across all voxels of all subjects.
The ONLY subject/voxel-specific component is the 256-dim voxel embedding vector.
This embedding captures the *functional role* of each brain voxel — what visual
features it cares about.

**What the paper showed:**
- Joint training on 8 NSD subjects beats single-subject models (Section 4.1)
- Transfer learning to a new subject requires only ~100 image-fMRI pairs and
  only the voxel embeddings are optimized (Section 4.3)
- Learned voxel embeddings cluster by functional role, not by subject identity,
  and these clusters correspond to known brain regions like FFA (faces), PPA
  (places), EBA (bodies) etc. (Section 5, Figs 7, S14-S18)

### The gap: still needs *some* fMRI data

Even the transfer learning result needs ~100 fMRI scans per new subject. The
voxel embeddings are initialized randomly and optimized against real fMRI data.
For a truly new subject with zero functional data, there's no signal to learn
embeddings from.

### Our hypothesis

If voxel embeddings encode functional roles, and functional roles are
(partially) predictable from brain anatomy, then we can **predict voxel
embeddings from anatomical features alone**. The strong clustering by brain
region in the t-SNE visualizations (Figs S17, S18) suggests this is feasible —
nearby anatomical regions tend to have similar embeddings.

---

## 2. The Proposed Experiment

### Overview

```
Phase 1: Train the Universal Brain Encoder (already implemented)
  - Train on N subjects from NSD (e.g., 7 out of 8)
  - Result: shared weights + learned voxel embeddings for all trained subjects

Phase 2: Train the Anatomy-to-Embedding Hypernetwork (NEW)
  - Input: anatomical features of each voxel (from structural MRI)
  - Target: the learned voxel embeddings from Phase 1
  - Train on the same N subjects used in Phase 1
  - Result: a model that predicts voxel embeddings from anatomy

Phase 3: Zero-Shot Evaluation (NEW)
  - Take the held-out subject (not seen in Phase 1 or 2)
  - Predict their voxel embeddings from anatomy alone (Phase 2 model)
  - Plug predicted embeddings into the frozen Universal Encoder (Phase 1)
  - Evaluate encoding performance (Pearson correlation, retrieval accuracy)
  - Compare against: random embeddings, ROI-average baselines, and
    transfer learning with varying amounts of fMRI data
```

### Leave-One-Subject-Out Protocol

For rigorous evaluation, we do 8-fold leave-one-out:

```
For each subject S_i in {subj01, ..., subj08}:
  1. Train Universal Encoder on the other 7 subjects → shared weights + 
     7 subjects' worth of learned embeddings
  2. Train hypernetwork on those 7 subjects' (anatomy → embedding) pairs
  3. For held-out S_i: predict embeddings from anatomy only
  4. Evaluate encoding quality on S_i's test set (shared ~1000 images)
  5. Also evaluate: transfer learning with 10, 50, 100 fMRI pairs for
     comparison (showing where zero-shot sits on the data-efficiency curve)
```

---

## 3. Anatomical Features (Hypernetwork Input)

For each brain voxel, we extract a feature vector from the subject's structural
MRI and FreeSurfer parcellation. The NSD dataset provides all of these.

### Available per-voxel features from NSD:

| Feature | Source | Dim | Description |
|---------|--------|-----|-------------|
| MNI coordinates | FreeSurfer transform | 3 | Standardized brain position (x, y, z) |
| Cortical curvature | FreeSurfer | 1 | Local folding geometry |
| Sulcal depth | FreeSurfer | 1 | Depth within a sulcus |
| Cortical thickness | FreeSurfer | 1 | Gray matter thickness |
| ROI label (one-hot) | NSD ROI masks | ~20-40 | Which visual region (V1, V2, V3, hV4, FFA, PPA, EBA, etc.) |
| pRF parameters | NSD prf fits | 4 | Population receptive field: x, y, size, eccentricity |
| R² (signal quality) | NSD | 1 | Explained variance of the GLM fit |

The pRF (population receptive field) parameters are especially interesting —
they describe what part of the visual field each voxel responds to, which is
highly relevant to what the spatial attention in the encoder needs to learn.

**Total feature dimension per voxel:** ~30-50 depending on ROI encoding.

### Where these files live on Isambard:

```
# Structural / anatomical
/projects/b6ac/brain/nsd_data/nsddata/ppdata/subj01/func1pt8mm/roi/

# pRF parameters (already downloaded)
/projects/b6ac/brain/nsd_data/nsddata/ppdata/subj01/func1pt8mm/prf_angle.nii.gz
/projects/b6ac/brain/nsd_data/nsddata/ppdata/subj01/func1pt8mm/prf_eccentricity.nii.gz
/projects/b6ac/brain/nsd_data/nsddata/ppdata/subj01/func1pt8mm/prf_size.nii.gz
/projects/b6ac/brain/nsd_data/nsddata/ppdata/subj01/func1pt8mm/prf_R2.nii.gz

# FreeSurfer anatomy (may need to download)
/projects/b6ac/brain/nsd_data/nsddata/ppdata/subj01/anat/

# Transforms to MNI space
/projects/b6ac/brain/nsd_data/nsddata/ppdata/subj01/transforms/

# Betas (functional data for Phase 1 training)
/projects/b6ac/brain/nsd_data/nsddata_betas/ppdata/subj01/func1pt8mm/betas_fithrf_GLMdenoise_RR/

# Stimuli (73k COCO images)
/projects/b6ac/brain/nsd_data/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5

# Experimental design (trial → stimulus mapping)
/projects/b6ac/brain/nsd_data/nsddata/experiments/nsd/nsd_expdesign.mat
```

---

## 4. Hypernetwork Architecture

The hypernetwork is intentionally simple — we want to show that anatomy
*predicts* function, not that we built an elaborate architecture.

```
Voxel Anatomical Features (dim ~30-50)
         │
         ▼
   Linear(in_features, 512)
   LayerNorm + GELU
         │
         ▼
   Linear(512, 512)
   LayerNorm + GELU
         │
         ▼
   Linear(512, 256)
         │
         ▼
Predicted Voxel Embedding (256-dim)
```

**Training:**
- Input: anatomical feature vector for each voxel
- Target: the learned embedding for that voxel (from Phase 1)
- Loss: MSE + cosine similarity between predicted and learned embeddings
  (mirroring the encoder's own loss philosophy)
- Data: ~40,000 voxels × 7 subjects = ~280,000 training pairs
- This is a tiny model — trains in minutes

**Possible upgrades (try after baseline works):**
- Graph neural network over neighbouring voxels (spatial context)
- Separate prediction heads for different brain regions
- Contrastive loss to preserve embedding space structure
- Transformer over local anatomical patches

---

## 5. Evaluation Protocol

### Metrics (same as the paper)

1. **Pearson Correlation (per voxel):** Correlation between predicted and 
   ground-truth fMRI activations across all test images. Report median, 25th,
   75th percentiles.

2. **Image Retrieval (per image):** For each real test fMRI, retrieve the 
   matching image from N=1000 candidates using the encoder's predicted fMRIs.
   Report Top-1 and Top-5 accuracy.

### Comparisons (what to put in the results table)

| Model | fMRI data needed | Description |
|-------|-----------------|-------------|
| Random embeddings | 0 | Random 256-dim vectors (lower bound) |
| ROI-average embeddings | 0 | Mean embedding per ROI from trained subjects |
| Nearest-MNI embeddings | 0 | Copy embedding from nearest trained voxel in MNI space |
| **Hypernetwork (ours)** | **0** | **Predicted from anatomy** |
| Transfer learning (100) | 100 images | Paper's transfer learning with 100 pairs |
| Transfer learning (full) | ~9000 images | Paper's transfer learning with all data |
| Full single-subject | ~9000 images | Trained from scratch on all subject data |
| Universal multi-subject | ~72000 images | Trained on all 8 subjects jointly |

The key comparison is where our zero-shot result falls relative to transfer
learning with N fMRI pairs. If zero-shot matches transfer-100, that's already
a strong result. If it's between transfer-50 and transfer-200, that's
publishable and practically useful.

### Statistical testing

- Repeat with all 8 leave-one-out folds
- Report mean ± std across folds
- Paired t-test between zero-shot and baselines
- FDR correction for multiple comparisons (following the paper's protocol)

---

## 6. What's Already Built

### Existing codebase (`universal_brain_encoder/`)

| File | Status | Description |
|------|--------|-------------|
| `model.py` | ✅ Complete | Full encoder architecture: LoRA-DINO, voxel embeddings, cross-attention |
| `dataset.py` | ✅ Complete | NSD data loaders (Algonauts format + raw NSD betas) |
| `train.py` | ✅ Complete | Training loop with multi-subject support + transfer learning |
| `explore_embeddings.py` | ✅ Complete | Post-training embedding clustering and t-SNE visualization |
| `slurm_train.sh` | ✅ Complete | SLURM script for Isambard GH200 nodes |

### Data on Isambard (`/projects/b6ac/brain/nsd_data/`)

| Data | Status | Path |
|------|--------|------|
| NSD stimuli (73k images) | ✅ Downloaded | `nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5` (36.8 GB) |
| NSD betas (all 8 subj) | ✅ Downloaded | `nsddata_betas/ppdata/subj{01-08}/func1pt8mm/betas_fithrf_GLMdenoise_RR/` |
| ROI masks | ✅ Downloaded | `nsddata/ppdata/subj{01-08}/func1pt8mm/roi/` |
| Experimental design | ✅ Downloaded | `nsddata/experiments/nsd/nsd_expdesign.mat` |
| pRF parameters | ✅ Downloaded | `nsddata/ppdata/subj{01-08}/func1pt8mm/prf_*.nii.gz` |
| FreeSurfer anatomy | ⚠️ Check | `nsddata/ppdata/subj{01-08}/anat/` — may need `freesurfer/` too |
| MNI transforms | ⚠️ Check | `nsddata/ppdata/subj{01-08}/transforms/` |

### What still needs to be built

| Component | Priority | Description |
|-----------|----------|-------------|
| `preprocess_nsd.py` | 🔴 HIGH | Convert raw nii.gz betas + ROI masks → flat .npy arrays per subject |
| `preprocess_anatomy.py` | 🔴 HIGH | Extract per-voxel anatomical features from NSD structural data |
| `hypernetwork.py` | 🔴 HIGH | The anatomy → embedding prediction model |
| `train_hypernetwork.py` | 🔴 HIGH | Training loop for the hypernetwork |
| `eval_zeroshot.py` | 🟡 MEDIUM | Full leave-one-out evaluation pipeline |
| `baselines.py` | 🟡 MEDIUM | ROI-average and nearest-MNI baseline methods |
| Update `dataset.py` | 🟡 MEDIUM | Update beta path to `nsddata_betas/` (current code assumes `nsddata/`) |

---

## 7. Execution Plan

### Step 1: Preprocessing (do this first)

Write `preprocess_nsd.py` to:
1. Load ROI masks from `nsddata/ppdata/subj{XX}/func1pt8mm/roi/` to select
   visual cortex voxels (~40k per subject, matching Algonauts/Gifford et al.)
2. Load all beta sessions from `nsddata_betas/ppdata/subj{XX}/func1pt8mm/
   betas_fithrf_GLMdenoise_RR/betas_session{NN}.nii.gz` (or .hdf5)
3. Apply ROI mask to extract only visual voxels
4. Z-score normalize per session/run
5. Map trials to stimulus IDs using `nsd_expdesign.mat`
6. Average repeated presentations of the same image
7. Save as flat arrays: `{subject}_fmri.npy` (N_images × N_voxels) and
   `{subject}_image_ids.npy`
8. Identify the ~1000 shared test images (seen by all 8 subjects)

**Output:** One directory per subject with clean numpy arrays ready for training.

### Step 2: Train the Universal Encoder (Phase 1)

Use the existing `train.py` with all 8 subjects (or 7 for leave-one-out).
This is the standard paper replication — ~1 day on a single GPU.

```bash
python train.py \
    --data_root /projects/b6ac/brain/nsd_data/preprocessed \
    --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --epochs 30 --batch_size 32 \
    --output_dir checkpoints/universal_8subj
```

### Step 3: Extract Anatomical Features

Write `preprocess_anatomy.py` to:
1. For each subject, load the ROI masks (same voxel selection as Step 1)
2. Extract per-voxel: MNI coordinates, pRF params, ROI labels, curvature, etc.
3. Save as `{subject}_anatomy.npy` (N_voxels × feature_dim)

### Step 4: Train the Hypernetwork (Phase 2)

```bash
python train_hypernetwork.py \
    --encoder_checkpoint checkpoints/universal_7subj/best_model.pt \
    --anatomy_dir /projects/b6ac/brain/nsd_data/preprocessed \
    --train_subjects subj02 subj03 subj04 subj05 subj06 subj07 subj08 \
    --output_dir checkpoints/hypernetwork_loso_subj01
```

### Step 5: Zero-Shot Evaluation (Phase 3)

```bash
python eval_zeroshot.py \
    --encoder_checkpoint checkpoints/universal_7subj/best_model.pt \
    --hypernetwork_checkpoint checkpoints/hypernetwork_loso_subj01/best.pt \
    --test_subject subj01 \
    --data_root /projects/b6ac/brain/nsd_data/preprocessed
```

---

## 8. Risks & Mitigations

**Risk: Anatomy doesn't predict function well enough.**
Mitigation: The t-SNE plots strongly suggest it does at least at the ROI level.
Even coarse prediction (getting the right brain region) should outperform
random. The pRF parameters are especially powerful features since they directly
describe receptive field properties that the spatial attention learns.

**Risk: Hypernetwork overfits to 7 training subjects.**
Mitigation: With ~280k voxels and a small MLP, overfitting is unlikely. Use
dropout and early stopping. The leave-one-out protocol ensures we measure
generalization to unseen anatomy.

**Risk: MNI alignment is too coarse.**
Mitigation: We use multiple feature types (not just coordinates). ROI labels
and pRF parameters are computed in native space before alignment, so they carry
subject-specific functional information even if MNI coordinates are imprecise.

**Risk: Some brain regions are harder to predict than others.**
Mitigation: Report per-ROI results (as the paper does in Fig S10). Early visual
cortex (V1-V3) should be easiest since pRF parameters are highly informative
there. Higher areas (PPA, FFA) may need the ROI labels more.

---

## 9. Why This Matters

1. **Practical:** Structural MRI is cheaper, faster, and more widely available
   than fMRI. If you can predict brain encoding from a 5-minute structural scan
   instead of hours in an fMRI machine, it opens up brain-computer interfaces
   to far more people.

2. **Scientific:** Demonstrates a quantitative link between brain anatomy and
   visual function that goes beyond traditional ROI-level parcellations. The
   hypernetwork learns a continuous mapping from anatomical features to
   functional properties.

3. **For the PhD:** Connects to the "countering digital impersonation" theme —
   if neural encoding patterns are predictable from structural MRI, this has
   implications for neural privacy and biometric security. Also fits well as
   an ACCV 2026 submission alongside the diffusion model work.

---

## 10. Key References

- Beliy et al. (2025). "The Wisdom of a Crowd of Brains: A Universal Brain
  Encoder." arXiv:2406.12179v3. **[The base paper we build on]**
- Allen et al. (2022). "A massive 7T fMRI dataset to bridge cognitive
  neuroscience and AI." Nature Neuroscience. **[The NSD dataset]**
- Gifford et al. (2023). "The Algonauts Project 2023 Challenge."
  **[Visual voxel selection we follow]**
- Ha et al. (2017). "HyperNetworks." ICLR. **[Hypernetwork concept]**
- Scotti et al. (2024). "MindEye2: Shared-subject models enable fMRI-to-image
  with 1 hour of data." **[Related multi-subject approach, but for decoding]**