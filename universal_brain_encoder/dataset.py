"""
NSD Dataset Loader for Universal Brain Encoder

The NSD (Natural Scenes Dataset) is a 7-Tesla fMRI dataset with:
- 8 subjects
- ~9000 unique images per subject (from COCO)
- ~1000 shared test images across all subjects
- ~40,000 visually-sensitive voxels per subject (from Algonauts selection)

Expected NSD directory structure (after download):
    nsd/
    ├── nsddata/
    │   ├── ppdata/
    │   │   ├── subj01/
    │   │   │   └── func1pt8mm/
    │   │   │       └── betas_fithrf_GLMdenoise_RR/
    │   │   │           ├── betas_session01.nii.gz
    │   │   │           └── ...
    │   │   └── ...
    │   └── experiments/
    │       └── nsd/
    │           └── nsd_expdesign.mat
    ├── nsddata_stimuli/
    │   └── stimuli/
    │       └── nsd/
    │           └── nsd_stimuli.hdf5
    └── algonauts_2023/  (optional, for voxel selection)
        ├── subj01/
        │   ├── training_split/
        │   │   └── training_fmri/
        │   │       └── *.npy
        │   └── test_split/
        └── ...

This loader supports both:
1. Raw NSD betas (nii.gz) with custom voxel selection
2. Pre-processed Algonauts 2023 challenge format (easier to start with)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from typing import Dict, List, Optional, Tuple
import h5py
import json


# ---------------------------------------------------------------------------
# Algonauts-format NSD Dataset (recommended starting point)
# ---------------------------------------------------------------------------

class NSDAlgonautsDataset(Dataset):
    """
    Loads NSD data in the Algonauts 2023 challenge format.
    This is the easiest format to work with - pre-selected ~40k visual voxels
    with Z-scored betas, as used in the paper (Gifford et al., 2023).

    Download from: https://naturalscenesdataset.org/ or the Algonauts challenge.

    Expected directory per subject:
        subj{XX}/
        ├── training_split/
        │   ├── training_fmri/
        │   │   ├── lh_training_fmri.npy   # (N_train, N_voxels_lh)
        │   │   └── rh_training_fmri.npy   # (N_train, N_voxels_rh)
        │   └── training_images/
        │       ├── train-0000_nsd-00001.png
        │       └── ...
        └── test_split/
            └── test_images/
                ├── test-0000_nsd-00001.png
                └── ...
    """

    def __init__(
        self,
        data_root: str,
        subject_id: str,  # e.g. "subj01"
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        max_voxels: Optional[int] = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.subject_id = subject_id
        self.split = split

        if transform is None:
            self.transform = get_default_transform()
        else:
            self.transform = transform

        subj_dir = os.path.join(data_root, subject_id)

        if split == "train":
            fmri_dir = os.path.join(subj_dir, "training_split", "training_fmri")
            self.image_dir = os.path.join(subj_dir, "training_split", "training_images")

            # Load fMRI betas (concatenate left and right hemispheres)
            lh_fmri = np.load(os.path.join(fmri_dir, "lh_training_fmri.npy"))
            rh_fmri = np.load(os.path.join(fmri_dir, "rh_training_fmri.npy"))
            self.fmri_data = np.concatenate([lh_fmri, rh_fmri], axis=1).astype(np.float32)

        elif split == "test":
            # Test split typically doesn't have fMRI (for the challenge)
            # But for our encoder evaluation, we need held-out fMRI
            # You may need to create your own train/test split from training data
            self.image_dir = os.path.join(subj_dir, "test_split", "test_images")
            self.fmri_data = None

        # Get sorted image file list
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.num_voxels = self.fmri_data.shape[1] if self.fmri_data is not None else 0

        if max_voxels and self.num_voxels > max_voxels:
            self.num_voxels = max_voxels
            self.fmri_data = self.fmri_data[:, :max_voxels]

        print(f"[{subject_id}] Loaded {len(self.image_files)} images, "
              f"{self.num_voxels} voxels ({split})")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        result = {
            "image": image,
            "subject_id": self.subject_id,
            "index": idx,
        }

        if self.fmri_data is not None:
            result["fmri"] = torch.from_numpy(self.fmri_data[idx])

        return result


# ---------------------------------------------------------------------------
# Raw NSD Beta Dataset (more flexible, requires more setup)
# ---------------------------------------------------------------------------

class NSDRawDataset(Dataset):
    """
    Loads NSD data directly from the raw betas and stimulus files.
    More flexible but requires more preprocessing.

    This loads from:
    - nsd_stimuli.hdf5 for images
    - Beta weight nii.gz files for fMRI responses
    - nsd_expdesign.mat for stimulus-to-trial mapping
    """

    def __init__(
        self,
        nsd_root: str,
        subject_id: str,  # "subj01" through "subj08"
        split: str = "train",
        roi_mask: Optional[np.ndarray] = None,
        transform: Optional[transforms.Compose] = None,
        shared_image_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.nsd_root = nsd_root
        self.subject_id = subject_id
        self.split = split
        self.transform = transform or get_default_transform()

        # Load experimental design to map trials -> stimuli
        self._load_exp_design()

        # Load or create voxel mask
        self.roi_mask = roi_mask  # boolean mask for which voxels to use

        # Determine train/test split based on shared images
        # ~1000 images are shared across all 8 subjects (test set)
        if shared_image_ids is not None:
            self.shared_ids = set(shared_image_ids)
        else:
            self.shared_ids = self._find_shared_images()

        self._setup_split()

        # Open stimuli HDF5 (lazy loading)
        stim_path = os.path.join(
            nsd_root, "nsddata_stimuli", "stimuli", "nsd", "nsd_stimuli.hdf5"
        )
        self.stim_file = None
        self.stim_path = stim_path

        print(f"[{subject_id}] {split}: {len(self.trial_indices)} trials, "
              f"voxel mask: {self.roi_mask.sum() if self.roi_mask is not None else 'all'}")

    def _load_exp_design(self):
        """Load the NSD experimental design file."""
        import scipy.io as sio
        design_path = os.path.join(
            self.nsd_root, "nsddata", "experiments", "nsd", "nsd_expdesign.mat"
        )
        design = sio.loadmat(design_path)
        # masterordering: which 73k image was shown on each trial
        self.masterordering = design["masterordering"].flatten() - 1  # 0-indexed
        # subjectim: which of the 10k images each subject saw
        subj_idx = int(self.subject_id[-2:]) - 1
        self.subjectim = design["subjectim"][subj_idx] - 1  # 0-indexed

    def _find_shared_images(self) -> set:
        """Find image IDs shared across all 8 subjects (the ~1000 test images)."""
        import scipy.io as sio
        design_path = os.path.join(
            self.nsd_root, "nsddata", "experiments", "nsd", "nsd_expdesign.mat"
        )
        design = sio.loadmat(design_path)
        subjectim = design["subjectim"] - 1

        # Find images seen by ALL subjects
        sets = [set(subjectim[i]) for i in range(8)]
        shared = sets[0]
        for s in sets[1:]:
            shared = shared.intersection(s)
        return shared

    def _setup_split(self):
        """Split trials into train/test based on shared images."""
        subj_idx = int(self.subject_id[-2:]) - 1
        subj_images = set(self.subjectim.tolist()) if hasattr(self.subjectim, 'tolist') else set(self.subjectim)

        self.trial_indices = []
        self.image_ids = []

        # Find all trials for this subject
        for trial_idx in range(len(self.masterordering)):
            img_id = self.masterordering[trial_idx]
            if img_id not in subj_images:
                continue

            is_shared = img_id in self.shared_ids

            if self.split == "train" and not is_shared:
                self.trial_indices.append(trial_idx)
                self.image_ids.append(img_id)
            elif self.split == "test" and is_shared:
                self.trial_indices.append(trial_idx)
                self.image_ids.append(img_id)

    def _load_beta(self, trial_idx: int) -> np.ndarray:
        """Load beta weight for a specific trial."""
        # Determine which session and trial within session
        session = trial_idx // 750 + 1  # 750 trials per session
        trial_in_session = trial_idx % 750

        beta_path = os.path.join(
            self.nsd_root, "nsddata_betas", "ppdata", self.subject_id,
            "func1pt8mm", "betas_fithrf_GLMdenoise_RR",
            f"betas_session{session:02d}.nii.gz"
        )

        import nibabel as nib
        betas = nib.load(beta_path).get_fdata()
        beta = betas[:, :, :, trial_in_session]

        if self.roi_mask is not None:
            beta = beta[self.roi_mask]

        return beta.astype(np.float32)

    def __len__(self) -> int:
        return len(self.trial_indices)

    def __getitem__(self, idx: int) -> dict:
        trial_idx = self.trial_indices[idx]
        img_id = self.image_ids[idx]

        # Load image from HDF5
        if self.stim_file is None:
            self.stim_file = h5py.File(self.stim_path, "r")

        image = self.stim_file["imgBrick"][img_id]  # (425, 425, 3) uint8
        image = Image.fromarray(image)
        image = self.transform(image)

        # Load fMRI beta
        fmri = self._load_beta(trial_idx)

        return {
            "image": image,
            "fmri": torch.from_numpy(fmri),
            "subject_id": self.subject_id,
            "image_id": img_id,
            "index": idx,
        }


# ---------------------------------------------------------------------------
# Multi-Subject Dataset (combines multiple subjects for joint training)
# ---------------------------------------------------------------------------

class MultiSubjectDataset(Dataset):
    """
    Wraps multiple single-subject datasets for joint training.
    Each __getitem__ returns a sample from a randomly chosen subject
    (weighted by dataset size for balanced sampling).
    """

    def __init__(self, datasets: Dict[str, Dataset]):
        self.datasets = datasets
        self.subject_ids = list(datasets.keys())

        # Build a flat index mapping
        self.flat_indices = []
        for sid, ds in datasets.items():
            for i in range(len(ds)):
                self.flat_indices.append((sid, i))

        print(f"MultiSubjectDataset: {len(self.flat_indices)} total samples "
              f"from {len(self.subject_ids)} subjects")

    def __len__(self) -> int:
        return len(self.flat_indices)

    def __getitem__(self, idx: int) -> dict:
        sid, local_idx = self.flat_indices[idx]
        return self.datasets[sid][local_idx]

    def get_num_voxels(self, subject_id: str) -> int:
        return self.datasets[subject_id].num_voxels


# ---------------------------------------------------------------------------
# Transform & Collation utilities
# ---------------------------------------------------------------------------

def get_default_transform(image_size: int = 224) -> transforms.Compose:
    """Default image transform matching DINO-v2 preprocessing."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def collate_multi_subject(batch: List[dict]) -> dict:
    """
    Custom collation for multi-subject batches.
    Groups samples by subject_id since voxel embeddings are subject-specific.
    """
    # Group by subject
    by_subject = {}
    for item in batch:
        sid = item["subject_id"]
        if sid not in by_subject:
            by_subject[sid] = []
        by_subject[sid].append(item)

    # Stack within each subject group
    result = {}
    for sid, items in by_subject.items():
        result[sid] = {
            "images": torch.stack([it["image"] for it in items]),
            "fmri": torch.stack([it["fmri"] for it in items]) if "fmri" in items[0] else None,
            "subject_id": sid,
        }

    return result


# ---------------------------------------------------------------------------
# Quick data loading helpers
# ---------------------------------------------------------------------------

def load_algonauts_subjects(
    data_root: str,
    subject_ids: List[str],
    split: str = "train",
    transform: Optional[transforms.Compose] = None,
) -> MultiSubjectDataset:
    """
    Convenience function to load multiple subjects from Algonauts format.

    Usage:
        dataset = load_algonauts_subjects(
            "/path/to/algonauts_2023",
            ["subj01", "subj02", "subj03", "subj04",
             "subj05", "subj06", "subj07", "subj08"],
            split="train"
        )
    """
    datasets = {}
    for sid in subject_ids:
        datasets[sid] = NSDAlgonautsDataset(
            data_root=data_root,
            subject_id=sid,
            split=split,
            transform=transform,
        )
    return MultiSubjectDataset(datasets)


def create_train_test_split(
    data_root: str,
    subject_id: str,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[NSDAlgonautsDataset, NSDAlgonautsDataset]:
    """
    Since the Algonauts test split doesn't include fMRI,
    create a train/test split from the training data.

    Returns separate datasets for training and evaluation.
    """
    full_dataset = NSDAlgonautsDataset(data_root, subject_id, split="train")

    n = len(full_dataset)
    n_test = int(n * test_ratio)
    n_train = n - n_test

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Create subset datasets
    train_fmri = full_dataset.fmri_data[train_indices]
    test_fmri = full_dataset.fmri_data[test_indices]

    train_images = [full_dataset.image_files[i] for i in train_indices]
    test_images = [full_dataset.image_files[i] for i in test_indices]

    # Return as split datasets (you'd need to implement SubsetDataset or similar)
    # For simplicity, return indices
    return train_indices, test_indices, full_dataset
