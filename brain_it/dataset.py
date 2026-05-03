"""
Brain-IT Dataset Loader.

Supports two data sources (combined during training):
  1. NSD paired data (real fMRI + viewed images) — from Algonauts prepared format
  2. COCO unlabeled images + synthetic fMRI predicted by the frozen encoder

Data format (Algonauts prepared):
  subj{XX}/
  ├── training_split/
  │   ├── training_fmri/
  │   │   ├── lh_training_fmri.npy  (N_train, N_vox_lh)
  │   │   └── rh_training_fmri.npy  (N_train, N_vox_rh)
  │   └── training_images/
  │       └── *.png
  └── test_split/
      ├── test_fmri/
      │   ├── lh_test_fmri.npy  (N_test, N_vox_lh)
      │   └── rh_test_fmri.npy  (N_test, N_vox_rh)
      └── test_images/
          └── *.png

fMRI arrays are concatenated (LH + RH) → (N_images, N_total_voxels).

During training:
  - 15K voxels are randomly sampled per image (with replacement from ~40K)
  - Returns sampled fMRI values, their global voxel indices, and their cluster assignments
  - Images are returned as [0,1] float tensors at 256×256 (downscaled as needed)

COCO unlabeled data:
  Predicted fMRI is pre-computed offline by `predict_coco_fmri.py` and stored
  as a single numpy memmap per subject:
    {coco_fmri_dir}/subj{XX}_fmri.npy  (N_coco, N_total_voxels)
    {coco_fmri_dir}/image_paths.txt     one path per line
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image transform
# ---------------------------------------------------------------------------

def get_image_transform(size: int = 256) -> transforms.Compose:
    """Standard transform: resize to `size`×`size`, convert to float [0, 1]."""
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),  # → [0, 1] float
    ])


# ---------------------------------------------------------------------------
# Core: NSD single-subject dataset
# ---------------------------------------------------------------------------

class NSDSubjectDataset(Dataset):
    """
    Loads one subject's NSD fMRI + image pairs.

    Returns per-sample dicts:
      {
        "image":              (3, H, W) float32 in [0, 1],
        "fmri":               (N_v_sampled,) float32  — sampled voxel activations,
        "voxel_indices":      (N_v_sampled,) int64     — global voxel indices,
        "cluster_assignments":(N_v_sampled,) int64     — V2C cluster per voxel,
        "subject_id":         str,
        "n_voxels":           int  — total number of voxels for this subject,
      }
    """

    def __init__(
        self,
        data_root: str,
        subject_id: str,
        v2c_dir: str,
        split: str = "train",
        n_voxels_sample: int = 15_000,
        image_size: int = 256,
        seed: Optional[int] = None,
    ):
        self.subject_id = subject_id
        self.n_voxels_sample = n_voxels_sample
        self.transform = get_image_transform(image_size)
        self.rng = np.random.default_rng(seed)

        subj_dir = Path(data_root) / subject_id

        # Load fMRI arrays (LH + RH concatenated)
        if split == "train":
            fmri_dir = subj_dir / "training_split" / "training_fmri"
            self.image_dir = subj_dir / "training_split" / "training_images"
        else:
            fmri_dir = subj_dir / "test_split" / "test_fmri"
            self.image_dir = subj_dir / "test_split" / "test_images"

        # Filenames use "training" for train split and "test" for test split
        split_name = "training" if split == "train" else split
        lh = np.load(fmri_dir / f"lh_{split_name}_fmri.npy")
        rh = np.load(fmri_dir / f"rh_{split_name}_fmri.npy")
        # Concatenate hemispheres: (N_images, N_vox_total)
        self.fmri = np.concatenate([lh, rh], axis=1).astype(np.float32)
        self.n_total_voxels = self.fmri.shape[1]

        # Load image file list (sorted for determinism)
        self.image_files = sorted(self.image_dir.glob("*.png")) + sorted(self.image_dir.glob("*.jpg"))
        assert len(self.image_files) == self.fmri.shape[0], (
            f"{subject_id} {split}: {len(self.image_files)} images but {self.fmri.shape[0]} fMRI rows"
        )

        # Load V2C cluster assignments for this subject
        v2c_path = Path(v2c_dir) / f"v2c_{subject_id}.npy"
        self.cluster_assignments = np.load(v2c_path).astype(np.int64)
        assert len(self.cluster_assignments) == self.n_total_voxels, (
            f"V2C assignments ({len(self.cluster_assignments)}) != n_voxels ({self.n_total_voxels})"
        )

        log.info(
            "%s [%s]: %d images, %d voxels total",
            subject_id, split, len(self.image_files), self.n_total_voxels,
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        # Load image
        img = Image.open(self.image_files[idx]).convert("RGB")
        image = self.transform(img)

        # Sample voxel indices
        n_sample = min(self.n_voxels_sample, self.n_total_voxels)
        vox_idx = self.rng.integers(0, self.n_total_voxels, size=n_sample)

        # Load fMRI activations for sampled voxels
        fmri_sample = torch.from_numpy(self.fmri[idx][vox_idx])  # (N_v,)
        vox_idx_t = torch.from_numpy(vox_idx.astype(np.int64))
        cluster_t = torch.from_numpy(self.cluster_assignments[vox_idx])

        return {
            "image": image,
            "fmri": fmri_sample,
            "voxel_indices": vox_idx_t,
            "cluster_assignments": cluster_t,
            "subject_id": self.subject_id,
            "n_voxels": self.n_total_voxels,
        }


# ---------------------------------------------------------------------------
# Multi-subject dataset (ConcatDataset with subject ID tracking)
# ---------------------------------------------------------------------------

class MultiSubjectBITDataset(Dataset):
    """
    Combines NSD datasets from multiple subjects.
    Each item is sampled uniformly across all subjects.
    """

    def __init__(
        self,
        data_root: str,
        subjects: list[str],
        v2c_dir: str,
        split: str = "train",
        n_voxels_sample: int = 15_000,
        image_size: int = 256,
        seed: Optional[int] = None,
    ):
        self.datasets = {}
        self.subject_offsets = {}
        cumulative = 0

        for subj in subjects:
            ds = NSDSubjectDataset(
                data_root=data_root,
                subject_id=subj,
                v2c_dir=v2c_dir,
                split=split,
                n_voxels_sample=n_voxels_sample,
                image_size=image_size,
                seed=seed,
            )
            self.datasets[subj] = ds
            self.subject_offsets[subj] = cumulative
            cumulative += len(ds)

        # Build flat index: global_idx → (subject_id, local_idx)
        self._index = []
        for subj, ds in self.datasets.items():
            for local_idx in range(len(ds)):
                self._index.append((subj, local_idx))

        log.info(
            "MultiSubjectBITDataset: %d subjects, %d total samples",
            len(subjects), len(self._index),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        subj, local_idx = self._index[idx]
        return self.datasets[subj][local_idx]

    def get_n_voxels(self, subject_id: str) -> int:
        return self.datasets[subject_id].n_total_voxels


# ---------------------------------------------------------------------------
# COCO unlabeled dataset with pre-computed synthetic fMRI
# ---------------------------------------------------------------------------

class COCOSyntheticDataset(Dataset):
    """
    COCO unlabeled images with fMRI responses predicted by the frozen encoder.

    Pre-computed by `predict_coco_fmri.py`. For each image, fMRI predictions
    from one randomly selected subject are used (per the paper's approach).

    Expected files in coco_fmri_dir:
      subj{XX}_fmri.npy   — float16, shape (N_coco, N_voxels_XX)
                            (may be a memmap for large files)
      image_paths.txt     — one absolute image path per line (N_coco lines)
      n_voxels_{XX}.txt   — total voxel count per subject (one line)
      v2c_subj{XX}.npy    — cluster assignments, shape (N_voxels_XX,)
    """

    def __init__(
        self,
        coco_fmri_dir: str,
        v2c_dir: str,
        subjects: list[str],
        n_voxels_sample: int = 15_000,
        image_size: int = 256,
        seed: Optional[int] = None,
    ):
        self.subjects = subjects
        self.n_voxels_sample = n_voxels_sample
        self.transform = get_image_transform(image_size)
        self.rng = np.random.default_rng(seed)
        d = Path(coco_fmri_dir)
        v2c = Path(v2c_dir)

        # Load image paths
        paths_file = d / "image_paths.txt"
        if not paths_file.exists():
            raise FileNotFoundError(f"Missing {paths_file}. Run predict_coco_fmri.py first.")
        with open(paths_file) as f:
            self.image_paths = [Path(line.strip()) for line in f]
        self.n_images = len(self.image_paths)

        # Load per-subject fMRI (memmap for memory efficiency)
        self.fmri = {}
        self.n_voxels = {}
        self.cluster_assignments = {}
        for subj in subjects:
            fmri_path = d / f"{subj}_fmri.npy"
            if not fmri_path.exists():
                raise FileNotFoundError(
                    f"Missing {fmri_path}. Run predict_coco_fmri.py for {subj}."
                )
            arr = np.load(fmri_path, mmap_mode="r")
            self.fmri[subj] = arr
            self.n_voxels[subj] = arr.shape[1]
            self.cluster_assignments[subj] = np.load(v2c / f"v2c_{subj}.npy").astype(np.int64)

        log.info(
            "COCOSyntheticDataset: %d images, %d subjects",
            self.n_images, len(subjects),
        )

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, idx: int) -> dict:
        # Randomly pick a subject for this image
        subj = self.rng.choice(self.subjects)

        img = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(img)

        n_total = self.n_voxels[subj]
        n_sample = min(self.n_voxels_sample, n_total)
        vox_idx = self.rng.integers(0, n_total, size=n_sample)

        fmri_row = self.fmri[subj][idx].astype(np.float32)
        fmri_sample = torch.from_numpy(fmri_row[vox_idx])
        vox_idx_t = torch.from_numpy(vox_idx.astype(np.int64))
        cluster_t = torch.from_numpy(self.cluster_assignments[subj][vox_idx])

        return {
            "image": image,
            "fmri": fmri_sample,
            "voxel_indices": vox_idx_t,
            "cluster_assignments": cluster_t,
            "subject_id": subj,
            "n_voxels": n_total,
        }


# ---------------------------------------------------------------------------
# Combined dataset (NSD + COCO)
# ---------------------------------------------------------------------------

class CombinedBITDataset(Dataset):
    """
    Interleaves NSD paired data and COCO synthetic data.
    Iterates through the full NSD data once per epoch and the full COCO
    data once per epoch (they may have different sizes; both are iterated
    in full and wrapped around as needed).
    """

    def __init__(
        self,
        nsd_dataset: MultiSubjectBITDataset,
        coco_dataset: Optional[COCOSyntheticDataset] = None,
    ):
        self.nsd = nsd_dataset
        self.coco = coco_dataset
        self._n_nsd = len(nsd_dataset)
        self._n_coco = len(coco_dataset) if coco_dataset is not None else 0
        self._total = self._n_nsd + self._n_coco
        log.info(
            "CombinedBITDataset: %d NSD + %d COCO = %d total",
            self._n_nsd, self._n_coco, self._total,
        )

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict:
        if idx < self._n_nsd:
            return self.nsd[idx]
        else:
            coco_idx = (idx - self._n_nsd) % self._n_coco
            return self.coco[coco_idx]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def bit_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate that handles variable subject_id strings and
    stacks tensors by subject for the BIT forward pass.

    Returns a flat batch; the training loop splits by subject.
    """
    images = torch.stack([b["image"] for b in batch])
    fmri = torch.stack([b["fmri"] for b in batch])
    voxel_indices = torch.stack([b["voxel_indices"] for b in batch])
    cluster_assignments = torch.stack([b["cluster_assignments"] for b in batch])
    subject_ids = [b["subject_id"] for b in batch]
    n_voxels = [b["n_voxels"] for b in batch]

    return {
        "images": images,
        "fmri": fmri,
        "voxel_indices": voxel_indices,
        "cluster_assignments": cluster_assignments,
        "subject_ids": subject_ids,
        "n_voxels": n_voxels,
    }


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=bit_collate_fn,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


# ---------------------------------------------------------------------------
# Test dataset (evaluation only — full voxels, no sampling)
# ---------------------------------------------------------------------------

class NSDTestDataset(Dataset):
    """
    Loads test fMRI and images for evaluation.
    Returns full fMRI vectors (all voxels) — used for computing Pearson r etc.
    """

    def __init__(
        self,
        data_root: str,
        subject_id: str,
        v2c_dir: str,
        image_size: int = 256,
    ):
        self.subject_id = subject_id
        self.transform = get_image_transform(image_size)

        subj_dir = Path(data_root) / subject_id
        fmri_dir = subj_dir / "test_split" / "test_fmri"
        self.image_dir = subj_dir / "test_split" / "test_images"

        lh = np.load(fmri_dir / "lh_test_fmri.npy")
        rh = np.load(fmri_dir / "rh_test_fmri.npy")
        self.fmri = np.concatenate([lh, rh], axis=1).astype(np.float32)
        self.n_total_voxels = self.fmri.shape[1]

        self.image_files = sorted(self.image_dir.glob("*.png")) + sorted(self.image_dir.glob("*.jpg"))
        assert len(self.image_files) == self.fmri.shape[0]

        # Full cluster assignments (all voxels)
        v2c_path = Path(v2c_dir) / f"v2c_{subject_id}.npy"
        self.cluster_assignments = np.load(v2c_path).astype(np.int64)

        # All voxel indices (for full-voxel evaluation)
        self.all_voxel_indices = np.arange(self.n_total_voxels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.image_files[idx]).convert("RGB")
        image = self.transform(img)
        fmri = torch.from_numpy(self.fmri[idx])
        voxel_indices = torch.from_numpy(self.all_voxel_indices)
        cluster_assignments = torch.from_numpy(self.cluster_assignments)

        return {
            "image": image,
            "fmri": fmri,
            "voxel_indices": voxel_indices,
            "cluster_assignments": cluster_assignments,
            "subject_id": self.subject_id,
            "n_voxels": self.n_total_voxels,
        }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    DATA_ROOT = "/projects/b6ac/brain/algonauts_prepared_data"
    V2C_DIR = "/projects/b6ac/brain/brain_it/v2c"

    if not Path(V2C_DIR).exists():
        print("V2C directory not found. Run v2c_mapping.py first.")
        sys.exit(0)

    ds = NSDSubjectDataset(
        data_root=DATA_ROOT,
        subject_id="subj01",
        v2c_dir=V2C_DIR,
        split="train",
        n_voxels_sample=15_000,
    )
    print(f"NSDSubjectDataset: {len(ds)} samples")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v}")

    # Test multi-subject
    multi = MultiSubjectBITDataset(
        data_root=DATA_ROOT,
        subjects=["subj01", "subj02"],
        v2c_dir=V2C_DIR,
        split="train",
    )
    print(f"\nMultiSubjectBITDataset: {len(multi)} samples")

    loader = make_dataloader(multi, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    print("\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {v}")
    print("Dataset OK")
