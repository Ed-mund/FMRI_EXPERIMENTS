"""
Preprocess anatomical features for the zero-shot hypernetwork.

For each subject, extracts a feature vector per cortical vertex in the
Algonauts challenge space (the same surface-vertex indexing used by the
Universal Brain Encoder's VoxelEmbeddingStore).

Features extracted (per vertex):
  [0:3]   Sphere coordinates (x, y, z) from fsaverage sphere — shared space
  [3]     Curvature (fsaverage avg) — shared
  [4]     Sulcal depth (fsaverage avg_sulc) — shared
  [5]     Cortical thickness (fsaverage avg_thickness) — shared
  [6]     Hemisphere indicator (0=LH, 1=RH)
  [7:14]  prf-visualrois one-hot label (7 labels: 0=none, 1=V1v, ..., 7=hV4)
  [14:22] streams one-hot label (8 labels: 0=none, 1-7=streams regions)
  [22:26] floc-faces one-hot label (4 labels)
  [24:28] floc-bodies one-hot label (4 labels)  -- NB indices overlap, kept flat
  [28:32] floc-places one-hot label (4 labels)
  [32:36] floc-words one-hot label (4 labels)
  [36:41] prf-eccrois one-hot label (5 labels)

  Total: ~41 features per vertex

Output (per subject):
  algonauts_prepared_data/{subject}/anatomy_features.npy
    shape: (N_voxels, N_features) — N_voxels = lh + rh challenge voxels
  algonauts_prepared_data/{subject}/anatomy_feature_names.json
    names of each feature column

Usage:
  python preprocess_anatomy.py --data_root /projects/b6ac/brain \\
      --subjects subj01 subj02 ... subj08

  # Or SLURM array (subj index via SLURM_ARRAY_TASK_ID):
  python preprocess_anatomy.py --data_root /projects/b6ac/brain \\
      --subject_index $SLURM_ARRAY_TASK_ID
"""

import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np


SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]

ROI_FILES = [
    "prf-visualrois",
    "streams",
    "floc-faces",
    "floc-bodies",
    "floc-places",
    "floc-words",
    "prf-eccrois",
]

# Max label value per ROI type (from inspecting the files)
ROI_MAX_LABELS = {
    "prf-visualrois": 7,   # 0=none, 1=V1v, 2=V1d, 3=V2v, 4=V2d, 5=V3v, 6=V3d, 7=hV4
    "streams":        7,   # 0=none, 1-7
    "floc-faces":     5,   # 0=none, 1-5
    "floc-bodies":    3,   # 0=none, 1-3
    "floc-places":    3,   # 0=none, 1-3
    "floc-words":     4,   # 0=none, 1-4
    "prf-eccrois":    5,   # 0=none, 1-5 eccentricity bands
}


def load_fsaverage_features(fsaverage_dir: str) -> dict:
    """
    Load shared fsaverage surface features (same for all subjects).
    Returns dict: hemisphere -> (N_vertices, n_surface_features) arrays.
    """
    hemi_features = {}
    for hemi in ("lh", "rh"):
        surf_dir = os.path.join(fsaverage_dir, "surf")

        # Sphere coordinates (radius-100 normalised sphere)
        coords, _ = nib.freesurfer.read_geometry(
            os.path.join(surf_dir, f"{hemi}.sphere")
        )
        # Normalise to unit sphere (coords are on a 100-radius sphere)
        coords_norm = coords / 100.0  # shape (163842, 3)

        # Curvature (signed, positive = sulcal)
        curv = nib.freesurfer.read_morph_data(
            os.path.join(surf_dir, f"{hemi}.curv")
        )  # (163842,)

        # Average sulcal depth
        sulc = nib.freesurfer.read_morph_data(
            os.path.join(surf_dir, f"{hemi}.avg_sulc")
        )  # (163842,)

        # Average cortical thickness
        thickness = nib.freesurfer.read_morph_data(
            os.path.join(surf_dir, f"{hemi}.avg_thickness")
        )  # (163842,)

        # Stack: (163842, 6) = [x, y, z, curv, sulc, thickness]
        feats = np.column_stack([
            coords_norm,
            curv[:, np.newaxis],
            sulc[:, np.newaxis],
            thickness[:, np.newaxis],
        ]).astype(np.float32)

        hemi_features[hemi] = feats  # (163842, 6)

    return hemi_features


def build_roi_onehot(roi_array: np.ndarray, max_label: int) -> np.ndarray:
    """
    Convert an integer label array to one-hot encoding.
    Label 0 encodes as all-zeros (background).
    Returns shape (N_vertices, max_label).
    """
    N = len(roi_array)
    onehot = np.zeros((N, max_label), dtype=np.float32)
    labels = roi_array.astype(int)
    # Clamp negatives (label -1 means "other cortex") to 0
    labels = np.clip(labels, 0, max_label)
    valid = labels > 0
    onehot[valid, labels[valid] - 1] = 1.0
    return onehot


def extract_subject_anatomy(
    subject: str,
    algonauts_dir: str,
    fsaverage_features: dict,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract anatomy feature matrix for one subject.

    Returns:
        features: (N_voxels, N_features) float32 array
        feature_names: list of column name strings
    """
    subj_dir = os.path.join(algonauts_dir, subject)
    roi_dir = os.path.join(subj_dir, "roi_masks")

    all_feats = []
    feature_names: list[str] = []
    names_built = False  # Only build names on first (lh) pass

    for hemi_idx, hemi in enumerate(("lh", "rh")):
        # --- Vertex selection mask: which fsaverage vertices are in challenge space ---
        mask_path = os.path.join(roi_dir, f"{hemi}.all-vertices_fsaverage_space.npy")
        vertex_mask = np.load(mask_path).astype(bool)  # (163842,)
        n_voxels = vertex_mask.sum()

        # --- Shared fsaverage surface features ---
        surf_feats = fsaverage_features[hemi][vertex_mask]  # (n_voxels, 6)

        if not names_built:
            feature_names += ["sphere_x", "sphere_y", "sphere_z",
                               "curvature", "sulcal_depth", "thickness"]

        # --- Hemisphere indicator ---
        hemi_feat = np.full((n_voxels, 1), float(hemi_idx), dtype=np.float32)
        if not names_built:
            feature_names += ["hemisphere"]

        # --- ROI one-hot labels ---
        roi_feats_list = []
        for roi_name in ROI_FILES:
            roi_path = os.path.join(roi_dir, f"{hemi}.{roi_name}_challenge_space.npy")
            if not os.path.exists(roi_path):
                # ROI not available for this subject/hemisphere — zero fill
                max_label = ROI_MAX_LABELS[roi_name]
                roi_onehot = np.zeros((n_voxels, max_label), dtype=np.float32)
            else:
                roi_vals = np.load(roi_path).astype(float)  # (n_voxels,)
                max_label = ROI_MAX_LABELS[roi_name]
                roi_onehot = build_roi_onehot(roi_vals, max_label)

            roi_feats_list.append(roi_onehot)
            if not names_built:
                feature_names += [f"{roi_name}_{i}" for i in range(ROI_MAX_LABELS[roi_name])]

        names_built = True  # Don't duplicate names for rh

        roi_feats = np.concatenate(roi_feats_list, axis=1)  # (n_voxels, sum_labels)

        # Concatenate all feature groups
        subj_feats = np.concatenate([surf_feats, hemi_feat, roi_feats], axis=1)
        all_feats.append(subj_feats)

    features = np.concatenate(all_feats, axis=0).astype(np.float32)
    return features, feature_names


def normalize_features(features: np.ndarray, stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """
    Z-score normalize continuous features (first 6: sphere xyz, curv, sulc, thick).
    One-hot and hemisphere features are left unchanged.
    Returns normalized features and stats dict for later use on test subjects.
    """
    n_continuous = 6  # sphere_x, sphere_y, sphere_z, curvature, sulcal_depth, thickness
    feats = features.copy()

    if stats is None:
        mean = feats[:, :n_continuous].mean(axis=0)
        std = feats[:, :n_continuous].std(axis=0) + 1e-8
        stats = {"mean": mean.tolist(), "std": std.tolist(), "n_continuous": n_continuous}

    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    feats[:, :n_continuous] = (feats[:, :n_continuous] - mean) / std

    return feats, stats


def process_subject(
    subject: str,
    algonauts_dir: str,
    fsaverage_features: dict,
    norm_stats: dict | None,
    output_dir: str,
) -> dict:
    """Process one subject: extract, optionally normalize, and save."""
    print(f"\n[{subject}] Extracting anatomy features...")
    features, feature_names = extract_subject_anatomy(
        subject, algonauts_dir, fsaverage_features
    )
    print(f"[{subject}] Raw features shape: {features.shape}")

    features_norm, stats = normalize_features(features, norm_stats)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{subject}_anatomy.npy")
    np.save(out_path, features_norm)
    print(f"[{subject}] Saved to {out_path}")

    return stats, feature_names


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-vertex anatomical features for the zero-shot hypernetwork"
    )
    parser.add_argument(
        "--data_root",
        default="/projects/b6ac/brain",
        help="Root directory containing algonauts_prepared_data/ and nsd_data/",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=SUBJECTS,
        help="Subject IDs to process (default: all 8)",
    )
    parser.add_argument(
        "--subject_index",
        type=int,
        default=None,
        help="1-based subject index (for SLURM array jobs)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to save anatomy .npy files (default: data_root/anatomy_features/)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Z-score normalize continuous features (default: True)",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Skip Z-score normalization (save raw features)",
    )
    parser.add_argument(
        "--norm_subjects",
        nargs="+",
        default=None,
        help="Subjects to compute normalization stats from (default: same as --subjects). "
             "Set to training subjects only to avoid data leakage.",
    )
    args = parser.parse_args()

    algonauts_dir = os.path.join(args.data_root, "algonauts_prepared_data")
    fsaverage_dir = os.path.join(
        args.data_root, "nsd_data", "nsddata", "freesurfer", "fsaverage"
    )
    output_dir = args.output_dir or os.path.join(args.data_root, "anatomy_features")

    if args.subject_index is not None:
        subjects = [SUBJECTS[args.subject_index - 1]]
    else:
        subjects = args.subjects

    print(f"Processing subjects: {subjects}")
    print(f"Fsaverage directory: {fsaverage_dir}")
    print(f"Output directory: {output_dir}")

    # Load fsaverage surface features once (shared across all subjects)
    print("\nLoading fsaverage surface features (shared across subjects)...")
    fsaverage_features = load_fsaverage_features(fsaverage_dir)
    for hemi, feats in fsaverage_features.items():
        print(f"  {hemi}: {feats.shape}")

    # Compute normalisation stats from a reference set of subjects
    # to avoid data leakage when evaluating leave-one-out
    norm_stats = None
    if args.normalize:
        norm_subjects = args.norm_subjects or subjects
        print(f"\nComputing normalisation stats from: {norm_subjects}")

        all_raw = []
        for subj in norm_subjects:
            feats, _ = extract_subject_anatomy(subj, algonauts_dir, fsaverage_features)
            all_raw.append(feats)
        all_raw = np.concatenate(all_raw, axis=0)

        n_continuous = 6
        mean = all_raw[:, :n_continuous].mean(axis=0)
        std = all_raw[:, :n_continuous].std(axis=0) + 1e-8
        norm_stats = {"mean": mean.tolist(), "std": std.tolist(), "n_continuous": n_continuous}
        print(f"  Normalisation stats from {len(all_raw)} total vertices")

    # Process each subject
    feature_names = None
    for subj in subjects:
        stats, feat_names = process_subject(
            subj, algonauts_dir, fsaverage_features, norm_stats, output_dir
        )
        if feature_names is None:
            feature_names = feat_names

    # Save metadata (feature names and norm stats)
    os.makedirs(output_dir, exist_ok=True)

    if feature_names is not None:
        names_path = os.path.join(output_dir, "feature_names.json")
        with open(names_path, "w") as f:
            json.dump({"feature_names": feature_names, "n_features": len(feature_names)}, f, indent=2)
        print(f"\nFeature names ({len(feature_names)} total) saved to {names_path}")

    if norm_stats is not None:
        stats_path = os.path.join(output_dir, "norm_stats.json")
        with open(stats_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        print(f"Normalisation stats saved to {stats_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
