"""
Voxel-to-Cluster (V2C) Mapping for Brain-IT.

Extracts the 256-dim voxel embeddings learned by the Universal Brain Encoder,
fits a Gaussian Mixture Model (GMM) with 128 components on the combined
embeddings from all subjects, and assigns each voxel to its most likely cluster.

This is the bridge between Stage 1 (encoding) and Stage 2 (Brain-IT decoding):
  - The encoder learns voxel embeddings that capture each voxel's visual function
  - The GMM groups functionally similar voxels across subjects into shared clusters
  - Brain-IT then operates on these 128 clusters rather than 40K individual voxels

Usage:
    python v2c_mapping.py \\
        --encoder_checkpoint /projects/b6ac/brain/checkpoints/brain_encoder_20260318/best_model.pt \\
        --output_dir /projects/b6ac/brain/brain_it/v2c \\
        --n_clusters 128

The output directory will contain:
    gmm.pkl                       - fitted sklearn GMM (128 components)
    v2c_{subject}.npy             - int32 array of shape (N_voxels,) with cluster idx
    embeddings_{subject}.npy      - float32 array of shape (N_voxels, 256) embeddings
    metadata.json                 - subjects, n_clusters, embedding_dim, etc.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load voxel embeddings from encoder checkpoint
# ---------------------------------------------------------------------------

def load_encoder_embeddings(checkpoint_path: str) -> dict[str, np.ndarray]:
    """
    Load voxel embedding tensors from a UniversalBrainEncoder checkpoint.

    Returns:
        dict mapping subject_id -> np.ndarray of shape (N_voxels, embedding_dim)
    """
    log.info("Loading encoder checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # The checkpoint may be the full model state_dict or a dict with 'model_state_dict'
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    embeddings = {}
    prefix = "voxel_store.embeddings."

    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            subject_id = key[len(prefix):]
            emb = tensor.detach().float().numpy()  # (N_voxels, E)
            embeddings[subject_id] = emb
            log.info("  %s: %s  (mean norm=%.3f)", subject_id, emb.shape,
                     float(np.linalg.norm(emb, axis=1).mean()))

    if not embeddings:
        raise ValueError(
            f"No voxel embeddings found in checkpoint {checkpoint_path}. "
            "Expected keys like 'voxel_store.embeddings.subj01'."
        )

    return embeddings


# ---------------------------------------------------------------------------
# Fit GMM
# ---------------------------------------------------------------------------

def fit_gmm(
    embeddings: dict[str, np.ndarray],
    n_clusters: int = 128,
    random_state: int = 42,
    max_iter: int = 300,
    subsample: int | None = None,
) -> "sklearn.mixture.GaussianMixture":
    """
    Fit a GMM on the combined embeddings from all subjects.

    Args:
        embeddings: subject_id -> (N_voxels, E)
        n_clusters: number of GMM components (128 in the paper)
        random_state: for reproducibility
        max_iter: GMM EM iterations
        subsample: if set, randomly sample this many voxels per subject for faster fitting

    Returns:
        fitted GaussianMixture object
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import normalize

    # Stack all embeddings
    all_embs = np.concatenate(list(embeddings.values()), axis=0)  # (total_voxels, E)

    # L2-normalize for better GMM clustering (embeddings encode directions)
    all_embs_norm = normalize(all_embs, norm="l2")

    if subsample is not None and subsample < all_embs_norm.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(all_embs_norm.shape[0], size=subsample, replace=False)
        fit_data = all_embs_norm[idx]
        log.info("Fitting GMM on %d / %d voxels (subsampled)", subsample, all_embs_norm.shape[0])
    else:
        fit_data = all_embs_norm
        log.info("Fitting GMM on all %d voxels", all_embs_norm.shape[0])

    log.info("GMM: %d components, max_iter=%d", n_clusters, max_iter)
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",  # diagonal faster + less memory than full
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
        verbose_interval=20,
        n_init=1,
        init_params="kmeans",
    )
    gmm.fit(fit_data)
    log.info("GMM converged: %s  (lower_bound=%.4f)", gmm.converged_, gmm.lower_bound_)
    return gmm


# ---------------------------------------------------------------------------
# Assign voxels to clusters
# ---------------------------------------------------------------------------

def assign_clusters(
    gmm: "sklearn.mixture.GaussianMixture",
    embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    For each subject, assign each voxel to its most probable GMM cluster.

    Returns:
        dict mapping subject_id -> int32 array of shape (N_voxels,)
    """
    from sklearn.preprocessing import normalize

    assignments = {}
    for subject_id, emb in embeddings.items():
        emb_norm = normalize(emb, norm="l2")
        cluster_ids = gmm.predict(emb_norm).astype(np.int32)
        assignments[subject_id] = cluster_ids

        # Log cluster size distribution
        unique, counts = np.unique(cluster_ids, return_counts=True)
        log.info(
            "  %s: %d voxels → %d clusters used  (min=%d, max=%d, mean=%.1f voxels/cluster)",
            subject_id, len(cluster_ids), len(unique), counts.min(), counts.max(), counts.mean(),
        )

    return assignments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_v2c(
    encoder_checkpoint: str,
    output_dir: str,
    n_clusters: int = 128,
    random_state: int = 42,
    subsample: int | None = 100_000,
) -> dict:
    """
    Full V2C pipeline: load embeddings → fit GMM → assign clusters → save.

    Returns metadata dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load embeddings
    embeddings = load_encoder_embeddings(encoder_checkpoint)
    subjects = sorted(embeddings.keys())
    embedding_dim = next(iter(embeddings.values())).shape[1]

    # 2. Fit GMM
    gmm = fit_gmm(
        embeddings,
        n_clusters=n_clusters,
        random_state=random_state,
        subsample=subsample,
    )

    # 3. Assign clusters
    assignments = assign_clusters(gmm, embeddings)

    # 4. Save GMM
    gmm_path = out / "gmm.pkl"
    with open(gmm_path, "wb") as f:
        pickle.dump(gmm, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Saved GMM → %s", gmm_path)

    # 5. Save per-subject assignments and embeddings
    for subject_id in subjects:
        np.save(out / f"v2c_{subject_id}.npy", assignments[subject_id])
        np.save(out / f"embeddings_{subject_id}.npy", embeddings[subject_id])
        log.info("Saved v2c_%s.npy and embeddings_%s.npy", subject_id, subject_id)

    # 6. Save metadata
    meta = {
        "encoder_checkpoint": str(encoder_checkpoint),
        "subjects": subjects,
        "n_clusters": n_clusters,
        "embedding_dim": embedding_dim,
        "random_state": random_state,
        "gmm_converged": bool(gmm.converged_),
        "gmm_lower_bound": float(gmm.lower_bound_),
        "voxels_per_subject": {s: int(embeddings[s].shape[0]) for s in subjects},
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved metadata.json")
    log.info("V2C mapping complete → %s", out)
    return meta


# ---------------------------------------------------------------------------
# Runtime helpers: load pre-computed mapping
# ---------------------------------------------------------------------------

def load_v2c(v2c_dir: str, subjects: list[str] | None = None) -> dict:
    """
    Load a pre-computed V2C mapping from disk.

    Returns:
        {
          "gmm":         GaussianMixture,
          "metadata":    dict,
          "assignments": {subject_id: int32 ndarray (N_voxels,)},
          "embeddings":  {subject_id: float32 ndarray (N_voxels, E)},
        }
    """
    d = Path(v2c_dir)
    with open(d / "gmm.pkl", "rb") as f:
        gmm = pickle.load(f)
    with open(d / "metadata.json") as f:
        meta = json.load(f)

    if subjects is None:
        subjects = meta["subjects"]

    assignments = {}
    embeddings_dict = {}
    for s in subjects:
        assignments[s] = np.load(d / f"v2c_{s}.npy")
        emb_path = d / f"embeddings_{s}.npy"
        if emb_path.exists():
            embeddings_dict[s] = np.load(emb_path)

    return {
        "gmm": gmm,
        "metadata": meta,
        "assignments": assignments,
        "embeddings": embeddings_dict,
    }


def get_cluster_assignments_tensor(
    v2c_assignments: np.ndarray,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert cluster assignment array to a long tensor for indexing."""
    return torch.from_numpy(v2c_assignments).long().to(device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build V2C mapping for Brain-IT")
    p.add_argument(
        "--encoder_checkpoint",
        default="/projects/b6ac/brain/checkpoints/brain_encoder_20260318/best_model.pt",
    )
    p.add_argument(
        "--output_dir",
        default="/projects/b6ac/brain/brain_it/v2c",
    )
    p.add_argument("--n_clusters", type=int, default=128)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--subsample",
        type=int,
        default=100_000,
        help="Max voxels to use for GMM fitting (for speed). 0 = use all.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    subsample = args.subsample if args.subsample > 0 else None
    meta = build_v2c(
        encoder_checkpoint=args.encoder_checkpoint,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        subsample=subsample,
    )
    print(json.dumps(meta, indent=2))
