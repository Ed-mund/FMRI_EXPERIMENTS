"""
Explore Brain Functionality via Voxel Embeddings (Paper Section 5)

After training, cluster the learned voxel embeddings to discover
functional brain regions. Find which images maximally activate each cluster.

Usage:
    python explore_embeddings.py \
        --checkpoint checkpoints/universal_8subj/best_model.pt \
        --data_root /path/to/algonauts_2023 \
        --subjects subj01 subj02 \
        --n_clusters 20
"""

import argparse
import os
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import UniversalBrainEncoder


def load_model(checkpoint_path: str, device: torch.device) -> UniversalBrainEncoder:
    """Load a trained Universal Brain Encoder."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})

    model = UniversalBrainEncoder(
        embedding_dim=args.get("embedding_dim", 256),
        projection_dim=args.get("projection_dim", 128),
        lora_rank=args.get("lora_rank", 16),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def cluster_embeddings(
    model: UniversalBrainEncoder,
    subject_ids: list,
    n_clusters: int = 20,
    seed: int = 42,
) -> dict:
    """
    Apply k-means clustering to voxel embeddings from multiple subjects.
    Returns cluster assignments and centroids.
    """
    all_embeddings = []
    all_labels = []  # (subject_id, voxel_idx) for each embedding

    for sid in subject_ids:
        embs = model.voxel_store.get_all_embeddings(sid).detach().cpu().numpy()
        all_embeddings.append(embs)
        for i in range(len(embs)):
            all_labels.append((sid, i))

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Clustering {len(all_embeddings)} voxel embeddings into {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_ids = kmeans.fit_predict(all_embeddings)

    # Report cluster sizes
    for c in range(n_clusters):
        mask = cluster_ids == c
        n_voxels = mask.sum()
        # Count per-subject
        subj_counts = {}
        for idx in np.where(mask)[0]:
            sid = all_labels[idx][0]
            subj_counts[sid] = subj_counts.get(sid, 0) + 1
        print(f"  Cluster {c}: {n_voxels} voxels | {subj_counts}")

    return {
        "embeddings": all_embeddings,
        "cluster_ids": cluster_ids,
        "centroids": kmeans.cluster_centers_,
        "labels": all_labels,
    }


def find_top_activating_images(
    model: UniversalBrainEncoder,
    dataloader,
    cluster_info: dict,
    device: torch.device,
    top_k: int = 5,
) -> dict:
    """
    For each cluster, find the images that produce the highest
    average activation across all voxels in that cluster.
    """
    n_clusters = cluster_info["centroids"].shape[0]

    # Group voxel indices by (subject, cluster)
    cluster_voxels = {}  # cluster_id -> {subject_id: [voxel_indices]}
    for idx, (sid, vidx) in enumerate(cluster_info["labels"]):
        cid = cluster_info["cluster_ids"][idx]
        if cid not in cluster_voxels:
            cluster_voxels[cid] = {}
        if sid not in cluster_voxels[cid]:
            cluster_voxels[cid][sid] = []
        cluster_voxels[cid][sid].append(vidx)

    # For each cluster, accumulate activations across images
    print("Computing per-cluster activations on all images...")
    # This would iterate through the dataset and compute activations
    # Left as a template - fill in with your specific dataloader setup

    return cluster_voxels


def visualize_tsne(
    cluster_info: dict,
    output_path: str = "tsne_embeddings.png",
    max_points: int = 50000,
    seed: int = 42,
):
    """
    Create t-SNE visualization of voxel embeddings colored by cluster.
    (Reproduces Fig. S17/S18 from the paper)
    """
    embeddings = cluster_info["embeddings"]
    cluster_ids = cluster_info["cluster_ids"]
    labels = cluster_info["labels"]

    # Subsample if too many points
    if len(embeddings) > max_points:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[idx]
        cluster_ids = cluster_ids[idx]
        labels = [labels[i] for i in idx]

    print(f"Running t-SNE on {len(embeddings)} embeddings...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    # Color by cluster
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Color by cluster
    scatter1 = axes[0].scatter(
        coords[:, 0], coords[:, 1],
        c=cluster_ids, cmap="tab20", alpha=0.5, s=2
    )
    axes[0].set_title("Colored by Cluster")
    plt.colorbar(scatter1, ax=axes[0])

    # Plot 2: Color by subject
    subject_ids = [l[0] for l in labels]
    unique_sids = sorted(set(subject_ids))
    sid_to_int = {s: i for i, s in enumerate(unique_sids)}
    sid_colors = [sid_to_int[s] for s in subject_ids]

    scatter2 = axes[1].scatter(
        coords[:, 0], coords[:, 1],
        c=sid_colors, cmap="Set1", alpha=0.5, s=2
    )
    axes[1].set_title("Colored by Subject")
    plt.colorbar(scatter2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved t-SNE visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subjects", type=str, nargs="+", default=["subj01", "subj02"])
    parser.add_argument("--n_clusters", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="exploration")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint, device)

    # Cluster embeddings
    cluster_info = cluster_embeddings(model, args.subjects, args.n_clusters)

    # t-SNE visualization
    visualize_tsne(
        cluster_info,
        output_path=os.path.join(args.output_dir, "tsne_embeddings.png"),
    )

    # Save cluster assignments
    np.savez(
        os.path.join(args.output_dir, "clusters.npz"),
        cluster_ids=cluster_info["cluster_ids"],
        centroids=cluster_info["centroids"],
    )

    print("Done! Explore the clusters by finding top-activating images per cluster.")


if __name__ == "__main__":
    main()
