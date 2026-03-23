"""
Zero-Shot Brain Encoding Evaluation (Phase 3).

Evaluates the full zero-shot pipeline on the held-out subject:
  1. Load the frozen Universal Brain Encoder (shared weights from Phase 1)
  2. Load anatomy features for the held-out subject
  3. Predict voxel embeddings via the trained hypernetwork
  4. Run the frozen encoder with those predicted embeddings
  5. Measure encoding quality (Pearson r, Top-1/Top-5 retrieval)

Also evaluates three baselines that require zero fMRI data:
  - random:      random 256-dim embeddings (lower bound)
  - roi_average: mean embedding per ROI from training subjects' learned embeddings
  - nearest_fsaverage: copy embedding from nearest fsaverage vertex in training set

Results are printed as a comparison table and saved to JSON.

Usage:
    python eval_zeroshot.py \\
        --encoder_checkpoint /path/to/brain_encoder/best_model.pt \\
        --hypernetwork_checkpoint /path/to/hypernetwork/best_model.pt \\
        --anatomy_dir /projects/b6ac/brain/anatomy_features \\
        --algonauts_dir /projects/b6ac/brain/algonauts_prepared_data \\
        --test_subject subj08 \\
        --train_subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 \\
        --output_dir /projects/b6ac/brain/results/zeroshot_subj08

    # With W&B:
    python eval_zeroshot.py ... --wandb_project brain-zeroshot
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from model import UniversalBrainEncoder
from dataset import NSDAlgonautsDataset, get_default_transform
from hypernetwork import AnatomyToEmbeddingNet, load_voxel_embeddings_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder forward with custom embeddings (bypasses VoxelEmbeddingStore)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_embeddings(
    encoder: UniversalBrainEncoder,
    images: torch.Tensor,
    voxel_embeddings: torch.Tensor,
    chunk_size: int = 5000,
) -> torch.Tensor:
    """
    Run the frozen encoder using externally-supplied voxel embeddings.

    Args:
        encoder:          trained UniversalBrainEncoder (weights frozen)
        images:           (B, 3, 224, 224) batch of images
        voxel_embeddings: (N, E=256) per-voxel embeddings (from hypernetwork, or baseline)
        chunk_size:       voxels per cross-attention chunk (memory management)

    Returns:
        predictions: (N, B) predicted fMRI activations, on CPU
    """
    features = encoder.feature_extractor(images)  # (B, L, P, C)

    N = voxel_embeddings.shape[0]
    all_preds = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = voxel_embeddings[start:end].to(features.device)
        preds = encoder.cross_attention(features, chunk)  # (chunk, B)
        all_preds.append(preds.cpu())

    return torch.cat(all_preds, dim=0)  # (N, B)


# ---------------------------------------------------------------------------
# Evaluation metrics (matches train.py evaluate_subject)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_embeddings(
    encoder: UniversalBrainEncoder,
    dataloader: DataLoader,
    voxel_embeddings: torch.Tensor,
    device: torch.device,
    chunk_size: int = 5000,
    label: str = "",
) -> dict:
    """
    Evaluate encoding quality using provided voxel embeddings.

    Returns dict with Pearson correlation stats and retrieval accuracy.
    """
    encoder.eval()

    all_pred = []
    all_gt   = []

    for batch in dataloader:
        images = batch["image"].to(device)   # (B, 3, H, W)
        fmri   = batch["fmri"].to(device)    # (B, V)

        preds = predict_with_embeddings(encoder, images, voxel_embeddings, chunk_size)
        all_pred.append(preds)         # (V, B)
        all_gt.append(fmri.T.cpu())    # (V, B)

    if not all_pred:
        return {}

    all_pred = torch.cat(all_pred, dim=1).float()  # (V, N_images)
    all_gt   = torch.cat(all_gt,   dim=1).float()  # (V, N_images)

    V, N = all_pred.shape

    # --- Pearson r per voxel ---
    pred_c = all_pred - all_pred.mean(dim=1, keepdim=True)
    gt_c   = all_gt   - all_gt.mean(dim=1, keepdim=True)

    num   = (pred_c * gt_c).sum(dim=1)
    denom = pred_c.norm(dim=1) * gt_c.norm(dim=1) + 1e-8
    corrs = num / denom  # (V,)

    median_r = corrs.median().item()
    p25_r    = corrs.quantile(0.25).item()
    p75_r    = corrs.quantile(0.75).item()

    # --- Retrieval ---
    pred_n = F.normalize(pred_c, dim=0)
    gt_n   = F.normalize(gt_c,   dim=0)
    sim    = gt_n.T @ pred_n  # (N, N)

    ranks = []
    for i in range(N):
        rank = (sim[i] > sim[i, i]).sum().item() + 1
        ranks.append(rank)
    ranks = np.array(ranks)

    top1  = (ranks == 1).mean() * 100
    top5  = (ranks <= 5).mean() * 100
    mean_r = ranks.mean()

    results = {
        "median_r":   median_r,
        "p25_r":      p25_r,
        "p75_r":      p75_r,
        "top1":       top1,
        "top5":       top5,
        "mean_rank":  mean_r,
        "n_images":   N,
        "n_voxels":   V,
        "_corrs":     corrs.numpy(),
    }

    logger.info(
        f"  [{label:28s}] r={median_r:.4f} [{p25_r:.3f},{p75_r:.3f}]  "
        f"Top1={top1:.1f}%  Top5={top5:.1f}%  MeanRank={mean_r:.1f}/{N}"
    )
    return results


# ---------------------------------------------------------------------------
# Baseline embedding generators
# ---------------------------------------------------------------------------

def random_baseline(n_voxels: int, embedding_dim: int = 256, seed: int = 42) -> torch.Tensor:
    """Random 256-dim embeddings (lower bound baseline)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randn(n_voxels, embedding_dim, generator=rng)


def roi_average_baseline(
    train_embeddings: dict[str, torch.Tensor],
    train_roi_labels: dict[str, np.ndarray],
    test_roi_labels: np.ndarray,
    n_roi_types: int,
    embedding_dim: int = 256,
) -> torch.Tensor:
    """
    For each ROI label, compute the mean embedding across training subjects.
    Assign each test voxel the mean embedding of its ROI.

    Args:
        train_embeddings:  {subject -> (N_train, 256)} learned embeddings
        train_roi_labels:  {subject -> (N_train,)} concatenated ROI labels
        test_roi_labels:   (N_test,) ROI labels for the test subject
        n_roi_types:       number of unique ROI categories
        embedding_dim:     E=256

    Returns:
        (N_test, 256) predicted embeddings
    """
    # Accumulate mean embedding per ROI across training subjects
    roi_emb_sum   = torch.zeros(n_roi_types + 1, embedding_dim)
    roi_emb_count = torch.zeros(n_roi_types + 1)

    for subj, embs in train_embeddings.items():
        labels = train_roi_labels[subj]  # (N,) int labels
        for roi_id in range(n_roi_types + 1):
            mask = labels == roi_id
            if mask.sum() > 0:
                roi_emb_sum[roi_id]   += embs[mask].float().mean(dim=0)
                roi_emb_count[roi_id] += 1

    # Average across subjects
    valid = roi_emb_count > 0
    roi_means = torch.zeros(n_roi_types + 1, embedding_dim)
    roi_means[valid] = roi_emb_sum[valid] / roi_emb_count[valid].unsqueeze(1)

    # Fall back to global mean for unseen ROIs
    global_mean = roi_means[valid].mean(dim=0)
    roi_means[~valid] = global_mean

    # Assign to test voxels
    test_preds = roi_means[test_roi_labels.astype(int).clip(0, n_roi_types)]
    return test_preds


def nearest_fsaverage_baseline(
    train_embeddings: dict[str, torch.Tensor],
    train_anatomy: dict[str, np.ndarray],
    test_anatomy: np.ndarray,
    k: int = 5,
) -> torch.Tensor:
    """
    For each test voxel, find the k nearest training voxels in fsaverage
    sphere space (first 3 anatomy features) and average their embeddings.

    Args:
        train_embeddings: {subject -> (N_train, 256)}
        train_anatomy:    {subject -> (N_train, F)} anatomy arrays
        test_anatomy:     (N_test, F) test subject anatomy
        k:                number of nearest neighbours to average

    Returns:
        (N_test, 256) predicted embeddings
    """
    # Stack all training voxels: coords and embeddings
    all_train_coords = []
    all_train_embs   = []
    for subj in sorted(train_embeddings.keys()):
        all_train_coords.append(train_anatomy[subj][:, :3])  # sphere xyz
        all_train_embs.append(train_embeddings[subj].float().numpy())

    all_train_coords = np.concatenate(all_train_coords, axis=0)  # (M, 3)
    all_train_embs   = np.concatenate(all_train_embs,   axis=0)  # (M, 256)

    test_coords = test_anatomy[:, :3]  # (N_test, 3)

    # Batch nearest-neighbour (L2 distance in sphere space)
    logger.info(f"  Computing nearest-fsaverage distances ({test_coords.shape[0]} test, {all_train_coords.shape[0]} train)...")

    test_t  = torch.from_numpy(test_coords).float()
    train_t = torch.from_numpy(all_train_coords).float()

    # Process in chunks to avoid OOM
    chunk = 2000
    preds = []
    for i in range(0, len(test_coords), chunk):
        tc = test_t[i:i+chunk]  # (chunk, 3)
        dists = torch.cdist(tc, train_t)  # (chunk, M)
        topk  = dists.topk(k, dim=1, largest=False).indices  # (chunk, k)
        near_embs = torch.from_numpy(all_train_embs)[topk].float()  # (chunk, k, 256)
        preds.append(near_embs.mean(dim=1))  # (chunk, 256)

    return torch.cat(preds, dim=0)


# ---------------------------------------------------------------------------
# Load encoder with validation
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path: str, device: torch.device) -> UniversalBrainEncoder:
    """Load the UniversalBrainEncoder from a checkpoint file."""
    logger.info(f"Loading encoder from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # Infer registered subjects from state dict
    import re
    subjects = sorted(set(
        re.match(r"voxel_store\.embeddings\.(\w+)", k).group(1)
        for k in state if re.match(r"voxel_store\.embeddings\.(\w+)", k)
    ))
    logger.info(f"  Encoder trained on: {subjects}")

    encoder = UniversalBrainEncoder()
    for subj in subjects:
        n = state[f"voxel_store.embeddings.{subj}"].shape[0]
        encoder.register_subject(subj, n)

    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"  Missing keys: {missing[:5]}")
    if unexpected:
        logger.warning(f"  Unexpected keys: {unexpected[:5]}")

    encoder.to(device)
    encoder.eval()

    # Freeze everything
    for p in encoder.parameters():
        p.requires_grad_(False)

    logger.info(f"  Encoder loaded and frozen.")
    return encoder


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot brain encoding evaluation"
    )

    # Paths
    parser.add_argument("--encoder_checkpoint", required=True)
    parser.add_argument("--hypernetwork_checkpoint", required=True)
    parser.add_argument("--anatomy_dir",
                        default="/projects/b6ac/brain/anatomy_features")
    parser.add_argument("--algonauts_dir",
                        default="/projects/b6ac/brain/algonauts_prepared_data")
    parser.add_argument("--output_dir",
                        default="/projects/b6ac/brain/results/zeroshot_subj08")

    # Subjects
    parser.add_argument("--test_subject", default="subj08")
    parser.add_argument(
        "--train_subjects", nargs="+",
        default=["subj01","subj02","subj03","subj04","subj05","subj06","subj07"],
    )

    # Evaluation settings
    parser.add_argument("--n_eval_images", type=int, default=1000,
                        help="Number of images to use for eval (last N from training set)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=5000,
                        help="Voxels per cross-attention chunk")

    # Baselines to run
    parser.add_argument("--skip_random", action="store_true")
    parser.add_argument("--skip_roi_average", action="store_true")
    parser.add_argument("--skip_nearest", action="store_true")

    # W&B
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    use_wandb = _WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"zeroshot_{args.test_subject}",
            entity=args.wandb_entity,
            config=vars(args),
        )

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------

    logger.info(f"\nLoading test subject data: {args.test_subject}")
    test_ds = NSDAlgonautsDataset(
        data_root=args.algonauts_dir,
        subject_id=args.test_subject,
        split="train",  # only split with fMRI ground truth
        transform=get_default_transform(),
    )
    n_total = len(test_ds)
    n_eval  = min(args.n_eval_images, n_total)
    eval_indices = list(range(n_total - n_eval, n_total))  # last N images
    eval_ds = Subset(test_ds, eval_indices)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    n_voxels = test_ds.num_voxels
    logger.info(f"  {n_eval} eval images, {n_voxels} voxels")

    # Load anatomy features
    logger.info(f"Loading anatomy features...")
    test_anatomy = np.load(
        os.path.join(args.anatomy_dir, f"{args.test_subject}_anatomy.npy")
    ).astype(np.float32)
    logger.info(f"  test anatomy shape: {test_anatomy.shape}")

    # -------------------------------------------------------------------
    # Load encoder
    # -------------------------------------------------------------------
    encoder = load_encoder(args.encoder_checkpoint, device)

    # -------------------------------------------------------------------
    # Condition 1: Zero-shot (hypernetwork predicted embeddings)
    # -------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("Loading hypernetwork and predicting embeddings...")

    hn_ckpt = torch.load(args.hypernetwork_checkpoint, map_location="cpu")
    in_features   = hn_ckpt.get("in_features", test_anatomy.shape[1])
    hidden_dim    = hn_ckpt.get("hidden_dim", 512)
    embedding_dim = hn_ckpt.get("embedding_dim", 256)

    hypernetwork = AnatomyToEmbeddingNet(
        in_features=in_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=0.0,
    )
    hypernetwork.load_state_dict(hn_ckpt["model_state_dict"])
    hypernetwork.eval()

    test_feats = torch.from_numpy(test_anatomy)
    with torch.no_grad():
        zeroshot_embeddings = hypernetwork(test_feats)  # (N_voxels, 256)
    logger.info(f"  Predicted embeddings: {zeroshot_embeddings.shape}")

    # -------------------------------------------------------------------
    # Evaluate all conditions
    # -------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("Evaluating conditions:")
    logger.info(f"  {'Condition':<30} {'r_med':>6} {'r_p25':>6} {'r_p75':>6} {'Top1%':>6} {'Top5%':>6}")
    logger.info(f"  {'-'*60}")

    all_results = {}

    # --- Zero-shot (hypernetwork) ---
    logger.info("\nCondition: zero_shot_hypernetwork")
    results_zs = evaluate_embeddings(
        encoder, eval_loader, zeroshot_embeddings.to(device),
        device, args.chunk_size, label="zero_shot_hypernetwork"
    )
    all_results["zero_shot_hypernetwork"] = {
        k: v.tolist() if hasattr(v, "tolist") else v
        for k, v in results_zs.items()
        if not k.startswith("_")
    }

    # --- Baseline: Random embeddings ---
    if not args.skip_random:
        logger.info("\nCondition: baseline_random")
        random_embs = random_baseline(n_voxels, embedding_dim=256).to(device)
        results_rnd = evaluate_embeddings(
            encoder, eval_loader, random_embs,
            device, args.chunk_size, label="baseline_random"
        )
        all_results["baseline_random"] = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in results_rnd.items()
            if not k.startswith("_")
        }

    # --- Baseline: ROI-average embeddings ---
    if not args.skip_roi_average:
        logger.info("\nCondition: baseline_roi_average")
        logger.info("  Loading train embeddings and ROI labels...")

        train_embs = load_voxel_embeddings_from_checkpoint(
            args.encoder_checkpoint,
            subjects=args.train_subjects,
            device="cpu",
        )

        # Build ROI label arrays for train subjects and test subject
        # Use prf-visualrois index (features 7:14 in anatomy) + raw label from roi mask files
        # Simplest: use the argmax of the first ROI block (prf-visualrois) in anatomy features
        # Features [7:14] are the prf-visualrois one-hot, [14:22] are streams, etc.
        # Reconstructed label = argmax + 1 (0 = none/background)
        def get_roi_labels_from_anatomy(anatomy: np.ndarray) -> np.ndarray:
            """Recover dominant ROI label from one-hot anatomy features."""
            # Use combined ROI presence: max over all ROI one-hots [7:]
            roi_block = anatomy[:, 7:]  # all ROI features
            has_roi   = roi_block.sum(axis=1) > 0
            labels = np.zeros(len(anatomy), dtype=int)
            labels[has_roi] = roi_block[has_roi].argmax(axis=1) + 1
            return labels

        train_roi_labels = {}
        for subj in args.train_subjects:
            train_anat = np.load(
                os.path.join(args.anatomy_dir, f"{subj}_anatomy.npy")
            ).astype(np.float32)
            train_roi_labels[subj] = get_roi_labels_from_anatomy(train_anat)

        test_roi_labels = get_roi_labels_from_anatomy(test_anatomy)
        n_roi_types = int(test_roi_labels.max()) + 1

        roi_avg_embs = roi_average_baseline(
            train_embs, train_roi_labels, test_roi_labels, n_roi_types
        ).to(device)

        results_roi = evaluate_embeddings(
            encoder, eval_loader, roi_avg_embs,
            device, args.chunk_size, label="baseline_roi_average"
        )
        all_results["baseline_roi_average"] = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in results_roi.items()
            if not k.startswith("_")
        }

    # --- Baseline: Nearest fsaverage vertex ---
    if not args.skip_nearest:
        logger.info("\nCondition: baseline_nearest_fsaverage")
        if "train_embs" not in locals():
            train_embs = load_voxel_embeddings_from_checkpoint(
                args.encoder_checkpoint,
                subjects=args.train_subjects,
                device="cpu",
            )
        train_anatomy_dict = {}
        for subj in args.train_subjects:
            train_anatomy_dict[subj] = np.load(
                os.path.join(args.anatomy_dir, f"{subj}_anatomy.npy")
            ).astype(np.float32)

        nearest_embs = nearest_fsaverage_baseline(
            train_embs, train_anatomy_dict, test_anatomy, k=5
        ).to(device)

        results_near = evaluate_embeddings(
            encoder, eval_loader, nearest_embs,
            device, args.chunk_size, label="baseline_nearest_fsaverage"
        )
        all_results["baseline_nearest_fsaverage"] = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in results_near.items()
            if not k.startswith("_")
        }

    # --- Reference: Oracle (held-out subject's own learned embeddings, if in checkpoint) ---
    try:
        oracle_embs_dict = load_voxel_embeddings_from_checkpoint(
            args.encoder_checkpoint,
            subjects=[args.test_subject],
            device="cpu",
        )
        logger.info(f"\nCondition: oracle_learned_embeddings (upper bound)")
        oracle_embs = oracle_embs_dict[args.test_subject].to(device)
        results_oracle = evaluate_embeddings(
            encoder, eval_loader, oracle_embs,
            device, args.chunk_size, label="oracle_learned_embeddings"
        )
        all_results["oracle_learned_embeddings"] = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in results_oracle.items()
            if not k.startswith("_")
        }
    except KeyError:
        logger.info(
            f"  ('{args.test_subject}' not in encoder checkpoint — "
            f"oracle upper bound not available for this LOSO split)"
        )

    # -------------------------------------------------------------------
    # Print results table
    # -------------------------------------------------------------------
    logger.info(f"\n{'='*65}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*65}")
    logger.info(f"{'Condition':<35} {'r_med':>7} {'r_p25':>7} {'r_p75':>7} {'Top1%':>7} {'Top5%':>7}")
    logger.info(f"{'-'*65}")
    for cond, res in all_results.items():
        logger.info(
            f"{cond:<35} "
            f"{res.get('median_r', 0):7.4f} "
            f"{res.get('p25_r', 0):7.4f} "
            f"{res.get('p75_r', 0):7.4f} "
            f"{res.get('top1', 0):7.2f} "
            f"{res.get('top5', 0):7.2f}"
        )

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {out_path}")

    # Also save the predicted embeddings for later analysis
    emb_path = os.path.join(args.output_dir, f"{args.test_subject}_zeroshot_embeddings.pt")
    torch.save(zeroshot_embeddings.cpu(), emb_path)
    logger.info(f"Predicted embeddings saved to: {emb_path}")

    if use_wandb:
        flat = {}
        for cond, res in all_results.items():
            for metric, val in res.items():
                if not isinstance(val, list):
                    flat[f"{cond}/{metric}"] = val
        wandb.log(flat)
        wandb.finish()


if __name__ == "__main__":
    main()
