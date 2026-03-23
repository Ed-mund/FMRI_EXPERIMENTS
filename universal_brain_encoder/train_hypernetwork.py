"""
Training Script for the Anatomy-to-Embedding Hypernetwork (Phase 2).

Loads learned voxel embeddings from a trained Universal Brain Encoder
checkpoint, then trains a small MLP to predict those embeddings from
per-voxel anatomical features.

Usage:
    python train_hypernetwork.py \\
        --encoder_checkpoint /projects/b6ac/brain/checkpoints/brain_encoder_*/best_model.pt \\
        --anatomy_dir /projects/b6ac/brain/anatomy_features \\
        --train_subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 \\
        --held_out_subject subj08 \\
        --output_dir /projects/b6ac/brain/checkpoints/hypernetwork_loso_subj08

    # With W&B logging:
    python train_hypernetwork.py ... --wandb_project brain-zeroshot

Notes:
    - The hypernetwork is tiny (~600k params) and trains in <15 min on CPU,
      ~2 min on GPU. A single A100/GH200 is overkill but fine.
    - 10% of training voxels (randomly held out per subject) are used for
      early stopping — these are voxels, not images, so there's no data
      leakage from the fMRI side.
    - The anatomy_dir must contain {subject}_anatomy.npy files produced by
      preprocess_anatomy.py.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from hypernetwork import (
    AnatomyToEmbeddingNet,
    HypernetworkLoss,
    load_voxel_embeddings_from_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SUBJECTS_ALL = [f"subj{i:02d}" for i in range(1, 9)]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_anatomy_features(anatomy_dir: str, subjects: list[str]) -> dict[str, np.ndarray]:
    """Load anatomy feature arrays for a list of subjects."""
    features = {}
    for subj in subjects:
        path = os.path.join(anatomy_dir, f"{subj}_anatomy.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Anatomy features not found: {path}\n"
                f"Run preprocess_anatomy.py first."
            )
        arr = np.load(path).astype(np.float32)
        features[subj] = arr
        logger.info(f"  [{subj}] anatomy shape: {arr.shape}")
    return features


def build_dataset(
    anatomy_features: dict[str, np.ndarray],
    embeddings: dict[str, torch.Tensor],
    subjects: list[str],
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """
    Build train/val TensorDatasets by concatenating across subjects and
    splitting 10% of voxels per subject for validation.

    Returns:
        train_dataset, val_dataset
    """
    rng = np.random.default_rng(seed)

    train_x_list, train_y_list = [], []
    val_x_list,   val_y_list   = [], []

    for subj in subjects:
        x = torch.from_numpy(anatomy_features[subj])  # (N, F)
        y = embeddings[subj].float()                  # (N, 256)

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"[{subj}] anatomy voxels ({x.shape[0]}) != "
                f"embedding voxels ({y.shape[0]}). "
                f"Ensure preprocess_anatomy.py was run with the same "
                f"algonauts_prepared_data as the encoder."
            )

        N = x.shape[0]
        idx = rng.permutation(N)
        n_val = max(1, int(N * val_fraction))

        val_idx   = idx[:n_val]
        train_idx = idx[n_val:]

        train_x_list.append(x[train_idx])
        train_y_list.append(y[train_idx])
        val_x_list.append(x[val_idx])
        val_y_list.append(y[val_idx])

        logger.info(f"  [{subj}] train: {len(train_idx)}, val: {len(val_idx)}")

    train_x = torch.cat(train_x_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)
    val_x   = torch.cat(val_x_list, dim=0)
    val_y   = torch.cat(val_y_list, dim=0)

    logger.info(f"Total: {train_x.shape[0]} train voxels, {val_x.shape[0]} val voxels")

    return (
        TensorDataset(train_x, train_y),
        TensorDataset(val_x,   val_y),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: AnatomyToEmbeddingNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: HypernetworkLoss,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> dict:
    model.train()
    total_loss = 0.0
    total_mse  = 0.0
    total_cos  = 0.0
    total_cos_sim = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = model(x)
                loss, metrics = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss, metrics = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss    += loss.item()
        total_mse     += metrics["mse_loss"]
        total_cos     += metrics["cosine_loss"]
        total_cos_sim += metrics["cosine_sim_mean"]
        n_batches += 1

    return {
        "loss":        total_loss    / n_batches,
        "mse":         total_mse     / n_batches,
        "cosine_loss": total_cos     / n_batches,
        "cosine_sim":  total_cos_sim / n_batches,
    }


@torch.no_grad()
def eval_epoch(
    model: AnatomyToEmbeddingNet,
    loader: DataLoader,
    loss_fn: HypernetworkLoss,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss    = 0.0
    total_mse     = 0.0
    total_cos     = 0.0
    total_cos_sim = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss, metrics = loss_fn(pred, y)

        total_loss    += loss.item()
        total_mse     += metrics["mse_loss"]
        total_cos     += metrics["cosine_loss"]
        total_cos_sim += metrics["cosine_sim_mean"]
        n_batches += 1

    return {
        "loss":        total_loss    / n_batches,
        "mse":         total_mse     / n_batches,
        "cosine_loss": total_cos     / n_batches,
        "cosine_sim":  total_cos_sim / n_batches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train the anatomy-to-embedding hypernetwork (Phase 2 of zero-shot pipeline)"
    )

    # Paths
    parser.add_argument(
        "--encoder_checkpoint",
        required=True,
        help="Path to best_model.pt from the Universal Brain Encoder training",
    )
    parser.add_argument(
        "--anatomy_dir",
        default="/projects/b6ac/brain/anatomy_features",
        help="Directory with {subject}_anatomy.npy files from preprocess_anatomy.py",
    )
    parser.add_argument(
        "--output_dir",
        default="/projects/b6ac/brain/checkpoints/hypernetwork_loso_subj08",
        help="Where to save hypernetwork checkpoints and logs",
    )

    # Subjects
    parser.add_argument(
        "--train_subjects",
        nargs="+",
        default=["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07"],
        help="Subjects to train on (should match the encoder's training subjects)",
    )
    parser.add_argument(
        "--held_out_subject",
        default="subj08",
        help="The subject held out for zero-shot evaluation (not used in training)",
    )

    # Architecture
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss_alpha", type=float, default=0.5,
                        help="Weight for MSE in combined loss (0=cosine only, 1=MSE only)")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--seed", type=int, default=42)

    # W&B
    parser.add_argument("--wandb_project", default=None,
                        help="W&B project name (omit to disable W&B)")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- W&B init ---
    use_wandb = _WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"hypernetwork_loso_{args.held_out_subject}",
            entity=args.wandb_entity,
            config=vars(args),
        )

    # --- Load voxel embeddings from encoder checkpoint ---
    logger.info(f"Loading voxel embeddings from: {args.encoder_checkpoint}")
    embeddings = load_voxel_embeddings_from_checkpoint(
        args.encoder_checkpoint,
        subjects=args.train_subjects,
        device="cpu",
    )

    # --- Load anatomy features ---
    logger.info(f"Loading anatomy features from: {args.anatomy_dir}")
    anatomy_features = load_anatomy_features(args.anatomy_dir, args.train_subjects)

    # Infer feature dim from data
    first_subj = args.train_subjects[0]
    in_features = anatomy_features[first_subj].shape[1]
    logger.info(f"Anatomy feature dimension: {in_features}")

    # Load feature names for logging (optional)
    names_path = os.path.join(args.anatomy_dir, "feature_names.json")
    if os.path.exists(names_path):
        with open(names_path) as f:
            feature_meta = json.load(f)
        logger.info(f"Feature names: {feature_meta['feature_names'][:7]} ... ({feature_meta['n_features']} total)")

    # --- Build dataset ---
    logger.info("Building train/val datasets...")
    train_ds, val_ds = build_dataset(
        anatomy_features,
        embeddings,
        args.train_subjects,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # --- Build model ---
    model = AnatomyToEmbeddingNet(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    loss_fn   = HypernetworkLoss(alpha=args.loss_alpha)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # --- Training loop with early stopping ---
    best_val_loss  = float("inf")
    best_epoch     = 0
    patience_count = 0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")

    logger.info(f"Starting training for up to {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_metrics   = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} cos_sim={train_metrics['cosine_sim']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} cos_sim={val_metrics['cosine_sim']:.3f} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/mse": train_metrics["mse"],
                "train/cosine_loss": train_metrics["cosine_loss"],
                "train/cosine_sim": train_metrics["cosine_sim"],
                "val/loss": val_metrics["loss"],
                "val/mse": val_metrics["mse"],
                "val/cosine_loss": val_metrics["cosine_loss"],
                "val/cosine_sim": val_metrics["cosine_sim"],
                "lr": lr_now,
            })

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss  = val_metrics["loss"]
            best_epoch     = epoch
            patience_count = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_cosine_sim": val_metrics["cosine_sim"],
                    "train_subjects": args.train_subjects,
                    "held_out_subject": args.held_out_subject,
                    "in_features": in_features,
                    "hidden_dim": args.hidden_dim,
                    "embedding_dim": args.embedding_dim,
                    "config": vars(args),
                },
                best_ckpt_path,
            )
            logger.info(f"  -> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(best was epoch {best_epoch}, val_loss={best_val_loss:.4f})"
                )
                break

    # --- Save final model alongside best ---
    final_ckpt_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_metrics["loss"],
            "train_subjects": args.train_subjects,
            "held_out_subject": args.held_out_subject,
            "in_features": in_features,
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
        },
        final_ckpt_path,
    )

    logger.info(f"\nTraining complete.")
    logger.info(f"  Best model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    logger.info(f"  Saved to: {best_ckpt_path}")

    # --- Sanity check: predict embeddings for held-out subject if anatomy is available ---
    held_out_path = os.path.join(args.anatomy_dir, f"{args.held_out_subject}_anatomy.npy")
    if os.path.exists(held_out_path):
        logger.info(f"\nSanity check: predicting embeddings for held-out {args.held_out_subject}...")
        held_feats = torch.from_numpy(
            np.load(held_out_path).astype(np.float32)
        ).to(device)

        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        with torch.no_grad():
            pred_embeddings = model(held_feats).cpu()

        out_emb_path = os.path.join(args.output_dir, f"{args.held_out_subject}_predicted_embeddings.pt")
        torch.save(pred_embeddings, out_emb_path)
        logger.info(f"  Predicted embeddings shape: {pred_embeddings.shape}")
        logger.info(f"  Saved to: {out_emb_path}")

        if use_wandb:
            wandb.run.summary["best_epoch"]   = best_epoch
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["held_out"]      = args.held_out_subject
    else:
        logger.info(
            f"  (anatomy features for {args.held_out_subject} not found — "
            f"run preprocess_anatomy.py for that subject first)"
        )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
