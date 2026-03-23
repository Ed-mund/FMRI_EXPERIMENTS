"""
Training Script: Low-Level BIT (VGG Feature Prediction).

Trains the LowLevelBIT model to predict VGG-16+BN features from fMRI
activations using InfoNCE loss applied independently per VGG layer.

Training config (from paper Appendix D.1):
  - Loss:          InfoNCE (contrastive, per VGG layer)
  - Optimiser:     AdamW, lr=5e-4, weight_decay=1e-2
  - Warmup:        15 epochs (linear)
  - Scheduler:     ReduceLROnPlateau (factor=0.1, patience=5, min_lr=1e-6)
  - Epochs:        60
  - Batch size:    64
  - Voxel sample:  15K per image
  - AMP:           bfloat16
  - Checkpoint:    every 5 epochs + best val loss

Usage:
    python train_lowlevel.py \\
        --data_root /projects/b6ac/brain/algonauts_prepared_data \\
        --v2c_dir /projects/b6ac/brain/brain_it/v2c \\
        --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 \\
        --output_dir /projects/b6ac/brain/checkpoints/bit_lowlevel_$(date +%Y%m%d) \\
        --wandb

    # Resume from checkpoint:
    python train_lowlevel.py ... --resume

    # With COCO unlabeled data:
    python train_lowlevel.py ... --coco_fmri_dir /projects/b6ac/brain/coco_fmri
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

try:
    from torch.amp import GradScaler as _GradScaler
    def _make_scaler():
        return _GradScaler("cuda")
except ImportError:
    from torch.cuda.amp import GradScaler as _GradScaler
    def _make_scaler():
        return _GradScaler()

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from bit_model import LowLevelBIT, InfoNCELoss, count_parameters
from vgg_features import VGGTargetExtractor, VGG_TRAIN_SAMPLES, VGG_LAYER_CONFIG
from dataset import (
    MultiSubjectBITDataset,
    COCOSyntheticDataset,
    CombinedBITDataset,
    NSDTestDataset,
    make_dataloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Signal for SLURM pre-emption (sent 5 min before time limit)
_SAVE_AND_EXIT = False


def _handle_usr1(signum, frame):
    global _SAVE_AND_EXIT
    log.warning("Received SIGUSR1 — will save checkpoint and exit after this epoch.")
    _SAVE_AND_EXIT = True


signal.signal(signal.SIGUSR1, _handle_usr1)
signal.signal(signal.SIGTERM, _handle_usr1)


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

def get_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float = 1e-6,
    total_epochs: int = 60,
):
    """Linear warmup then cosine decay."""
    import math
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: LowLevelBIT,
    optimizer,
    scaler,
    epoch: int,
    best_val_loss: float,
    history: list,
    args: argparse.Namespace,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_val_loss": best_val_loss,
            "history": history,
            "args": vars(args),
        },
        path,
    )
    log.info("Saved checkpoint → %s", path)


def load_checkpoint(
    path: Path,
    model: LowLevelBIT,
    optimizer,
    scaler,
    device,
) -> tuple[int, float, list]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    epoch = ckpt["epoch"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    history = ckpt.get("history", [])
    log.info("Resumed from epoch %d (best_val_loss=%.4f)", epoch, best_val_loss)
    return epoch + 1, best_val_loss, history


# ---------------------------------------------------------------------------
# W&B step-level logging helper
# ---------------------------------------------------------------------------

def _wandb_log(metrics: dict, step: int, use_wandb: bool):
    if use_wandb and _WANDB_AVAILABLE:
        wandb.log(metrics, step=step)


def _log_image_panel(
    images: torch.Tensor,
    caption_prefix: str,
    step: int,
    use_wandb: bool,
    n: int = 8,
):
    """Log a grid of raw training images to W&B (ground-truth context)."""
    if not (use_wandb and _WANDB_AVAILABLE):
        return
    imgs = images[:n].detach().cpu().clamp(0, 1)
    wandb_images = [
        wandb.Image(img.permute(1, 2, 0).numpy(), caption=f"{caption_prefix}_{i}")
        for i, img in enumerate(imgs)
    ]
    wandb.log({f"images/{caption_prefix}": wandb_images}, step=step)


def _log_layer_cosine_panel(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    step: int,
    use_wandb: bool,
    layer_names: list[str],
):
    """Log per-layer token cosine similarity (aligned N + D, same as InfoNCE)."""
    if not (use_wandb and _WANDB_AVAILABLE):
        return
    import torch.nn.functional as F
    try:
        metrics = {}
        for pred, tgt, name in zip(preds, targets, layer_names):
            _, N_pred, D_pred = pred.shape
            _, N_tgt, D_tgt = tgt.shape
            N_use = min(N_pred, N_tgt)
            D_use = min(D_pred, D_tgt)
            if N_pred > N_use:
                idx = torch.randperm(N_pred, device=pred.device)[:N_use]
                p = pred[:, idx, :D_use]
            else:
                p = pred[:, :N_use, :D_use]
            t = tgt[:, :N_use, :D_use]
            p_n = F.normalize(p.float().detach(), dim=-1)
            t_n = F.normalize(t.float().detach(), dim=-1)
            cos = (p_n * t_n).sum(dim=-1).cpu().numpy().flatten()
            metrics[f"token_cosine/{name}_mean"] = float(cos.mean())
            metrics[f"token_cosine/{name}_min"] = float(cos.min())
        wandb.log(metrics, step=step)
    except Exception as e:
        log.warning("W&B token cosine panel skipped: %s", e)


def _gpu_stats(device: torch.device) -> dict:
    if not device.type == "cuda":
        return {}
    alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    return {"gpu/mem_alloc_gb": alloc, "gpu/mem_reserved_gb": reserved}


def _grad_norm(model: nn.Module) -> dict:
    """Compute total grad norm + per named-module norms for key components."""
    total_norm = 0.0
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            pn = param.grad.detach().norm(2).item()
            total_norm += pn ** 2
            # Track top-level module norms (tokenizer vs transformer)
            top = name.split(".")[0]
            norms[top] = norms.get(top, 0.0) + pn ** 2
    total_norm = total_norm ** 0.5
    result = {"grad/total_norm": total_norm}
    for k, v in norms.items():
        result[f"grad/{k}_norm"] = v ** 0.5
    return result


def _per_layer_infonce(
    criterion: InfoNCELoss,
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """
    Compute InfoNCE per VGG layer; return total loss + per-layer values.

    Predictions have full token counts (e.g. 3136, 3025, ...) and layer channel dims
    (64, 128, 256, 512, 512). Targets are sampled (512, 512, 128, 64, 16) and 512-dim.
    We sample preds to match target token count and slice target to pred channel dim.
    """
    layer_losses = []
    for pred, tgt in zip(preds, targets):
        B, N_pred, D_pred = pred.shape
        _, N_tgt, D_tgt = tgt.shape
        N_use = min(N_pred, N_tgt)
        D_use = min(D_pred, D_tgt)

        # Sample N_use token positions from predictions (random during training)
        if N_pred > N_use:
            idx = torch.randperm(N_pred, device=pred.device, generator=generator)[:N_use]
            pred_use = pred[:, idx, :]  # (B, N_use, D_pred)
        else:
            pred_use = pred[:, :N_use, :]

        pred_use = pred_use[..., :D_use]  # (B, N_use, D_use)
        tgt_use = tgt[:, :N_use, :D_use]  # (B, N_use, D_use)

        layer_losses.append(criterion(pred_use, tgt_use))
    total = sum(layer_losses) / len(layer_losses)
    return total, [l.item() for l in layer_losses]


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: LowLevelBIT,
    vgg_extractor: VGGTargetExtractor,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> tuple[float, dict, int]:
    """
    Returns:
        avg_loss:     float
        extra_metrics dict (per-layer losses, per-subject losses, throughput)
        global_step:  updated step counter
    """
    model.train()
    vgg_extractor.eval()
    criterion = InfoNCELoss(temperature=0.07)

    total_loss = 0.0
    n_batches = len(loader)
    t0 = time.time()
    images_seen = 0

    # Accumulators for extra metrics
    layer_names = [cfg[3] for cfg in VGG_LAYER_CONFIG]
    layer_loss_accum = {name: 0.0 for name in layer_names}
    subj_loss_accum: dict[str, list[float]] = {}
    n_layer_accum = 0

    use_wandb = args.wandb and _WANDB_AVAILABLE

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        fmri = batch["fmri"].to(device, non_blocking=True)
        voxel_indices = batch["voxel_indices"].to(device, non_blocking=True)
        cluster_assignments = batch["cluster_assignments"].to(device, non_blocking=True)
        subject_ids = batch["subject_ids"]

        # Log training image panel once per epoch (first batch only)
        if batch_idx == 0 and epoch % args.log_images_every == 0:
            _log_image_panel(images, f"train_epoch{epoch:03d}", global_step, use_wandb)

        # VGG targets (frozen)
        with torch.no_grad():
            vgg_targets = vgg_extractor(images, sample=True, sample_counts=VGG_TRAIN_SAMPLES)

        optimizer.zero_grad(set_to_none=True)

        unique_subjects = list(dict.fromkeys(subject_ids))
        batch_loss = torch.tensor(0.0, device=device)
        n_subj_batches = 0

        for subj in unique_subjects:
            subj_mask = torch.tensor(
                [i for i, s in enumerate(subject_ids) if s == subj], device=device
            )
            if len(subj_mask) == 0:
                continue

            fmri_s = fmri[subj_mask]
            vox_s = voxel_indices[subj_mask[0]]
            clust_s = cluster_assignments[subj_mask[0]]
            vgg_t = [t[subj_mask] for t in vgg_targets]

            with autocast("cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                preds = model(fmri_s, vox_s, clust_s, subj)
                loss, layer_vals = _per_layer_infonce(criterion, preds, vgg_t)

            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            batch_loss = batch_loss + loss.detach()
            n_subj_batches += 1

            # Track per-layer and per-subject
            for name, lv in zip(layer_names, layer_vals):
                layer_loss_accum[name] += lv
            subj_loss_accum.setdefault(subj, []).append(loss.item())
            n_layer_accum += 1

        if args.use_amp:
            scaler.unscale_(optimizer)
            gnorm_dict = _grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            gnorm_dict = _grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_loss = (batch_loss / max(1, n_subj_batches)).item()
        total_loss += avg_loss
        images_seen += len(images)
        global_step += 1

        # Step-level W&B logging every log_every_steps
        if use_wandb and batch_idx % args.log_every_steps == 0:
            step_metrics = {
                "train/loss_step": avg_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                **{f"train/layer_{name}": layer_loss_accum[name] / max(1, n_layer_accum)
                   for name in layer_names},
                **gnorm_dict,
                **_gpu_stats(device),
            }
            _wandb_log(step_metrics, step=global_step, use_wandb=use_wandb)

        if batch_idx % 50 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            imgs_per_sec = images_seen / max(1e-3, elapsed)
            log.info(
                "  Epoch %d | Batch %d/%d | Loss: %.4f | %.0f img/s | %.1fs",
                epoch, batch_idx, n_batches, avg_loss, imgs_per_sec, elapsed,
            )

    elapsed_total = time.time() - t0
    extra = {
        "throughput_imgs_per_sec": images_seen / max(1e-3, elapsed_total),
        "layer_losses": {
            name: layer_loss_accum[name] / max(1, n_layer_accum)
            for name in layer_names
        },
        "subject_losses": {
            subj: float(np.mean(vals)) for subj, vals in subj_loss_accum.items()
        },
    }
    return total_loss / max(1, n_batches), extra, global_step


@torch.no_grad()
def validate(
    model: LowLevelBIT,
    vgg_extractor: VGGTargetExtractor,
    loader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, dict]:
    """Returns (avg_loss, extra_metrics).

    extra_metrics includes:
      - layer_losses:    per-VGG-layer InfoNCE losses
      - subject_losses:  per-subject average loss
      - _sample_preds:   first-batch predicted tokens (for cosine viz)
      - _sample_targets: first-batch target tokens
      - _sample_images:  first-batch ground-truth images
    """
    model.eval()
    criterion = InfoNCELoss(temperature=0.07)

    layer_names = [cfg[3] for cfg in VGG_LAYER_CONFIG]
    layer_loss_accum = {name: 0.0 for name in layer_names}
    subj_loss_accum: dict[str, list[float]] = {}
    total_loss = 0.0
    n_batches = 0

    # Keep the first batch's outputs for image/cosine visualisation
    sample_preds: list[torch.Tensor] | None = None
    sample_targets: list[torch.Tensor] | None = None
    sample_images: torch.Tensor | None = None

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        fmri = batch["fmri"].to(device, non_blocking=True)
        voxel_indices = batch["voxel_indices"].to(device, non_blocking=True)
        cluster_assignments = batch["cluster_assignments"].to(device, non_blocking=True)
        subject_ids = batch["subject_ids"]

        vgg_targets = vgg_extractor(images, sample=True, sample_counts=VGG_TRAIN_SAMPLES)

        unique_subjects = list(dict.fromkeys(subject_ids))
        for subj in unique_subjects:
            subj_mask = torch.tensor(
                [i for i, s in enumerate(subject_ids) if s == subj], device=device
            )
            if len(subj_mask) == 0:
                continue

            fmri_s = fmri[subj_mask]
            vox_s = voxel_indices[subj_mask[0]]
            clust_s = cluster_assignments[subj_mask[0]]
            vgg_t = [t[subj_mask] for t in vgg_targets]

            with autocast("cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                preds = model(fmri_s, vox_s, clust_s, subj)
                loss, layer_vals = _per_layer_infonce(criterion, preds, vgg_t)

            total_loss += loss.item()
            n_batches += 1
            for name, lv in zip(layer_names, layer_vals):
                layer_loss_accum[name] += lv
            subj_loss_accum.setdefault(subj, []).append(loss.item())

            # Save first-batch sample for vizualisation
            if sample_preds is None:
                sample_preds = [p[:8].cpu() for p in preds]
                sample_targets = [t[:8].cpu() for t in vgg_t]
                sample_images = images[:8].cpu()

    extra = {
        "layer_losses": {
            name: layer_loss_accum[name] / max(1, n_batches)
            for name in layer_names
        },
        "subject_losses": {
            subj: float(np.mean(vals)) for subj, vals in subj_loss_accum.items()
        },
        "_sample_preds": sample_preds,
        "_sample_targets": sample_targets,
        "_sample_images": sample_images,
    }
    return total_loss / max(1, n_batches), extra


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    if args.wandb and _WANDB_AVAILABLE:
        run_id_path = out_dir / "wandb_run_id.txt"
        run_id = run_id_path.read_text().strip() if run_id_path.exists() else None
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"lowlevel_{Path(args.output_dir).name}",
            id=run_id,
            resume="allow",
            config=vars(args),
        )
        run_id_path.write_text(wandb.run.id)
        # Log cluster statistics once at the start (useful context)
        import json as _json
        v2c_meta_path = Path(args.v2c_dir) / "metadata.json"
        if v2c_meta_path.exists():
            meta = _json.loads(v2c_meta_path.read_text())
            wandb.config.update({"v2c": meta}, allow_val_change=True)

    # Dataset
    log.info("Loading datasets...")
    train_ds = MultiSubjectBITDataset(
        data_root=args.data_root,
        subjects=args.subjects,
        v2c_dir=args.v2c_dir,
        split="train",
        n_voxels_sample=args.voxels_per_image,
        image_size=256,
    )

    # 90/10 train/val split
    n_val = max(1, int(0.1 * len(train_ds)))
    n_train = len(train_ds) - n_val
    train_split, val_split = torch.utils.data.random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Optional COCO unlabeled data
    if args.coco_fmri_dir and Path(args.coco_fmri_dir).exists():
        log.info("Adding COCO synthetic data from %s", args.coco_fmri_dir)
        coco_ds = COCOSyntheticDataset(
            coco_fmri_dir=args.coco_fmri_dir,
            v2c_dir=args.v2c_dir,
            subjects=args.subjects,
            n_voxels_sample=args.voxels_per_image,
        )
        combined_train = CombinedBITDataset(train_split, coco_ds)
    else:
        combined_train = train_split

    train_loader = make_dataloader(
        combined_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = make_dataloader(
        val_split, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    log.info("Train: %d batches | Val: %d batches", len(train_loader), len(val_loader))

    # Model: register all subjects first, then .to(device) so per-subject
    # Embedding layers are on the same device (new modules stay on CPU otherwise).
    model = LowLevelBIT(
        n_clusters=args.n_clusters,
        token_dim=args.token_dim,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    for subj in args.subjects:
        from dataset import NSDSubjectDataset
        ds_tmp = NSDSubjectDataset(
            data_root=args.data_root, subject_id=subj,
            v2c_dir=args.v2c_dir, split="train", n_voxels_sample=1,
        )
        model.register_subject(subj, ds_tmp.n_total_voxels)
    model = model.to(device)

    params = count_parameters(model)
    log.info("Model params: %s", params)
    if args.wandb and _WANDB_AVAILABLE:
        wandb.config.update({"model_params": params}, allow_val_change=True)

    # VGG extractor (frozen)
    vgg_extractor = VGGTargetExtractor().to(device)

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = _make_scaler() if args.use_amp else None

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    history = []
    global_step = 0

    latest_ckpt = out_dir / "checkpoint_latest.pt"
    if args.resume and latest_ckpt.exists():
        start_epoch, best_val_loss, history = load_checkpoint(
            latest_ckpt, model, optimizer, scaler, device
        )
        global_step = sum(len(loader) for _ in range(start_epoch))
    elif args.resume:
        log.info("--resume specified but no checkpoint found. Starting fresh.")

    # Reduce-on-plateau scheduler (applied after warmup)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=args.plateau_patience,
        min_lr=1e-6,
    )

    use_wandb = args.wandb and _WANDB_AVAILABLE
    layer_names = [cfg[3] for cfg in VGG_LAYER_CONFIG]

    # Training
    log.info("Starting training from epoch %d", start_epoch)
    for epoch in range(start_epoch, args.epochs):
        # LR schedule: linear warmup then ReduceLROnPlateau takes over
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        current_lr = optimizer.param_groups[0]["lr"]

        t_start = time.time()
        train_loss, train_extra, global_step = train_epoch(
            model, vgg_extractor, train_loader, optimizer, scaler,
            device, epoch, global_step, args,
        )
        val_loss, val_extra = validate(model, vgg_extractor, val_loader, device, args)
        epoch_time = time.time() - t_start

        # ReduceLROnPlateau (only after warmup)
        if epoch >= args.warmup_epochs:
            plateau_scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            log.info("New best val_loss=%.4f → best_model.pt", best_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # Log per-layer and per-subject breakdowns to console
        log.info(
            "Epoch %d/%d | train=%.4f | val=%.4f | lr=%.2e | %.0f img/s | %.1fs",
            epoch, args.epochs, train_loss, val_loss, current_lr,
            train_extra["throughput_imgs_per_sec"], epoch_time,
        )
        for name in layer_names:
            log.info(
                "  val layer %s: %.4f", name,
                val_extra["layer_losses"].get(name, float("nan")),
            )
        for subj, sloss in sorted(val_extra["subject_losses"].items()):
            log.info("  val %s: %.4f", subj, sloss)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "time": epoch_time,
            "train_layer_losses": train_extra["layer_losses"],
            "val_layer_losses": val_extra["layer_losses"],
            "val_subject_losses": val_extra["subject_losses"],
        }
        history.append(record)

        # Epoch-level W&B metrics
        if use_wandb:
            epoch_metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/lr": current_lr,
                "train/throughput_imgs_per_sec": train_extra["throughput_imgs_per_sec"],
                "train/epoch_time_s": epoch_time,
                "val/is_best": float(is_best),
                "val/best_loss": best_val_loss,
            }
            # Per-layer losses (train + val)
            for name in layer_names:
                epoch_metrics[f"train/layer_{name}"] = train_extra["layer_losses"].get(name, 0)
                epoch_metrics[f"val/layer_{name}"] = val_extra["layer_losses"].get(name, 0)
            # Per-subject val losses
            for subj, sloss in val_extra["subject_losses"].items():
                epoch_metrics[f"val/subj_{subj}"] = sloss
            for subj, sloss in train_extra["subject_losses"].items():
                epoch_metrics[f"train/subj_{subj}"] = sloss
            # GPU memory
            epoch_metrics.update(_gpu_stats(device))
            wandb.log(epoch_metrics, step=global_step)

        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                out_dir / f"checkpoint_epoch{epoch:03d}.pt",
                model, optimizer, scaler, epoch, best_val_loss, history, args,
            )

        # Always save latest for resume
        save_checkpoint(
            latest_ckpt, model, optimizer, scaler, epoch, best_val_loss, history, args
        )

        # Save history JSON
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if use_wandb and epoch % args.log_images_every == 0:
            if val_extra.get("_sample_images") is not None:
                _log_image_panel(
                    val_extra["_sample_images"], f"val_gt_epoch{epoch:03d}",
                    global_step, use_wandb,
                )
            if val_extra.get("_sample_preds") and val_extra.get("_sample_targets"):
                _log_layer_cosine_panel(
                    val_extra["_sample_preds"], val_extra["_sample_targets"],
                    global_step, use_wandb, layer_names,
                )

        if _SAVE_AND_EXIT:
            log.info("SLURM pre-emption: saved checkpoint, exiting.")
            break

    log.info("Training complete. Best val_loss: %.4f", best_val_loss)
    log.info("Output dir: %s", out_dir)

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LowLevel BIT (VGG feature prediction)")
    p.add_argument("--data_root", default="/projects/b6ac/brain/algonauts_prepared_data")
    p.add_argument("--v2c_dir", default="/projects/b6ac/brain/brain_it/v2c")
    p.add_argument("--subjects", nargs="+",
                   default=["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07"])
    p.add_argument("--coco_fmri_dir", default=None,
                   help="Directory with pre-computed COCO fMRI predictions")
    p.add_argument("--output_dir",
                   default="/projects/b6ac/brain/checkpoints/bit_lowlevel_20260318")

    # Model
    p.add_argument("--n_clusters", type=int, default=128)
    p.add_argument("--token_dim", type=int, default=512)
    p.add_argument("--n_blocks", type=int, default=5)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--voxels_per_image", type=int, default=15_000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_epochs", type=int, default=15)
    p.add_argument("--plateau_patience", type=int, default=5)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_every", type=int, default=5)

    # Resume
    p.add_argument("--resume", action="store_true")

    # Logging
    p.add_argument("--log_every_steps", type=int, default=20,
                   help="Log step-level metrics to W&B every N batches")
    p.add_argument("--log_images_every", type=int, default=5,
                   help="Log image panels + token cosine viz to W&B every N epochs")

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="brain-it")
    p.add_argument("--wandb_group", default="lowlevel")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
