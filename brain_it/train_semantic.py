"""
Training Script: Semantic BIT (CLIP Token Prediction + Joint Diffusion).

Stage 1 — Feature Alignment:
  Trains SemanticBIT to predict 256 OpenCLIP ViT-bigG/14 spatial tokens
  using L2 loss. Aligns BIT output with MindEye2's unCLIP conditioning format.
  Config: 60 epochs, batch 128, AdamW lr=5e-4, 15-epoch warmup, ReduceLROnPlateau.

Stage 2 — Joint Training with Diffusion:
  Jointly trains SemanticBIT + MindEye2 unCLIP SDXL diffusion model.
  BIT's predicted CLIP tokens condition the diffusion model; both are optimised
  together with the standard diffusion (noise prediction) loss.
  Config: 10 epochs, batch 8, grad_accum 8 (effective batch 64),
          AdamW lr=1e-5, image size 256×256.

Usage:
    # Stage 1
    python train_semantic.py --stage 1 \\
        --data_root /projects/b6ac/brain/algonauts_prepared_data \\
        --v2c_dir /projects/b6ac/brain/brain_it/v2c \\
        --subjects subj01 subj02 subj03 subj04 subj05 subj06 subj07 \\
        --output_dir /projects/b6ac/brain/checkpoints/bit_semantic_$(date +%Y%m%d)

    # Stage 2 (after stage 1 completes)
    python train_semantic.py --stage 2 \\
        --stage1_checkpoint /projects/b6ac/brain/checkpoints/bit_semantic_*/best_model.pt \\
        --mindeye2_checkpoint /projects/b6ac/brain/checkpoints/mindeye2/diffusion_prior.pt \\
        --output_dir /projects/b6ac/brain/checkpoints/bit_semantic_joint_$(date +%Y%m%d)

    # Resume:
    python train_semantic.py --stage 1 ... --resume
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
import torch.nn.functional as F
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

from bit_model import SemanticBIT, count_parameters
from clip_features import CLIPTargetExtractor
from dataset import (
    MultiSubjectBITDataset,
    COCOSyntheticDataset,
    CombinedBITDataset,
    make_dataloader,
)


# ---------------------------------------------------------------------------
# Shared W&B / metric helpers (mirrors train_lowlevel.py)
# ---------------------------------------------------------------------------

def _wandb_log(metrics: dict, step: int, use_wandb: bool):
    if use_wandb and _WANDB_AVAILABLE:
        wandb.log(metrics, step=step)


def _gpu_stats(device: torch.device) -> dict:
    if not device.type == "cuda":
        return {}
    alloc = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    return {"gpu/mem_alloc_gb": alloc, "gpu/mem_reserved_gb": reserved}


def _grad_norm(model: nn.Module, prefix: str = "") -> dict:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    key = f"grad/{prefix}norm" if prefix else "grad/total_norm"
    return {key: total ** 0.5}


def _clip_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean cosine similarity between predicted and target CLIP tokens."""
    pred_n = F.normalize(pred.float(), dim=-1)
    tgt_n = F.normalize(target.float(), dim=-1)
    return (pred_n * tgt_n).sum(dim=-1).mean().item()


def _log_image_panel(
    images: torch.Tensor,
    caption_prefix: str,
    step: int,
    use_wandb: bool,
    n: int = 8,
):
    """Log a grid of images (ground truth) to W&B."""
    if not (use_wandb and _WANDB_AVAILABLE):
        return
    imgs = images[:n].detach().cpu().clamp(0, 1)
    wandb.log(
        {f"images/{caption_prefix}": [
            wandb.Image(img.permute(1, 2, 0).numpy(), caption=f"{caption_prefix}_{i}")
            for i, img in enumerate(imgs)
        ]},
        step=step,
    )


def _log_clip_token_panel(
    predicted: torch.Tensor,
    target: torch.Tensor,
    images: torch.Tensor,
    step: int,
    use_wandb: bool,
    n: int = 4,
):
    """
    For the first n images, log:
      - the ground-truth image
      - a heatmap of per-token cosine similarity between predicted and target CLIP tokens
        (16×16 spatial grid, values 0→1)
    Gives a spatial view of which image regions the brain decodes best.
    """
    if not (use_wandb and _WANDB_AVAILABLE):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        B = min(n, predicted.shape[0])
        pred_n = F.normalize(predicted[:B].float().detach().cpu(), dim=-1)
        tgt_n = F.normalize(target[:B].float().detach().cpu(), dim=-1)
        per_token_cos = (pred_n * tgt_n).sum(dim=-1)  # (B, 256)
        spatial = per_token_cos.reshape(B, 16, 16).numpy()

        fig, axes = plt.subplots(2, B, figsize=(3 * B, 6))
        if B == 1:
            axes = [[axes[0]], [axes[1]]]
        for i in range(B):
            # Top row: ground-truth image
            img = images[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            axes[0][i].imshow(img)
            axes[0][i].axis("off")
            axes[0][i].set_title(f"GT image {i}", fontsize=8)
            # Bottom row: per-token cosine heatmap
            im = axes[1][i].imshow(spatial[i], vmin=0, vmax=1, cmap="viridis")
            axes[1][i].axis("off")
            axes[1][i].set_title(f"CLIP cosine {spatial[i].mean():.2f}", fontsize=8)

        plt.colorbar(im, ax=axes[1][-1])
        plt.tight_layout()
        wandb.log({"images/clip_token_cosine": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as e:
        log.debug("CLIP token panel skipped: %s", e)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

_SAVE_AND_EXIT = False


def _handle_signal(signum, frame):
    global _SAVE_AND_EXIT
    log.warning("Signal received — will checkpoint and exit after this epoch.")
    _SAVE_AND_EXIT = True


signal.signal(signal.SIGUSR1, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Stage 1: CLIP alignment
# ---------------------------------------------------------------------------

def train_stage1_epoch(
    model: SemanticBIT,
    clip_extractor: CLIPTargetExtractor,
    loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> tuple[float, dict, int]:
    """Returns (avg_loss, extra_metrics, global_step)."""
    model.train()
    clip_extractor.eval()

    total_loss = 0.0
    total_cosine = 0.0
    n_batches = len(loader)
    t0 = time.time()
    images_seen = 0
    subj_loss_accum: dict[str, list[float]] = {}
    subj_cos_accum: dict[str, list[float]] = {}
    use_wandb = args.wandb and _WANDB_AVAILABLE

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        fmri = batch["fmri"].to(device, non_blocking=True)
        voxel_indices = batch["voxel_indices"].to(device, non_blocking=True)
        cluster_assignments = batch["cluster_assignments"].to(device, non_blocking=True)
        subject_ids = batch["subject_ids"]

        with torch.no_grad():
            clip_targets = clip_extractor(images)  # (B, 256, 1280)

        optimizer.zero_grad(set_to_none=True)

        unique_subjects = list(dict.fromkeys(subject_ids))
        batch_loss = torch.tensor(0.0, device=device)
        batch_cos = 0.0
        n_subj = 0

        for subj in unique_subjects:
            subj_mask = torch.tensor(
                [i for i, s in enumerate(subject_ids) if s == subj], device=device
            )
            if len(subj_mask) == 0:
                continue

            fmri_s = fmri[subj_mask]
            vox_s = voxel_indices[subj_mask[0]]
            clust_s = cluster_assignments[subj_mask[0]]
            targets_s = clip_targets[subj_mask]

            with autocast("cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                predicted = model(fmri_s, vox_s, clust_s, subj)  # (B_s, 256, 1280)
                loss = F.mse_loss(predicted.float(), targets_s.float())

            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            cos_val = _clip_cosine_similarity(predicted.detach(), targets_s)
            batch_loss = batch_loss + loss.detach()
            batch_cos += cos_val
            n_subj += 1
            subj_loss_accum.setdefault(subj, []).append(loss.item())
            subj_cos_accum.setdefault(subj, []).append(cos_val)

        if args.use_amp:
            scaler.unscale_(optimizer)
            gnorm_dict = _grad_norm(model, "bit_")
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            gnorm_dict = _grad_norm(model, "bit_")
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_loss = (batch_loss / max(1, n_subj)).item()
        avg_cos = batch_cos / max(1, n_subj)
        total_loss += avg_loss
        total_cosine += avg_cos
        images_seen += len(images)
        global_step += 1

        if use_wandb and batch_idx % args.log_every_steps == 0:
            _wandb_log({
                "train/loss_step": avg_loss,
                "train/clip_cosine_step": avg_cos,
                "train/lr": optimizer.param_groups[0]["lr"],
                **gnorm_dict,
                **_gpu_stats(device),
            }, step=global_step, use_wandb=use_wandb)

        if batch_idx % 50 == 0:
            log.info(
                "  Epoch %d | Batch %d/%d | L2=%.4f | cosine=%.3f | %.1fs",
                epoch, batch_idx, n_batches, avg_loss, avg_cos, time.time() - t0,
            )

    elapsed = time.time() - t0
    extra = {
        "throughput_imgs_per_sec": images_seen / max(1e-3, elapsed),
        "clip_cosine": total_cosine / max(1, n_batches),
        "subject_losses": {s: float(np.mean(v)) for s, v in subj_loss_accum.items()},
        "subject_cosines": {s: float(np.mean(v)) for s, v in subj_cos_accum.items()},
    }
    return total_loss / max(1, n_batches), extra, global_step


@torch.no_grad()
def validate_stage1(
    model: SemanticBIT,
    clip_extractor: CLIPTargetExtractor,
    loader,
    device,
    args: argparse.Namespace,
) -> tuple[float, dict]:
    """Returns (avg_loss, extra_metrics).

    extra includes _sample_pred, _sample_target, _sample_images for the
    first-batch CLIP token cosine heatmap logged to W&B.
    """
    model.eval()
    total_loss = 0.0
    total_cosine = 0.0
    n_batches = 0
    subj_loss_accum: dict[str, list[float]] = {}
    subj_cos_accum: dict[str, list[float]] = {}

    # First-batch samples for visualisation
    sample_pred: torch.Tensor | None = None
    sample_target: torch.Tensor | None = None
    sample_images: torch.Tensor | None = None

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        fmri = batch["fmri"].to(device, non_blocking=True)
        voxel_indices = batch["voxel_indices"].to(device, non_blocking=True)
        cluster_assignments = batch["cluster_assignments"].to(device, non_blocking=True)
        subject_ids = batch["subject_ids"]

        clip_targets = clip_extractor(images)

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
            targets_s = clip_targets[subj_mask]

            with autocast("cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                predicted = model(fmri_s, vox_s, clust_s, subj)
                loss = F.mse_loss(predicted.float(), targets_s.float())

            cos_val = _clip_cosine_similarity(predicted, targets_s)
            total_loss += loss.item()
            total_cosine += cos_val
            n_batches += 1
            subj_loss_accum.setdefault(subj, []).append(loss.item())
            subj_cos_accum.setdefault(subj, []).append(cos_val)

            # Capture first-batch sample for spatial cosine heatmap
            if sample_pred is None:
                sample_pred = predicted[:4].cpu()
                sample_target = targets_s[:4].cpu()
                sample_images = images[:4].cpu()

    extra = {
        "clip_cosine": total_cosine / max(1, n_batches),
        "subject_losses": {s: float(np.mean(v)) for s, v in subj_loss_accum.items()},
        "subject_cosines": {s: float(np.mean(v)) for s, v in subj_cos_accum.items()},
        "_sample_pred": sample_pred,
        "_sample_target": sample_target,
        "_sample_images": sample_images,
    }
    return total_loss / max(1, n_batches), extra


# ---------------------------------------------------------------------------
# Stage 2: Joint training with MindEye2 diffusion model
# ---------------------------------------------------------------------------

def _load_mindeye2_pipeline(checkpoint_dir: str, device: torch.device):
    """
    Load MindEye2's unCLIP SDXL diffusion pipeline.
    The pipeline conditions on 256 OpenCLIP ViT-bigG/14 spatial tokens.

    MindEye2 weights are available at: huggingface.co/datasets/pscotti/mindeyev2
    The relevant component is the 'versatile_diffusion' prior that accepts
    clip_image_embeds of shape (B, 256, 1280).
    """
    try:
        from diffusers import StableDiffusionXLPipeline, UnCLIPImageVariationPipeline
        from diffusers import DDIMScheduler
    except ImportError:
        raise ImportError(
            "diffusers is required for stage 2. Install with: pip install diffusers accelerate"
        )

    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"MindEye2 checkpoint not found at {checkpoint_dir}.\n"
            "Download from: https://huggingface.co/datasets/pscotti/mindeyev2\n"
            "Or use the MindEye2 repo's download script."
        )

    # Load the unCLIP-style SDXL pipeline used in MindEye2
    # This accepts 256 spatial CLIP tokens as conditioning
    try:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
        )
        pipe.to(device)
        pipe.enable_attention_slicing()
        unet = pipe.unet
        vae = pipe.vae
        noise_scheduler = pipe.scheduler
        return pipe, unet, vae, noise_scheduler
    except Exception as e:
        log.error("Failed to load MindEye2 pipeline: %s", e)
        raise


def train_stage2_epoch(
    bit_model: SemanticBIT,
    unet,
    vae,
    noise_scheduler,
    loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> tuple[float, dict, int]:
    """
    Joint training: SemanticBIT + diffusion UNet.
    Returns (avg_loss, extra_metrics, global_step).
    """
    bit_model.train()
    unet.train()
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    total_loss = 0.0
    n_batches = len(loader)
    t0 = time.time()
    images_seen = 0
    grad_accum_steps = args.grad_accum_steps
    use_wandb = args.wandb and _WANDB_AVAILABLE

    # Timestep-range loss accumulators (early/mid/late denoising phases)
    T = noise_scheduler.config.num_train_timesteps
    range_loss = {"early": [], "mid": [], "late": []}  # late=noisy, early=clean

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        fmri = batch["fmri"].to(device, non_blocking=True)
        voxel_indices = batch["voxel_indices"].to(device, non_blocking=True)
        cluster_assignments = batch["cluster_assignments"].to(device, non_blocking=True)
        subject_ids = batch["subject_ids"]

        unique_subjects = list(dict.fromkeys(subject_ids))
        batch_loss = torch.tensor(0.0, device=device)
        n_subj = 0

        for subj in unique_subjects:
            subj_mask = torch.tensor(
                [i for i, s in enumerate(subject_ids) if s == subj], device=device
            )
            if len(subj_mask) == 0:
                continue

            imgs_s = images[subj_mask]
            fmri_s = fmri[subj_mask]
            vox_s = voxel_indices[subj_mask[0]]
            clust_s = cluster_assignments[subj_mask[0]]
            B_s = len(subj_mask)

            with autocast("cuda", dtype=torch.bfloat16, enabled=args.use_amp):
                with torch.no_grad():
                    latents = vae.encode(imgs_s * 2 - 1).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, T, (B_s,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                clip_tokens = bit_model(fmri_s, vox_s, clust_s, subj)
                noise_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=clip_tokens,
                ).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

            # Track per-timestep-range loss (for W&B)
            with torch.no_grad():
                for i, t in enumerate(timesteps):
                    tv = t.item()
                    per_item_loss = F.mse_loss(
                        noise_pred[i:i+1].float(), noise[i:i+1].float()
                    ).item()
                    if tv < T // 3:
                        range_loss["early"].append(per_item_loss)
                    elif tv < 2 * T // 3:
                        range_loss["mid"].append(per_item_loss)
                    else:
                        range_loss["late"].append(per_item_loss)

            loss_scaled = loss / grad_accum_steps
            if args.use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            batch_loss = batch_loss + loss.detach()
            n_subj += 1

        # Optimizer step after accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            if args.use_amp:
                scaler.unscale_(optimizer)
                gnorm_bit = _grad_norm(bit_model, "bit_")
                gnorm_unet = _grad_norm(unet, "unet_")
                nn.utils.clip_grad_norm_(
                    list(bit_model.parameters()) + list(unet.parameters()), max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                gnorm_bit = _grad_norm(bit_model, "bit_")
                gnorm_unet = _grad_norm(unet, "unet_")
                nn.utils.clip_grad_norm_(
                    list(bit_model.parameters()) + list(unet.parameters()), max_norm=1.0,
                )
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if use_wandb and global_step % args.log_every_steps == 0:
                _wandb_log({
                    "train/loss_step": batch_loss.item() / max(1, n_subj),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    **gnorm_bit,
                    **gnorm_unet,
                    **_gpu_stats(device),
                }, step=global_step, use_wandb=use_wandb)

        avg_loss = (batch_loss / max(1, n_subj)).item()
        total_loss += avg_loss
        images_seen += len(images)

        if batch_idx % 20 == 0:
            log.info("  Epoch %d | Batch %d/%d | Diffusion loss: %.4f | %.1fs",
                     epoch, batch_idx, n_batches, avg_loss, time.time() - t0)

    elapsed = time.time() - t0
    extra = {
        "throughput_imgs_per_sec": images_seen / max(1e-3, elapsed),
        "diffusion_loss_early_t": float(np.mean(range_loss["early"])) if range_loss["early"] else 0.0,
        "diffusion_loss_mid_t": float(np.mean(range_loss["mid"])) if range_loss["mid"] else 0.0,
        "diffusion_loss_late_t": float(np.mean(range_loss["late"])) if range_loss["late"] else 0.0,
    }
    return total_loss / max(1, n_batches), extra, global_step


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scaler, epoch, best_val, history, args):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_val_loss": best_val,
        "history": history,
        "args": vars(args),
    }, path)
    log.info("Saved → %s", path)


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"] + 1, ckpt.get("best_val_loss", float("inf")), ckpt.get("history", [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Stage %d | Device: %s", args.stage, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    if args.wandb and _WANDB_AVAILABLE:
        run_id_path = out_dir / "wandb_run_id.txt"
        run_id = run_id_path.read_text().strip() if run_id_path.exists() else None
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"semantic_s{args.stage}_{Path(args.output_dir).name}",
            id=run_id,
            resume="allow",
            config=vars(args),
        )
        if run_id is None:
            run_id_path.write_text(wandb.run.id)

    # Dataset
    log.info("Loading dataset...")
    train_ds = MultiSubjectBITDataset(
        data_root=args.data_root,
        subjects=args.subjects,
        v2c_dir=args.v2c_dir,
        split="train",
        n_voxels_sample=args.voxels_per_image,
        image_size=256,
    )
    n_val = max(1, int(0.1 * len(train_ds)))
    n_train = len(train_ds) - n_val
    train_split, val_split = torch.utils.data.random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    if args.coco_fmri_dir and Path(args.coco_fmri_dir).exists():
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
        combined_train, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )
    val_loader = make_dataloader(
        val_split, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # Model
    # Register all subjects before .to(device) so per-subject Embedding layers
    # are on the same device (new submodules added after .to(device) stay on CPU).
    model = SemanticBIT(
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

    # Load stage 1 checkpoint for stage 2
    if args.stage == 2:
        if not args.stage1_checkpoint:
            raise ValueError("--stage1_checkpoint required for stage 2")
        ckpt = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        log.info("Loaded stage 1 weights from %s", args.stage1_checkpoint)

    log.info("Model params: %s", count_parameters(model))

    use_wandb = args.wandb and _WANDB_AVAILABLE
    global_step = 0

    # Log V2C metadata once
    if use_wandb:
        import json as _json
        v2c_meta_path = Path(args.v2c_dir) / "metadata.json"
        if v2c_meta_path.exists():
            meta = _json.loads(v2c_meta_path.read_text())
            wandb.config.update({"v2c": meta, "model_params": count_parameters(model)},
                                 allow_val_change=True)

    # ----- Stage 1: CLIP alignment -----
    if args.stage == 1:
        clip_extractor = CLIPTargetExtractor().to(device).eval()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scaler = _make_scaler() if args.use_amp else None
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=args.plateau_patience, min_lr=1e-6
        )

        start_epoch, best_val, history = 0, float("inf"), []
        latest = out_dir / "checkpoint_latest.pt"
        if args.resume and latest.exists():
            start_epoch, best_val, history = load_checkpoint(latest, model, optimizer, scaler, device)
            global_step = start_epoch * len(train_loader)

        for epoch in range(start_epoch, args.epochs):
            if epoch < args.warmup_epochs:
                lr = args.lr * (epoch + 1) / args.warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            t0 = time.time()
            train_loss, train_extra, global_step = train_stage1_epoch(
                model, clip_extractor, train_loader, optimizer, scaler,
                device, epoch, global_step, args,
            )
            val_loss, val_extra = validate_stage1(
                model, clip_extractor, val_loader, device, args
            )
            epoch_time = time.time() - t0

            if epoch >= args.warmup_epochs:
                plateau.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                log.info("New best val_loss=%.4f → best_model.pt", best_val)

            log.info(
                "Epoch %d/%d | train_L2=%.4f cos=%.3f | val_L2=%.4f cos=%.3f | lr=%.2e | %.0f img/s",
                epoch, args.epochs, train_loss, train_extra["clip_cosine"],
                val_loss, val_extra["clip_cosine"], current_lr,
                train_extra["throughput_imgs_per_sec"],
            )
            for subj, sloss in sorted(val_extra["subject_losses"].items()):
                log.info("  val %s: L2=%.4f cos=%.3f",
                         subj, sloss, val_extra["subject_cosines"].get(subj, 0.0))

            record = {
                "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
                "train_clip_cosine": train_extra["clip_cosine"],
                "val_clip_cosine": val_extra["clip_cosine"],
                "lr": current_lr, "time": epoch_time,
            }
            history.append(record)

            if use_wandb:
                epoch_metrics = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/clip_cosine": train_extra["clip_cosine"],
                    "val/clip_cosine": val_extra["clip_cosine"],
                    "train/lr": current_lr,
                    "train/throughput_imgs_per_sec": train_extra["throughput_imgs_per_sec"],
                    "val/is_best": float(is_best),
                    "val/best_loss": best_val,
                }
                for subj, sloss in val_extra["subject_losses"].items():
                    epoch_metrics[f"val/subj_{subj}_loss"] = sloss
                for subj, scos in val_extra["subject_cosines"].items():
                    epoch_metrics[f"val/subj_{subj}_cosine"] = scos
                for subj, sloss in train_extra["subject_losses"].items():
                    epoch_metrics[f"train/subj_{subj}_loss"] = sloss
                epoch_metrics.update(_gpu_stats(device))
                wandb.log(epoch_metrics, step=global_step)

                # Image panels every N epochs
                if epoch % args.log_images_every == 0:
                    if val_extra.get("_sample_images") is not None:
                        _log_image_panel(
                            val_extra["_sample_images"],
                            f"val_gt_epoch{epoch:03d}", global_step, use_wandb,
                        )
                    if (val_extra.get("_sample_pred") is not None
                            and val_extra.get("_sample_target") is not None):
                        _log_clip_token_panel(
                            val_extra["_sample_pred"],
                            val_extra["_sample_target"],
                            val_extra["_sample_images"],
                            global_step, use_wandb,
                        )

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(out_dir / f"checkpoint_epoch{epoch:03d}.pt",
                                model, optimizer, scaler, epoch, best_val, history, args)
            save_checkpoint(latest, model, optimizer, scaler, epoch, best_val, history, args)

            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            if _SAVE_AND_EXIT:
                log.info("SLURM signal: exiting after epoch %d", epoch)
                break

    # ----- Stage 2: Joint training with diffusion -----
    elif args.stage == 2:
        pipe, unet, vae, noise_scheduler = _load_mindeye2_pipeline(
            args.mindeye2_checkpoint, device
        )

        params_to_train = list(model.parameters()) + list(unet.parameters())
        optimizer = torch.optim.AdamW(params_to_train, lr=args.lr, weight_decay=1e-4)
        scaler = _make_scaler() if args.use_amp else None

        start_epoch, best_val, history = 0, float("inf"), []
        latest = out_dir / "checkpoint_latest.pt"
        if args.resume and latest.exists():
            start_epoch, best_val, history = load_checkpoint(latest, model, optimizer, scaler, device)
            global_step = start_epoch * (len(train_loader) // args.grad_accum_steps)

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_loss, train_extra, global_step = train_stage2_epoch(
                model, unet, vae, noise_scheduler,
                train_loader, optimizer, scaler, device, epoch, global_step, args,
            )
            epoch_time = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            log.info(
                "Epoch %d/%d | diff_loss=%.4f (early=%.4f mid=%.4f late=%.4f) | %.0f img/s",
                epoch, args.epochs, train_loss,
                train_extra["diffusion_loss_early_t"],
                train_extra["diffusion_loss_mid_t"],
                train_extra["diffusion_loss_late_t"],
                train_extra["throughput_imgs_per_sec"],
            )

            record = {
                "epoch": epoch, "train_loss": train_loss, "lr": current_lr,
                "time": epoch_time,
                "diffusion_loss_early_t": train_extra["diffusion_loss_early_t"],
                "diffusion_loss_mid_t": train_extra["diffusion_loss_mid_t"],
                "diffusion_loss_late_t": train_extra["diffusion_loss_late_t"],
            }
            history.append(record)

            torch.save(model.state_dict(), out_dir / "bit_latest.pt")
            torch.save(unet.state_dict(), out_dir / "unet_latest.pt")

            if (epoch + 1) % args.save_every == 0:
                torch.save(model.state_dict(), out_dir / f"bit_epoch{epoch:03d}.pt")
                torch.save(unet.state_dict(), out_dir / f"unet_epoch{epoch:03d}.pt")

            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/lr": current_lr,
                    "train/diffusion_loss_early_t": train_extra["diffusion_loss_early_t"],
                    "train/diffusion_loss_mid_t": train_extra["diffusion_loss_mid_t"],
                    "train/diffusion_loss_late_t": train_extra["diffusion_loss_late_t"],
                    "train/throughput_imgs_per_sec": train_extra["throughput_imgs_per_sec"],
                    **_gpu_stats(device),
                }, step=global_step)

            if _SAVE_AND_EXIT:
                log.info("SLURM signal: exiting after epoch %d", epoch)
                break

    log.info("Stage %d complete → %s", args.stage, out_dir)
    if args.wandb and _WANDB_AVAILABLE:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Semantic BIT")
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--data_root", default="/projects/b6ac/brain/algonauts_prepared_data")
    p.add_argument("--v2c_dir", default="/projects/b6ac/brain/brain_it/v2c")
    p.add_argument("--subjects", nargs="+",
                   default=["subj01", "subj02", "subj03", "subj04", "subj05", "subj06", "subj07"])
    p.add_argument("--coco_fmri_dir", default=None)
    p.add_argument("--output_dir",
                   default="/projects/b6ac/brain/checkpoints/bit_semantic_20260318")

    # Stage 2 specific
    p.add_argument("--stage1_checkpoint", default=None,
                   help="Path to best_model.pt from stage 1")
    p.add_argument("--mindeye2_checkpoint",
                   default="/projects/b6ac/brain/checkpoints/mindeye2",
                   help="Directory containing MindEye2 unCLIP SDXL weights")

    # Model
    p.add_argument("--n_clusters", type=int, default=128)
    p.add_argument("--token_dim", type=int, default=512)
    p.add_argument("--n_blocks", type=int, default=5)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=60,
                   help="60 for stage 1, 10 for stage 2")
    p.add_argument("--batch_size", type=int, default=128,
                   help="128 for stage 1, 8 for stage 2")
    p.add_argument("--grad_accum_steps", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    p.add_argument("--voxels_per_image", type=int, default=15_000)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="5e-4 for stage 1, 1e-5 for stage 2")
    p.add_argument("--warmup_epochs", type=int, default=15)
    p.add_argument("--plateau_patience", type=int, default=5)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log_every_steps", type=int, default=20,
                   help="Log step-level metrics to W&B every N optimizer steps")
    p.add_argument("--log_images_every", type=int, default=5,
                   help="Log image panels + CLIP cosine heatmaps to W&B every N epochs")

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="brain-it")
    p.add_argument("--wandb_group", default="semantic")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
