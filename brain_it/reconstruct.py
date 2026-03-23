"""
Brain-IT Inference Pipeline: fMRI → Image Reconstruction.

Full reconstruction pipeline (Figure 2 in paper):
  1. [BIT] Predict VGG tokens (low-level) and CLIP tokens (semantic) from fMRI
  2. [DIP] Invert predicted VGG tokens → coarse 112×112 image via 2K iter optimisation
  3. [Diffusion init] Upsample DIP output → 256×256, encode via VAE, add noise at step 14/38
  4. [SDXL denoising] 38-step DDIM/DDPM denoising conditioned on BIT CLIP tokens
  5. [SDXL refinement] Optional high-res refinement pass (paper: one additional pass)

Output: 256×256 reconstructed image

Usage:
    python reconstruct.py \\
        --lowlevel_checkpoint /projects/b6ac/brain/checkpoints/bit_lowlevel_*/best_model.pt \\
        --semantic_checkpoint /projects/b6ac/brain/checkpoints/bit_semantic_*/bit_latest.pt \\
        --unet_checkpoint     /projects/b6ac/brain/checkpoints/bit_semantic_joint_*/unet_latest.pt \\
        --mindeye2_checkpoint /projects/b6ac/brain/checkpoints/mindeye2 \\
        --v2c_dir             /projects/b6ac/brain/brain_it/v2c \\
        --data_root           /projects/b6ac/brain/algonauts_prepared_data \\
        --subject             subj01 \\
        --output_dir          /projects/b6ac/brain/brain_it/reconstructions/subj01 \\
        --split               test
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from bit_model import LowLevelBIT, SemanticBIT
from dip import DIPInverter
from vgg_features import VGGTargetExtractor, VGGFeatureExtractor
from v2c_mapping import load_v2c, get_cluster_assignments_tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_lowlevel_bit(
    checkpoint_path: str,
    subjects: list[str],
    v2c_metadata: dict,
    device: torch.device,
) -> LowLevelBIT:
    model = LowLevelBIT(
        n_clusters=v2c_metadata["n_clusters"],
        token_dim=512,
        n_blocks=5,
        n_heads=8,
    ).to(device)

    for subj in subjects:
        n_vox = v2c_metadata["voxels_per_subject"][subj]
        model.register_subject(subj, n_vox)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    log.info("Loaded LowLevelBIT from %s", checkpoint_path)
    return model


def load_semantic_bit(
    checkpoint_path: str,
    subjects: list[str],
    v2c_metadata: dict,
    device: torch.device,
) -> SemanticBIT:
    model = SemanticBIT(
        n_clusters=v2c_metadata["n_clusters"],
        token_dim=512,
        n_blocks=5,
        n_heads=8,
    ).to(device)

    for subj in subjects:
        n_vox = v2c_metadata["voxels_per_subject"][subj]
        model.register_subject(subj, n_vox)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    log.info("Loaded SemanticBIT from %s", checkpoint_path)
    return model


def load_diffusion_pipeline(
    mindeye2_checkpoint: str,
    unet_checkpoint: str | None,
    device: torch.device,
):
    """
    Load MindEye2's unCLIP SDXL pipeline.
    Optionally replace the UNet with fine-tuned weights from stage 2.
    """
    try:
        from diffusers import DiffusionPipeline, DDIMScheduler
    except ImportError:
        raise ImportError("Install diffusers: pip install diffusers accelerate")

    log.info("Loading MindEye2 diffusion pipeline from %s", mindeye2_checkpoint)
    pipe = DiffusionPipeline.from_pretrained(
        mindeye2_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.to(device)

    # Replace scheduler with DDIM for deterministic decoding
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if unet_checkpoint and Path(unet_checkpoint).exists():
        unet_sd = torch.load(unet_checkpoint, map_location=device, weights_only=False)
        pipe.unet.load_state_dict(unet_sd)
        log.info("Loaded fine-tuned UNet from %s", unet_checkpoint)

    pipe.unet.eval()
    pipe.vae.eval()
    return pipe


# ---------------------------------------------------------------------------
# Core reconstruction function
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_from_fmri(
    fmri_activations: torch.Tensor,     # (1, N_voxels_total) full fMRI
    voxel_indices_all: torch.Tensor,    # (N_voxels_total,) all voxel indices
    cluster_assignments: torch.Tensor,  # (N_voxels_total,) all cluster assignments
    subject_id: str,
    ll_bit: LowLevelBIT,
    sem_bit: SemanticBIT,
    dip_inverter: DIPInverter,
    pipe,                               # diffusion pipeline
    device: torch.device,
    n_voxels_sample: int = 15_000,
    diffusion_steps: int = 38,
    init_step: int = 14,
    output_size: int = 256,
    seed: int = 42,
) -> torch.Tensor:
    """
    Reconstruct one image from fMRI activations.

    Args:
        fmri_activations: (1, N_vox) full fMRI vector
        ...

    Returns:
        reconstructed: (1, 3, output_size, output_size) in [0, 1]
    """
    rng = torch.Generator(device=device).manual_seed(seed)

    # 1. Sample voxels for BIT (same subset for both branches for consistency)
    N_vox = voxel_indices_all.shape[0]
    n_sample = min(n_voxels_sample, N_vox)
    perm = torch.randperm(N_vox, generator=rng, device=device)[:n_sample]

    fmri_s = fmri_activations[:, perm]    # (1, n_sample)
    vox_s = voxel_indices_all[perm]        # (n_sample,)
    clust_s = cluster_assignments[perm]    # (n_sample,)

    # 2. Low-level BIT: predict VGG tokens
    log.debug("Running LowLevelBIT...")
    vgg_predicted = ll_bit(fmri_s, vox_s, clust_s, subject_id)
    # vgg_predicted: list of 5 tensors [(1, N_l, D_l)]

    # 3. DIP: invert VGG tokens → coarse image
    log.debug("Running DIP inversion (2000 iters)...")
    coarse_image = dip_inverter.invert(vgg_predicted, verbose=False)
    # coarse_image: (1, 3, 256, 256) in [0, 1]

    # 4. Semantic BIT: predict CLIP tokens
    log.debug("Running SemanticBIT...")
    clip_tokens = sem_bit(fmri_s, vox_s, clust_s, subject_id)
    # clip_tokens: (1, 256, 1280) — conditioning for diffusion

    # 5. Encode DIP output via VAE → latent space
    vae = pipe.vae
    scheduler = pipe.scheduler
    unet = pipe.unet

    coarse_vae_input = coarse_image.to(dtype=torch.float16) * 2 - 1  # [-1, 1]
    latents = vae.encode(coarse_vae_input).latent_dist.sample()
    latents = latents * vae.config.scaling_factor  # (1, 4, H/8, W/8)

    # 6. Add noise at init_step (partial inversion to get noisy latent)
    # This allows the diffusion model to "correct" the coarse DIP image
    # while being conditioned on the BIT CLIP tokens
    scheduler.set_timesteps(diffusion_steps)
    timesteps = scheduler.timesteps  # e.g. [999, 972, ..., 27]

    # Add noise as if at step `init_step` out of `diffusion_steps`
    noise = torch.randn(latents.shape, device=device, dtype=torch.float16, generator=rng)
    t_init = timesteps[init_step]
    noisy_latents = scheduler.add_noise(latents, noise, t_init.unsqueeze(0))

    # 7. DDIM denoising from step `init_step` onwards
    latents_denoised = noisy_latents
    clip_cond = clip_tokens.to(dtype=torch.float16)

    for t in timesteps[init_step:]:
        noise_pred = unet(
            latents_denoised,
            t,
            encoder_hidden_states=clip_cond,
        ).sample

        step_output = scheduler.step(noise_pred, t, latents_denoised)
        latents_denoised = step_output.prev_sample

    # 8. Decode latents → image
    latents_denoised = latents_denoised / vae.config.scaling_factor
    decoded = vae.decode(latents_denoised).sample  # (1, 3, H, W) in [-1, 1]
    reconstructed = (decoded.clamp(-1, 1) + 1) / 2  # → [0, 1]

    return reconstructed.float()


# ---------------------------------------------------------------------------
# Batch reconstruction over test set
# ---------------------------------------------------------------------------

def reconstruct_subject(
    args: argparse.Namespace,
    subject_id: str,
    ll_bit: LowLevelBIT,
    sem_bit: SemanticBIT,
    dip_inverter: DIPInverter,
    pipe,
    v2c_data: dict,
    device: torch.device,
    out_dir: Path,
) -> list[str]:
    """Run reconstruction for all test images of one subject."""
    from dataset import NSDTestDataset

    test_ds = NSDTestDataset(
        data_root=args.data_root,
        subject_id=subject_id,
        v2c_dir=args.v2c_dir,
        image_size=256,
    )
    log.info("%s: %d test images", subject_id, len(test_ds))

    subj_out = out_dir / subject_id
    subj_out.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        fmri = sample["fmri"].unsqueeze(0).to(device)         # (1, N_vox)
        voxel_idx = sample["voxel_indices"].to(device)         # (N_vox,)
        cluster_a = sample["cluster_assignments"].to(device)   # (N_vox,)

        log.info("  Reconstructing %s [%d/%d]...", subject_id, idx + 1, len(test_ds))

        recon = reconstruct_from_fmri(
            fmri_activations=fmri,
            voxel_indices_all=voxel_idx,
            cluster_assignments=cluster_a,
            subject_id=subject_id,
            ll_bit=ll_bit,
            sem_bit=sem_bit,
            dip_inverter=dip_inverter,
            pipe=pipe,
            device=device,
            n_voxels_sample=args.voxels_per_image,
            diffusion_steps=args.diffusion_steps,
            init_step=args.init_step,
            output_size=args.output_size,
            seed=args.seed + idx,
        )  # (1, 3, 256, 256)

        # Save reconstructed image
        img_name = test_ds.image_files[idx].stem
        out_path = subj_out / f"{img_name}_reconstructed.png"
        pil_img = to_pil_image(recon[0].clamp(0, 1))
        pil_img.save(out_path)
        saved_paths.append(str(out_path))

        # Also save ground truth for comparison
        gt_path = subj_out / f"{img_name}_gt.png"
        gt_img = Image.open(test_ds.image_files[idx]).convert("RGB")
        gt_img.save(gt_path)

    log.info("Saved %d reconstructions → %s", len(saved_paths), subj_out)
    return saved_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects

    # Load V2C
    v2c_data = load_v2c(args.v2c_dir, subjects=subjects)
    meta = v2c_data["metadata"]

    # Load BIT models
    ll_bit = load_lowlevel_bit(args.lowlevel_checkpoint, subjects, meta, device)
    sem_bit = load_semantic_bit(args.semantic_checkpoint, subjects, meta, device)

    # DIP inverter
    vgg_ext = VGGFeatureExtractor().to(device).eval()
    dip_inverter = DIPInverter(
        vgg_extractor=vgg_ext,
        device=device,
        image_size=112,
        n_iters=args.dip_iters,
        output_size=args.output_size,
    )

    # Diffusion pipeline
    pipe = load_diffusion_pipeline(
        args.mindeye2_checkpoint,
        args.unet_checkpoint,
        device,
    )

    # Reconstruct for each subject
    all_paths = {}
    for subj in subjects:
        paths = reconstruct_subject(
            args, subj, ll_bit, sem_bit, dip_inverter, pipe, v2c_data, device, out_dir
        )
        all_paths[subj] = paths

    # Save manifest
    manifest_path = out_dir / "reconstructions.json"
    with open(manifest_path, "w") as f:
        json.dump(all_paths, f, indent=2)
    log.info("Saved manifest → %s", manifest_path)
    log.info("All reconstructions complete → %s", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruct images from fMRI using Brain-IT")
    p.add_argument("--lowlevel_checkpoint", required=True)
    p.add_argument("--semantic_checkpoint", required=True)
    p.add_argument("--unet_checkpoint", default=None,
                   help="Fine-tuned UNet from semantic stage 2 (optional)")
    p.add_argument("--mindeye2_checkpoint",
                   default="/projects/b6ac/brain/checkpoints/mindeye2")
    p.add_argument("--v2c_dir", default="/projects/b6ac/brain/brain_it/v2c")
    p.add_argument("--data_root", default="/projects/b6ac/brain/algonauts_prepared_data")
    p.add_argument("--subjects", nargs="+", default=["subj01", "subj02", "subj05"])
    p.add_argument("--output_dir",
                   default="/projects/b6ac/brain/brain_it/reconstructions")
    p.add_argument("--split", choices=["train", "test"], default="test")

    # Inference config
    p.add_argument("--voxels_per_image", type=int, default=15_000)
    p.add_argument("--diffusion_steps", type=int, default=38)
    p.add_argument("--init_step", type=int, default=14,
                   help="Diffusion step to initialise from DIP output (paper: 14/38)")
    p.add_argument("--output_size", type=int, default=256)
    p.add_argument("--dip_iters", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
