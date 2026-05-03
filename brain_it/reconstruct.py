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

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from bit_model import LowLevelBIT, SemanticBIT
from dip import DIPInverter
from train_semantic import CLIPProjection
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
    model.load_state_dict(sd, strict=False)
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
    model.load_state_dict(sd, strict=False)
    model.eval()
    log.info("Loaded SemanticBIT from %s", checkpoint_path)
    return model


def load_clip_proj(
    checkpoint_path: str,
    device: torch.device,
) -> CLIPProjection:
    """Load the trained CLIPProjection (1280→1664) saved alongside SemanticBIT stage 2."""
    proj = CLIPProjection(in_dim=1280, out_dim=1664).to(device)
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    proj.load_state_dict(sd)
    proj.eval()
    log.info("Loaded CLIPProjection from %s", checkpoint_path)
    return proj


class _SgmVAEWrapper:
    """Wraps sgm AutoencoderKL(InferenceWrapper) in a diffusers-compatible API.

    reconstruct_from_fmri expects:
        vae.encode(x).latent_dist.sample()
        vae.config.scaling_factor
        vae.decode(z).sample
    """
    def __init__(self, vae, scale_factor: float):
        self._vae = vae
        self.config = type("_Cfg", (), {"scaling_factor": scale_factor})()

    def encode(self, x):
        # Cast to the VAE's native dtype (fp32) regardless of what the caller passes
        x = x.to(dtype=next(self._vae.parameters()).dtype)
        result = self._vae.encode(x)
        # sgm AutoencoderKL returns a DiagonalGaussianDistribution with .sample(),
        # but AutoencoderKLInferenceWrapper may return the sampled tensor directly.
        if isinstance(result, torch.Tensor):
            latent = result
            posterior = type("_Post", (), {"sample": lambda self, generator=None: latent})()
        else:
            posterior = result  # DiagonalGaussianDistribution — has .sample()
        return type("_Enc", (), {"latent_dist": posterior})()

    def decode(self, z):
        p = next(self._vae.parameters())
        z = z.to(device=p.device, dtype=p.dtype)
        out = self._vae.decode(z)
        # sgm InferenceWrapper may return a tuple; take first element
        if isinstance(out, (tuple, list)):
            out = out[0]
        return type("_Dec", (), {"sample": out})()

    def eval(self):
        self._vae.eval()
        return self


class _SgmUNetWrapper:
    """Wraps sgm UNetModel in a diffusers-compatible API.

    reconstruct_from_fmri calls:
        unet(latents, t, encoder_hidden_states=clip_cond).sample

    sgm UNetModel expects:
        forward(x, timesteps, context=None, y=None)
    """
    def __init__(self, unet, vector_suffix: torch.Tensor):
        self._unet = unet
        self._vec = vector_suffix  # (1, 1024) fixed conditioning

    def __call__(self, x, t, encoder_hidden_states=None):
        # Cast all inputs to the UNet's native dtype and device
        p = next(self._unet.parameters())
        unet_dtype, unet_device = p.dtype, p.device
        x = x.to(device=unet_device, dtype=unet_dtype)
        t_in = t.unsqueeze(0) if t.dim() == 0 else t
        t_in = t_in.to(device=unet_device)
        batch = x.shape[0]
        y = self._vec.expand(batch, -1).to(device=unet_device, dtype=unet_dtype)
        ctx = encoder_hidden_states.to(device=unet_device, dtype=unet_dtype) if encoder_hidden_states is not None else None
        out = self._unet(
            x,
            timesteps=t_in,
            context=ctx,
            y=y,
        )
        return type("_Out", (), {"sample": out})()

    def load_state_dict(self, sd, strict=True):
        self._unet.load_state_dict(sd, strict=strict)

    def eval(self):
        self._unet.eval()
        return self


class _SgmPipeline:
    """Minimal diffusers-compatible container for sgm components."""
    def __init__(self, unet: _SgmUNetWrapper, vae: _SgmVAEWrapper, scheduler):
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler


def load_diffusion_pipeline(
    mindeye2_checkpoint: str,
    unet_checkpoint: str | None,
    device: torch.device,
) -> _SgmPipeline:
    """
    Load MindEye2's unCLIP model directly from the raw sgm checkpoints.

    The mindeye2 directory contains:
        unclip6_epoch0_step110000.ckpt  — UNet + VAE weights (sgm Lightning ckpt)
        sd_image_var_autoenc.pth        — VAE-only weights (fallback, unused)

    These are loaded via the same _load_unclip_components helper used during
    Stage 2 training, then wrapped in thin adapters so reconstruct_from_fmri
    can call them with the diffusers-style API it was written against.
    """
    try:
        from diffusers import DDIMScheduler
    except ImportError:
        raise ImportError("Install diffusers: pip install diffusers accelerate")

    # Reuse the same loader that Stage 2 training uses — keeps model loading
    # logic in one place and guarantees we load exactly the same architecture.
    from train_semantic import _load_unclip_components
    unet_raw, vae_raw, vector_suffix, scale_factor = _load_unclip_components(
        mindeye2_checkpoint, device
    )

    # Build DDIM scheduler matching sgm's LegacyDDPMDiscretization
    # (scaled-linear betas, 1000 training steps, identical to SDXL schedule)
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        num_train_timesteps=1000,
    )

    unet = _SgmUNetWrapper(unet_raw, vector_suffix)
    vae  = _SgmVAEWrapper(vae_raw, scale_factor)

    if unet_checkpoint and Path(unet_checkpoint).exists():
        unet_sd = torch.load(unet_checkpoint, map_location=device, weights_only=False)
        # Stage 2 only saves the 140 cross-attention K/V projections (attn2.to_k/to_v).
        # All other UNet weights come from the base unclip6 checkpoint already loaded
        # above, so strict=False is correct here.
        unet.load_state_dict(unet_sd, strict=False)
        log.info("Loaded fine-tuned cross-attention (%d tensors) from %s", len(unet_sd), unet_checkpoint)

    unet.eval()
    vae.eval()
    return _SgmPipeline(unet, vae, scheduler)


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
    clip_proj: CLIPProjection,          # projects 1280-dim → 1664-dim for UNet
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

    # 3. DIP: invert VGG tokens → coarse image.
    # reconstruct_from_fmri is decorated @torch.no_grad(), but DIP needs a live
    # autograd graph to optimise its parameters.  Enable grad just for this call.
    log.debug("Running DIP inversion (2000 iters)...")
    with torch.enable_grad():
        coarse_image = dip_inverter.invert(vgg_predicted, verbose=False)
    # coarse_image: (1, 3, 256, 256) in [0, 1]

    # 4. Semantic BIT: predict CLIP tokens, then project to UNet context_dim
    log.debug("Running SemanticBIT...")
    clip_tokens = sem_bit(fmri_s, vox_s, clust_s, subject_id)
    # clip_tokens: (1, 256, 1280) — project to (1, 256, 1664) for UNet cross-attn
    clip_tokens = clip_proj(clip_tokens)

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

def _init_wandb(args: argparse.Namespace, out_dir: Path) -> bool:
    """Initialise a resumable W&B run for reconstruction image logging."""
    if not args.wandb:
        return False
    if not _WANDB_AVAILABLE:
        log.warning("W&B requested but wandb is not installed; skipping W&B logging.")
        return False

    run_id_path = out_dir / "wandb_run_id.txt"
    run_id = run_id_path.read_text().strip() if run_id_path.exists() else None
    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=f"reconstruct_{out_dir.name}",
        id=run_id,
        resume="allow",
        config=vars(args),
    )
    if run_id is None:
        run_id_path.write_text(wandb.run.id)
    return True


def reconstruct_subject(
    args: argparse.Namespace,
    subject_id: str,
    ll_bit: LowLevelBIT,
    sem_bit: SemanticBIT,
    clip_proj: CLIPProjection,
    dip_inverter: DIPInverter,
    pipe,
    v2c_data: dict,
    device: torch.device,
    out_dir: Path,
    use_wandb: bool,
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
    wandb_table = None
    if use_wandb:
        wandb_table = wandb.Table(columns=[
            "subject",
            "index",
            "image_id",
            "reconstruction",
            "ground_truth",
            "reconstruction_path",
            "ground_truth_path",
        ])

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
            clip_proj=clip_proj,
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

        should_log_image = args.wandb_log_every > 0 and (
            idx % args.wandb_log_every == 0 or idx == len(test_ds) - 1
        )
        if use_wandb and should_log_image:
            recon_wb = wandb.Image(
                pil_img,
                caption=f"{subject_id} {idx + 1}/{len(test_ds)} {img_name} reconstruction",
            )
            gt_wb = wandb.Image(
                gt_img,
                caption=f"{subject_id} {idx + 1}/{len(test_ds)} {img_name} ground truth",
            )
            wandb_table.add_data(
                subject_id,
                idx + 1,
                img_name,
                recon_wb,
                gt_wb,
                str(out_path),
                str(gt_path),
            )
            wandb.log({
                f"reconstructions/{subject_id}/latest_pair": [recon_wb, gt_wb],
                f"reconstructions/{subject_id}/completed": idx + 1,
            })

    log.info("Saved %d reconstructions → %s", len(saved_paths), subj_out)
    if use_wandb and wandb_table is not None:
        wandb.log({
            f"reconstructions/{subject_id}/table": wandb_table,
            f"reconstructions/{subject_id}/total": len(saved_paths),
        })
    return saved_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_wandb = _init_wandb(args, out_dir)

    subjects = args.subjects

    # Load V2C
    v2c_data = load_v2c(args.v2c_dir, subjects=subjects)
    meta = v2c_data["metadata"]

    # Load BIT models
    ll_bit = load_lowlevel_bit(args.lowlevel_checkpoint, subjects, meta, device)
    sem_bit = load_semantic_bit(args.semantic_checkpoint, subjects, meta, device)

    # CLIPProjection: 1280-dim SemanticBIT output → 1664-dim UNet context
    # Checkpoint lives alongside the SemanticBIT weights in the same directory
    clip_proj_path = str(Path(args.semantic_checkpoint).parent / "clip_proj_latest.pt")
    clip_proj = load_clip_proj(clip_proj_path, device)

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
            args, subj, ll_bit, sem_bit, clip_proj, dip_inverter, pipe, v2c_data, device, out_dir, use_wandb
        )
        all_paths[subj] = paths

    # Save manifest
    manifest_path = out_dir / "reconstructions.json"
    with open(manifest_path, "w") as f:
        json.dump(all_paths, f, indent=2)
    log.info("Saved manifest → %s", manifest_path)
    log.info("All reconstructions complete → %s", out_dir)
    if use_wandb:
        wandb.log({"reconstructions/manifest": str(manifest_path)})
        wandb.finish()


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

    # Logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="brain-it")
    p.add_argument("--wandb_group", default="reconstruct")
    p.add_argument("--wandb_log_every", type=int, default=1,
                   help="Log every N reconstruction pairs to W&B; <=0 disables image uploads")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
