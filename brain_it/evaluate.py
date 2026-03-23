"""
Evaluation Script for Brain-IT Reconstructions.

Computes all 8 standard fMRI decoding metrics used in the literature
(see Table 1 in Brain-IT paper, also used in MindEye2 and related work):

  Low-level metrics (image quality):
    PixCorr   — pixel-wise Pearson correlation (higher is better)
    SSIM      — structural similarity (higher is better)
    Alex(2)   — AlexNet layer-2 cosine similarity (higher is better)
    Alex(5)   — AlexNet layer-5 cosine similarity (higher is better)

  High-level metrics (semantic fidelity):
    Incep     — InceptionV3 cosine similarity (higher is better)
    CLIP      — CLIP ViT-L/14 cosine similarity (higher is better)
    Eff       — EfficientNet-B1 cosine similarity (higher is better)
    SwAV      — ResNet50 + SwAV cosine similarity (higher is better)

All metrics are computed on matched reconstructed-vs-ground-truth image pairs.
Results are averaged over each subject and then over the specified subjects
(default: subj01, subj02, subj05 — as in Brain-IT Table 1).

Usage:
    python evaluate.py \\
        --recon_dir  /projects/b6ac/brain/brain_it/reconstructions \\
        --gt_dir     /projects/b6ac/brain/algonauts_prepared_data \\
        --subjects   subj01 subj02 subj05 \\
        --output_dir /projects/b6ac/brain/brain_it/results

    # Reconstruction directory structure expected:
    # recon_dir/
    #   subj01/
    #     test-0001_nsd-*_reconstructed.png
    #     test-0001_nsd-*_gt.png
    #     ...
    #   subj02/
    #     ...
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading + preprocessing
# ---------------------------------------------------------------------------

def load_image(path: str, size: int = 256) -> torch.Tensor:
    """Load image as (3, size, size) float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    return transform(img)


# ---------------------------------------------------------------------------
# Metric: PixCorr
# ---------------------------------------------------------------------------

def pixel_correlation(recon: torch.Tensor, target: torch.Tensor) -> float:
    """
    Pearson correlation across all pixels of a pair of images.
    Both inputs: (3, H, W) in [0, 1].
    """
    r = recon.flatten()
    t = target.flatten()
    r_c = r - r.mean()
    t_c = t - t.mean()
    corr = (r_c * t_c).sum() / (r_c.norm() * t_c.norm() + 1e-8)
    return corr.item()


# ---------------------------------------------------------------------------
# Metric: SSIM
# ---------------------------------------------------------------------------

def structural_similarity(recon: torch.Tensor, target: torch.Tensor) -> float:
    """
    Structural Similarity Index (SSIM) computed on greyscale images.
    Both inputs: (3, H, W) in [0, 1].
    """
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
        return ssim_fn(recon.unsqueeze(0), target.unsqueeze(0)).item()
    except ImportError:
        # Manual SSIM fallback (simplified, single-scale)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        r = recon.mean(0)  # (H, W) greyscale
        t = target.mean(0)
        mu_r, mu_t = r.mean(), t.mean()
        sigma_r = ((r - mu_r) ** 2).mean().sqrt()
        sigma_t = ((t - mu_t) ** 2).mean().sqrt()
        sigma_rt = ((r - mu_r) * (t - mu_t)).mean()
        num = (2 * mu_r * mu_t + C1) * (2 * sigma_rt + C2)
        den = (mu_r ** 2 + mu_t ** 2 + C1) * (sigma_r ** 2 + sigma_t ** 2 + C2)
        return (num / (den + 1e-8)).item()


# ---------------------------------------------------------------------------
# Feature extractor for AlexNet / Inception / CLIP / EfficientNet / SwAV
# ---------------------------------------------------------------------------

class AlexNetExtractor(nn.Module):
    """AlexNet features at layers 2 and 5."""
    def __init__(self):
        super().__init__()
        alex = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = alex.features
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (layer2_feat, layer5_feat) as 1-D vectors."""
        # AlexNet features sequential indices:
        # 0: Conv1, 1: ReLU, 2: MaxPool    → layer1
        # 3: Conv2, 4: ReLU, 5: MaxPool    → layer2
        # 6: Conv3, 7: ReLU                → layer3
        # 8: Conv4, 9: ReLU                → layer4
        # 10: Conv5, 11: ReLU, 12: MaxPool → layer5
        layer2_out = self.features[:6](x)    # after pool1 + conv2 + relu + pool2
        layer5_out = self.features(x)        # full features
        return (
            layer2_out.flatten(1),
            layer5_out.flatten(1),
        )


class InceptionExtractor(nn.Module):
    """InceptionV3 feature vector (before final classifier)."""
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )
        # Remove final FC
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


class CLIPExtractor(nn.Module):
    """CLIP ViT-L/14 image features."""
    def __init__(self):
        super().__init__()
        import clip
        model, _ = clip.load("ViT-L/14")
        self.model = model.visual
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EfficientNetExtractor(nn.Module):
    """EfficientNet-B1 feature vector."""
    def __init__(self):
        super().__init__()
        eff = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        self.features = eff.features
        self.pool = eff.avgpool
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


class SwAVExtractor(nn.Module):
    """ResNet50 + SwAV feature vector (self-supervised)."""
    def __init__(self):
        super().__init__()
        # Load SwAV from torch.hub
        self.model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        self.model.fc = nn.Identity()  # remove final classifier
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Preprocessing helpers (each model has different normalisation)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def preprocess_imagenet(imgs: torch.Tensor, size: int = 224) -> torch.Tensor:
    """(B, 3, H, W) [0,1] → ImageNet normalised at given size."""
    x = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
    return (x - _IMAGENET_MEAN.to(x)) / _IMAGENET_STD.to(x)


def preprocess_clip(imgs: torch.Tensor, size: int = 224) -> torch.Tensor:
    x = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
    return (x - _CLIP_MEAN.to(x)) / _CLIP_STD.to(x)


# ---------------------------------------------------------------------------
# Load all evaluators
# ---------------------------------------------------------------------------

class BrainITEvaluator:
    """
    Loads all 8 metric models once and evaluates reconstructions.
    """

    def __init__(self, device: torch.device):
        self.device = device
        log.info("Loading evaluation models...")

        self.alex = AlexNetExtractor().to(device).eval()
        self.inception = InceptionExtractor().to(device).eval()
        self.efficientnet = EfficientNetExtractor().to(device).eval()

        try:
            self.clip_extractor = CLIPExtractor().to(device).eval()
            self._has_clip = True
        except Exception as e:
            log.warning("CLIP not available: %s. CLIP metric will be skipped.", e)
            self._has_clip = False

        try:
            self.swav = SwAVExtractor().to(device).eval()
            self._has_swav = True
        except Exception as e:
            log.warning("SwAV not available: %s. SwAV metric will be skipped.", e)
            self._has_swav = False

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return (a * b).sum(dim=-1).mean().item()

    def evaluate_pair(
        self,
        recon: torch.Tensor,   # (3, H, W) float in [0, 1]
        target: torch.Tensor,  # (3, H, W) float in [0, 1]
    ) -> dict[str, float]:
        """Compute all 8 metrics for a single image pair."""
        metrics = {}

        # PixCorr
        metrics["PixCorr"] = pixel_correlation(recon, target)

        # SSIM
        metrics["SSIM"] = structural_similarity(recon, target)

        # AlexNet metrics
        r_in = preprocess_imagenet(recon.unsqueeze(0).to(self.device))
        t_in = preprocess_imagenet(target.unsqueeze(0).to(self.device))
        with torch.no_grad():
            r_alex2, r_alex5 = self.alex(r_in)
            t_alex2, t_alex5 = self.alex(t_in)
        metrics["Alex2"] = self._cosine_similarity(r_alex2, t_alex2)
        metrics["Alex5"] = self._cosine_similarity(r_alex5, t_alex5)

        # InceptionV3
        r_inc = preprocess_imagenet(recon.unsqueeze(0).to(self.device), size=299)
        t_inc = preprocess_imagenet(target.unsqueeze(0).to(self.device), size=299)
        with torch.no_grad():
            r_incep = self.inception(r_inc)
            t_incep = self.inception(t_inc)
        metrics["Incep"] = self._cosine_similarity(r_incep, t_incep)

        # CLIP
        if self._has_clip:
            r_cl = preprocess_clip(recon.unsqueeze(0).to(self.device))
            t_cl = preprocess_clip(target.unsqueeze(0).to(self.device))
            with torch.no_grad():
                r_clip = self.clip_extractor(r_cl)
                t_clip = self.clip_extractor(t_cl)
            metrics["CLIP"] = self._cosine_similarity(r_clip, t_clip)
        else:
            metrics["CLIP"] = float("nan")

        # EfficientNet
        with torch.no_grad():
            r_eff = self.efficientnet(r_in)
            t_eff = self.efficientnet(t_in)
        metrics["Eff"] = self._cosine_similarity(r_eff, t_eff)

        # SwAV
        if self._has_swav:
            with torch.no_grad():
                r_sw = self.swav(r_in)
                t_sw = self.swav(t_in)
            metrics["SwAV"] = self._cosine_similarity(r_sw, t_sw)
        else:
            metrics["SwAV"] = float("nan")

        return metrics

    def evaluate_subject(
        self,
        recon_dir: Path,
        gt_dir: Path,
        subject_id: str,
    ) -> dict:
        """Compute per-image and aggregate metrics for one subject."""
        subj_recon_dir = recon_dir / subject_id
        gt_image_dir = gt_dir / subject_id / "test_split" / "test_images"

        # Find all reconstructed images
        recon_files = sorted(subj_recon_dir.glob("*_reconstructed.png"))
        if not recon_files:
            log.warning("No reconstructed images found in %s", subj_recon_dir)
            return {}

        per_image_metrics = []
        for recon_path in recon_files:
            # Derive GT file name (same stem without _reconstructed suffix)
            stem = recon_path.stem.replace("_reconstructed", "")
            gt_path = subj_recon_dir / f"{stem}_gt.png"

            if not gt_path.exists():
                # Try looking in original GT dir
                pattern = f"{stem}.png"
                candidates = list(gt_image_dir.glob(pattern))
                if not candidates:
                    log.warning("GT not found for %s", recon_path.name)
                    continue
                gt_path = candidates[0]

            recon_img = load_image(str(recon_path))
            gt_img = load_image(str(gt_path))

            metrics = self.evaluate_pair(recon_img, gt_img)
            metrics["image"] = stem
            per_image_metrics.append(metrics)

        # Aggregate
        if not per_image_metrics:
            return {}

        metric_keys = ["PixCorr", "SSIM", "Alex2", "Alex5", "Incep", "CLIP", "Eff", "SwAV"]
        aggregated = {}
        for key in metric_keys:
            vals = [m[key] for m in per_image_metrics if not np.isnan(m.get(key, float("nan")))]
            aggregated[key] = float(np.mean(vals)) if vals else float("nan")

        log.info("%s results:", subject_id)
        for k, v in aggregated.items():
            log.info("  %s = %.4f", k, v)

        return {
            "per_image": per_image_metrics,
            "aggregate": aggregated,
            "n_images": len(per_image_metrics),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    recon_dir = Path(args.recon_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = BrainITEvaluator(device)

    all_results = {}
    for subj in args.subjects:
        log.info("Evaluating %s...", subj)
        results = evaluator.evaluate_subject(recon_dir, gt_dir, subj)
        all_results[subj] = results

    # Average across subjects
    metric_keys = ["PixCorr", "SSIM", "Alex2", "Alex5", "Incep", "CLIP", "Eff", "SwAV"]
    avg_results = {}
    for key in metric_keys:
        vals = [
            all_results[s]["aggregate"][key]
            for s in args.subjects
            if s in all_results and "aggregate" in all_results[s]
            and not np.isnan(all_results[s]["aggregate"].get(key, float("nan")))
        ]
        avg_results[key] = float(np.mean(vals)) if vals else float("nan")

    log.info("\n" + "=" * 50)
    log.info("AVERAGE across subjects: %s", ", ".join(args.subjects))
    log.info("=" * 50)
    for k, v in avg_results.items():
        log.info("  %s = %.4f", k, v)
    log.info("=" * 50)

    # Save results
    output = {
        "subjects": args.subjects,
        "per_subject": {s: all_results[s].get("aggregate", {}) for s in args.subjects},
        "average": avg_results,
        "recon_dir": str(recon_dir),
    }
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved results → %s", results_path)

    # Also save per-image metrics for detailed analysis
    for subj in args.subjects:
        if subj in all_results and "per_image" in all_results[subj]:
            per_img_path = out_dir / f"per_image_{subj}.json"
            with open(per_img_path, "w") as f:
                json.dump(all_results[subj]["per_image"], f, indent=2)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Brain-IT reconstructions")
    p.add_argument(
        "--recon_dir",
        default="/projects/b6ac/brain/brain_it/reconstructions",
        help="Directory with reconstructed images (one subdir per subject)"
    )
    p.add_argument(
        "--gt_dir",
        default="/projects/b6ac/brain/algonauts_prepared_data",
        help="Algonauts data root (for GT images)"
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        default=["subj01", "subj02", "subj05"],
        help="Subjects to evaluate (default: subj01, subj02, subj05 as in paper)"
    )
    p.add_argument(
        "--output_dir",
        default="/projects/b6ac/brain/brain_it/results"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
