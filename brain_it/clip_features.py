"""
OpenCLIP ViT-bigG/14 Feature Extractor for the Brain-IT Semantic Branch.

Extracts 256 spatial patch tokens (16×16 grid) from the penultimate layer
of OpenCLIP ViT-bigG/14. These tokens are used as:
  - Training targets for the SemanticBIT (stage 1, L2 loss)
  - Conditioning signal for the MindEye2 unCLIP SDXL diffusion model

Model: OpenCLIP ViT-bigG/14 (open_clip_torch)
  - Hidden dim:       1280
  - Patch size:       14 × 14
  - Input resolution: 224 × 224
  - Spatial tokens:   (224 / 14)² = 256  (CLS excluded)

The 256 spatial tokens are extracted from the last transformer block's output
(before the final projection head), matching MindEye2's conditioning format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lazy import of open_clip to avoid hard dependency at import time
# ---------------------------------------------------------------------------

def _load_openclip():
    try:
        import open_clip
        return open_clip
    except ImportError as e:
        raise ImportError(
            "open_clip_torch is required for the semantic branch. "
            "Install with: pip install open-clip-torch"
        ) from e


# ---------------------------------------------------------------------------
# CLIPTokenExtractor
# ---------------------------------------------------------------------------

class CLIPTokenExtractor(nn.Module):
    """
    Extracts 256 spatial tokens (dim=1280) from OpenCLIP ViT-bigG/14.

    The spatial tokens are the patch embeddings from the final transformer
    layer (CLS token excluded). These are identical to the tokens used as
    conditioning in MindEye2's unCLIP SDXL model.

    All parameters are frozen; this module is only used as a target extractor.
    """

    MODEL_NAME = "ViT-bigG-14"
    PRETRAINED = "laion2b_s39b_b160k"
    N_TOKENS = 256    # 16×16 spatial patches
    TOKEN_DIM = 1280

    def __init__(self, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED):
        super().__init__()
        open_clip = _load_openclip()

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.visual  # Use the visual encoder only

        # Register the preprocessing transform (used externally)
        self.preprocess = preprocess

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Enable output_tokens so forward() returns (pooled, spatial_tokens)
        self.model.output_tokens = True

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224) — preprocessed with CLIP normalization

        Returns:
            spatial_tokens: (B, 256, 1280)
        """
        # output_tokens=True -> (pooled, tokens)
        # tokens: (B, N_patches, width=1664) [CLS excluded by _global_pool]
        _, tokens = self.model(images)

        # Project each spatial token from internal width (1664) to embed_dim (1280)
        # using the same projection matrix applied to the CLS token
        spatial = tokens @ self.model.proj  # (B, 256, 1280)
        return spatial


def preprocess_for_clip(images: torch.Tensor, size: int = 224) -> torch.Tensor:
    """
    Resize and normalize images for OpenCLIP ViT-bigG/14.

    Args:
        images: (B, 3, H, W) in [0, 1]

    Returns:
        (B, 3, 224, 224) with CLIP normalization
    """
    if images.shape[-1] != size or images.shape[-2] != size:
        images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)

    # OpenCLIP normalization (same as standard CLIP)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


class CLIPTargetExtractor(nn.Module):
    """
    Convenience wrapper: takes raw images [0,1], returns 256 spatial CLIP tokens.
    Used during training to generate targets on-the-fly (or offline in a cache).
    """

    def __init__(self, model_name: str = CLIPTokenExtractor.MODEL_NAME,
                 pretrained: str = CLIPTokenExtractor.PRETRAINED):
        super().__init__()
        self.extractor = CLIPTokenExtractor(model_name=model_name, pretrained=pretrained)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) in [0, 1]

        Returns:
            (B, 256, 1280) spatial CLIP tokens
        """
        x = preprocess_for_clip(images)
        return self.extractor(x)


# ---------------------------------------------------------------------------
# Pre-computation utility: cache CLIP tokens for entire dataset
# ---------------------------------------------------------------------------

def precompute_clip_tokens(
    image_paths: list[str],
    output_path: str,
    batch_size: int = 64,
    device: str = "cuda",
) -> None:
    """
    Pre-compute and cache CLIP tokens for a list of images.
    Saves as a numpy memmap for fast random-access loading during training.

    Args:
        image_paths: list of image file paths
        output_path: path to save .npy file (shape: N × 256 × 1280)
        batch_size:  images per GPU batch
        device:      compute device
    """
    import numpy as np
    from pathlib import Path
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    extractor = CLIPTargetExtractor().to(device).eval()
    N = len(image_paths)

    all_tokens = np.zeros((N, 256, 1280), dtype=np.float16)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_imgs = []
        for p in image_paths[start:end]:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(to_tensor(img))
        batch = torch.stack(batch_imgs).to(device)

        tokens = extractor(batch)  # (B, 256, 1280)
        all_tokens[start:end] = tokens.cpu().float().numpy().astype(np.float16)

        if (start // batch_size) % 50 == 0:
            print(f"  CLIP precompute: {end}/{N}")

    np.save(output_path, all_tokens)
    print(f"Saved CLIP tokens → {output_path}  shape={all_tokens.shape}")


if __name__ == "__main__":
    print("Testing CLIPTargetExtractor (requires open_clip_torch and model download)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        extractor = CLIPTargetExtractor().to(device).eval()
        imgs = torch.rand(2, 3, 256, 256).to(device)
        tokens = extractor(imgs)
        print(f"CLIP tokens shape: {tuple(tokens.shape)}  (expected (2, 256, 1280))")
        print("CLIP feature extraction OK")
    except ImportError as e:
        print(f"Skipping: {e}")
