"""
VGG-16+BN Feature Extractor for the Brain-IT Low-Level Branch.

Extracts features from 5 intermediate layers of VGG-16 with batch normalization,
operating on 112×112 images as specified in the paper (Appendix D.2).

Token layout (from paper Sec. D.2):
  relu1_2: 112×112 spatial → merge 2×2 no-overlap  → 56×56 = 3136 tokens, 64 ch
  relu2_2: 56×56  spatial  → merge 2×2 with overlap (stride 1) → 55×55 = 3025 tokens, 128 ch
  relu3_3: 28×28  spatial  → 28×28 = 784 tokens, 256 ch
  relu4_3: 14×14  spatial  → 14×14 = 196 tokens, 512 ch
  relu5_3: 7×7    spatial  → 7×7   = 49  tokens, 512 ch

"Tokens with fewer than 512 dimensions are replicated until reaching size 512."
→ relu1_2 (64ch) and relu2_2 (128ch) are replicated; relu3_3+ are already ≥256ch.
  The paper says "replicated until reaching size 512", so:
  64→512 (8×), 128→512 (4×), 256→512 (2×), 512→512 (1×).

During training: randomly sample subsets of tokens per layer to reduce memory.
At inference: predict all tokens.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Token counts (must match bit_model.VGGTokenConfig)
# ---------------------------------------------------------------------------

VGG_LAYER_CONFIG = [
    # (n_tokens, raw_ch, padded_ch, layer_name, merge_type)
    (56 * 56,  64,  512, "relu1_2", "no_overlap"),
    (55 * 55, 128,  512, "relu2_2", "overlap"),
    (28 * 28, 256,  512, "relu3_3", "none"),
    (14 * 14, 512,  512, "relu4_3", "none"),
    ( 7 *  7, 512,  512, "relu5_3", "none"),
]

# Training sample counts per layer (paper: 512, 512, 128, 64, 16)
VGG_TRAIN_SAMPLES = [512, 512, 128, 64, 16]

VGG_IMAGE_SIZE = 112  # VGG input resolution used in Brain-IT


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class VGGFeatureExtractor(nn.Module):
    """
    Frozen VGG-16+BN backbone that returns intermediate feature maps
    from layers relu1_2, relu2_2, relu3_3, relu4_3, relu5_3.

    All parameters are frozen; this is only used as a target/loss network.
    """

    # VGG-16 layer indices for the target relu activations
    # torchvision VGG-16+BN features sequential:
    # 0:Conv,1:BN,2:ReLU → relu1_1
    # 3:Conv,4:BN,5:ReLU → relu1_2  (idx=5)
    # 6:MaxPool
    # 7:Conv,8:BN,9:ReLU → relu2_1
    # 10:Conv,11:BN,12:ReLU → relu2_2 (idx=12)
    # 13:MaxPool
    # 14:Conv,15:BN,16:ReLU → relu3_1
    # 17:Conv,18:BN,19:ReLU → relu3_2
    # 20:Conv,21:BN,22:ReLU → relu3_3 (idx=22)
    # 23:MaxPool
    # 24:Conv,25:BN,26:ReLU → relu4_1
    # 27:Conv,28:BN,29:ReLU → relu4_2
    # 30:Conv,31:BN,32:ReLU → relu4_3 (idx=32)
    # 33:MaxPool
    # 34:Conv,35:BN,36:ReLU → relu5_1
    # 37:Conv,38:BN,39:ReLU → relu5_2
    # 40:Conv,41:BN,42:ReLU → relu5_3 (idx=42)
    LAYER_INDICES = {
        "relu1_2": 5,
        "relu2_2": 12,
        "relu3_3": 22,
        "relu4_3": 32,
        "relu5_3": 42,
    }

    def __init__(self):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.features = vgg.features

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self._extract_at = sorted(self.LAYER_INDICES.values())

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, 3, 112, 112)  — pre-normalized images

        Returns:
            list of 5 feature maps in layer order:
              [(B, 64, 112, 112), (B, 128, 56, 56),
               (B, 256, 28, 28),  (B, 512, 14, 14),  (B, 512, 7, 7)]
        """
        outputs = []
        extract_set = set(self._extract_at)

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in extract_set:
                outputs.append(x)

        return outputs  # 5 feature maps


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _replicate_channels(feat: torch.Tensor, target_dim: int = 512) -> torch.Tensor:
    """
    Replicate feature channels until reaching target_dim.
    e.g. 64-dim → repeat 8 times = 512-dim.
    """
    ch = feat.shape[-1]
    if ch >= target_dim:
        return feat[..., :target_dim]
    reps = math.ceil(target_dim / ch)
    return feat.repeat(*([1] * (feat.ndim - 1)), reps)[..., :target_dim]


def tokenize_relu1_2(feat_map: torch.Tensor) -> torch.Tensor:
    """
    relu1_2: (B, 64, 112, 112) → merge 2×2 no-overlap → (B, 56*56, 512)
    """
    B, C, H, W = feat_map.shape  # 64, 112, 112
    # Merge 2×2 no-overlap via unfold
    # After merge: B, C*4, 56, 56
    merged = feat_map.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, C, 56, 56, 2, 2)
    merged = merged.contiguous().view(B, C, 56, 56, 4)
    merged = merged.permute(0, 2, 3, 4, 1).contiguous()  # (B, 56, 56, 4, C)
    merged = merged.view(B, 56 * 56, C * 4)  # (B, 3136, 256)
    # Replicate to 512
    tokens = _replicate_channels(merged, 512)  # (B, 3136, 512)
    return tokens


def tokenize_relu2_2(feat_map: torch.Tensor) -> torch.Tensor:
    """
    relu2_2: (B, 128, 56, 56) → merge 2×2 with overlap (stride 1) → (B, 55*55, 512)
    """
    B, C, H, W = feat_map.shape  # 128, 56, 56
    # Merge 2×2 with stride 1 (overlap): unfold gives 55×55 windows
    merged = feat_map.unfold(2, 2, 1).unfold(3, 2, 1)  # (B, C, 55, 55, 2, 2)
    merged = merged.contiguous().view(B, C, 55, 55, 4)
    merged = merged.permute(0, 2, 3, 4, 1).contiguous()  # (B, 55, 55, 4, C)
    merged = merged.view(B, 55 * 55, C * 4)  # (B, 3025, 512)
    # 128*4=512 → already 512
    tokens = _replicate_channels(merged, 512)
    return tokens


def tokenize_simple(feat_map: torch.Tensor) -> torch.Tensor:
    """
    relu3_3 / relu4_3 / relu5_3: (B, C, H, W) → (B, H*W, 512)
    Channels are replicated/truncated to 512.
    """
    B, C, H, W = feat_map.shape
    tokens = feat_map.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
    tokens = _replicate_channels(tokens, 512)
    return tokens


def vgg_maps_to_tokens(feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Convert 5 VGG feature maps to token lists following Brain-IT tokenisation.

    Args:
        feature_maps: list of 5 tensors in layer order (output of VGGFeatureExtractor)

    Returns:
        list of 5 token tensors:
          [(B, 3136, 512), (B, 3025, 512), (B, 784, 512), (B, 196, 512), (B, 49, 512)]
    """
    assert len(feature_maps) == 5
    return [
        tokenize_relu1_2(feature_maps[0]),
        tokenize_relu2_2(feature_maps[1]),
        tokenize_simple(feature_maps[2]),
        tokenize_simple(feature_maps[3]),
        tokenize_simple(feature_maps[4]),
    ]


# ---------------------------------------------------------------------------
# Training-time random token sampling
# ---------------------------------------------------------------------------

def sample_tokens(
    token_list: list[torch.Tensor],
    sample_counts: list[int] = VGG_TRAIN_SAMPLES,
    generator: Optional[torch.Generator] = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Randomly sample token subsets during training to reduce memory.

    Returns:
        sampled_tokens: list of (B, n_samples_l, 512)
        indices:        list of (n_samples_l,) for each layer (for computing targets)
    """
    sampled, indices = [], []
    for tokens, n_sample in zip(token_list, sample_counts):
        B, N, D = tokens.shape
        n = min(n_sample, N)
        idx = torch.randperm(N, generator=generator, device=tokens.device)[:n]
        sampled.append(tokens[:, idx, :])
        indices.append(idx)
    return sampled, indices


# ---------------------------------------------------------------------------
# Image preprocessing for VGG
# ---------------------------------------------------------------------------

def preprocess_for_vgg(images: torch.Tensor, size: int = VGG_IMAGE_SIZE) -> torch.Tensor:
    """
    Resize and normalize images for VGG-16+BN.

    Args:
        images: (B, 3, H, W) in [0, 1]
        size:   target spatial size (112 for Brain-IT)

    Returns:
        (B, 3, size, size) normalized with ImageNet mean/std
    """
    if images.shape[-1] != size or images.shape[-2] != size:
        images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


# ---------------------------------------------------------------------------
# All-in-one helper
# ---------------------------------------------------------------------------

class VGGTargetExtractor(nn.Module):
    """
    Convenience wrapper: takes raw images, returns tokenised VGG features.
    Used during training to generate targets on-the-fly.
    """

    def __init__(self):
        super().__init__()
        self.vgg = VGGFeatureExtractor()

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        sample: bool = False,
        sample_counts: list[int] = VGG_TRAIN_SAMPLES,
    ) -> list[torch.Tensor]:
        """
        Args:
            images:        (B, 3, H, W) in [0, 1]
            sample:        if True, randomly sample tokens per layer
            sample_counts: tokens to sample per layer (only used if sample=True)

        Returns:
            list of 5 token tensors (B, N_l, 512)
        """
        x = preprocess_for_vgg(images)
        feat_maps = self.vgg(x)
        tokens = vgg_maps_to_tokens(feat_maps)
        if sample:
            tokens, _ = sample_tokens(tokens, sample_counts)
        return tokens


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    extractor = VGGTargetExtractor().to(device)
    images = torch.rand(4, 3, 256, 256, device=device)

    tokens = extractor(images, sample=False)
    for tok, (n, ch, _, name, _) in zip(tokens, VGG_LAYER_CONFIG):
        print(f"  {name}: {tuple(tok.shape)}  (expected (4, {n}, 512))")

    tokens_s, _ = sample_tokens(tokens, VGG_TRAIN_SAMPLES)
    for tok, n_s, (_, _, _, name, _) in zip(tokens_s, VGG_TRAIN_SAMPLES, VGG_LAYER_CONFIG):
        print(f"  {name} (sampled {n_s}): {tuple(tok.shape)}")

    print("VGG feature extraction OK")
