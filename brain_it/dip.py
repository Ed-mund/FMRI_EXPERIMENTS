"""
Deep Image Prior (DIP) for Brain-IT Low-Level Image Reconstruction.

Implements the DIP framework (Ulyanov et al., 2018) used in Brain-IT to invert
predicted VGG features back into an image. The DIP model is a U-Net with
convolutional inductive biases that naturally produces natural-looking images.

Architecture (from paper Appendix D.2):
  - U-Net with input dimensionality 32, internal dimensionality 128
  - 3 scales (encoder: 3 downsampling blocks, decoder: 3 upsampling blocks)
  - Bilinear interpolation for up/downsampling
  - Input: fixed random noise tensor (32-dim channels, 112×112)
  - Output: reconstructed image (3 channels, 112×112)

Inference-time optimisation (per test fMRI):
  - Freeze the VGG-16+BN network
  - Optimise DIP weights to minimise L2 between DIP(noise) → VGG and predicted VGG
  - 2000 iterations, Adam lr=0.01
  - Initial noise scale: 0.1
  - Regularisation noise added each iteration: std = 1/30
  - Exponential Moving Average (EMA) with factor 0.99 on the output image
  - Output: upsampled from 112×112 → 256×256 to initialise diffusion
"""

import math
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# U-Net DIP Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv → BN → LeakyReLU (×2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDIP(nn.Module):
    """
    U-Net Deep Image Prior.

    Input:  (B, in_ch=32, H, W)  — fixed random noise, same shape as output
    Output: (B, 3, H, W)         — predicted image in [0, 1]

    Architecture (3 scales, internal dim 128):
      Encoder: 32 → 128 → 128 → 128  (with bilinear downsampling ×2 between scales)
      Bottleneck: 128 → 128
      Decoder: 128+128 → 128 → 128+128 → 128 → 128+128 → 128 → 3
               (skip connections from encoder, bilinear upsampling ×2)
    """

    def __init__(self, in_ch: int = 32, hidden: int = 128, n_scales: int = 3):
        super().__init__()
        self.n_scales = n_scales

        # Encoder
        self.enc = nn.ModuleList()
        ch = in_ch
        enc_channels = []
        for _ in range(n_scales):
            self.enc.append(ConvBlock(ch, hidden))
            enc_channels.append(hidden)
            ch = hidden

        # Bottleneck
        self.bottleneck = ConvBlock(hidden, hidden)

        # Decoder (skip connections from encoder)
        self.dec = nn.ModuleList()
        for i in range(n_scales):
            skip_ch = enc_channels[-(i + 1)]
            self.dec.append(ConvBlock(hidden + skip_ch, hidden))

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_ch, H, W)

        Returns:
            (B, 3, H, W) in [0, 1]
        """
        skips = []

        # Encoder
        for enc_block in self.enc:
            x = enc_block(x)
            skips.append(x)
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, dec_block in enumerate(self.dec):
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            skip = skips[-(i + 1)]
            # Handle size mismatch from rounding
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        return self.out_conv(x)


# ---------------------------------------------------------------------------
# DIP Inference-Time Optimiser
# ---------------------------------------------------------------------------

class DIPInverter:
    """
    Optimises a DIP U-Net at inference time to reconstruct an image whose
    VGG features match predicted VGG features from the BIT model.

    Usage:
        inverter = DIPInverter(vgg_extractor, device=device)
        image = inverter.invert(predicted_vgg_tokens)

    This runs a fresh DIP optimisation for each test fMRI sample.
    """

    def __init__(
        self,
        vgg_extractor,   # VGGTargetExtractor or VGGFeatureExtractor
        device: torch.device | str = "cuda",
        image_size: int = 112,
        n_iters: int = 2000,
        lr: float = 0.01,
        noise_std: float = 0.1,
        reg_noise_std: float = 1 / 30,
        ema_decay: float = 0.99,
        output_size: int = 256,
    ):
        self.vgg = vgg_extractor
        self.device = torch.device(device)
        self.image_size = image_size
        self.n_iters = n_iters
        self.lr = lr
        self.noise_std = noise_std
        self.reg_noise_std = reg_noise_std
        self.ema_decay = ema_decay
        self.output_size = output_size

        # Freeze VGG
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.to(self.device).eval()

    @torch.no_grad()
    def _init_noise(self, size: int = 112) -> torch.Tensor:
        """Create fixed input noise for DIP."""
        return torch.randn(1, 32, size, size, device=self.device) * self.noise_std

    def _predicted_tokens_to_maps(self, predicted_tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Convert predicted token tensors back to spatial feature maps for L2 comparison.
        predicted_tokens: list of (1, N_tokens_l, ch_l) — output of LowLevelBIT
        Returns: list of (1, ch_l, H_l, W_l) feature maps
        """
        from vgg_features import VGG_LAYER_CONFIG
        maps = []
        spatial_shapes = [
            (56, 56),
            (55, 55),
            (28, 28),
            (14, 14),
            (7, 7),
        ]
        for tokens, (n_tok, ch, _, name, _), (H, W) in zip(
            predicted_tokens, VGG_LAYER_CONFIG, spatial_shapes
        ):
            # tokens: (1, N, 512) — take first `ch` dims (un-replicate)
            tok = tokens[0, :, :ch]  # (N, ch)
            feat_map = tok.reshape(H, W, ch).permute(2, 0, 1).unsqueeze(0)  # (1, ch, H, W)
            maps.append(feat_map.detach())
        return maps

    def invert(
        self,
        predicted_tokens: list[torch.Tensor],
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Run DIP optimisation to invert predicted VGG tokens back to an image.

        Args:
            predicted_tokens: list of 5 tensors (1, N_l, 512) from LowLevelBIT
            verbose:          print loss every 500 iters

        Returns:
            image: (1, 3, output_size, output_size) — EMA output, upsampled
        """
        from vgg_features import vgg_maps_to_tokens, preprocess_for_vgg
        from vgg_features import VGGFeatureExtractor

        # Target feature maps (detached from computational graph)
        target_maps = self._predicted_tokens_to_maps(predicted_tokens)

        # Build a fresh DIP model for this inversion
        dip = UNetDIP(in_ch=32, hidden=128, n_scales=3).to(self.device)
        dip.train()

        # Fixed input noise
        z = self._init_noise(self.image_size)

        # Optimiser
        optimizer = torch.optim.Adam(dip.parameters(), lr=self.lr)

        # EMA buffer
        ema_image = None

        for it in range(self.n_iters):
            optimizer.zero_grad()

            # Add regularisation noise to input
            z_noisy = z + torch.randn_like(z) * self.reg_noise_std

            # Generate image
            img = dip(z_noisy)  # (1, 3, H, W) in [0, 1]

            # Extract VGG features from generated image
            img_vgg = preprocess_for_vgg(img, size=self.image_size)
            feat_maps = self.vgg(img_vgg) if isinstance(self.vgg, VGGFeatureExtractor) else self.vgg.vgg(img_vgg)

            # L2 loss against target feature maps (equal weight per layer)
            loss = sum(
                F.mse_loss(gen_feat, tgt_feat)
                for gen_feat, tgt_feat in zip(feat_maps, target_maps)
            ) / len(feat_maps)

            loss.backward()
            optimizer.step()

            # EMA on output image
            with torch.no_grad():
                if ema_image is None:
                    ema_image = img.detach().clone()
                else:
                    ema_image = self.ema_decay * ema_image + (1 - self.ema_decay) * img.detach()

            if verbose and (it % 500 == 0 or it == self.n_iters - 1):
                print(f"  DIP iter {it:4d}/{self.n_iters}  loss={loss.item():.4f}")

        # Upsample EMA output to output_size
        if self.output_size != self.image_size:
            ema_image = F.interpolate(
                ema_image, size=(self.output_size, self.output_size),
                mode="bilinear", align_corners=False,
            )

        return ema_image.clamp(0, 1)


# ---------------------------------------------------------------------------
# Utility: save/load DIP U-Net weights (not usually needed — DIP is random-init per image)
# ---------------------------------------------------------------------------

def make_dip_model(device: torch.device | str = "cpu") -> UNetDIP:
    """Create a fresh DIP U-Net with the Brain-IT configuration."""
    return UNetDIP(in_ch=32, hidden=128, n_scales=3).to(device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test DIP forward pass
    dip = UNetDIP(in_ch=32, hidden=128, n_scales=3).to(device)
    z = torch.randn(1, 32, 112, 112, device=device)
    out = dip(z)
    print(f"DIP output shape: {tuple(out.shape)}  (expected (1, 3, 112, 112))")
    print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")

    n_params = sum(p.numel() for p in dip.parameters())
    print(f"DIP parameters: {n_params:,}")
    print("DIP model OK")
