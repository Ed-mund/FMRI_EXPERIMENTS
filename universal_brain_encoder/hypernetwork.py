"""
Anatomy-to-Embedding Hypernetwork for Zero-Shot Brain Encoding.

This module defines the hypernetwork that predicts voxel embeddings from
anatomical features, enabling zero-shot encoding for subjects with no fMRI
data.

Architecture:
    Anatomical features (N_features) → Linear(512) → LN+GELU
                                      → Linear(512) → LN+GELU
                                      → Linear(256)
                                      → Predicted Voxel Embedding (256-dim)

Training targets: the learned 256-dim voxel embeddings from the Universal
Brain Encoder (VoxelEmbeddingStore) trained on N-1 subjects.

Loss: α * MSE + (1-α) * (1 - cosine_similarity), α=0.5
  - MSE: matches the embedding magnitudes/directions
  - Cosine: preserves the angular structure of the embedding space
    (the encoder's cross-attention relies heavily on directions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AnatomyToEmbeddingNet(nn.Module):
    """
    Maps per-voxel anatomical features to predicted voxel embeddings.

    Args:
        in_features: dimension of the anatomical feature vector (e.g. 41)
        hidden_dim: hidden layer dimension (default 512)
        embedding_dim: output dimension, must match the encoder's E=256
        dropout: dropout rate applied after each hidden layer
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_features) anatomical features

        Returns:
            embeddings: (N, embedding_dim) predicted voxel embeddings
        """
        return self.net(x)


class HypernetworkLoss(nn.Module):
    """
    Combined MSE + cosine loss for hypernetwork training.

    Loss = α * MSE(pred, target) + (1-α) * (1 - cosine_sim(pred, target))

    The cosine term preserves the angular structure of the embedding space
    that the encoder's cross-attention relies on, while MSE anchors the scale.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred:   (N, E) predicted embeddings
            target: (N, E) ground-truth embeddings from trained encoder

        Returns:
            loss: scalar tensor
            metrics: dict with mse_loss, cosine_loss, cosine_sim (mean)
        """
        mse = F.mse_loss(pred, target)
        cos_sim = F.cosine_similarity(pred, target, dim=-1)  # (N,)
        cosine_loss = (1.0 - cos_sim).mean()

        loss = self.alpha * mse + (1.0 - self.alpha) * cosine_loss

        metrics = {
            "mse_loss": mse.item(),
            "cosine_loss": cosine_loss.item(),
            "cosine_sim_mean": cos_sim.mean().item(),
            "cosine_sim_median": cos_sim.median().item(),
        }
        return loss, metrics


def load_voxel_embeddings_from_checkpoint(
    checkpoint_path: str,
    subjects: list[str],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Extract VoxelEmbeddingStore parameters from a trained encoder checkpoint.

    Args:
        checkpoint_path: path to best_model.pt or checkpoint_epochN.pt
        subjects: list of subject IDs to extract (e.g. ["subj01", ..., "subj07"])
        device: torch device

    Returns:
        dict mapping subject_id -> (N_voxels, 256) embedding tensor
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # The checkpoint may be wrapped in different ways
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    embeddings = {}
    for subj in subjects:
        key = f"voxel_store.embeddings.{subj}"
        if key not in state:
            raise KeyError(
                f"Subject '{subj}' not found in checkpoint. "
                f"Available keys: {[k for k in state if 'voxel_store' in k]}"
            )
        emb = state[key].to(device)
        embeddings[subj] = emb
        print(f"  [{subj}] embedding shape: {emb.shape}")

    return embeddings


def build_hypernetwork_from_checkpoint(
    checkpoint_path: str,
    anatomy_feature_dim: int,
    device: str = "cpu",
) -> "AnatomyToEmbeddingNet":
    """
    Convenience: build and load a saved hypernetwork from checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = AnatomyToEmbeddingNet(
        in_features=anatomy_feature_dim,
        hidden_dim=ckpt.get("hidden_dim", 512),
        embedding_dim=ckpt.get("embedding_dim", 256),
        dropout=0.0,  # no dropout at inference
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


if __name__ == "__main__":
    # Quick sanity check
    model = AnatomyToEmbeddingNet(in_features=41, hidden_dim=512, embedding_dim=256)
    x = torch.randn(1000, 41)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")

    loss_fn = HypernetworkLoss(alpha=0.5)
    target = torch.randn(1000, 256)
    loss, metrics = loss_fn(out, target)
    print(f"Loss: {loss.item():.4f}, metrics: {metrics}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
