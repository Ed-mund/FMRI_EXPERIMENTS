"""
Universal Brain Encoder - Model Architecture
Based on: "The Wisdom of a Crowd of Brains: A Universal Brain Encoder"
(Beliy et al., 2025)

Architecture:
  (a) Feature Extraction Block - DINO-v2 ViT-L/14 with LoRA, multi-scale features
  (b) Per-Voxel Embeddings - 256-dim learned vectors per brain voxel
  (c) Cross-Attention Block - Spatial attention + MLPs + Functional attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# LoRA Adapter for DINO-v2
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """Low-rank adaptation applied to the output projection of self-attention."""
    def __init__(self, in_features: int, out_features: int, rank: int = 16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.lora_A @ self.lora_B


class LoRADINO(nn.Module):
    """
    DINO-v2 ViT-L/14 with LoRA on the output projection weights (Wo)
    of self-attention blocks. Extracts features from L=5 intermediate layers.

    Paper: layers 1, 6, 12, 18, 24 of ViT-L/14
    """
    EXTRACT_LAYERS = [1, 6, 12, 18, 24]  # 0-indexed: 0, 5, 11, 17, 23

    def __init__(
        self,
        projection_dim: int = 128,
        lora_rank: int = 16,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        # Load DINO-v2 ViT-L/14
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        self.hidden_dim = self.dino.embed_dim  # 1024 for ViT-L

        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False

        # LoRA adapters on output projection of each extracted layer
        # Paper says they only modify Wo (output projection)
        self.lora_layers = nn.ModuleDict()
        extract_indices = [l - 1 for l in self.EXTRACT_LAYERS]  # Convert to 0-indexed
        for idx in extract_indices:
            self.lora_layers[str(idx)] = LoRALayer(
                self.hidden_dim, self.hidden_dim, rank=lora_rank
            )

        # Per-layer projection to lower dimension C
        self.num_layers = len(self.EXTRACT_LAYERS)
        self.projection_dim = projection_dim
        self.projections = nn.ModuleList([
            nn.Linear(self.hidden_dim, projection_dim)
            for _ in range(self.num_layers)
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224) normalized images

        Returns:
            features: (B, L, P, C) where L=5 layers, P=num_patches, C=projection_dim
        """
        B = images.shape[0]

        # Get intermediate features from DINO
        # We need to hook into intermediate layers
        features_per_layer = []
        extract_indices = set(l - 1 for l in self.EXTRACT_LAYERS)

        # Patch embed
        x = self.dino.prepare_tokens_with_masks(images, masks=None)

        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)
            if i in extract_indices:
                # Apply LoRA to the output
                lora_delta = self.lora_layers[str(i)](x)
                feat = x + lora_delta
                # Remove CLS token, keep only patch tokens
                patch_feat = feat[:, 1:, :]  # (B, P, hidden_dim)
                features_per_layer.append(patch_feat)

        # Project each layer's features to dimension C
        projected = []
        for layer_idx, feat in enumerate(features_per_layer):
            proj = self.projections[layer_idx](feat)  # (B, P, C)
            projected.append(proj)

        # Stack along layer dimension: (B, L, P, C)
        output = torch.stack(projected, dim=1)
        return output


# ---------------------------------------------------------------------------
# Voxel Embedding Store
# ---------------------------------------------------------------------------

class VoxelEmbeddingStore(nn.Module):
    """
    Manages per-voxel embeddings for all subjects.
    Each brain voxel gets a unique E=256 dimensional embedding vector,
    initialized randomly and optimized during training.

    Subjects can have different numbers of voxels.
    """
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ParameterDict()
        self._subject_sizes = {}

    def register_subject(self, subject_id: str, num_voxels: int):
        """Register a new subject with their number of voxels."""
        # Initialize randomly (normal distribution)
        emb = nn.Parameter(torch.randn(num_voxels, self.embedding_dim) * 0.02)
        self.embeddings[subject_id] = emb
        self._subject_sizes[subject_id] = num_voxels

    def get_embeddings(
        self, subject_id: str, voxel_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            subject_id: which subject
            voxel_indices: (N,) indices of voxels to retrieve

        Returns:
            embeddings: (N, E) voxel embeddings
        """
        return self.embeddings[subject_id][voxel_indices]

    def get_all_embeddings(self, subject_id: str) -> torch.Tensor:
        """Return all voxel embeddings for a subject: (num_voxels, E)"""
        return self.embeddings[subject_id]

    @property
    def subject_ids(self) -> List[str]:
        return list(self.embeddings.keys())


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """
    Spatial attention: lets each voxel embedding attend to specific
    spatial locations within the image features.

    For each layer l in L:
        output_l = softmax(q @ K_l^T) @ V_l

    q = linear(voxel_embedding)  -> (1, E)
    K_l = features_l + positional_embedding  -> (P, E)
    V_l = features_l  -> (P, C)
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        num_patches: int = 256,  # 16x16 for 224/14
        num_layers: int = 5,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Query projection for voxel embedding
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)

        # Learned positional embedding for patches
        self.positional_embedding = nn.Parameter(
            torch.randn(num_patches, embedding_dim) * 0.02
        )

        # Key projection: project features to embedding size E
        self.key_proj = nn.Linear(feature_dim, embedding_dim)

        self.scale = math.sqrt(embedding_dim)

    def forward(
        self,
        image_features: torch.Tensor,
        voxel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: (B, L, P, C) multi-scale image features
            voxel_embeddings: (N, E) voxel embeddings (N voxels in this batch)

        Returns:
            spatially_attended: (N, B, L, C) features after spatial attention
        """
        B, L, P, C = image_features.shape
        N, E = voxel_embeddings.shape

        # Query from voxel embeddings: (N, E)
        q = self.query_proj(voxel_embeddings)  # (N, E)

        # Process each layer separately
        outputs = []
        for l in range(L):
            feat_l = image_features[:, l, :, :]  # (B, P, C)

            # Keys: project features + add positional embedding
            k = self.key_proj(feat_l) + self.positional_embedding.unsqueeze(0)  # (B, P, E)

            # Values: raw features
            v = feat_l  # (B, P, C)

            # Attention: q(N,E) x k(B,P,E)^T -> (N, B, P)
            # For each voxel, for each image in batch, attention over patches
            attn = torch.einsum("ne,bpe->nbp", q, k) / self.scale  # (N, B, P)
            attn = F.softmax(attn, dim=-1)  # (N, B, P)

            # Weighted sum: (N, B, P) x (B, P, C) -> (N, B, C)
            out = torch.einsum("nbp,bpc->nbc", attn, v)  # (N, B, C)
            outputs.append(out)

        # Stack layers: (N, B, L, C)
        spatially_attended = torch.stack(outputs, dim=2)
        return spatially_attended


class FunctionalAttention(nn.Module):
    """
    Functional attention: weighted summation of spatially-attended features
    to produce a single scalar voxel activation.

    v = flattened MLP output (1, L*C)
    q = voxel embedding (1, E)
    K = learned functional embedding (L*C, E)

    output = (q @ K^T) @ v^T  -> scalar
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        num_layers: int = 5,
        feature_dim: int = 128,
    ):
        super().__init__()
        lc = num_layers * feature_dim
        # Learned functional embedding
        self.functional_embedding = nn.Parameter(
            torch.randn(lc, embedding_dim) * 0.02
        )
        self.scale = math.sqrt(embedding_dim)

    def forward(
        self,
        mlp_output: torch.Tensor,
        voxel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mlp_output: (N, B, L*C) flattened MLP output
            voxel_embeddings: (N, E)

        Returns:
            activations: (N, B) predicted voxel activations
        """
        N, B, LC = mlp_output.shape

        # q @ K^T: (N, E) x (LC, E)^T -> (N, LC)
        attn = torch.einsum("ne,le->nl", voxel_embeddings, self.functional_embedding) / self.scale
        attn = F.softmax(attn, dim=-1)  # (N, LC)

        # (N, LC) x (N, B, LC) -> (N, B) via einsum
        activations = torch.einsum("nl,nbl->nb", attn, mlp_output)

        return activations


class CrossAttentionBlock(nn.Module):
    """
    Full cross-attention block combining:
    1. Spatial attention
    2. Per-layer MLPs
    3. Functional attention
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        num_patches: int = 256,
        num_layers: int = 5,
        feature_dim: int = 128,
        mlp_hidden_mult: int = 2,
    ):
        super().__init__()

        self.spatial_attention = SpatialAttention(
            embedding_dim=embedding_dim,
            num_patches=num_patches,
            num_layers=num_layers,
            feature_dim=feature_dim,
        )

        # Per-layer 2-layer MLPs
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * mlp_hidden_mult),
                nn.GELU(),
                nn.Linear(feature_dim * mlp_hidden_mult, feature_dim),
            )
            for _ in range(num_layers)
        ])

        self.functional_attention = FunctionalAttention(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            feature_dim=feature_dim,
        )

        self.num_layers = num_layers
        self.feature_dim = feature_dim

    def forward(
        self,
        image_features: torch.Tensor,
        voxel_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: (B, L, P, C)
            voxel_embeddings: (N, E)

        Returns:
            activations: (N, B) predicted scalar activations
        """
        # 1. Spatial attention: (N, B, L, C)
        spatially_attended = self.spatial_attention(image_features, voxel_embeddings)

        # 2. Per-layer MLPs
        mlp_outputs = []
        for l in range(self.num_layers):
            feat_l = spatially_attended[:, :, l, :]  # (N, B, C)
            mlp_out = self.mlps[l](feat_l)  # (N, B, C)
            mlp_outputs.append(mlp_out)

        # Flatten: (N, B, L*C)
        mlp_output = torch.cat(mlp_outputs, dim=-1)

        # 3. Functional attention: (N, B)
        activations = self.functional_attention(mlp_output, voxel_embeddings)

        return activations


# ---------------------------------------------------------------------------
# Universal Brain Encoder (full model)
# ---------------------------------------------------------------------------

class UniversalBrainEncoder(nn.Module):
    """
    The full Universal Brain Encoder.

    Input: an image + a set of voxel indices (for a given subject)
    Output: predicted fMRI activation for each voxel on that image

    Architecture:
        - DINO-v2 ViT-L/14 with LoRA -> multi-scale features (L, P, C)
        - Per-voxel learned embeddings (E=256)
        - Cross-attention block -> scalar activation per voxel
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        projection_dim: int = 128,
        lora_rank: int = 16,
        num_layers: int = 5,
        mlp_hidden_mult: int = 2,
        image_size: int = 224,
        patch_size: int = 14,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = (image_size // patch_size) ** 2  # 256 for 224/14

        # (a) Feature Extraction Block
        self.feature_extractor = LoRADINO(
            projection_dim=projection_dim,
            lora_rank=lora_rank,
            freeze_backbone=True,
        )

        # (b) Voxel Embedding Store
        self.voxel_store = VoxelEmbeddingStore(embedding_dim=embedding_dim)

        # (c) Cross-Attention Block
        self.cross_attention = CrossAttentionBlock(
            embedding_dim=embedding_dim,
            num_patches=self.num_patches,
            num_layers=num_layers,
            feature_dim=projection_dim,
            mlp_hidden_mult=mlp_hidden_mult,
        )

    def register_subject(self, subject_id: str, num_voxels: int):
        """Register a subject with their number of brain voxels."""
        self.voxel_store.register_subject(subject_id, num_voxels)

    def forward(
        self,
        images: torch.Tensor,
        subject_id: str,
        voxel_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224) batch of images
            subject_id: which subject's voxels to predict
            voxel_indices: (N,) which voxels to predict for

        Returns:
            predictions: (N, B) predicted fMRI activation per voxel per image
        """
        # Extract multi-scale image features
        features = self.feature_extractor(images)  # (B, L, P, C)

        # Get voxel embeddings
        voxel_embs = self.voxel_store.get_embeddings(
            subject_id, voxel_indices
        )  # (N, E)

        # Cross-attention to predict activations
        predictions = self.cross_attention(features, voxel_embs)  # (N, B)

        return predictions

    def predict_all_voxels(
        self,
        images: torch.Tensor,
        subject_id: str,
        chunk_size: int = 5000,
    ) -> torch.Tensor:
        """
        Predict all voxel activations for a subject. Processes in chunks
        to manage memory.

        Returns:
            predictions: (num_voxels, B)
        """
        features = self.feature_extractor(images)
        all_embs = self.voxel_store.get_all_embeddings(subject_id)
        num_voxels = all_embs.shape[0]

        all_preds = []
        for start in range(0, num_voxels, chunk_size):
            end = min(start + chunk_size, num_voxels)
            chunk_embs = all_embs[start:end]
            preds = self.cross_attention(features, chunk_embs)
            all_preds.append(preds)

        return torch.cat(all_preds, dim=0)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class BrainEncoderLoss(nn.Module):
    """
    Combined MSE + Cosine loss from the paper:
    L(r_hat, r) = alpha * MSE(r_hat, r) - (1 - alpha) * cos(r_hat, r)

    where alpha = 0.1
    """
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (N, B) predicted activations
            target: (N, B) ground truth activations

        Returns:
            loss: scalar
        """
        # MSE component
        mse_loss = F.mse_loss(predicted, target)

        # Cosine similarity component (per-voxel across images)
        # Normalize along the batch dimension
        cos_sim = F.cosine_similarity(predicted, target, dim=1).mean()

        loss = self.alpha * mse_loss - (1 - self.alpha) * cos_sim
        return loss


# ---------------------------------------------------------------------------
# Utility: count parameters
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict:
    """Count trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
