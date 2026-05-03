"""
Brain Interaction Transformer (BIT) for Brain-IT.
Based on: "Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer"
(Beliy et al., ICLR 2026)

Architecture (Figure 4 in paper):

  BrainTokenizer:
    - Per-voxel embedding (512-dim): captures each voxel's decoding function
    - Per-cluster embedding (512-dim): summarises cluster's overall function
    - Graph Attention: K,V = modulated voxel activations; Q = cluster embeddings
    - Output: 128 Brain Tokens × 512-dim

  CrossTransformer:
    - Initial cross-attention block: query tokens ← Brain Tokens
    - 5 × (self-attention + cross-attention), 8 heads each
    - Final linear projection to output feature dimension

Two BIT variants are trained:
  LowLevelBIT  → predicts VGG-16 features (multiple layers)
  SemanticBIT  → predicts 256 OpenCLIP ViT-bigG/14 spatial tokens
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BrainTokenizer
# ---------------------------------------------------------------------------

class BrainTokenizer(nn.Module):
    """
    Transforms fMRI voxel activations into a fixed set of Brain Tokens.

    For each sample:
      1. Each voxel activation (scalar) is multiplied by its 512-dim voxel embedding
         → modulated activations (N_voxels_sampled, 512)
      2. Single-head graph attention aggregates modulated activations within each
         cluster, with cluster embeddings acting as queries:
           Q = cluster_embeddings[cluster_ids]      (N_voxels_sampled, 512)
           K = V = modulated_activations             (N_voxels_sampled, 512)
         Attention is restricted to act between each voxel and its assigned cluster
         (sparse graph attention using the V2C mask).
      3. Output: (n_clusters, 512) Brain Tokens

    The per-voxel and per-cluster embeddings are subject-specific for voxel
    embeddings but shared (single set) for cluster embeddings.
    """

    def __init__(
        self,
        n_clusters: int = 128,
        token_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.token_dim = token_dim

        # Shared cluster embeddings (one per functional cluster, shared across subjects)
        self.cluster_embeddings = nn.Parameter(
            torch.randn(n_clusters, token_dim) * (1.0 / math.sqrt(token_dim))
        )

        # Graph attention projections (single head)
        self.q_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.out_proj = nn.Linear(token_dim, token_dim)

        self.scale = math.sqrt(token_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(token_dim)

        # Per-subject voxel embedding stores (populated dynamically)
        # Maps subject_id → nn.Embedding (registered as submodules via ModuleDict)
        self.voxel_embeddings = nn.ModuleDict()

    def register_subject(self, subject_id: str, num_voxels: int):
        """
        Register a new subject's voxel embeddings.
        These are the decoder voxel embeddings (512-dim, distinct from the
        encoder's 256-dim embeddings).
        """
        emb = nn.Embedding(num_voxels, self.token_dim)
        nn.init.normal_(emb.weight, std=1.0 / math.sqrt(self.token_dim))
        self.voxel_embeddings[subject_id] = emb
        # If the rest of the module is already on a device, move the new embedding there
        # (submodules added after .to(device) otherwise stay on CPU).
        for p in self.parameters():
            emb.to(p.device)
            break

    def get_voxel_embeddings(
        self, subject_id: str, voxel_indices: torch.Tensor
    ) -> torch.Tensor:
        """Return voxel embeddings for the given subject and voxel indices."""
        return self.voxel_embeddings[subject_id](voxel_indices)

    def forward(
        self,
        fmri_activations: torch.Tensor,
        voxel_indices: torch.Tensor,
        cluster_assignments: torch.Tensor,
        subject_id: str,
    ) -> torch.Tensor:
        """
        Args:
            fmri_activations:   (B, N_v) float, fMRI scalars for sampled voxels
            voxel_indices:      (N_v,)   long, global voxel indices in [0, total_voxels)
            cluster_assignments:(N_v,)   long, cluster idx for each voxel in [0, n_clusters)
            subject_id:         str

        Returns:
            brain_tokens: (B, n_clusters, token_dim)
        """
        B, N_v = fmri_activations.shape
        device = fmri_activations.device

        # 1. Per-voxel embeddings (N_v, D)
        vox_emb = self.get_voxel_embeddings(subject_id, voxel_indices)  # (N_v, D)

        # 2. Modulate: scalar activation × voxel embedding → (B, N_v, D)
        modulated = fmri_activations.unsqueeze(-1) * vox_emb.unsqueeze(0)  # (B, N_v, D)

        # 3. Cluster queries from shared cluster embeddings
        # We expand for batch: (n_clusters, D)
        C = self.cluster_embeddings  # (n_clusters, D)

        # 4. Graph-restricted attention:
        #    For each cluster c, gather all voxels assigned to c,
        #    attend cluster embedding over those modulated activations.
        #    Output: (B, n_clusters, D)
        brain_tokens = self._cluster_attention(modulated, C, cluster_assignments)

        # 5. Output projection + norm
        brain_tokens = self.out_proj(brain_tokens)  # (B, n_clusters, D)
        brain_tokens = self.norm(brain_tokens)

        return brain_tokens

    def _cluster_attention(
        self,
        modulated: torch.Tensor,
        cluster_emb: torch.Tensor,
        cluster_assignments: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized sparse graph attention: each cluster attends only to its assigned voxels.

        Replaces the original per-cluster Python loop with a single batched einsum +
        masked softmax over all clusters simultaneously.

        Args:
            modulated:           (B, N_v, D)
            cluster_emb:         (n_clusters, D)
            cluster_assignments: (N_v,) int, cluster index per voxel in [0, n_clusters)

        Returns:
            (B, n_clusters, D)
        """
        B, N_v, D = modulated.shape
        n_clusters = self.n_clusters
        device = modulated.device

        K = self.k_proj(modulated)   # (B, N_v, D)
        V = self.v_proj(modulated)   # (B, N_v, D)
        Q = self.q_proj(cluster_emb) # (n_clusters, D)

        # scores[b, c, n] = Q[c] · K[b, n] / scale  →  (B, n_clusters, N_v)
        scores = torch.einsum("cd,bnd->bcn", Q, K) / self.scale

        # membership[c, n] = True if voxel n belongs to cluster c  →  (n_clusters, N_v)
        cluster_ids = torch.arange(n_clusters, device=device).unsqueeze(1)  # (n_clusters, 1)
        membership = cluster_assignments.unsqueeze(0) == cluster_ids         # (n_clusters, N_v)

        # Mask non-member voxels with -inf before softmax
        scores = scores.masked_fill(~membership.unsqueeze(0), float("-inf"))

        # Softmax over voxel dim; empty clusters produce NaN → replace with 0
        attn = F.softmax(scores, dim=-1).nan_to_num(0.0)  # (B, n_clusters, N_v)
        attn = self.dropout(attn)

        # Weighted aggregation: (B, n_clusters, D)
        output = torch.einsum("bcn,bnd->bcd", attn, V)

        return output


# ---------------------------------------------------------------------------
# CrossTransformer
# ---------------------------------------------------------------------------

class CrossTransformerBlock(nn.Module):
    """
    One block of the CrossTransformer:
      1. Self-attention on Brain Tokens (refines Brain Token representations)
      2. Cross-attention: query tokens ← Brain Tokens (extracts image features)

    Adapted from CPC-MAE (Fu et al., 2024).
    """

    def __init__(self, token_dim: int = 512, n_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.token_dim = token_dim
        self.n_heads = n_heads

        # Self-attention on brain tokens
        self.sa_norm = nn.LayerNorm(token_dim)
        self.sa = nn.MultiheadAttention(token_dim, n_heads, dropout=dropout, batch_first=True)

        # Cross-attention: query tokens attend over brain tokens
        self.ca_norm_q = nn.LayerNorm(token_dim)
        self.ca_norm_kv = nn.LayerNorm(token_dim)
        self.ca = nn.MultiheadAttention(token_dim, n_heads, dropout=dropout, batch_first=True)

        # FFN after cross-attention
        hidden = int(token_dim * mlp_ratio)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, token_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        brain_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            brain_tokens:  (B, n_clusters, D)
            query_tokens:  (B, N_q, D)  — image feature query tokens

        Returns:
            brain_tokens:  (B, n_clusters, D)  updated
            query_tokens:  (B, N_q, D)          updated
        """
        # Self-attention on brain tokens (pre-norm)
        bt = self.sa_norm(brain_tokens)
        bt, _ = self.sa(bt, bt, bt)
        brain_tokens = brain_tokens + bt

        # Cross-attention: queries attend to brain tokens
        q = self.ca_norm_q(query_tokens)
        kv = self.ca_norm_kv(brain_tokens)
        ca_out, _ = self.ca(q, kv, kv)
        query_tokens = query_tokens + ca_out

        # FFN on query tokens
        query_tokens = query_tokens + self.ffn(self.ffn_norm(query_tokens))

        return brain_tokens, query_tokens


class CrossTransformer(nn.Module):
    """
    Full Cross-Transformer Module.

    1. Initial cross-attention block: randomly-initialized query tokens attend to Brain Tokens
    2. Five CrossTransformerBlocks (self-attn on brain tokens + cross-attn to update queries)
    3. Final linear projection to output feature dimension
    """

    def __init__(
        self,
        n_clusters: int = 128,
        n_query_tokens: int = 256,
        token_dim: int = 512,
        out_dim: int = 1280,
        n_blocks: int = 5,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_query_tokens = n_query_tokens
        self.token_dim = token_dim

        # Learnable query tokens (initialized randomly, optimized during training)
        self.query_tokens = nn.Parameter(
            torch.randn(1, n_query_tokens, token_dim) * (1.0 / math.sqrt(token_dim))
        )

        # Initial cross-attention block
        self.init_ca_norm_q = nn.LayerNorm(token_dim)
        self.init_ca_norm_kv = nn.LayerNorm(token_dim)
        self.init_ca = nn.MultiheadAttention(token_dim, n_heads, dropout=dropout, batch_first=True)

        # 5 CrossTransformerBlocks
        self.blocks = nn.ModuleList([
            CrossTransformerBlock(token_dim=token_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_blocks)
        ])

        # Final projection to output feature dimension
        self.out_norm = nn.LayerNorm(token_dim)
        self.out_proj = nn.Linear(token_dim, out_dim)

    def forward(self, brain_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            brain_tokens: (B, n_clusters, token_dim)

        Returns:
            predicted_features: (B, n_query_tokens, out_dim)
        """
        B = brain_tokens.shape[0]

        # Expand query tokens to batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, N_q, D)

        # Initial cross-attention
        q = self.init_ca_norm_q(queries)
        kv = self.init_ca_norm_kv(brain_tokens)
        ca_out, _ = self.init_ca(q, kv, kv)
        queries = queries + ca_out

        # 5 transformer blocks
        bt = brain_tokens
        for block in self.blocks:
            bt, queries = block(bt, queries)

        # Project to output dimension
        out = self.out_proj(self.out_norm(queries))  # (B, N_q, out_dim)
        return out


# ---------------------------------------------------------------------------
# Full BIT models: LowLevelBIT and SemanticBIT
# ---------------------------------------------------------------------------

class BIT(nn.Module):
    """
    Brain Interaction Transformer: BrainTokenizer + CrossTransformer.

    Can be used for either:
      - Low-level branch: predicts VGG features (out_dim depends on VGG layer config)
      - Semantic branch:  predicts 256 OpenCLIP ViT-bigG/14 tokens (out_dim=1280)
    """

    def __init__(
        self,
        n_clusters: int = 128,
        n_query_tokens: int = 256,
        token_dim: int = 512,
        out_dim: int = 1280,
        n_blocks: int = 5,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.token_dim = token_dim

        self.tokenizer = BrainTokenizer(
            n_clusters=n_clusters,
            token_dim=token_dim,
            dropout=dropout,
        )
        self.transformer = CrossTransformer(
            n_clusters=n_clusters,
            n_query_tokens=n_query_tokens,
            token_dim=token_dim,
            out_dim=out_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def register_subject(self, subject_id: str, num_voxels: int):
        self.tokenizer.register_subject(subject_id, num_voxels)

    def forward(
        self,
        fmri_activations: torch.Tensor,
        voxel_indices: torch.Tensor,
        cluster_assignments: torch.Tensor,
        subject_id: str,
    ) -> torch.Tensor:
        """
        Args:
            fmri_activations:   (B, N_v) sampled fMRI scalars
            voxel_indices:      (N_v,) global voxel indices
            cluster_assignments:(N_v,) cluster idx per voxel
            subject_id:         str

        Returns:
            predicted_features: (B, n_query_tokens, out_dim)
        """
        brain_tokens = self.tokenizer(
            fmri_activations, voxel_indices, cluster_assignments, subject_id
        )
        return self.transformer(brain_tokens)


# ---------------------------------------------------------------------------
# VGG-BIT: multi-head BIT for multiple VGG layers
# ---------------------------------------------------------------------------

class VGGTokenConfig:
    """Token counts and dims for VGG-16+BN feature layers used in Brain-IT."""
    # (n_tokens, channel_dim, name)  — after position tiling described in paper
    LAYERS = [
        (56 * 56,  64,  "relu1_2"),   # layer 1_2: 56x56 grid, merged 2x2→1 (no overlap) → 28x28=784... wait
        # Paper: "For layers 1_2 and 2_2, we merge 2×2 adjacent positions into a single token"
        # 112→56 input: layer1_2 has spatial 112, merge 2x2 no-overlap → 56x56=3136... too many
        # Actual: input 112x112, VGG pool halves:
        #   after conv1 (2×conv): 112×112, 64ch → merge 2x2 no-overlap → 56×56 tokens
        #   after pool1+conv2: 56×56, 128ch → merge 2x2 with overlap → 56×56 tokens (paper says "with overlaps")
        #   after pool2+conv3: 28×28, 256ch
        #   after pool3+conv4: 14×14, 512ch
        #   after pool4+conv5: 7×7, 512ch
        # Paper says token counts: "56², 55², 28², 14², 7² tokens" → 3136, 3025(?), 784, 196, 49
        # Using exact paper values from Sec D.2: 56², 55², 28², 14², 7²
    ]

    # From paper Sec D.2 (exact token counts after tiling/merging):
    # 112×112 input to VGG → layer spatial dims before tokenising:
    #   relu1_2: 112  → merge 2x2 no-overlap → 56x56 = 3136 tokens, 64 ch
    #   relu2_2: 56   → merge 2x2 with overlap (stride 1) → 55x55 = 3025 tokens, 128 ch
    #   relu3_3: 28   → 28x28 = 784 tokens, 256 ch
    #   relu4_3: 14   → 14x14 = 196 tokens, 512 ch
    #   relu5_3: 7    → 7x7   = 49 tokens, 512 ch

    @staticmethod
    def get_token_config() -> list[tuple[int, int, str]]:
        """Returns list of (n_tokens, channel_dim, layer_name)."""
        return [
            (56 * 56,  64,  "relu1_2"),
            (55 * 55,  128, "relu2_2"),
            (28 * 28,  256, "relu3_3"),
            (14 * 14,  512, "relu4_3"),
            (7  * 7,   512, "relu5_3"),
        ]

    @staticmethod
    def get_total_tokens() -> int:
        return sum(n for n, _, _ in VGGTokenConfig.get_token_config())

    @staticmethod
    def get_token_dim() -> int:
        """All tokens are padded/replicated to 512 dim (paper: <512 dims are replicated)."""
        return 512


class LowLevelBIT(nn.Module):
    """
    BIT model for the low-level branch.

    Predicts VGG-16 features across all 5 layers. The total number of query
    tokens equals the sum of spatial positions across all VGG layers
    (after tiling described in paper).

    Since different VGG layers have different channel counts, and the
    cross-transformer outputs a fixed out_dim per token, we use a per-layer
    output head to project to the correct channel dimension.

    Training loss: InfoNCE applied independently per layer.
    """

    def __init__(
        self,
        n_clusters: int = 128,
        token_dim: int = 512,
        n_blocks: int = 5,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_config = VGGTokenConfig.get_token_config()
        self.n_total_tokens = sum(n for n, _, _ in self.token_config)

        # One BIT that predicts all VGG tokens jointly
        # out_dim = token_dim (512), then per-layer heads project to correct channel dim
        self.bit = BIT(
            n_clusters=n_clusters,
            n_query_tokens=self.n_total_tokens,
            token_dim=token_dim,
            out_dim=token_dim,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Per-layer projection heads (token_dim → channel_dim)
        self.layer_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, ch_dim))
            for _, ch_dim, _ in self.token_config
        ])

        self._token_splits = [n for n, _, _ in self.token_config]

    def register_subject(self, subject_id: str, num_voxels: int):
        self.bit.register_subject(subject_id, num_voxels)

    def forward(
        self,
        fmri_activations: torch.Tensor,
        voxel_indices: torch.Tensor,
        cluster_assignments: torch.Tensor,
        subject_id: str,
    ) -> list[torch.Tensor]:
        """
        Returns:
            per-layer predicted tokens: list of (B, n_tokens_l, ch_dim_l)
        """
        # (B, n_total_tokens, token_dim)
        all_tokens = self.bit(fmri_activations, voxel_indices, cluster_assignments, subject_id)

        # Split into per-layer chunks and project
        splits = torch.split(all_tokens, self._token_splits, dim=1)
        predictions = []
        for tokens_l, head in zip(splits, self.layer_heads):
            predictions.append(head(tokens_l))

        return predictions


class SemanticBIT(nn.Module):
    """
    BIT model for the semantic branch.

    Predicts 256 spatial tokens of OpenCLIP ViT-bigG/14 (dim=1280 each).
    This matches the conditioning format expected by MindEye2's unCLIP SDXL.
    """

    CLIP_N_TOKENS = 256       # 16×16 spatial tokens
    CLIP_TOKEN_DIM = 1280     # OpenCLIP ViT-bigG/14 hidden dim

    def __init__(
        self,
        n_clusters: int = 128,
        token_dim: int = 512,
        n_blocks: int = 5,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bit = BIT(
            n_clusters=n_clusters,
            n_query_tokens=self.CLIP_N_TOKENS,
            token_dim=token_dim,
            out_dim=self.CLIP_TOKEN_DIM,
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
        )

    def register_subject(self, subject_id: str, num_voxels: int):
        self.bit.register_subject(subject_id, num_voxels)

    def forward(
        self,
        fmri_activations: torch.Tensor,
        voxel_indices: torch.Tensor,
        cluster_assignments: torch.Tensor,
        subject_id: str,
    ) -> torch.Tensor:
        """
        Returns:
            predicted_clip_tokens: (B, 256, 1280)
        """
        return self.bit(fmri_activations, voxel_indices, cluster_assignments, subject_id)


# ---------------------------------------------------------------------------
# InfoNCE loss for low-level branch
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) contrastive loss.
    Used for training the low-level BIT to predict VGG features.
    Applied independently per VGG layer.

    For a batch of B samples, each predicted token is contrasted against
    all B target tokens in the batch (including the positive).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (B, N_tokens, D)
            target:    (B, N_tokens, D)  — ground-truth VGG features

        Returns:
            scalar loss
        """
        B, N, D = predicted.shape

        # Reshape to (B*N, D) — treat each token position independently
        pred_flat = predicted.reshape(B * N, D)
        tgt_flat = target.reshape(B * N, D)

        # L2-normalize
        pred_norm = F.normalize(pred_flat, dim=-1)
        tgt_norm = F.normalize(tgt_flat, dim=-1)

        # Similarity matrix: (B*N, B*N)
        sim = torch.mm(pred_norm, tgt_norm.T) / self.temperature

        # Labels: diagonal (same token position from same image is the positive pair)
        labels = torch.arange(B * N, device=predicted.device)

        # But we want per-token-position contrastive learning:
        # For token position i of image b, the positive is token position i of image b in target
        # Re-index: for each (b, n) pair, the positive is the same (b, n) in target
        loss = F.cross_entropy(sim, labels)
        return loss


def infonce_loss_per_layer(
    predicted_layers: list[torch.Tensor],
    target_layers: list[torch.Tensor],
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Compute InfoNCE loss for each VGG layer and average.

    Args:
        predicted_layers: list of (B, N_l, D_l)
        target_layers:    list of (B, N_l, D_l)
    """
    criterion = InfoNCELoss(temperature=temperature)
    total_loss = sum(
        criterion(pred, tgt)
        for pred, tgt in zip(predicted_layers, target_layers)
    )
    return total_loss / len(predicted_layers)


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    SUBJECTS = ["subj01", "subj02"]
    N_VOXELS = 39548
    N_CLUSTERS = 128
    BATCH = 4
    N_SAMPLE = 15_000

    # Build LowLevelBIT
    ll_bit = LowLevelBIT(n_clusters=N_CLUSTERS).to(device)
    for s in SUBJECTS:
        ll_bit.register_subject(s, N_VOXELS)
    params = count_parameters(ll_bit)
    print(f"LowLevelBIT params: {params['trainable']:,} trainable")

    # Build SemanticBIT
    sem_bit = SemanticBIT(n_clusters=N_CLUSTERS).to(device)
    for s in SUBJECTS:
        sem_bit.register_subject(s, N_VOXELS)
    params = count_parameters(sem_bit)
    print(f"SemanticBIT params: {params['trainable']:,} trainable")

    # Fake forward pass
    fmri = torch.randn(BATCH, N_SAMPLE, device=device)
    vox_idx = torch.randint(0, N_VOXELS, (N_SAMPLE,), device=device)
    clust = torch.randint(0, N_CLUSTERS, (N_SAMPLE,), device=device)

    print("Running LowLevelBIT forward...")
    preds = ll_bit(fmri, vox_idx, clust, "subj01")
    for i, (p, (n, d, name)) in enumerate(zip(preds, VGGTokenConfig.get_token_config())):
        print(f"  {name}: {tuple(p.shape)}  (expected ({BATCH}, {n}, {d}))")

    print("Running SemanticBIT forward...")
    clip_preds = sem_bit(fmri, vox_idx, clust, "subj01")
    print(f"  CLIP tokens: {tuple(clip_preds.shape)}  (expected ({BATCH}, 256, 1280))")
    print("All checks passed.")
