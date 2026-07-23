"""
Cross-Attention Feature Interactive Block (FIB)
================================================
Bidirectional cross-attention for CNN ↔ Transformer feature exchange.

Replaces the paper's simple concat/addition with learned cross-modal attention:

  CNN → Transformer: Transformer queries CNN features for local detail
  Transformer → CNN: CNN queries Transformer features for global context

Key design choices:
  - Spatial pooling to fixed 32×32 for memory-efficient cross-attention
  - Learnable gating (starts near zero) to preserve pretrained features
  - Pre-LayerNorm for training stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFIB(nn.Module):
    """
    Bidirectional Cross-Attention Feature Interactive Block.

    Exchanges information between CNN spatial features and Transformer tokens
    using multi-head cross-attention in both directions.

    Args:
        cnn_dim: Number of channels in CNN feature maps at this stage
        trans_dim: Transformer embedding dimension (768 for DeiT-III-Base)
        num_heads: Number of attention heads (auto-adjusted per stage)
        spatial_size: Fixed spatial size for cross-attention pooling
        dropout: Dropout rate in attention
    """

    def __init__(
        self,
        cnn_dim: int,
        trans_dim: int = 768,
        num_heads: int = 8,
        spatial_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.trans_dim = trans_dim
        self.spatial_size = spatial_size

        # Auto-adjust heads to ensure head_dim >= 32
        self.heads_c2t = min(num_heads, trans_dim // 32)
        self.heads_t2c = min(num_heads, cnn_dim // 32)
        # Ensure at least 1 head and divisibility
        self.heads_c2t = max(1, self.heads_c2t)
        self.heads_t2c = max(1, self.heads_t2c)
        while trans_dim % self.heads_c2t != 0:
            self.heads_c2t -= 1
        while cnn_dim % self.heads_t2c != 0:
            self.heads_t2c -= 1

        # ============================================================
        # CNN → Transformer direction
        # Transformer tokens (Q) attend to CNN features (K, V)
        # ============================================================
        self.cnn_proj_c2t = nn.Sequential(
            nn.Linear(cnn_dim, trans_dim),
            nn.GELU(),
        )
        self.norm_trans_q = nn.LayerNorm(trans_dim)
        self.cross_attn_c2t = nn.MultiheadAttention(
            embed_dim=trans_dim,
            num_heads=self.heads_c2t,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_trans = nn.Sequential(
            nn.Linear(trans_dim, trans_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trans_dim * 2, trans_dim),
            nn.Dropout(dropout),
        )
        self.norm_trans_ffn = nn.LayerNorm(trans_dim)

        # ============================================================
        # Transformer → CNN direction
        # CNN features (Q) attend to Transformer tokens (K, V)
        # ============================================================
        self.trans_proj_t2c = nn.Sequential(
            nn.Linear(trans_dim, cnn_dim),
            nn.GELU(),
        )
        self.norm_cnn_q = nn.LayerNorm(cnn_dim)
        self.cross_attn_t2c = nn.MultiheadAttention(
            embed_dim=cnn_dim,
            num_heads=self.heads_t2c,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_cnn = nn.Sequential(
            nn.Linear(cnn_dim, cnn_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_dim * 2, cnn_dim),
            nn.Dropout(dropout),
        )
        self.norm_cnn_ffn = nn.LayerNorm(cnn_dim)

        # ============================================================
        # Learnable gating (initialized near 0 to preserve pretrained)
        # ============================================================
        self.gate_trans = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ≈ 0.047
        self.gate_cnn = nn.Parameter(torch.tensor(-3.0))

    def forward(
        self,
        f_cnn: torch.Tensor,    # (B, C, H, W)
        f_trans: torch.Tensor,  # (B, N+1, D) including CLS token
    ) -> tuple:
        """
        Bidirectional cross-attention feature exchange.

        Args:
            f_cnn: CNN feature map (B, C, H, W)
            f_trans: Transformer tokens (B, N+1, D) with CLS token

        Returns:
            f_cnn_enriched: (B, C, H, W) — CNN features enriched with global context
            f_trans_enriched: (B, N+1, D) — Transformer tokens enriched with local detail
        """
        B, C, H, W = f_cnn.shape

        # Separate CLS token from patch tokens
        cls_token = f_trans[:, :1, :]    # (B, 1, D)
        patch_tokens = f_trans[:, 1:, :] # (B, N, D)

        # ============================================================
        # Pool CNN features to fixed spatial size for memory efficiency
        # ============================================================
        if H != self.spatial_size or W != self.spatial_size:
            f_cnn_pooled = F.adaptive_avg_pool2d(f_cnn, self.spatial_size)
        else:
            f_cnn_pooled = f_cnn

        # CNN spatial → sequence: (B, C, S, S) → (B, S*S, C)
        f_cnn_seq = f_cnn_pooled.flatten(2).transpose(1, 2)

        # ============================================================
        # Direction 1: CNN → Transformer
        # Transformer tokens query CNN features for local spatial info
        # ============================================================
        cnn_kv = self.cnn_proj_c2t(f_cnn_seq)  # (B, S*S, D)
        trans_q = self.norm_trans_q(patch_tokens)

        delta_trans, _ = self.cross_attn_c2t(
            query=trans_q,
            key=cnn_kv,
            value=cnn_kv,
        )
        # Post-attention FFN with residual
        delta_trans = delta_trans + self.ffn_trans(self.norm_trans_ffn(delta_trans))

        # Gated residual connection
        alpha_trans = torch.sigmoid(self.gate_trans)
        patch_tokens_enriched = patch_tokens + alpha_trans * delta_trans

        # Reattach CLS token
        f_trans_enriched = torch.cat([cls_token, patch_tokens_enriched], dim=1)

        # ============================================================
        # Direction 2: Transformer → CNN
        # CNN features query Transformer tokens for global context
        # ============================================================
        trans_kv = self.trans_proj_t2c(patch_tokens)  # (B, N, C)
        cnn_q = self.norm_cnn_q(f_cnn_seq)

        delta_cnn, _ = self.cross_attn_t2c(
            query=cnn_q,
            key=trans_kv,
            value=trans_kv,
        )
        # Post-attention FFN with residual
        delta_cnn = delta_cnn + self.ffn_cnn(self.norm_cnn_ffn(delta_cnn))

        # Reshape back to spatial: (B, S*S, C) → (B, C, S, S)
        delta_cnn = delta_cnn.transpose(1, 2).reshape(
            B, C, self.spatial_size, self.spatial_size
        )

        # Interpolate back to original CNN spatial size if needed
        if H != self.spatial_size or W != self.spatial_size:
            delta_cnn = F.interpolate(
                delta_cnn, size=(H, W), mode="bilinear", align_corners=False
            )

        # Gated residual connection
        alpha_cnn = torch.sigmoid(self.gate_cnn)
        f_cnn_enriched = f_cnn + alpha_cnn * delta_cnn

        return f_cnn_enriched, f_trans_enriched


if __name__ == "__main__":
    # Test all 4 stages
    stages = [
        {"cnn_dim": 128, "h": 128, "w": 128},
        {"cnn_dim": 256, "h": 64, "w": 64},
        {"cnn_dim": 512, "h": 32, "w": 32},
        {"cnn_dim": 1024, "h": 16, "w": 16},
    ]
    for i, s in enumerate(stages):
        fib = CrossAttentionFIB(cnn_dim=s["cnn_dim"], trans_dim=768)
        f_cnn = torch.randn(1, s["cnn_dim"], s["h"], s["w"])
        f_trans = torch.randn(1, 1025, 768)
        out_cnn, out_trans = fib(f_cnn, f_trans)
        print(f"FIB Stage {i}: CNN {out_cnn.shape}, Trans {out_trans.shape}")
