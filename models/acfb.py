"""
Improved Attentional ConvTrans Fusion Block (ACFB)
==================================================
Replaces the paper's CBAM (2018) + SENet (2017) with modern attention:

  (a) CNN Branch → Coordinate Attention (CVPR 2021)
      Two-axis spatial attention (H-attention × W-attention)
      Better positional encoding than single spatial map

  (b) Transformer Branch → ECA-Net (CVPR 2020)
      Efficient 1D-conv channel attention with adaptive kernel size
      No dimension reduction → no information bottleneck

  (c) Fusion → Gated Fusion
      Learnable sigmoid gate decides per-channel mixing ratio
      Replaces static concatenation
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (Hou et al., CVPR 2021).

    Decomposes spatial attention into two 1D operations along H and W axes.
    This preserves precise positional information better than standard
    spatial attention (which collapses to a single 2D map).

    Process:
        1. Pool along W → (B, C, H, 1) — captures vertical position
        2. Pool along H → (B, C, 1, W) — captures horizontal position
        3. Concatenate, reduce, split, sigmoid → attention maps
        4. Multiply: F_out = F_in × attn_H × attn_W
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(channels // reduction, 32)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (B, C, 1, W)

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Directional pooling
        x_h = self.pool_h(x)                          # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)     # (B, C, W, 1)

        # Concatenate along spatial dim, reduce channels
        y = torch.cat([x_h, x_w], dim=2)              # (B, C, H+W, 1)
        y = self.conv_reduce(y)                        # (B, C', H+W, 1)

        # Split back
        x_h, x_w = torch.split(y, [H, W], dim=2)     # (B, C', H, 1), (B, C', W, 1)

        # Generate attention weights
        attn_h = self.conv_h(x_h).sigmoid()            # (B, C, H, 1)
        attn_w = self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()  # (B, C, 1, W)

        return x * attn_h * attn_w


class ECAAttention(nn.Module):
    """
    Efficient Channel Attention (Wang et al., CVPR 2020).

    Replaces SENet's FC→ReLU→FC with a single 1D convolution
    whose kernel size adapts to the channel count.
    This avoids the information bottleneck of channel reduction.

    Kernel size: k = |log2(C)/γ + b/γ| (must be odd)
    """

    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        # Adaptive kernel size
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        k = t if t % 2 else t + 1  # Ensure odd
        k = max(k, 3)  # Minimum kernel size of 3

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial feature map

        Returns:
            Channel-attended feature map (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Global average pooling → (B, C, 1)
        y = x.flatten(2).mean(dim=2, keepdim=True)  # (B, C, 1)

        # 1D conv for channel interaction: (B, 1, C)
        y = y.transpose(1, 2)  # (B, 1, C)
        y = self.conv(y)       # (B, 1, C)
        y = y.transpose(1, 2)  # (B, C, 1)

        # Channel attention weights
        attn = y.sigmoid().unsqueeze(-1)  # (B, C, 1, 1)

        return x * attn


class GatedFusion(nn.Module):
    """
    Learnable gated fusion of two feature maps.

    gate = σ(W @ [F_a; F_b])
    F_out = gate * F_a + (1 - gate) * F_b

    The model learns per-channel which branch to trust more.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, f_a: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_conv(torch.cat([f_a, f_b], dim=1)))
        return gate * f_a + (1.0 - gate) * f_b


class ACFB(nn.Module):
    """
    Improved Attentional ConvTrans Fusion Block.

    Takes final CNN and Transformer features, applies:
      1. Coordinate Attention on CNN branch (spatial)
      2. ECA-Net on Transformer branch (channel)
      3. Gated fusion to combine both

    Args:
        cnn_channels: CNN output channels (1024 for ConvNeXt-V2-Base stage 3)
        trans_dim: Transformer embedding dim (768 for DeiT-III-Base)
        trans_grid_size: Spatial grid size of transformer tokens (32)
        output_channels: Fused output channels
        reduction: Reduction ratio for Coordinate Attention
    """

    def __init__(
        self,
        cnn_channels: int = 1024,
        trans_dim: int = 768,
        trans_grid_size: int = 32,
        output_channels: int = 1024,
        reduction: int = 16,
    ):
        super().__init__()
        self.trans_grid_size = trans_grid_size
        self.output_channels = output_channels

        # Project transformer features to match CNN channel count
        # and downsample spatially to match CNN feature map size
        self.trans_proj = nn.Sequential(
            nn.Conv2d(trans_dim, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

        # CNN branch: Coordinate Attention (spatial)
        self.coord_attn = CoordinateAttention(output_channels, reduction=reduction)

        # Transformer branch: ECA-Net (channel)
        self.eca_attn = ECAAttention(output_channels)

        # Gated fusion
        self.fusion = GatedFusion(output_channels)

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.final_act = nn.GELU()

    def forward(
        self,
        f_cnn: torch.Tensor,    # (B, C_cnn, H, W)  e.g., (B, 1024, 16, 16)
        f_trans: torch.Tensor,  # (B, N+1, D)         e.g., (B, 1025, 768)
    ) -> torch.Tensor:
        """
        Fuse CNN and Transformer features.

        Returns:
            Fused feature map (B, output_channels, H_cnn, W_cnn)
        """
        B, C_cnn, H_cnn, W_cnn = f_cnn.shape

        # ============================================================
        # Prepare Transformer features as spatial map
        # ============================================================
        # Remove CLS token, reshape to spatial grid
        patch_tokens = f_trans[:, 1:, :]  # (B, N, D)
        f_trans_spatial = patch_tokens.transpose(1, 2).reshape(
            B, -1, self.trans_grid_size, self.trans_grid_size
        )  # (B, D, 32, 32)

        # Project channels and resize to match CNN spatial dims
        f_trans_proj = self.trans_proj(f_trans_spatial)  # (B, output_channels, 32, 32)
        if f_trans_proj.shape[2:] != (H_cnn, W_cnn):
            f_trans_proj = F.interpolate(
                f_trans_proj, size=(H_cnn, W_cnn),
                mode="bilinear", align_corners=False,
            )
        # (B, output_channels, H_cnn, W_cnn)

        # If CNN channels differ from output, project
        if C_cnn != self.output_channels:
            # This shouldn't happen with our architecture (both 1024)
            # but handle gracefully
            f_cnn = F.conv2d(
                f_cnn,
                torch.randn(self.output_channels, C_cnn, 1, 1, device=f_cnn.device),
            )

        # ============================================================
        # Apply attention
        # ============================================================
        f_cnn_attended = self.coord_attn(f_cnn)          # Spatial attention
        f_trans_attended = self.eca_attn(f_trans_proj)    # Channel attention

        # ============================================================
        # Gated fusion
        # ============================================================
        f_fused = self.fusion(f_cnn_attended, f_trans_attended)

        # Residual refinement
        f_out = f_fused + self.refine(f_fused)
        f_out = self.final_act(f_out)

        return f_out


if __name__ == "__main__":
    acfb = ACFB(cnn_channels=1024, trans_dim=768, trans_grid_size=32)
    f_cnn = torch.randn(1, 1024, 16, 16)
    f_trans = torch.randn(1, 1025, 768)
    out = acfb(f_cnn, f_trans)
    print(f"ACFB output: {out.shape}")
    # Expected: torch.Size([1, 1024, 16, 16])
