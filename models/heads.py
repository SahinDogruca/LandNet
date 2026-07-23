"""
Regression Heads for Roll, Pitch, and Yaw (High-Precision FP32)
================================================================
Deep residual MLP heads with:
  - Unit Circle Normalization (sin²+cos²=1 guaranteed)
  - FP32 enforcement (critical for 0.01° precision under AMP)
  - Separate independent heads per angle

Architecture per head:
    GAP → float32 cast
        → FC(dim, 2048) → LayerNorm → GELU → Dropout(0.3)
        → [ResBlock(2048)] × 2
        → FC(2048, 1024) → LayerNorm → GELU → Dropout(0.1)
        → FC(1024, 2) → Unit Circle Normalize → [sin(θ), cos(θ)]

FP32 Precision:
    FP16 epsilon ≈ 5e-4, but 0.01° = 1.75e-4 radians.
    FP16 would round sub-degree precision away.
    All head computations run in float32 regardless of AMP context.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm and GELU."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x + residual


class AngleHead(nn.Module):
    """
    Single angle regression head with FP32 enforcement.

    Predicts sin(θ) and cos(θ) from pooled feature vector,
    with output constrained to the unit circle.

    All computations run in float32 for precision, even under AMP.

    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden layer dimension (default 2048)
        num_residual_blocks: Number of residual blocks for depth
        dropout: Dropout rates (input, residual, output)
        normalize: Whether to apply unit circle normalization
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 2048,
        num_residual_blocks: int = 2,
        dropout: tuple = (0.3, 0.2, 0.1),
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout[0]),
        )

        # Residual blocks for depth
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout=dropout[1])
            for _ in range(num_residual_blocks)
        ])

        # Reduction and output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout[2]),
            nn.Linear(hidden_dim // 2, 2),  # [sin(θ), cos(θ)]
        )

        # Initialize final layer with small weights for stable start
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.01)
        nn.init.zeros_(self.output_proj[-1].bias)

    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — always in FP32 regardless of AMP context.

        Args:
            x: (B, in_features) pooled feature vector

        Returns:
            (B, 2) → [sin(θ), cos(θ)] on the unit circle, dtype=float32
        """
        x = self.input_proj(x)
        x = self.residual_blocks(x)
        x = self.output_proj(x)

        # Unit Circle Normalization: enforce sin²+cos² = 1
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x


class AngleHeads(nn.Module):
    """
    Multi-angle regression heads for roll, pitch, and optionally yaw.

    Takes a fused feature map, applies global average pooling,
    casts to FP32, then feeds to separate per-angle heads.

    Args:
        in_channels: Number of channels in the fused feature map
        hidden_dim: Hidden dimension in each head
        num_angles: Number of angles (2=roll+pitch, 3=roll+pitch+yaw)
        num_residual_blocks: Number of residual blocks per head
        dropout: Dropout rates (input, residual, output)
        normalize: Whether to apply unit circle normalization
    """

    def __init__(
        self,
        in_channels: int = 1024,
        hidden_dim: int = 2048,
        num_angles: int = 3,
        num_residual_blocks: int = 2,
        dropout: tuple = (0.3, 0.2, 0.1),
        normalize: bool = True,
    ):
        super().__init__()
        self.num_angles = num_angles

        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

        self.roll_head = AngleHead(
            in_channels, hidden_dim, num_residual_blocks, dropout, normalize
        )
        self.pitch_head = AngleHead(
            in_channels, hidden_dim, num_residual_blocks, dropout, normalize
        )
        if num_angles >= 3:
            self.yaw_head = AngleHead(
                in_channels, hidden_dim, num_residual_blocks, dropout, normalize
            )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Fused feature map (B, C, H, W) — can be FP16 or FP32

        Returns:
            roll_sincos: (B, 2) float32 → [sin(roll), cos(roll)]
            pitch_sincos: (B, 2) float32 → [sin(pitch), cos(pitch)]
            yaw_sincos: (B, 2) float32 → [sin(yaw), cos(yaw)] or None
        """
        # Global Average Pooling → FP32 cast for precision
        x = self.pool(x).flatten(1).float()

        roll = self.roll_head(x)
        pitch = self.pitch_head(x)
        yaw = self.yaw_head(x) if self.num_angles >= 3 else None

        return roll, pitch, yaw

    @staticmethod
    def decode_angles(
        roll_sincos: torch.Tensor,
        pitch_sincos: torch.Tensor,
        yaw_sincos: torch.Tensor = None,
    ) -> tuple:
        """
        Convert sin/cos predictions to angles in degrees.

        Returns:
            roll_deg, pitch_deg[, yaw_deg]: Angles in degrees
        """
        roll_rad = torch.atan2(roll_sincos[:, 0], roll_sincos[:, 1])
        pitch_rad = torch.atan2(pitch_sincos[:, 0], pitch_sincos[:, 1])

        roll_deg = roll_rad * (180.0 / math.pi)
        pitch_deg = pitch_rad * (180.0 / math.pi)

        if yaw_sincos is not None:
            yaw_rad = torch.atan2(yaw_sincos[:, 0], yaw_sincos[:, 1])
            yaw_deg = yaw_rad * (180.0 / math.pi)
            return roll_deg, pitch_deg, yaw_deg

        return roll_deg, pitch_deg

    @staticmethod
    def encode_angles(
        roll_deg: torch.Tensor,
        pitch_deg: torch.Tensor,
        yaw_deg: torch.Tensor = None,
    ) -> tuple:
        """
        Convert degree angles to sin/cos encoding.

        Returns:
            roll_sincos, pitch_sincos[, yaw_sincos]: (B, 2) tensors
        """
        roll_rad = roll_deg * (math.pi / 180.0)
        pitch_rad = pitch_deg * (math.pi / 180.0)

        roll_sincos = torch.stack([torch.sin(roll_rad), torch.cos(roll_rad)], dim=1)
        pitch_sincos = torch.stack([torch.sin(pitch_rad), torch.cos(pitch_rad)], dim=1)

        if yaw_deg is not None:
            yaw_rad = yaw_deg * (math.pi / 180.0)
            yaw_sincos = torch.stack([torch.sin(yaw_rad), torch.cos(yaw_rad)], dim=1)
            return roll_sincos, pitch_sincos, yaw_sincos

        return roll_sincos, pitch_sincos


# Backward compatibility alias
DualAngleHeads = AngleHeads


if __name__ == "__main__":
    # Test 3-angle heads
    heads = AngleHeads(in_channels=1024, hidden_dim=2048, num_angles=3, normalize=True)
    x = torch.randn(4, 1024, 16, 16)
    roll, pitch, yaw = heads(x)
    print(f"Roll: {roll.shape} dtype={roll.dtype} norm={roll.norm(dim=-1)}")
    print(f"Pitch: {pitch.shape} dtype={pitch.dtype} norm={pitch.norm(dim=-1)}")
    print(f"Yaw: {yaw.shape} dtype={yaw.dtype} norm={yaw.norm(dim=-1)}")

    # Test decode
    roll_deg, pitch_deg, yaw_deg = heads.decode_angles(roll, pitch, yaw)
    print(f"Roll: {roll_deg}, Pitch: {pitch_deg}, Yaw: {yaw_deg}")

    # Test with FP16 input (simulating AMP)
    x_fp16 = x.half()
    roll_fp32, pitch_fp32, yaw_fp32 = heads(x_fp16)
    print(f"\nFP16 input → output dtype: {roll_fp32.dtype}")  # Should be float32!

    # Count params
    n = sum(p.numel() for p in heads.parameters()) / 1e6
    print(f"Head params: {n:.1f}M")
