"""
LandNet-V2: Dual-Encoder Pose Estimation Model (High-Precision)
=================================================================
Main model combining all components with Multi-Scale Feature Aggregation.

    Input (3×768×768)
        │
        ├──→ ConvNeXt-V2-Base ──→ [S1, S2, S3, S4]
        │                           ↕    ↕    ↕    ↕    Cross-Attention FIB × 4
        └──→ DeiT-III-Base ────→ [S1, S2, S3, S4]
                                    ↓    ↓    ↓    ↓
                              Multi-Scale Aggregation (learnable weights)
                                           │
                                     ACFB (Gated Fusion)
                                           │
                                     ┌─────┴─────┐
                                 Roll Head    Pitch Head
                                     │            │
                               (sin,cos)_r   (sin,cos)_p
                                     ↓            ↓
                              Unit Circle Norm  Unit Circle Norm

Key improvement over v1: Multi-Scale Aggregation uses features from ALL 4
stages (not just the last), providing both fine-grained spatial cues and
high-level semantic features for precise angle estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_cnn import ConvNeXtV2Backbone
from .backbone_trans import DeiTIIIBackbone
from .fib import CrossAttentionFIB
from .acfb import ACFB
from .heads import AngleHeads


class MultiScaleAggregator(nn.Module):
    """
    Learnable multi-scale feature aggregation.

    Aggregates features from all 4 backbone stages into a single
    representation using learnable importance weights and channel
    projection. This provides richer features than using only the
    last stage.

    CNN: Different spatial sizes → pool to target → project → weighted sum
    Transformer: Same token count → project → weighted sum

    Args:
        cnn_channels: Channel dims per stage [128, 256, 512, 1024]
        trans_dim: Transformer embedding dim (768)
        output_channels: Output channel dimension
        num_stages: Number of stages (4)
    """

    def __init__(
        self,
        cnn_channels: list = [128, 256, 512, 1024],
        trans_dim: int = 768,
        output_channels: int = 1024,
        num_stages: int = 4,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.output_channels = output_channels

        # CNN: per-stage channel projection to output_channels
        self.cnn_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
            )
            for c in cnn_channels
        ])

        # Transformer: per-stage projection (same dim for all stages)
        self.trans_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trans_dim, trans_dim),
                nn.LayerNorm(trans_dim),
                nn.GELU(),
            )
            for _ in range(num_stages)
        ])

        # Learnable importance weights (softmax-normalized)
        self.cnn_weights = nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        self.trans_weights = nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        # Post-aggregation refinement
        self.cnn_refine = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(
        self,
        cnn_features: list,   # [f0, f1, f2, f3] different spatial sizes
        trans_features: list,  # [f0, f1, f2, f3] all same token count
    ) -> tuple:
        """
        Aggregate multi-scale features.

        Returns:
            agg_cnn: (B, output_channels, H_last, W_last)
            agg_trans: (B, N+1, trans_dim) — keeps CLS token
        """
        # Target spatial size = last CNN stage
        target_h, target_w = cnn_features[-1].shape[2:]

        # Normalized weights
        w_cnn = F.softmax(self.cnn_weights, dim=0)
        w_trans = F.softmax(self.trans_weights, dim=0)

        # Aggregate CNN features
        agg_cnn = None
        for i, f in enumerate(cnn_features):
            projected = self.cnn_projs[i](f)  # (B, output_channels, H_i, W_i)
            # Pool to target spatial size
            if projected.shape[2:] != (target_h, target_w):
                projected = F.adaptive_avg_pool2d(projected, (target_h, target_w))
            weighted = w_cnn[i] * projected
            agg_cnn = weighted if agg_cnn is None else agg_cnn + weighted

        agg_cnn = self.cnn_refine(agg_cnn)

        # Aggregate Transformer features
        agg_trans = None
        for i, f in enumerate(trans_features):
            projected = self.trans_projs[i](f)  # (B, N+1, D)
            weighted = w_trans[i] * projected
            agg_trans = weighted if agg_trans is None else agg_trans + weighted

        return agg_cnn, agg_trans


class LandNetV2(nn.Module):
    """
    LandNet-V2: Dual-Encoder architecture for high-precision
    roll and pitch estimation from aerial landing images.

    Supports 512, 768, and 1024 input sizes.

    Args:
        cnn_model_name: timm model name for CNN backbone
        trans_model_name: timm model name for Transformer backbone
        pretrained: Whether to load pretrained weights
        img_size: Input image size (512, 768, or 1024)
        fib_heads: Number of cross-attention heads in FIB
        fib_spatial_size: Spatial size for FIB pooling (auto: img_size//patch_size)
        fib_dropout: Dropout in FIB cross-attention
        acfb_reduction: Reduction ratio in ACFB coordinate attention
        head_hidden_dim: Hidden dimension in regression heads
        head_dropout: Dropout rates in regression heads
        head_normalize: Unit circle normalization on output
        multi_scale: Use multi-scale feature aggregation
        gradient_checkpointing: Enable gradient checkpointing to save VRAM
    """

    def __init__(
        self,
        cnn_model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        trans_model_name: str = "deit3_small_patch16_384.fb_in22k_ft_in1k",
        pretrained: bool = True,
        img_size: int = 768,
        fib_heads: int = 8,
        fib_spatial_size: int = None,  # Auto: img_size // patch_size
        fib_dropout: float = 0.1,
        acfb_reduction: int = 16,
        head_hidden_dim: int = 1024,
        head_dropout: tuple = (0.3, 0.2, 0.1),
        head_normalize: bool = True,
        num_angles: int = 3,
        multi_scale: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.multi_scale = multi_scale

        patch_size = 16  # DeiT-III patch size
        trans_grid = img_size // patch_size  # 48 for 768, 64 for 1024

        if fib_spatial_size is None:
            fib_spatial_size = trans_grid  # Match transformer grid

        # ============================================================
        # Dual Backbones
        # ============================================================
        self.cnn = ConvNeXtV2Backbone(
            model_name=cnn_model_name,
            pretrained=pretrained,
        )
        self.trans = DeiTIIIBackbone(
            model_name=trans_model_name,
            pretrained=pretrained,
            img_size=img_size,
        )

        # Channel dimensions at each stage
        cnn_channels = self.cnn.channels  # [128, 256, 512, 1024]
        trans_dim = self.trans.embed_dim   # 768

        # ============================================================
        # Feature Interactive Blocks (one per stage)
        # ============================================================
        self.fibs = nn.ModuleList([
            CrossAttentionFIB(
                cnn_dim=cnn_channels[i],
                trans_dim=trans_dim,
                num_heads=fib_heads,
                spatial_size=fib_spatial_size,
                dropout=fib_dropout,
            )
            for i in range(4)
        ])

        # ============================================================
        # Multi-Scale Feature Aggregation (optional)
        # ============================================================
        if multi_scale:
            self.multi_scale_agg = MultiScaleAggregator(
                cnn_channels=cnn_channels,
                trans_dim=trans_dim,
                output_channels=cnn_channels[-1],  # 1024
            )

        # ============================================================
        # Attentional ConvTrans Fusion Block
        # ============================================================
        self.acfb = ACFB(
            cnn_channels=cnn_channels[-1],   # 1024
            trans_dim=trans_dim,              # 768
            trans_grid_size=trans_grid,       # 48 for 768
            output_channels=cnn_channels[-1], # 1024
            reduction=acfb_reduction,
        )

        # ============================================================
        # Angle Regression Heads (Deep + Unit Circle + FP32)
        # ============================================================
        self.heads = AngleHeads(
            in_channels=cnn_channels[-1],  # 1024
            hidden_dim=head_hidden_dim,
            num_angles=num_angles,
            num_residual_blocks=2,
            dropout=head_dropout,
            normalize=head_normalize,
        )

    def _interleaved_forward(self, x: torch.Tensor) -> tuple:
        """
        Interleaved forward pass through both backbones with FIB exchange.
        Collects features from ALL 4 stages for multi-scale aggregation.
        """
        # ============================================================
        # Initial embeddings
        # ============================================================
        f_cnn = self.cnn.forward_stem(x)
        f_trans = self.trans.forward_embed(x)

        # ============================================================
        # Stage-by-stage with FIB interleaving
        # Collect intermediate features for multi-scale ACFB
        # ============================================================
        cnn_features = []
        trans_features = []

        for i in range(4):
            # Run backbone stages
            if self.gradient_checkpointing and self.training:
                f_cnn = torch.utils.checkpoint.checkpoint(
                    self.cnn.forward_stage, f_cnn, i,
                    use_reentrant=False,
                )
                f_trans = torch.utils.checkpoint.checkpoint(
                    self.trans.forward_stage, f_trans, i,
                    use_reentrant=False,
                )
            else:
                f_cnn = self.cnn.forward_stage(f_cnn, i)
                f_trans = self.trans.forward_stage(f_trans, i)

            # FIB cross-attention exchange
            if self.gradient_checkpointing and self.training:
                f_cnn, f_trans = torch.utils.checkpoint.checkpoint(
                    self.fibs[i], f_cnn, f_trans,
                    use_reentrant=False,
                )
            else:
                f_cnn, f_trans = self.fibs[i](f_cnn, f_trans)

            # Collect for multi-scale aggregation
            cnn_features.append(f_cnn)
            trans_features.append(f_trans)

        return cnn_features, trans_features

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass.

        Args:
            x: Input image (B, 3, img_size, img_size)

        Returns:
            roll_sincos: (B, 2) float32 → [sin(roll), cos(roll)]
            pitch_sincos: (B, 2) float32 → [sin(pitch), cos(pitch)]
            yaw_sincos: (B, 2) float32 → [sin(yaw), cos(yaw)] or None
        """
        # Interleaved backbone + FIB
        cnn_features, trans_features = self._interleaved_forward(x)

        # Multi-scale aggregation or single-scale
        if self.multi_scale:
            f_cnn_agg, f_trans_agg = self.multi_scale_agg(
                cnn_features, trans_features
            )
        else:
            f_cnn_agg = cnn_features[-1]
            f_trans_agg = trans_features[-1]

        # ACFB fusion
        f_fused = self.acfb(f_cnn_agg, f_trans_agg)

        # Regression heads (internally casts to FP32)
        roll_sincos, pitch_sincos, yaw_sincos = self.heads(f_fused)

        return roll_sincos, pitch_sincos, yaw_sincos

    def get_parameter_groups(
        self,
        lr_backbone: float = 1e-4,
        lr_head: float = 5e-4,
        weight_decay: float = 0.05,
    ) -> list:
        """
        Create parameter groups with different learning rates.

        - Backbone parameters: lower LR (pretrained, fine-tuning)
        - FIB/ACFB/Head/Aggregator: higher LR (training from scratch)
        """
        backbone_params = []
        head_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases and layer norms
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay_params.append(param)
            elif "cnn." in name or "trans." in name:
                # Backbone parameters (pretrained)
                backbone_params.append(param)
            else:
                # FIB, ACFB, Heads, Aggregator (training from scratch)
                head_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
            {"params": no_decay_params, "lr": lr_head, "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def predict_degrees(self, x: torch.Tensor) -> tuple:
        """
        Inference helper: returns roll and pitch in degrees.

        Args:
            x: Input image (B, 3, img_size, img_size)

        Returns:
            roll_deg, pitch_deg[, yaw_deg]: Angles in degrees
        """
        self.eval()
        roll_sincos, pitch_sincos, yaw_sincos = self.forward(x)
        return AngleHeads.decode_angles(roll_sincos, pitch_sincos, yaw_sincos)


if __name__ == "__main__":
    # Quick architecture test
    for img_size in [768, 1024]:
        print(f"\n{'='*60}")
        print(f"Testing with img_size={img_size}")
        print(f"{'='*60}")

        model = LandNetV2(pretrained=False, img_size=img_size, multi_scale=True)

        # Count parameters
        total = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Total params: {total:.1f}M, Trainable: {trainable:.1f}M")

        # Forward pass test
        x = torch.randn(1, 3, img_size, img_size)
        roll, pitch = model(x)
        print(f"Roll: {roll.shape} norm={roll.norm(dim=-1).item():.4f}")
        print(f"Pitch: {pitch.shape} norm={pitch.norm(dim=-1).item():.4f}")

        # Decode to degrees
        roll_deg, pitch_deg = model.predict_degrees(x)
        print(f"Roll: {roll_deg.item():.4f}°, Pitch: {pitch_deg.item():.4f}°")

        # Parameter groups
        groups = model.get_parameter_groups()
        for i, g in enumerate(groups):
            n = sum(p.numel() for p in g["params"]) / 1e6
            print(f"Group {i}: {n:.1f}M params, lr={g['lr']}, wd={g['weight_decay']}")
