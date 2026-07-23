"""
ConvNeXt-V2-Base Backbone Wrapper
=================================
Extracts multi-scale features from 4 stages for FIB interaction.
Pretrained on ImageNet-22k → fine-tuned on ImageNet-1k.

Stage outputs for 512×512 input (Tiny):
    Stage 0: (B, 96,  128, 128)
    Stage 1: (B, 192, 64,  64)
    Stage 2: (B, 384, 32,  32)
    Stage 3: (B, 768, 16,  16)
"""
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm>=0.9.0")


class ConvNeXtV2Backbone(nn.Module):
    """
    ConvNeXt-V2-Base backbone with stage-by-stage feature extraction.

    Unlike using features_only=True (which runs the full model at once),
    this wrapper exposes individual stages so FIB can be interleaved
    between CNN and Transformer stages.
    """

    def __init__(
        self,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
    ):
        super().__init__()

        # Load full model
        model = timm.create_model(model_name, pretrained=pretrained)

        # Extract components
        self.stem = model.stem          # 4x downsample: (B,3,H,W) → (B,128,H/4,W/4)
        self.stages = model.stages      # 4 ConvNeXtStage modules

        # Store channel dims for each stage
        self.channels = [96, 192, 384, 768]

        # Remove classification head (not needed)
        del model

    def forward_stage(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Run a single stage. Use this for interleaved FIB execution."""
        return self.stages[stage_idx](x)

    def forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem (initial 4x downsampling)."""
        return self.stem(x)

    def forward(self, x: torch.Tensor) -> list:
        """
        Full forward pass, returns features from all 4 stages.
        Use forward_stem + forward_stage for interleaved FIB.
        """
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


if __name__ == "__main__":
    # Quick test
    model = ConvNeXtV2Backbone(pretrained=False)
    x = torch.randn(1, 3, 512, 512)
    features = model(x)
    for i, f in enumerate(features):
        print(f"Stage {i}: {f.shape}")
    # Expected:
    # Stage 0: torch.Size([1, 96, 128, 128])
    # Stage 1: torch.Size([1, 192, 64, 64])
    # Stage 2: torch.Size([1, 384, 32, 32])
    # Stage 3: torch.Size([1, 768, 16, 16])
