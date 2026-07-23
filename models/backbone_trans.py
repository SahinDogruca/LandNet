"""
DeiT-III-Base Backbone Wrapper
==============================
Extracts intermediate Transformer features at 4 stage boundaries
for FIB interaction with CNN branch.

Splits 12 Transformer blocks into 4 groups of 3:
    Stage 0: Blocks  0-2  → (B, N+1, 768)
    Stage 1: Blocks  3-5  → (B, N+1, 768)
    Stage 2: Blocks  6-8  → (B, N+1, 768)
    Stage 3: Blocks  9-11 → (B, N+1, 768)

Where N = (img_size / patch_size)² = (512/16)² = 1024 patches.
The +1 is for the CLS token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm>=0.9.0")


class DeiTIIIBackbone(nn.Module):
    """
    DeiT-III-Base backbone with stage-by-stage feature extraction.

    Handles positional embedding interpolation for 512×512 input
    (DeiT-III is pretrained at 384×384 → pos_embed resized to 512).
    """

    def __init__(
        self,
        model_name: str = "deit3_base_patch16_384.fb_in22k_ft_in1k",
        pretrained: bool = True,
        img_size: int = 512,
        patch_size: int = 16,
        blocks_per_stage: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = 768
        self.num_patches = (img_size // patch_size) ** 2  # 1024
        self.grid_size = img_size // patch_size            # 32
        self.blocks_per_stage = blocks_per_stage

        # Load pretrained model
        model = timm.create_model(model_name, pretrained=pretrained)

        # Extract components
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_drop = model.pos_drop

        # Determine num_prefix_tokens for interpolation
        self.num_prefix_tokens = getattr(model, 'num_prefix_tokens', 1)

        # Interpolate positional embeddings for new image size
        self.pos_embed = nn.Parameter(
            self._interpolate_pos_embed(
                model.pos_embed, img_size, patch_size, self.num_prefix_tokens
            )
        )

        # Split transformer blocks into 4 stages
        all_blocks = list(model.blocks)
        assert len(all_blocks) == 12, f"Expected 12 blocks, got {len(all_blocks)}"

        self.stage0 = nn.Sequential(*all_blocks[0:3])
        self.stage1 = nn.Sequential(*all_blocks[3:6])
        self.stage2 = nn.Sequential(*all_blocks[6:9])
        self.stage3 = nn.Sequential(*all_blocks[9:12])
        self.stages = [self.stage0, self.stage1, self.stage2, self.stage3]

        self.norm = model.norm  # Final LayerNorm

        # Cleanup
        del model

    def _interpolate_pos_embed(
        self,
        pos_embed: torch.Tensor,
        new_img_size: int,
        patch_size: int,
        num_prefix_tokens: int = 1,
    ) -> torch.Tensor:
        """Interpolate positional embeddings for a different image size."""
        total_tokens = pos_embed.shape[1]
        
        # Detect if pos_embed includes the prefix (CLS) token
        grid_size = int(math.sqrt(total_tokens))
        if grid_size * grid_size == total_tokens:
            has_prefix = False
            cls_pos = None
            patch_pos = pos_embed
        else:
            has_prefix = True
            cls_pos = pos_embed[:, :num_prefix_tokens, :]
            patch_pos = pos_embed[:, num_prefix_tokens:, :]

        old_num_patches = patch_pos.shape[1]
        old_grid = int(old_num_patches ** 0.5)
        new_grid = new_img_size // patch_size  # 32

        if old_grid == new_grid:
            return pos_embed

        # Reshape to spatial grid, interpolate, flatten back
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos.float(),
            size=(new_grid, new_grid),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)

        if has_prefix:
            return torch.cat([cls_pos, patch_pos], dim=1)
        return patch_pos

    def forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embedding + positional embedding + CLS token."""
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Handle pos_embed based on whether it has the CLS token embedded
        if self.pos_embed.shape[1] == x.shape[1]:
            # pos_embed only contains patch pos embeddings (no CLS)
            x = x + self.pos_embed
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)          # (B, N+1, D)
        else:
            # pos_embed includes CLS pos embedding
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)          # (B, N+1, D)
            x = x + self.pos_embed
        x = self.pos_drop(x)

        return x

    def forward_stage(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Run a single stage (3 transformer blocks)."""
        return self.stages[stage_idx](x)

    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens only (remove CLS token)."""
        return x[:, 1:, :]  # (B, N, D)

    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token only."""
        return x[:, :1, :]  # (B, 1, D)

    def forward(self, x: torch.Tensor) -> list:
        """
        Full forward pass. Returns features from all 4 stages.
        Use forward_embed + forward_stage for interleaved FIB.
        """
        x = self.forward_embed(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


if __name__ == "__main__":
    # Quick test
    model = DeiTIIIBackbone(pretrained=False)
    x = torch.randn(1, 3, 512, 512)
    features = model(x)
    for i, f in enumerate(features):
        print(f"Stage {i}: {f.shape}")
    # Expected (all same shape since ViT doesn't downsample):
    # Stage 0: torch.Size([1, 1025, 768])  (1024 patches + 1 CLS)
    # Stage 1: torch.Size([1, 1025, 768])
    # Stage 2: torch.Size([1, 1025, 768])
    # Stage 3: torch.Size([1, 1025, 768])
