"""
LARD_V2 XPlane Dataset Loader
==============================
Loads DEEL-AI/LARD_V2 (xplane config) from Hugging Face.

Features:
  - 1024×1024 images resized to 512×512
  - Roll and pitch angles encoded as sin(θ)/cos(θ)
  - Angle-safe augmentations only (no rotation/flip)
  - Train/Val/Test splits
"""
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image, ImageFilter

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("datasets is required: pip install datasets")


class GaussianNoise:
    """Add random Gaussian noise to tensor."""

    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std


class RandomCutout:
    """
    Random rectangular cutout (erasing) on tensor.
    Safer than RandomErasing as it always uses zero fill.
    """

    def __init__(self, p: float = 0.2, min_ratio: float = 0.02, max_ratio: float = 0.1):
        self.p = p
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        _, h, w = tensor.shape
        area = h * w
        target_area = random.uniform(self.min_ratio, self.max_ratio) * area
        aspect_ratio = random.uniform(0.5, 2.0)
        cut_h = int(math.sqrt(target_area * aspect_ratio))
        cut_w = int(math.sqrt(target_area / aspect_ratio))
        cut_h = min(cut_h, h)
        cut_w = min(cut_w, w)
        top = random.randint(0, h - cut_h)
        left = random.randint(0, w - cut_w)
        tensor[:, top : top + cut_h, left : left + cut_w] = 0.0
        return tensor


class LARDDataset(Dataset):
    """
    LARD_V2 XPlane Dataset for roll and pitch regression.

    Loads images and encodes angles as [sin(θ), cos(θ)] pairs.

    Args:
        split: 'train', 'val', or 'test'
        img_size: Target image size (default 512)
        augment: Whether to apply augmentations (True for train)
        cache_dir: HuggingFace cache directory
        val_ratio: Ratio of train data to use for validation
        seed: Random seed for reproducible splits
    """

    # ImageNet normalization for pretrained backbones
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        split: str = "train",
        img_size: int = 512,
        augment: bool = None,
        cache_dir: str = None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.split = split
        self.img_size = img_size
        self.augment = augment if augment is not None else (split == "train")

        # Load dataset from HuggingFace
        print(f"Loading LARD_V2 XPlane ({split})...")
        raw_dataset = load_dataset(
            "DEEL-AI/LARD_V2",
            "xplane",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # Handle splits
        if split == "test" and "test" in raw_dataset:
            self.data = raw_dataset["test"]
        elif split in ("train", "val"):
            train_data = raw_dataset["train"]
            # Create train/val split
            n = len(train_data)
            indices = list(range(n))
            rng = random.Random(seed)
            rng.shuffle(indices)
            val_size = int(n * val_ratio)

            if split == "val":
                self.data = train_data.select(indices[:val_size])
            else:
                self.data = train_data.select(indices[val_size:])
        else:
            # Fallback: use whatever split name is provided
            self.data = raw_dataset[split]

        print(f"  → {len(self.data)} samples loaded for '{split}'")

        # Identify angle columns
        self._detect_columns()

        # Build transforms
        self.transform = self._build_transforms()

    def _detect_columns(self):
        """Detect column names for roll and pitch angles."""
        cols = self.data.column_names
        print(f"  → Available columns: {cols}")

        # Common column names in LARD_V2
        self.image_col = "image"
        self.roll_col = None
        self.pitch_col = None
        self.yaw_col = None

        # Try to find roll/pitch/yaw columns
        for col in cols:
            col_lower = col.lower()
            if "roll" in col_lower:
                self.roll_col = col
            elif "pitch" in col_lower:
                self.pitch_col = col
            elif "yaw" in col_lower:
                self.yaw_col = col

        # Fallback: check for camera_roll, camera_pitch, etc.
        if self.roll_col is None:
            for candidate in ["roll", "camera_roll", "Roll", "ROLL"]:
                if candidate in cols:
                    self.roll_col = candidate
                    break

        if self.pitch_col is None:
            for candidate in ["pitch", "camera_pitch", "Pitch", "PITCH"]:
                if candidate in cols:
                    self.pitch_col = candidate
                    break

        if self.yaw_col is None:
            for candidate in ["yaw", "camera_yaw", "Yaw", "YAW", "heading"]:
                if candidate in cols:
                    self.yaw_col = candidate
                    break

        if self.roll_col is None or self.pitch_col is None:
            raise ValueError(
                f"Could not find roll/pitch columns in: {cols}. "
                f"Found roll={self.roll_col}, pitch={self.pitch_col}"
            )

        print(f"  → Using columns: roll='{self.roll_col}', pitch='{self.pitch_col}', yaw='{self.yaw_col}'")

    def _build_transforms(self):
        """
        Build angle-safe augmentation pipeline.

        EXCLUDED (would change angles):
          - RandomRotation
          - RandomHorizontalFlip
          - RandomVerticalFlip
          - RandomAffine

        INCLUDED (safe):
          - ColorJitter
          - GaussianBlur
          - GaussianNoise
          - RandomCutout
          - RandomErasing
        """
        base = [
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ]

        if self.augment:
            augment_list = [
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.1,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))],
                    p=0.3,
                ),
                transforms.ToTensor(),
                GaussianNoise(std=0.01),
                RandomCutout(p=0.2, min_ratio=0.02, max_ratio=0.1),
                transforms.Normalize(mean=self.MEAN, std=self.STD),
            ]
            return transforms.Compose(augment_list)
        else:
            return transforms.Compose([
                *base,
                transforms.Normalize(mean=self.MEAN, std=self.STD),
            ])

    @staticmethod
    def encode_angle(angle_deg: float) -> torch.Tensor:
        """
        Encode angle (degrees) as [sin(θ), cos(θ)].

        Args:
            angle_deg: Angle in degrees

        Returns:
            (2,) tensor [sin(θ_rad), cos(θ_rad)]
        """
        angle_rad = angle_deg * (math.pi / 180.0)
        return torch.tensor([math.sin(angle_rad), math.cos(angle_rad)], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                'image': (3, img_size, img_size) normalized tensor
                'roll_sincos': (2,) [sin(roll), cos(roll)]
                'pitch_sincos': (2,) [sin(pitch), cos(pitch)]
                'roll_deg': float, original roll in degrees
                'pitch_deg': float, original pitch in degrees
        """
        sample = self.data[idx]

        # Load image
        image = sample[self.image_col]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        # Apply transforms
        image = self.transform(image)

        # Get angles
        roll_deg = float(sample[self.roll_col])
        pitch_deg = float(sample[self.pitch_col])
        yaw_deg = float(sample[self.yaw_col]) if self.yaw_col else 0.0

        # Encode as sin/cos
        roll_sincos = self.encode_angle(roll_deg)
        pitch_sincos = self.encode_angle(pitch_deg)
        yaw_sincos = self.encode_angle(yaw_deg)

        res = {
            "image": image,
            "roll_sincos": roll_sincos,
            "pitch_sincos": pitch_sincos,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
        }
        
        if self.yaw_col:
            res["yaw_sincos"] = yaw_sincos
            res["yaw_deg"] = yaw_deg
            
        return res


def create_dataloaders(
    batch_size: int = 8,
    img_size: int = 512,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    cache_dir: str = None,
    seed: int = 42,
) -> tuple:
    """
    Create train, validation, and test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = LARDDataset(
        split="train", img_size=img_size, augment=True,
        cache_dir=cache_dir, val_ratio=val_ratio, seed=seed,
    )
    val_ds = LARDDataset(
        split="val", img_size=img_size, augment=False,
        cache_dir=cache_dir, val_ratio=val_ratio, seed=seed,
    )
    test_ds = LARDDataset(
        split="test", img_size=img_size, augment=False,
        cache_dir=cache_dir, val_ratio=val_ratio, seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,  # Can use larger batch for eval
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    ds = LARDDataset(split="train", img_size=512)
    print(f"\nDataset size: {len(ds)}")

    sample = ds[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Roll: {sample['roll_deg']:.4f}° → sin/cos: {sample['roll_sincos']}")
    print(f"Pitch: {sample['pitch_deg']:.4f}° → sin/cos: {sample['pitch_sincos']}")
