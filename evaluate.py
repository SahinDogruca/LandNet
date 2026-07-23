"""
LandNet-V2 Evaluation & TTA
============================
Test evaluation with Test Time Augmentation (TTA):
  1. Original image → prediction
  2. 5-Crop (center + 4 corners at 90%) → averaged predictions
  3. Brightness/contrast variations → averaged predictions

Final prediction is the ensemble mean of all augmented predictions.

Usage:
    python -m landnet_v2.evaluate --checkpoint checkpoints/best_model.pth
    python -m landnet_v2.evaluate --checkpoint checkpoints/swa_model.pth --tta
"""
import os
import sys
import math
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landnet_v2 import config
from landnet_v2.models import LandNetV2
from landnet_v2.models.heads import AngleHeads
from landnet_v2.losses import LandNetV2Loss
from landnet_v2.dataset import LARDDataset
from landnet_v2.utils import (
    compute_angle_metrics,
    load_checkpoint,
    ModelEMA,
)


class TTAPredictor:
    """
    Test Time Augmentation for angle prediction.

    Augmentation strategies (angle-safe only):
      1. Original image (baseline)
      2. 5-Crop: center + 4 corners at crop_ratio of original size
      3. Brightness variations: ±10% brightness change
      4. Contrast variations: ±10% contrast change

    Predictions are averaged across all augmentations.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        img_size: int = 512,
        crop_ratio: float = 0.9,
        num_brightness: int = 3,
        use_amp: bool = True,
    ):
        self.model = model
        self.device = device
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.num_brightness = num_brightness
        self.use_amp = use_amp

        # Normalization (applied after augmentation)
        self.normalize = transforms.Normalize(
            mean=LARDDataset.MEAN, std=LARDDataset.STD
        )

    def _five_crop(self, images: torch.Tensor) -> list:
        """
        Generate 5 crops (center + 4 corners) from batch.

        Args:
            images: (B, 3, H, W) normalized images

        Returns:
            list of (B, 3, img_size, img_size) tensors
        """
        B, C, H, W = images.shape
        crop_h = int(H * self.crop_ratio)
        crop_w = int(W * self.crop_ratio)

        crops = []

        # Center crop
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        crops.append(images[:, :, top:top+crop_h, left:left+crop_w])

        # Top-left
        crops.append(images[:, :, :crop_h, :crop_w])

        # Top-right
        crops.append(images[:, :, :crop_h, W-crop_w:])

        # Bottom-left
        crops.append(images[:, :, H-crop_h:, :crop_w])

        # Bottom-right
        crops.append(images[:, :, H-crop_h:, W-crop_w:])

        # Resize all crops back to original size
        resized = []
        for crop in crops:
            resized.append(
                F.interpolate(crop, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
            )

        return resized

    def _brightness_variants(self, images: torch.Tensor) -> list:
        """Generate brightness/contrast variations."""
        variants = []

        # Brighter
        variants.append(torch.clamp(images * 1.1, 0, 1) if not self._is_normalized(images)
                        else images * 1.05)

        # Darker
        variants.append(torch.clamp(images * 0.9, 0, 1) if not self._is_normalized(images)
                        else images * 0.95)

        return variants

    def _is_normalized(self, images: torch.Tensor) -> bool:
        """Check if images are ImageNet-normalized (can have negative values)."""
        return images.min() < 0

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> tuple:
        """
        TTA prediction for a batch.

        Args:
            images: (B, 3, H, W) normalized images

        Returns:
            roll_sincos: (B, 2) averaged predictions
            pitch_sincos: (B, 2) averaged predictions
        """
        self.model.eval()
        all_roll_preds = []
        all_pitch_preds = []
        all_yaw_preds = []

        # 1. Original image
        with autocast(enabled=self.use_amp):
            if config.NUM_ANGLES >= 3:
                roll, pitch, yaw = self.model(images)
            else:
                roll, pitch, _ = self.model(images)
                yaw = None
        all_roll_preds.append(roll)
        all_pitch_preds.append(pitch)
        if yaw is not None:
            all_yaw_preds.append(yaw)

        # 2. Five crops
        crops = self._five_crop(images)
        for crop in crops:
            with autocast(enabled=self.use_amp):
                if config.NUM_ANGLES >= 3:
                    roll, pitch, yaw = self.model(crop)
                else:
                    roll, pitch, _ = self.model(crop)
            all_roll_preds.append(roll)
            all_pitch_preds.append(pitch)
            if yaw is not None:
                all_yaw_preds.append(yaw)

        # 3. Brightness/contrast variations
        variants = self._brightness_variants(images)
        for var in variants:
            with autocast(enabled=self.use_amp):
                if config.NUM_ANGLES >= 3:
                    roll, pitch, yaw = self.model(var)
                else:
                    roll, pitch, _ = self.model(var)
            all_roll_preds.append(roll)
            all_pitch_preds.append(pitch)
            if yaw is not None:
                all_yaw_preds.append(yaw)

        # Average all predictions
        avg_roll = torch.stack(all_roll_preds).mean(dim=0)   # (B, 2)
        avg_pitch = torch.stack(all_pitch_preds).mean(dim=0) # (B, 2)
        avg_yaw = torch.stack(all_yaw_preds).mean(dim=0) if all_yaw_preds else None

        return avg_roll, avg_pitch, avg_yaw


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader,
    device: torch.device,
    use_tta: bool = False,
    use_amp: bool = True,
) -> dict:
    """
    Full evaluation on test set.

    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Device
        use_tta: Whether to use Test Time Augmentation
        use_amp: Whether to use mixed precision

    Returns:
        dict with detailed metrics for roll and pitch
    """
    model.eval()

    if use_tta:
        predictor = TTAPredictor(model, device, use_amp=use_amp)

    all_roll_sin_pred, all_roll_cos_pred = [], []
    all_pitch_sin_pred, all_pitch_cos_pred = [], []
    all_yaw_sin_pred, all_yaw_cos_pred = [], []
    
    all_roll_sin_gt, all_roll_cos_gt = [], []
    all_pitch_sin_gt, all_pitch_cos_gt = [], []
    all_yaw_sin_gt, all_yaw_cos_gt = [], []
    
    all_roll_deg_gt, all_pitch_deg_gt, all_yaw_deg_gt = [], [], []

    criterion = LandNetV2Loss().to(device)

    for batch in test_loader:
        images = batch["image"].to(device, non_blocking=True)
        roll_gt = batch["roll_sincos"].to(device, non_blocking=True)
        pitch_gt = batch["pitch_sincos"].to(device, non_blocking=True)
        if config.NUM_ANGLES >= 3:
            yaw_gt = batch["yaw_sincos"].to(device, non_blocking=True)

        if use_tta:
            roll_pred, pitch_pred, yaw_pred = predictor.predict(images)
        else:
            with autocast(enabled=use_amp):
                if config.NUM_ANGLES >= 3:
                    roll_pred, pitch_pred, yaw_pred = model(images)
                else:
                    roll_pred, pitch_pred, _ = model(images)

        # Collect predictions
        all_roll_sin_pred.append(roll_pred[:, 0].float())
        all_roll_cos_pred.append(roll_pred[:, 1].float())
        all_pitch_sin_pred.append(pitch_pred[:, 0].float())
        all_pitch_cos_pred.append(pitch_pred[:, 1].float())

        all_roll_sin_gt.append(roll_gt[:, 0])
        all_roll_cos_gt.append(roll_gt[:, 1])
        all_pitch_sin_gt.append(pitch_gt[:, 0])
        all_pitch_cos_gt.append(pitch_gt[:, 1])

        all_roll_deg_gt.extend(batch["roll_deg"])
        all_pitch_deg_gt.extend(batch["pitch_deg"])
        
        if config.NUM_ANGLES >= 3:
            all_yaw_sin_pred.append(yaw_pred[:, 0].float())
            all_yaw_cos_pred.append(yaw_pred[:, 1].float())
            all_yaw_sin_gt.append(yaw_gt[:, 0])
            all_yaw_cos_gt.append(yaw_gt[:, 1])
            all_yaw_deg_gt.extend(batch["yaw_deg"])

    # Concatenate all predictions
    roll_sin_pred = torch.cat(all_roll_sin_pred)
    roll_cos_pred = torch.cat(all_roll_cos_pred)
    pitch_sin_pred = torch.cat(all_pitch_sin_pred)
    pitch_cos_pred = torch.cat(all_pitch_cos_pred)

    roll_sin_gt = torch.cat(all_roll_sin_gt)
    roll_cos_gt = torch.cat(all_roll_cos_gt)
    pitch_sin_gt = torch.cat(all_pitch_sin_gt)
    pitch_cos_gt = torch.cat(all_pitch_cos_gt)

    # Compute metrics
    roll_metrics = compute_angle_metrics(
        roll_sin_pred, roll_cos_pred, roll_sin_gt, roll_cos_gt
    )
    pitch_metrics = compute_angle_metrics(
        pitch_sin_pred, pitch_cos_pred, pitch_sin_gt, pitch_cos_gt
    )

    # Decode predicted angles for per-sample analysis
    roll_pred_deg = torch.atan2(roll_sin_pred, roll_cos_pred) * (180.0 / math.pi)
    pitch_pred_deg = torch.atan2(pitch_sin_pred, pitch_cos_pred) * (180.0 / math.pi)
    roll_gt_deg = torch.atan2(roll_sin_gt, roll_cos_gt) * (180.0 / math.pi)
    pitch_gt_deg = torch.atan2(pitch_sin_gt, pitch_cos_gt) * (180.0 / math.pi)

    # Per-sample errors
    roll_errors = (roll_pred_deg - roll_gt_deg).abs()
    pitch_errors = (pitch_pred_deg - pitch_gt_deg).abs()

    results = {
        "roll": roll_metrics,
        "pitch": pitch_metrics,
        "combined_mae": roll_metrics["mae"] + pitch_metrics["mae"],
        "roll_errors": roll_errors.cpu(),
        "pitch_errors": pitch_errors.cpu(),
        "roll_pred_deg": roll_pred_deg.cpu(),
        "pitch_pred_deg": pitch_pred_deg.cpu(),
        "roll_gt_deg": roll_gt_deg.cpu(),
        "pitch_gt_deg": pitch_gt_deg.cpu(),
    }
    
    if config.NUM_ANGLES >= 3:
        yaw_sin_pred = torch.cat(all_yaw_sin_pred)
        yaw_cos_pred = torch.cat(all_yaw_cos_pred)
        yaw_sin_gt = torch.cat(all_yaw_sin_gt)
        yaw_cos_gt = torch.cat(all_yaw_cos_gt)
        
        yaw_metrics = compute_angle_metrics(
            yaw_sin_pred, yaw_cos_pred, yaw_sin_gt, yaw_cos_gt
        )
        
        yaw_pred_deg = torch.atan2(yaw_sin_pred, yaw_cos_pred) * (180.0 / math.pi)
        yaw_gt_deg = torch.atan2(yaw_sin_gt, yaw_cos_gt) * (180.0 / math.pi)
        yaw_errors = (yaw_pred_deg - yaw_gt_deg).abs()
        
        results.update({
            "yaw": yaw_metrics,
            "yaw_errors": yaw_errors.cpu(),
            "yaw_pred_deg": yaw_pred_deg.cpu(),
            "yaw_gt_deg": yaw_gt_deg.cpu(),
        })
        results["combined_mae"] += yaw_metrics["mae"]

    return results


def print_results(results: dict, title: str = "Test Results"):
    """Pretty print evaluation results."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    
    header = f"  {'Metric':<20} {'Roll (°)':<15} {'Pitch (°)':<15}"
    if config.NUM_ANGLES >= 3:
        header += f" {'Yaw (°)':<15}"
    print(f"\n{header}")
    print(f"  {'-'*65}")

    for metric in ["mae", "rmse", "median", "p99", "max"]:
        r = results["roll"][metric]
        p = results["pitch"][metric]
        line = f"  {metric.upper():<20} {r:<15.6f} {p:<15.6f}"
        if config.NUM_ANGLES >= 3:
            y = results["yaw"][metric]
            line += f" {y:<15.6f}"
        print(line)

    print(f"\n  Combined MAE: {results['combined_mae']:.6f}°")

    # Error distribution
    roll_errors = results["roll_errors"]
    pitch_errors = results["pitch_errors"]
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]

    print(f"\n  Error Distribution (% of samples below threshold):")
    
    dist_header = f"  {'Threshold':<15} {'Roll %':<15} {'Pitch %':<15}"
    if config.NUM_ANGLES >= 3:
        dist_header += f" {'Yaw %':<15}"
    print(dist_header)
    print(f"  {'-'*60}")
    
    for t in thresholds:
        r_pct = (roll_errors < t).float().mean().item() * 100
        p_pct = (pitch_errors < t).float().mean().item() * 100
        line = f"  < {t:<12.3f}° {r_pct:<15.1f} {p_pct:<15.1f}"
        if config.NUM_ANGLES >= 3:
            yaw_errors = results["yaw_errors"]
            y_pct = (yaw_errors < t).float().mean().item() * 100
            line += f" {y_pct:<15.1f}"
        print(line)

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LandNet-V2 Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tta", action="store_true",
                        help="Enable Test Time Augmentation")
    parser.add_argument("--swa", action="store_true",
                        help="Load SWA model (different state_dict format)")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE * 2)
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Creating model...")
    model = LandNetV2(pretrained=False)

    if args.swa:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        swa_model.load_state_dict(state_dict)
        model = swa_model
    else:
        # Load checkpoint and apply EMA weights
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            # Apply EMA if available
            if "ema_state_dict" in checkpoint:
                ema = ModelEMA(model)
                ema.load_state_dict(checkpoint["ema_state_dict"])
                ema.apply_shadow(model)
                print("Applied EMA weights")
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Load test data
    print("Loading test dataset...")
    test_ds = LARDDataset(
        split="test",
        img_size=config.INPUT_SIZE,
        augment=False,
        cache_dir=args.cache_dir,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    print(f"Test samples: {len(test_ds)}")

    # Evaluate
    print(f"\nEvaluating{'(with TTA)' if args.tta else ''}...")
    start_time = time.time()

    results = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        use_tta=args.tta,
        use_amp=config.USE_AMP,
    )

    eval_time = time.time() - start_time
    print(f"Evaluation time: {eval_time:.1f}s")

    # Print results
    title = f"Test Results ({'TTA' if args.tta else 'Standard'})"
    if args.swa:
        title += " [SWA Model]"
    print_results(results, title)

    return results


if __name__ == "__main__":
    main()
