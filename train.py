"""
LandNet-V2 Training Script
============================
Full training pipeline with:
  - Mixed Precision (FP16) for T4 GPU efficiency
  - EMA (Exponential Moving Average) for better generalization
  - SWA (Stochastic Weight Averaging) for last 20 epochs
  - Gradient Accumulation for effective batch size 64
  - Cosine Annealing with Linear Warmup
  - Gradient Clipping
  - Comprehensive logging and checkpointing

Usage:
    python -m landnet_v2.train --epochs 300 --batch_size 8
    python -m landnet_v2.train --epochs 1 --batch_size 2  # smoke test
"""
import os
import sys
import math
import time
import argparse
import logging
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landnet_v2 import config
from landnet_v2.models import LandNetV2
from landnet_v2.losses import LandNetV2Loss
from landnet_v2.dataset import create_dataloaders
from landnet_v2.utils import (
    ModelEMA,
    AverageMeter,
    get_cosine_schedule_with_warmup,
    compute_angle_metrics,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
)


def setup_logging(log_dir: str) -> logging.Logger:
    """Configure logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("landnet_v2")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: LandNetV2Loss,
    optimizer,
    scaler: torch.amp.GradScaler,
    ema: ModelEMA,
    epoch: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    """Train for one epoch."""
    model.train()

    meters = {
        "loss": AverageMeter("loss"),
        "huber": AverageMeter("huber"),
        "angular": AverageMeter("angular"),
        "wing": AverageMeter("wing"),
        "cosine": AverageMeter("cosine"),
        "roll_mae": AverageMeter("roll_mae"),
        "pitch_mae": AverageMeter("pitch_mae"),
    }
    if config.NUM_ANGLES >= 3:
        meters["yaw_mae"] = AverageMeter("yaw_mae")

    optimizer.zero_grad()
    num_batches = len(train_loader)
    start_time = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        roll_gt = batch["roll_sincos"].to(device, non_blocking=True)
        pitch_gt = batch["pitch_sincos"].to(device, non_blocking=True)

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type="cuda", enabled=config.USE_AMP):
            if config.NUM_ANGLES >= 3:
                yaw_gt = batch["yaw_sincos"].to(device, non_blocking=True)
                roll_pred, pitch_pred, yaw_pred = model(images)
                loss_dict = criterion(roll_pred, pitch_pred, roll_gt, pitch_gt, yaw_pred, yaw_gt)
            else:
                roll_pred, pitch_pred, _ = model(images)
                loss_dict = criterion(roll_pred, pitch_pred, roll_gt, pitch_gt)
                
            loss = loss_dict["total"] / grad_accum_steps

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
            # Unscale and clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA
            if ema is not None:
                ema.update(model)

        # Update meters
        bs = images.size(0)
        meters["loss"].update(loss_dict["total"].item(), bs)
        meters["huber"].update(loss_dict["huber"].item(), bs)
        meters["angular"].update(loss_dict["angular"].item(), bs)
        meters["wing"].update(loss_dict["wing"].item(), bs)
        meters["cosine"].update(loss_dict["cosine"].item(), bs)
        meters["roll_mae"].update(loss_dict["roll_mae_deg"].item(), bs)
        meters["pitch_mae"].update(loss_dict["pitch_mae_deg"].item(), bs)
        if config.NUM_ANGLES >= 3:
            meters["yaw_mae"].update(loss_dict["yaw_mae_deg"].item(), bs)

        # Log progress to tqdm
        desc = f"Loss: {loss_dict['total'].item():.4f} | Pitch: {loss_dict['pitch_mae_deg'].item():.2f}°"
        pbar.set_description(desc)

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion: LandNetV2Loss,
    device: torch.device,
) -> dict:
    """Validate model on validation set."""
    model.eval()

    meters = {
        "loss": AverageMeter("loss"),
        "roll_mae": AverageMeter("roll_mae"),
        "pitch_mae": AverageMeter("pitch_mae"),
    }
    if config.NUM_ANGLES >= 3:
        meters["yaw_mae"] = AverageMeter("yaw_mae")

    all_roll_sin_pred, all_roll_cos_pred = [], []
    all_pitch_sin_pred, all_pitch_cos_pred = [], []
    all_yaw_sin_pred, all_yaw_cos_pred = [], []
    
    all_roll_sin_gt, all_roll_cos_gt = [], []
    all_pitch_sin_gt, all_pitch_cos_gt = [], []
    all_yaw_sin_gt, all_yaw_cos_gt = [], []

    pbar = tqdm(val_loader, desc="Validating", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        roll_gt = batch["roll_sincos"].to(device, non_blocking=True)
        pitch_gt = batch["pitch_sincos"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=config.USE_AMP):
            if config.NUM_ANGLES >= 3:
                yaw_gt = batch["yaw_sincos"].to(device, non_blocking=True)
                roll_pred, pitch_pred, yaw_pred = model(images)
                loss_dict = criterion(roll_pred, pitch_pred, roll_gt, pitch_gt, yaw_pred, yaw_gt)
            else:
                roll_pred, pitch_pred, _ = model(images)
                loss_dict = criterion(roll_pred, pitch_pred, roll_gt, pitch_gt)

        bs = images.size(0)
        meters["loss"].update(loss_dict["total"].item(), bs)
        meters["roll_mae"].update(loss_dict["roll_mae_deg"].item(), bs)
        meters["pitch_mae"].update(loss_dict["pitch_mae_deg"].item(), bs)
        if config.NUM_ANGLES >= 3:
            meters["yaw_mae"].update(loss_dict["yaw_mae_deg"].item(), bs)

        # Collect predictions for detailed metrics
        all_roll_sin_pred.append(roll_pred[:, 0])
        all_roll_cos_pred.append(roll_pred[:, 1])
        all_pitch_sin_pred.append(pitch_pred[:, 0])
        all_pitch_cos_pred.append(pitch_pred[:, 1])
        all_roll_sin_gt.append(roll_gt[:, 0])
        all_roll_cos_gt.append(roll_gt[:, 1])
        all_pitch_sin_gt.append(pitch_gt[:, 0])
        all_pitch_cos_gt.append(pitch_gt[:, 1])
        
        if config.NUM_ANGLES >= 3:
            all_yaw_sin_pred.append(yaw_pred[:, 0])
            all_yaw_cos_pred.append(yaw_pred[:, 1])
            all_yaw_sin_gt.append(yaw_gt[:, 0])
            all_yaw_cos_gt.append(yaw_gt[:, 1])

    # Compute detailed metrics
    roll_metrics = compute_angle_metrics(
        torch.cat(all_roll_sin_pred), torch.cat(all_roll_cos_pred),
        torch.cat(all_roll_sin_gt), torch.cat(all_roll_cos_gt)
    )
    pitch_metrics = compute_angle_metrics(
        torch.cat(all_pitch_sin_pred), torch.cat(all_pitch_cos_pred),
        torch.cat(all_pitch_sin_gt), torch.cat(all_pitch_cos_gt)
    )

    res = {
        "loss": meters["loss"].avg,
        "roll_mae": roll_metrics["mae"],
        "roll_rmse": roll_metrics["rmse"],
        "roll_median": roll_metrics["median"],
        "roll_p99": roll_metrics["p99"],
        "pitch_mae": pitch_metrics["mae"],
        "pitch_rmse": pitch_metrics["rmse"],
        "pitch_median": pitch_metrics["median"],
        "pitch_p99": pitch_metrics["p99"],
    }
    
    if config.NUM_ANGLES >= 3:
        yaw_metrics = compute_angle_metrics(
            torch.cat(all_yaw_sin_pred), torch.cat(all_yaw_cos_pred),
            torch.cat(all_yaw_sin_gt), torch.cat(all_yaw_cos_gt)
        )
        res.update({
            "yaw_mae": yaw_metrics["mae"],
            "yaw_rmse": yaw_metrics["rmse"],
            "yaw_median": yaw_metrics["median"],
            "yaw_p99": yaw_metrics["p99"],
        })
        
    return res


def main():
    parser = argparse.ArgumentParser(description="LandNet-V2 Training")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr_backbone", type=float, default=config.LR_BACKBONE)
    parser.add_argument("--lr_head", type=float, default=config.LR_HEAD)
    parser.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM_STEPS)
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument("--log_dir", type=str, default=config.LOG_DIR)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to save VRAM (at the cost of speed)")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger = setup_logging(args.log_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    logger.info(f"Device: {device}, GPUs: {num_gpus}")

    # ================================================================
    # Data
    # ================================================================
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        img_size=config.INPUT_SIZE,
        num_workers=args.workers,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
                f"Test: {len(test_loader.dataset)}")

    # ================================================================
    # Model
    # ================================================================
    logger.info("Creating model...")
    model = LandNetV2(
        pretrained=True,
        img_size=config.INPUT_SIZE,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    param_info = count_parameters(model)
    logger.info(f"Parameters: {param_info['total_M']:.1f}M total, "
                f"{param_info['trainable_M']:.1f}M trainable")

    # Multi-GPU
    if num_gpus > 1:
        logger.info(f"Using DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # ================================================================
    # Optimizer, Scheduler, Loss
    # ================================================================
    # Get parameter groups with different LRs
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    param_groups = base_model.get_parameter_groups(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=config.WEIGHT_DECAY,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=config.BETAS)

    # Scheduler: Cosine with warmup
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = len(train_loader) * config.WARMUP_EPOCHS // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr=config.MIN_LR
    )

    criterion = LandNetV2Loss(
        huber_delta=config.HUBER_DELTA,
        lambda_angular=config.LAMBDA_ANGULAR,
        lambda_wing=config.LAMBDA_WING,
        lambda_cosine=config.LAMBDA_COSINE,
        wing_omega=config.WING_OMEGA,
        wing_epsilon=config.WING_EPSILON,
    ).to(device)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

    # EMA
    ema = ModelEMA(base_model, decay=config.EMA_DECAY)
    logger.info(f"EMA decay: {config.EMA_DECAY}")

    # SWA model (initialized later)
    swa_model = None
    swa_scheduler = None

    # ================================================================
    # Resume from checkpoint
    # ================================================================
    start_epoch = 0
    best_metric = float("inf")

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch, best_metric = load_checkpoint(
            args.resume, base_model, optimizer, scheduler, ema
        )
        logger.info(f"Resumed at epoch {start_epoch}, best metric: {best_metric:.6f}")

    # ================================================================
    # Training Loop
    # ================================================================
    logger.info("=" * 70)
    logger.info("Starting training")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size} × {num_gpus} GPUs × {args.grad_accum} accum = "
                f"{args.batch_size * max(1, num_gpus) * args.grad_accum}")
    logger.info(f"  LR backbone: {args.lr_backbone}, LR head: {args.lr_head}")
    logger.info(f"  SWA starts at epoch {config.SWA_START_EPOCH}")
    logger.info("=" * 70)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()

        # ============================================================
        # SWA setup (at SWA start epoch)
        # ============================================================
        if epoch == config.SWA_START_EPOCH and swa_model is None:
            logger.info(f"Epoch {epoch}: Starting SWA")
            swa_model = AveragedModel(base_model)
            swa_scheduler = SWALR(optimizer, swa_lr=config.SWA_LR)

        # ============================================================
        # Train
        # ============================================================
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            epoch=epoch,
            grad_accum_steps=args.grad_accum,
            max_grad_norm=config.MAX_GRAD_NORM,
            device=device,
            logger=logger,
        )

        # ============================================================
        # Scheduler step
        # ============================================================
        if swa_model is not None:
            swa_model.update_parameters(base_model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # ============================================================
        # Validate
        # ============================================================
        val_metrics = validate(model, val_loader, criterion, device)

        if ema is not None:
            ema.apply_shadow(base_model)
            val_ema_metrics = validate(model, val_loader, criterion, device)
            ema.restore(base_model)
        else:
            val_ema_metrics = None

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(f"--- Epoch {epoch} Summary ({epoch_time:.1f}s) | LR: {current_lr:.2e} ---")
        
        # Train Summary
        train_str = f"Train     | Loss: {train_metrics['loss']:.4f} | Roll: {train_metrics['roll_mae']:.4f}° | Pitch: {train_metrics['pitch_mae']:.4f}°"
        if config.NUM_ANGLES >= 3:
            train_str += f" | Yaw: {train_metrics['yaw_mae']:.4f}°"
        logger.info(train_str)
        
        # Val Summary
        val_str = f"Val       | Loss: {val_metrics['loss']:.4f} | Roll: {val_metrics['roll_mae']:.4f}° | Pitch: {val_metrics['pitch_mae']:.4f}°"
        if config.NUM_ANGLES >= 3:
            val_str += f" | Yaw: {val_metrics['yaw_mae']:.4f}°"
        logger.info(val_str)

        # Val EMA Summary
        if val_ema_metrics is not None:
            ema_str = f"Val (EMA) | Loss: {val_ema_metrics['loss']:.4f} | Roll: {val_ema_metrics['roll_mae']:.4f}° | Pitch: {val_ema_metrics['pitch_mae']:.4f}°"
            if config.NUM_ANGLES >= 3:
                ema_str += f" | Yaw: {val_ema_metrics['yaw_mae']:.4f}°"
            logger.info(ema_str)

        # ============================================================
        # Save checkpoints
        # ============================================================
        # Save best model
        metric = val_metrics["roll_mae"] + val_metrics["pitch_mae"]
        if config.NUM_ANGLES >= 3:
            metric += val_metrics["yaw_mae"]
        if metric < best_metric:
            best_metric = metric
            ema.apply_shadow(base_model)
            save_checkpoint(
                base_model, optimizer, scheduler, epoch, best_metric, ema,
                filepath=os.path.join(args.checkpoint_dir, "best_model.pth"),
            )
            ema.restore(base_model)
            logger.info(f"  ★ New best model saved (metric={best_metric:.6f})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                base_model, optimizer, scheduler, epoch, best_metric, ema,
                filepath=os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch}.pth"),
            )

    # ================================================================
    # Post-training: Update SWA BatchNorm and save
    # ================================================================
    if swa_model is not None:
        logger.info("Updating SWA BatchNorm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save(
            swa_model.state_dict(),
            os.path.join(args.checkpoint_dir, "swa_model.pth"),
        )
        logger.info("SWA model saved")

    # ================================================================
    # Final test evaluation
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Final Test Evaluation")
    logger.info("=" * 70)

    # Load best EMA model
    best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, base_model, ema=ema)
        ema.apply_shadow(base_model)

    test_metrics = validate(model, test_loader, criterion, device)

    logger.info(f"Test Results (Best EMA Model):")
    logger.info(f"  Roll  MAE: {test_metrics['roll_mae']:.6f}°  "
                f"RMSE: {test_metrics['roll_rmse']:.6f}°  "
                f"Median: {test_metrics['roll_median']:.6f}°  "
                f"P99: {test_metrics['roll_p99']:.6f}°")
    logger.info(f"  Pitch MAE: {test_metrics['pitch_mae']:.6f}°  "
                f"RMSE: {test_metrics['pitch_rmse']:.6f}°  "
                f"Median: {test_metrics['pitch_median']:.6f}°  "
                f"P99: {test_metrics['pitch_p99']:.6f}°")
    
    if config.NUM_ANGLES >= 3:
        logger.info(f"  Yaw   MAE: {test_metrics['yaw_mae']:.6f}°  "
                    f"RMSE: {test_metrics['yaw_rmse']:.6f}°  "
                    f"Median: {test_metrics['yaw_median']:.6f}°  "
                    f"P99: {test_metrics['yaw_p99']:.6f}°")

    logger.info("\nTraining complete!")
    return test_metrics


if __name__ == "__main__":
    main()
