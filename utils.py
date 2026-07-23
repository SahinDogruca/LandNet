"""
LandNet-V2 Utilities
====================
EMA (Exponential Moving Average), metrics, and helper functions.
"""
import math
import copy
import torch
import torch.nn as nn
from collections import OrderedDict


class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights updated as:
        shadow = decay * shadow + (1 - decay) * param

    At inference, the EMA weights typically generalize better.

    Usage:
        ema = ModelEMA(model, decay=0.9999)
        for batch in dataloader:
            loss.backward()
            optimizer.step()
            ema.update(model)
        # Evaluate with EMA weights
        ema.apply_shadow(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module):
        """Register model parameters for EMA tracking."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow weights with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """Replace model weights with EMA shadow weights (backup originals)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original model weights from backup."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_angle_metrics(
    sin_pred: torch.Tensor,
    cos_pred: torch.Tensor,
    sin_gt: torch.Tensor,
    cos_gt: torch.Tensor,
) -> dict:
    """
    Compute angular error metrics in degrees.

    Returns:
        dict with 'mae', 'rmse', 'median', 'p99' (all in degrees)
    """
    angle_pred = torch.atan2(sin_pred, cos_pred)
    angle_gt = torch.atan2(sin_gt, cos_gt)

    # Wrapped difference
    diff = angle_pred - angle_gt
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    diff_deg = diff.abs() * (180.0 / math.pi)

    return {
        "mae": diff_deg.mean().item(),
        "rmse": diff_deg.pow(2).mean().sqrt().item(),
        "median": diff_deg.median().item(),
        "p99": diff_deg.quantile(0.99).item() if len(diff_deg) > 1 else diff_deg.item(),
        "max": diff_deg.max().item(),
    }


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 1e-7,
):
    """
    Cosine annealing schedule with linear warmup.

    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr: Minimum learning rate at end of cosine decay
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model: nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    ema: ModelEMA = None,
    filepath: str = "checkpoint.pth",
):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
    }
    if ema is not None:
        state["ema_state_dict"] = ema.state_dict()
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: nn.Module, optimizer=None, scheduler=None, ema=None):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if ema and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", float("inf"))
