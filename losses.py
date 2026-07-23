"""
LandNet-V2 Loss Functions
=========================
Four-component loss for high-precision angular regression:
  1. Huber Loss — robust sin/cos regression
  2. Angular Loss — direct angular error minimization
  3. Wing Loss — logarithmic gradient for sub-degree convergence
  4. Cosine Similarity Loss — unit circle angular proximity (new)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WingLoss(nn.Module):
    """
    Wing Loss for fine-grained regression (adapted from facial landmark detection).

    For errors < omega:  loss = omega * ln(1 + |x|/epsilon)   → log gradient, never vanishes
    For errors >= omega: loss = |x| - C                       → linear

    This ensures strong gradients even for very small errors (< 0.01°),
    preventing the loss surface from flattening as predictions improve.

    Reference: Feng et al., "Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks", CVPR 2018.
    """

    def __init__(self, omega: float = 0.008727, epsilon: float = 0.000873):
        """
        Args:
            omega: Threshold below which log loss applies (degrees).
                   Default 0.5°
            epsilon: Curvature parameter. Default omega/10 (0.05°).
        """
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = omega - omega * math.log(1.0 + omega / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = (pred - target).abs()
        log_term = self.omega * torch.log(1.0 + delta / self.epsilon)
        linear_term = delta - self.C
        loss = torch.where(delta < self.omega, log_term, linear_term)
        return loss.mean()


class AngularLoss(nn.Module):
    """
    Direct angular error loss computed from sin/cos predictions.
    Reconstructs angles via atan2 and computes wrapped absolute difference.
    """

    def forward(
        self,
        sin_pred: torch.Tensor,
        cos_pred: torch.Tensor,
        sin_gt: torch.Tensor,
        cos_gt: torch.Tensor,
    ) -> torch.Tensor:
        angle_pred = torch.atan2(sin_pred, cos_pred)
        angle_gt = torch.atan2(sin_gt, cos_gt)

        # Wrapped angular difference → [-π, π]
        diff = angle_pred - angle_gt
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        
        # Convert to degrees
        diff_deg = diff * (180.0 / math.pi)

        return diff_deg.abs().mean()


class WingAngularLoss(nn.Module):
    """
    Wing Loss applied to reconstructed angular errors.
    Provides logarithmic gradients for sub-degree convergence.
    """

    def __init__(self, omega: float = 0.5, epsilon: float = 0.05):
        super().__init__()
        self.wing = WingLoss(omega=omega, epsilon=epsilon)

    def forward(
        self,
        sin_pred: torch.Tensor,
        cos_pred: torch.Tensor,
        sin_gt: torch.Tensor,
        cos_gt: torch.Tensor,
    ) -> torch.Tensor:
        angle_pred = torch.atan2(sin_pred, cos_pred)
        angle_gt = torch.atan2(sin_gt, cos_gt)

        # Wrapped angular difference
        diff = angle_pred - angle_gt
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        
        # Convert to degrees
        diff_deg = diff * (180.0 / math.pi)

        # Wing loss on the angular error in degrees (treat target as zero)
        return self.wing(diff_deg, torch.zeros_like(diff_deg))


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss for unit-circle normalized predictions.

    L = 1 - cos_sim(pred, gt)

    For unit vectors [sin(θ_pred), cos(θ_pred)] and [sin(θ_gt), cos(θ_gt)]:
        cos_sim = sin(θ_p)·sin(θ_g) + cos(θ_p)·cos(θ_g) = cos(θ_p - θ_g)

    So L = 1 - cos(Δθ) ≈ Δθ²/2 for small angles.

    This provides extremely smooth gradients near zero error and directly
    measures angular proximity on the unit circle.
    """

    def forward(
        self,
        pred: torch.Tensor,   # (B, 2) [sin, cos] — unit normalized
        target: torch.Tensor,  # (B, 2) [sin, cos]
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(pred, target, dim=-1)  # (B,)
        return (1.0 - cos_sim).mean()


class LandNetV2Loss(nn.Module):
    """
    Combined loss for LandNet-V2 roll/pitch regression.

    L_total = L_huber + λ_angular * L_angular + λ_wing * L_wing

    Components:
        - Huber: Robust sin/cos regression (main gradient source)
        - Angular: True angular error supervision
        - Wing: Log-scale gradients for fine-grained convergence
    """

    def __init__(
        self,
        huber_delta: float = 0.1,
        lambda_angular: float = 0.01,
        lambda_wing: float = 0.01,
        lambda_cosine: float = 0.5,
        wing_omega: float = 0.5,
        wing_epsilon: float = 0.05,
    ):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.angular_loss = AngularLoss()
        self.wing_loss = WingAngularLoss(omega=wing_omega, epsilon=wing_epsilon)
        self.cosine_loss = CosineSimilarityLoss()
        self.lambda_angular = lambda_angular
        self.lambda_wing = lambda_wing
        self.lambda_cosine = lambda_cosine

    def forward(
        self,
        roll_pred: torch.Tensor,   # (B, 2) → [sin_roll, cos_roll]
        pitch_pred: torch.Tensor,  # (B, 2) → [sin_pitch, cos_pitch]
        roll_gt: torch.Tensor,     # (B, 2) → [sin_roll, cos_roll]
        pitch_gt: torch.Tensor,    # (B, 2) → [sin_pitch, cos_pitch]
        yaw_pred: torch.Tensor = None,  # (B, 2) → [sin_yaw, cos_yaw]
        yaw_gt: torch.Tensor = None,    # (B, 2) → [sin_yaw, cos_yaw]
    ) -> dict:
        """
        Returns:
            dict with 'total', 'huber', 'angular', 'wing', 'cosine' loss values
            and per-angle MAE in degrees
        """
        # Force FP32 for all loss computations (precision-critical)
        roll_pred = roll_pred.float()
        pitch_pred = pitch_pred.float()
        roll_gt = roll_gt.float()
        pitch_gt = pitch_gt.float()

        # Unpack sin/cos
        sin_r_pred, cos_r_pred = roll_pred[:, 0], roll_pred[:, 1]
        sin_p_pred, cos_p_pred = pitch_pred[:, 0], pitch_pred[:, 1]
        sin_r_gt, cos_r_gt = roll_gt[:, 0], roll_gt[:, 1]
        sin_p_gt, cos_p_gt = pitch_gt[:, 0], pitch_gt[:, 1]

        # 1. Huber Loss on sin/cos values
        L_huber = (
            self.huber(sin_r_pred, sin_r_gt)
            + self.huber(cos_r_pred, cos_r_gt)
            + self.huber(sin_p_pred, sin_p_gt)
            + self.huber(cos_p_pred, cos_p_gt)
        )

        # 2. Angular Loss (actual angle error in radians)
        L_angular_roll = self.angular_loss(sin_r_pred, cos_r_pred, sin_r_gt, cos_r_gt)
        L_angular_pitch = self.angular_loss(sin_p_pred, cos_p_pred, sin_p_gt, cos_p_gt)
        L_angular = L_angular_roll + L_angular_pitch

        # 3. Wing Loss (fine-grained convergence for small errors)
        L_wing_roll = self.wing_loss(sin_r_pred, cos_r_pred, sin_r_gt, cos_r_gt)
        L_wing_pitch = self.wing_loss(sin_p_pred, cos_p_pred, sin_p_gt, cos_p_gt)
        L_wing = L_wing_roll + L_wing_pitch

        # 4. Cosine Similarity Loss (unit circle angular proximity)
        L_cosine = self.cosine_loss(roll_pred, roll_gt) + self.cosine_loss(pitch_pred, pitch_gt)

        # Yaw losses (optional)
        L_angular_yaw = torch.tensor(0.0, device=roll_pred.device)
        if yaw_pred is not None and yaw_gt is not None:
            yaw_pred = yaw_pred.float()
            yaw_gt = yaw_gt.float()
            sin_y_pred, cos_y_pred = yaw_pred[:, 0], yaw_pred[:, 1]
            sin_y_gt, cos_y_gt = yaw_gt[:, 0], yaw_gt[:, 1]

            L_huber = L_huber + self.huber(sin_y_pred, sin_y_gt) + self.huber(cos_y_pred, cos_y_gt)
            L_angular_yaw = self.angular_loss(sin_y_pred, cos_y_pred, sin_y_gt, cos_y_gt)
            L_angular = L_angular + L_angular_yaw
            L_wing = L_wing + self.wing_loss(sin_y_pred, cos_y_pred, sin_y_gt, cos_y_gt)
            L_cosine = L_cosine + self.cosine_loss(yaw_pred, yaw_gt)

        # Combined
        L_total = (
            L_huber
            + self.lambda_angular * L_angular
            + self.lambda_wing * L_wing
            + self.lambda_cosine * L_cosine
        )

        result = {
            "total": L_total,
            "huber": L_huber.detach(),
            "angular": L_angular.detach(),
            "wing": L_wing.detach(),
            "cosine": L_cosine.detach(),
            # Per-angle errors in degrees for logging (L_angular is now already in degrees)
            "roll_mae_deg": L_angular_roll.detach(),
            "pitch_mae_deg": L_angular_pitch.detach(),
        }

        if yaw_pred is not None:
            result["yaw_mae_deg"] = L_angular_yaw.detach()

        return result

