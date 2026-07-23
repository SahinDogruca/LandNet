"""
LandNet-V2 Configuration
========================
All hyperparameters and settings for training, model architecture, and evaluation.
"""
import math


# =============================================================================
# Model Architecture
# =============================================================================
INPUT_SIZE = 768

# CNN Backbone (ConvNeXt-V2-Base)
CNN_BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k"
CNN_CHANNELS = [128, 256, 512, 1024]  # Stage output channels

# Transformer Backbone (DeiT-III-Base)
TRANS_BACKBONE = "deit3_base_patch16_384.fb_in22k_ft_in1k"
TRANS_EMBED_DIM = 768
TRANS_DEPTH = 12
TRANS_NUM_HEADS = 12
TRANS_PATCH_SIZE = 16

# FIB (Feature Interactive Block)
FIB_CROSS_ATTN_HEADS = 8
FIB_SPATIAL_SIZE = INPUT_SIZE // TRANS_PATCH_SIZE  # 48 for 768, 64 for 1024
FIB_DROPOUT = 0.1

# ACFB (Attentional ConvTrans Fusion Block)
ACFB_REDUCTION = 16  # ECA-Net reduction ratio

# Regression Heads
HEAD_HIDDEN_DIM = 2048
HEAD_DROPOUT = 0.3
UNIT_CIRCLE_NORMALIZE = True  # Normalize output to sin²+cos²=1

# Multi-Scale Feature Aggregation
MULTI_SCALE_ACFB = True  # Use all 4 stages for ACFB, not just the last

# Angle Configuration
NUM_ANGLES = 3  # 2=roll+pitch only, 3=roll+pitch+yaw

# AMP Precision Control
AMP_HEAD_FP32 = True  # Force FP32 for regression heads even under AMP

# =============================================================================
# Training
# =============================================================================
EPOCHS = 300
BATCH_SIZE = 8           # per GPU (reduced for 768×768 input)
GRAD_ACCUM_STEPS = 2     # effective batch = 8 * num_gpus * 4
NUM_WORKERS = 4

# Optimizer
LR_BACKBONE = 1e-4
LR_HEAD = 5e-4
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.999)

# Scheduler
WARMUP_EPOCHS = 5
MIN_LR = 1e-6

# EMA
EMA_DECAY = 0.9999

# SWA
SWA_START_EPOCH = 280    # Last 20 epochs
SWA_LR = 1e-5

# Gradient clipping
MAX_GRAD_NORM = 1.0

# Mixed Precision
USE_AMP = True

# =============================================================================
# Loss Function
# =============================================================================
LAMBDA_ANGULAR = 0.01    # Weight for angular loss (scaled down as loss is now in degrees)
LAMBDA_WING = 0.01       # Weight for wing loss (scaled down as loss is now in degrees)
LAMBDA_COSINE = 0.5      # Weight for cosine similarity loss (unit circle)

# Wing Loss parameters (in degrees)
WING_OMEGA = 0.5         # 0.5 degrees
WING_EPSILON = 0.05      # 0.05 degrees

# Huber Loss delta
HUBER_DELTA = 0.1

# =============================================================================
# Data Augmentation (angle-safe only)
# =============================================================================
COLOR_JITTER_BRIGHTNESS = 0.3
COLOR_JITTER_CONTRAST = 0.3
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.1
GAUSSIAN_BLUR_KERNEL = (3, 7)
GAUSSIAN_BLUR_P = 0.3
RANDOM_ERASING_P = 0.2
NOISE_STD = 0.01

# ImageNet normalization (pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# TTA (Test Time Augmentation)
# =============================================================================
TTA_CROPS = 5           # center + 4 corners
TTA_CROP_RATIO = 0.9    # crop 90% of image, then resize back

# =============================================================================
# Paths
# =============================================================================
DATASET_NAME = "DEEL-AI/LARD_V2"
DATASET_CONFIG = "xplane"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# =============================================================================
# Validation
# =============================================================================
VAL_SPLIT_RATIO = 0.1   # 10% of train for validation if no val split exists
SEED = 42
