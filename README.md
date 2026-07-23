# LandNet-V2 ✈️

State-of-the-art aircraft landing pose estimation (Roll, Pitch, Yaw) using a dual-branch neural architecture. Designed to achieve **0.01° accuracy** for autonomous landing systems.

## 🌟 Key Features

*   **Dual-Branch Architecture**: Fuses local spatial features (ConvNeXt-V2) with global contextual features (DeiT-III) through a Cross-Attention Feature Integration Block (FIB).
*   **Multi-Scale Aggregation**: Extracts and pools features from all 4 backbone stages instead of just the last layer, retaining both fine-grained runway details and high-level semantics.
*   **Independent 3-Angle FP32 Regression**: Dedicated deep residual MLP heads for Roll, Pitch, and Yaw. Outputs are strictly constrained to the unit circle (`sin²θ + cos²θ = 1`). Computations are strictly isolated in `float32` to prevent sub-degree (`0.01°`) truncation errors inherent in FP16/AMP.
*   **Advanced Loss Function**: A weighted combination of Huber Loss, pure Angular Loss (radians), Wing Loss (for fine-grained sub-degree convergence), and Cosine Similarity Loss.
*   **T4 GPU Optimized**: Fully utilizes Mixed Precision (AMP), Gradient Accumulation, and Gradient Checkpointing to train effectively on Kaggle/Colab 16GB GPUs without sacrificing output precision.
*   **Aspect-Ratio Preserving Crop**: Dynamically resizes 1920x1080 images based on the shortest edge and crops the center, preventing any distortion of runway lines or horizon.
*   **Test Time Augmentation (TTA)**: Employs 5-Crop and brightness variance ensembles during inference to push the accuracy boundaries on the test set.

## 🛠️ Installation & Setup

We recommend using a virtual environment (e.g., Anaconda or `venv`). 

```bash
# Clone the repository
git clone <your-repo-url>
cd Bitirme/landnet_v2

# Install dependencies
pip install -r requirements.txt
```

> **Note on CUDA 13.2:** PyTorch currently does not publish an official `cu132` index. The `requirements.txt` targets the `cu124` build which is 100% forward-compatible and runs at peak performance on CUDA 13.2 architectures.

## 📂 Dataset

This project uses the **[DEEL-AI/LARD_V2](https://huggingface.co/datasets/DEEL-AI/LARD_V2/viewer/xplane)** dataset (XPlane config).
The code leverages the `datasets` library to automatically download and cache the dataset.

## 🚀 Usage

### 1. Training

To train the model from scratch on the default configuration (768x768 input, 300 epochs):

```bash
python -m landnet_v2.train --batch_size 4 --grad_accum 16
```

*   Logs will be saved to `logs/train.log`
*   Checkpoints and the EMA/SWA weights will be saved to `checkpoints/`

### 2. Evaluation & TTA

To evaluate the best model checkpoint on the test set:

```bash
python -m landnet_v2.evaluate --checkpoint checkpoints/best_model.pth
```

To evaluate using **Test Time Augmentation (TTA)** for maximum accuracy:

```bash
python -m landnet_v2.evaluate --checkpoint checkpoints/best_model.pth --tta
```

To evaluate the Stochastic Weight Averaging (SWA) model:

```bash
python -m landnet_v2.evaluate --checkpoint checkpoints/swa_model.pth --swa --tta
```

## 🏗️ Architecture Details

*   **Backbone 1**: ConvNeXt-V2 Base (ImageNet pre-trained)
*   **Backbone 2**: DeiT-III Base (ImageNet pre-trained)
*   **FIB**: 4-Stage Cross-Attention Feature Integration Block
*   **ACFB**: Asymmetric Convolutional Feature Block for final feature fusion
*   **Heads**: `[GAP] -> [FC -> LayerNorm -> GELU -> Dropout] x 2 -> [sin(θ), cos(θ)]` (Independent for Roll, Pitch, Yaw)

## 📊 Evaluation Metrics

The evaluation script reports the following metrics for Roll, Pitch, and Yaw:
*   **MAE**: Mean Absolute Error (Degrees)
*   **RMSE**: Root Mean Square Error
*   **Median Error**
*   **P99 Error**: 99th percentile error
*   **Error Distribution**: Percentage of predictions falling under strict thresholds (e.g., `< 0.01°`, `< 0.05°`).
