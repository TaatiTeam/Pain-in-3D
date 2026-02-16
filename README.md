# Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment

<a href="https://opensource.org/licenses/MIT" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://arxiv.org/pdf/2509.16727" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Paper-arXiv%202509.16727-B31B1B.svg" alt="Paper"></a>
<a href="https://huggingface.co/datasets/SoroushMehraban/3D-Pain" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Dataset-3D--Pain-blue" alt="Dataset"></a>
<a href="https://huggingface.co/xinlei55555/ViTPain" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Model-ViTPain-blue" alt="Model"></a>

Official implementation of the paper **Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment**. This codebase supports training on both synthetic 3D pain face datasets and the UNBC-McMaster Shoulder Pain Expression Archive.

## 📄 Overview

![Sample Pain Faces](figures/preamble.png)

This implementation provides tools for automated pain assessment through:
- Reference-guided Vision Transformers (ViTPain) for pain intensity estimation
- Multi-task learning combining PSPI regression and Action Unit prediction
- Support for both synthetic and real-world pain datasets
- Comprehensive evaluation metrics including regression, classification, and correlation measures

## 🖼️ Visual Overview

### Synthetic 3D Pain Face Generation
![Data Generation Pipeline](figures/3dpain_pipeline.png)
*Controllable 3D pain face synthesis using parametric facial models with AU-based deformations*

### Model Architecture
![ViTPain Architecture](figures/architecture.png)
*Reference-guided Vision Transformer with DinoV3 backbone, LoRA adapters, and AU query head for pain assessment*

## 📋 Table of Contents

- [Visual Overview](#️-visual-overview)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Training](#-training)
  - [Download Pretrained Weights](#download-pretrained-weights)
  - [Quick Start with Shell Scripts](#quick-start-with-shell-scripts)
  - [Stage 1: Pretrain on Synthetic Data](#stage-1-pretrain-on-synthetic-3dpain-dataset-optional)
  - [Stage 2: 5-Fold Cross-Validation](#stage-2-5-fold-cross-validation-on-unbc-mcmaster)
  - [Stage 3: Production Training](#stage-3-production-training-optional)
  - [Evaluation](#evaluation)
- [Model Architecture](#-model-architecture)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [License](#-license)

## 🚀 Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv pain_env
source pain_env/bin/activate  # On Windows: pain_env\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies:
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- Transformers (for ViT models)
- timm (for DinoV3 models)
- wandb (for experiment tracking)
- huggingface-hub (for downloading pretrained weights)

## 💾 Dataset Setup

The code expects datasets to be organized in a `datasets/` directory at the project root:

```
PainGeneration_clean/
├── datasets/
│   ├── UNBC-McMaster/          # UNBC-McMaster dataset
│   │   ├── frames_unbc_2020-09-21-05-42-04.hdf5
│   │   ├── annotations_unbc_2020-10-13-22-55-04.hdf5
│   │   └── UNBC_CVFolds_2019-05-16-15-16-36.hdf5
│   └── pain_faces/              # Synthetic 3D pain faces
│       ├── meshes_inpainted/    # RGB images
│       └── annotations/         # JSON annotations
├── data/
├── lib/
├── scripts/
└── ...
```

### Download UNBC-McMaster Dataset

The UNBC-McMaster Shoulder Pain Expression Archive Dataset should be obtained from the official source and converted to HDF5 format with the following files:
- `frames_unbc_2020-09-21-05-42-04.hdf5` - Face image frames
- `annotations_unbc_2020-10-13-22-55-04.hdf5` - AU annotations and PSPI scores
- `UNBC_CVFolds_2019-05-16-15-16-36.hdf5` - Cross-validation fold splits

### Download Synthetic Dataset

The 3D Pain synthetic dataset is available on Hugging Face:

```bash
# Using git-lfs
git lfs install
git clone https://huggingface.co/datasets/SoroushMehraban/3D-Pain datasets/pain_faces

# Or using the Hugging Face datasets library
python -c "from datasets import load_dataset; load_dataset('SoroushMehraban/3D-Pain')"
```

## 🎓 Training

Training follows a two-stage pipeline: (1) pretrain on synthetic 3DPain data, then (2) fine-tune on UNBC-McMaster using the pretrained checkpoint.

### Download Pretrained Weights

We provide pretrained weights on Hugging Face that were trained on the 3D-Pain synthetic dataset:

```python
from huggingface_hub import hf_hub_download

# Download pretrained checkpoint
checkpoint_path = hf_hub_download(
    repo_id="xinlei55555/ViTPain",
    filename="vitpain-epoch=141-val_regression_mae=1.859.ckpt",
    cache_dir="./checkpoints"
)
```

Or download via command line:
```bash
pip install huggingface-hub
huggingface-cli download xinlei55555/ViTPain \
    vitpain-epoch=141-val_regression_mae=1.859.ckpt \
    --local-dir ./experiment/vitpain_pretrain/checkpoints/
```

**Model Card**: [https://huggingface.co/xinlei55555/ViTPain](https://huggingface.co/xinlei55555/ViTPain)

### Quick Start with Shell Scripts

```bash
# Option A: Download pretrained weights (recommended, see above)
# Option B: Train from scratch on synthetic data
./scripts/train_synthetic_pretrain.sh

# Then: 5-fold cross-validation on UNBC
./scripts/train_unbc_5fold.sh

# Evaluate results
python scripts/evaluate_unbc.py experiment/unbc_5fold_cv
```

### Stage 1: Pretrain on Synthetic 3DPain Dataset (Optional)

**Skip this step if you downloaded pretrained weights above.**

Otherwise, pretrain the ViTPain model on synthetic 3D pain faces:

```bash
python train_vitpain.py \
    --data_dir datasets/pain_faces \
    --split_csv data/splits/uniform_data_70_20_10_split.csv \
    --model_size large_dinov3 \
    --batch_size 48 \
    --max_epochs 150 \
    --learning_rate 1e-4 \
    --weight_decay 1e-1 \
    --au_loss_weight 1.0 \
    --pspi_loss_weight 1.0 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --use_neutral_reference \
    --output_dir experiment/vitpain_pretrain
```

The best checkpoint is saved to `experiment/vitpain_pretrain/checkpoints/`.

### Stage 2: 5-Fold Cross-Validation on UNBC-McMaster

Run 5-fold CV on UNBC using a pretrained checkpoint (either downloaded or trained from scratch):

```bash
# Using downloaded Hugging Face checkpoint
python scripts/run_unbc_5fold_cv.py \
    --pretrained_checkpoint ./checkpoints/vitpain-epoch=141-val_regression_mae=1.859.ckpt \
    --data_dir datasets/UNBC-McMaster \
    --model_size large_dinov3 \
    --batch_size 100 \
    --max_epochs 50 \
    --au_loss_weight 0.1 \
    --use_weighted_sampling \
    --use_neutral_reference \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir experiment/unbc_5fold_cv

# Or using checkpoint trained from scratch in Stage 1
python scripts/run_unbc_5fold_cv.py \
    --pretrained_checkpoint experiment/vitpain_pretrain/checkpoints/best.ckpt \
    --data_dir datasets/UNBC-McMaster \
    --model_size large_dinov3 \
    --batch_size 100 \
    --max_epochs 50 \
    --au_loss_weight 0.1 \
    --use_weighted_sampling \
    --use_neutral_reference \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir experiment/unbc_5fold_cv
```

### Stage 3: Production Training (Optional)

Train a final model on ALL UNBC data for deployment (use downloaded or trained checkpoint):

```bash
# Using downloaded Hugging Face checkpoint
python scripts/train_unbc_production.py \
    --synthetic_pretrained_checkpoint ./checkpoints/vitpain-epoch=141-val_regression_mae=1.859.ckpt \
    --data_dir datasets/UNBC-McMaster \
    --batch_size 100 \
    --max_epochs 50 \
    --au_loss_weight 0.1 \
    --use_neutral_reference \
    --use_weighted_sampling \
    --output_dir experiment/unbc_production
```

### Evaluation

Evaluate 5-fold cross-validation results:

```bash
python scripts/evaluate_unbc.py experiment/unbc_5fold_cv
```

With multi-shot inference (averages predictions across N neutral references):
```bash
python scripts/evaluate_unbc.py experiment/unbc_5fold_cv \
    --use_neutral_reference --multi_shot_inference 3
```

Key metrics reported:
- **Pearson Correlation**: Mean across folds with 95% CI
- **AUROC**: At PSPI thresholds 1, 2, 3 (mean across folds)
- **Train-Calibrated F1**: Threshold tuned on train set, applied to test set
  - Most realistic evaluation for production use
  - Reported at PSPI thresholds 1, 2, 3 and macro average

For detailed metrics (test-calibrated F1, uncalibrated F1, combined metrics), use:
```bash
python scripts/evaluate_unbc_verbose.py experiment/unbc_5fold_cv
```

### Training Arguments

#### Model:
- `--model_size`: `small_dinov3`, `base_dinov3`, or `large_dinov3`
- `--use_neutral_reference`: Enable neutral reference images
- `--multi_shot_inference`: Number of neutral refs for ensemble (default: 1)
- `--lora_rank`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha (default: 16)

#### Training:
- `--batch_size`: Batch size per GPU (default: 48 for synthetic, 100 for UNBC)
- `--max_epochs`: Maximum epochs (default: 150 for synthetic, 50 for UNBC)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-1)
- `--precision`: 16 or 32 (default: 16)
- `--au_loss_weight`: Weight for AU prediction loss (default: 1.0)
- `--pspi_loss_weight`: Weight for PSPI regression loss (default: 1.0)

#### Data:
- `--data_dir`: Path to dataset
- `--fold`: CV fold for UNBC (0-4)
- `--split_csv`: Train/val/test split CSV (for synthetic data)
- `--use_weighted_sampling`: Handle class imbalance

#### Output:
- `--output_dir`: Checkpoint and log directory
- `--wandb_project`: W&B project name
- `--run_name`: Custom run name

### Directory Structure After Training

```
experiment/
├── vitpain_pretrain/              # Stage 1: Pretrained model
│   └── checkpoints/
│       └── best.ckpt
├── unbc_5fold_cv/                 # Stage 2: Cross-validation
│   ├── fold_0/
│   │   └── checkpoints/
│   ├── fold_1/
│   ├── ...
│   └── combined_evaluation_results_corr.txt
└── unbc_production/               # Stage 3: Production model
    └── checkpoints/
        └── best.ckpt
```

## 📁 Project Structure

```
PainGeneration_clean/
├── data/                                  # Data loaders
│   ├── unbc_loader.py                    # UNBC-McMaster dataset loader
│   ├── pain3d_loader.py                  # 3D synthetic pain face dataset loader
│   └── split_utils.py                    # Split utilities
├── lib/                                   # Library code
│   └── models/                           # Model definitions
│       ├── vitpain.py                    # ViTPain model
│       └── pspi_evaluator_mixin.py       # Evaluation metrics
├── scripts/                               # Training & evaluation scripts
│   ├── train_synthetic_pretrain.sh       # Pretrain on synthetic data
│   ├── train_unbc_5fold.sh              # 5-fold cross-validation
│   ├── train_unbc_production_simple.sh  # Production training
│   ├── run_unbc_5fold_cv.py             # 5-fold CV Python script
│   ├── train_unbc_production.py         # Production training script
│   ├── evaluate_unbc.py                 # Main evaluation (clean output)
│   └── evaluate_unbc_verbose.py         # Verbose evaluation (all metrics)
├── configs/                               # Configuration management
│   └── __init__.py                       # Config dataclasses and parser
├── train_vitpain.py                      # Train on synthetic data
├── train_unbc.py                         # Train on UNBC-McMaster
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## 🧠 Model Architecture

### ViTPain

ViTPain is a reference-based Vision Transformer designed for pain assessment:

- **Backbone**: DinoV3 vision transformer (always enabled)
- **Fine-tuning**: LoRA adapters for efficient training (always enabled)
- **AU Query Head**: Cross-attention with learnable queries for AU prediction (always enabled)
- **Input**: Target (pain) face + optional neutral reference face
- **Outputs**: 
  - PSPI score (0-16 regression)
  - Action Unit intensities (AU4, AU6, AU7, AU9, AU10, AU43)

### Key Features:
1. **DinoV3 Backbone**: State-of-the-art vision features
2. **LoRA Fine-tuning**: Memory-efficient training with adapters
3. **AU Query Head**: Attention-based AU prediction
4. **Multi-task Learning**: Joint prediction of PSPI and AUs
5. **Multi-Shot Inference**: Ensemble predictions with multiple neutral references (optional)

## 📊 Evaluation Metrics

The code reports the following core metrics:

### Regression Metrics:
- Mean Absolute Error (MAE)
- Pearson Correlation (Corr)

### Classification Metrics:
- Binary F1, Precision, Recall (pain vs. no-pain)
- AUROC

## 🔗 Citation

If you use this code, pretrained weights, or the 3DPain dataset in your research, please cite:

```bibtex
@article{lin2025pain,
  title={Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment},
  author={Lin, Xin Lei and Mehraban, Soroush and Moturu, Abhishek and Taati, Babak},
  journal={arXiv preprint arXiv:2509.16727},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License.
