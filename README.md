# Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

- [Visual Overview](#visual-overview)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Citation](#citation)

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

### Stage 1: Pretrain on Synthetic 3DPain Dataset

Pretrain the ViTPain model on synthetic 3D pain faces with a train/val/test split:

```bash
python train_vitpain.py \
    --data_dir datasets/pain_faces \
    --split_csv data/splits/uniform_data_70_20_10_split.csv \
    --model_size large_dinov3 \
    --batch_size 48 \
    --max_epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-1 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --log_every_n_steps 100 \
    --output_dir experiment/vitpain_pretrain \
    --wandb_project vitpain-pretrain
```

The best checkpoint (lowest `val/regression/mae`) is saved to `experiment/vitpain_pretrain/checkpoints/`.

### Stage 2: 5-Fold Cross-Validation on UNBC-McMaster

Run 5-fold CV on UNBC using the pretrained checkpoint from Stage 1:

```bash
python scripts/run_unbc_5fold_cv.py \
    --pretrained_checkpoint experiment/vitpain_pretrain/checkpoints/<best_checkpoint>.ckpt \
    --data_dir datasets/UNBC-McMaster \
    --model_size large_dinov3 \
    --batch_size 100 \
    --max_epochs 85 \
    --use_weighted_sampling \
    --lora_rank 8 \
    --lora_alpha 16 \
    --log_every_n_steps 10 \
    --wandb_project unbc-5fold-cv
```

Replace `<best_checkpoint>.ckpt` with the actual filename from Stage 1 (e.g. `vitpain-epoch=68-val_regression_mae=1.457.ckpt`).

### Train on UNBC-McMaster (Single Fold)

To train a single fold (e.g. fold 0) with a pretrained checkpoint:

```bash
python train_unbc.py \
    --data_dir datasets/UNBC-McMaster \
    --synthetic_pretrained_checkpoint experiment/vitpain_pretrain/checkpoints/<best_checkpoint>.ckpt \
    --model_size large_dinov3 \
    --fold 0 \
    --batch_size 100 \
    --max_epochs 85 \
    --use_weighted_sampling \
    --output_dir experiment/unbc_fold0 \
    --wandb_project unbc-training
```

### SLURM (Compute Canada)

On Compute Canada clusters, prefix commands with `srun` and request GPU resources:

```bash
srun python -u train_vitpain.py ...    # Stage 1
srun python -u scripts/run_unbc_5fold_cv.py ...  # Stage 2
```

### Key Training Arguments

#### Model Arguments:
- `--model_size`: DinoV3 model size (default: `large_dinov3`)
  - Options: `small_dinov3`, `base_dinov3`, `large_dinov3`
- `--use_neutral_reference`: Use neutral reference images
- `--multi_shot_inference`: Number of neutral refs for multi-shot inference (default: 1)
- `--lora_rank`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha (default: 16)


#### Training Arguments:
- `--batch_size`: Batch size per GPU (default: 32)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-1)
- `--precision`: Training precision (default: 16)
- `--synthetic_pretrained_checkpoint`: Path to pretrained checkpoint from Stage 1
- `--dropout_rate`: Dropout rate for UNBC fine-tuning (default: 0.5)

#### Data Arguments:
- `--data_dir`: Path to dataset directory
- `--fold`: Cross-validation fold (0-4 for UNBC)
- `--split_csv`: Path to CSV with train/val/test splits (for synthetic data)
- `--use_weighted_sampling`: Handle class imbalance

#### Output Arguments:
- `--output_dir`: Directory for checkpoints and logs
- `--resume_from_checkpoint`: Resume training from a Lightning checkpoint
- `--wandb_project`: Weights & Biases project name
- `--wandb_group`: Group name for organizing runs
- `--run_name`: Custom run name

## 📁 Project Structure

```
PainGeneration_clean/
├── data/                          # Data loaders
│   ├── unbc_loader.py            # UNBC-McMaster dataset loader
│   ├── pain3d_loader.py          # 3D synthetic pain face dataset loader
│   ├── split_utils.py            # Split utilities
│   └── utils/
│       └── threshold_calibration.py # Binary threshold calibration
├── lib/                           # Library code
│   ├── models/                   # Model definitions
│   │   ├── vitpain.py               # ViTPain model
│   │   └── pspi_evaluator_mixin.py  # Evaluation metrics
│   └── utils/                    # Utility functions
├── scripts/                       # Training scripts
│   └── run_unbc_5fold_cv.py      # 5-fold cross-validation
├── train_unbc.py                 # Train on UNBC-McMaster
├── train_vitpain.py              # Train on synthetic data
├── configs/                      # Configuration management
├── requirements.txt              # Python dependencies
└── README.md                     # This file
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
- AUROC and AUPR

## 🔗 Citation

If you use this code or the 3DPain dataset in your research, please cite:

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
