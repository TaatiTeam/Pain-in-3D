# Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment

[![arXiv](https://img.shields.io/badge/arXiv-2509.16727-b31b1b.svg)](https://arxiv.org/abs/2509.16727)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/SoroushMehraban/3D-Pain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation and dataset for the paper **"Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment"**.

## 📄 Abstract

Automated pain assessment from facial expressions is crucial for non-communicative patients but has been limited by data scarcity and ethical constraints. We present **3DPain**, a large-scale synthetic dataset designed to address the scarcity and demographic imbalance in automated pain assessment. 

The dataset features **82,500 samples across 2,500 synthetic identities**, generated using a novel three-stage framework that ensures precise control over facial action units (AUs), facial structure, and clinically validated pain levels (PSPI).

Key features include:
* **Size:** 82,500 images across 2,500 synthetic identities.
* **Diversity:** Balanced representation across age, gender, and ethnicity.
* **Annotations:** Exact AU configurations, PSPI scores, and paired neutral references for every pain expression.

We also introduce **ViTPain**, a reference-based Vision Transformer framework. Unlike standard baselines, ViTPain leverages **cross-attention with a neutral reference face** to explicitly disentangle dynamic pain expressions from static identity features, significantly improving estimation accuracy.

## 💾 Dataset

The dataset is hosted on Hugging Face: [**SoroushMehraban/3D-Pain**](https://huggingface.co/datasets/SoroushMehraban/3D-Pain)

### Directory Structure
The dataset is organized into four main directories:

    3D-Pain/
    ├── heatmaps/           # 25k colorized pain-region heatmaps
    │   ├── README.txt
    │   ├── heatmaps_00000.tar.gz
    │   └── ...
    ├── textured_meshes/    # 2.5k textured FLAME meshes
    │   ├── README.txt
    │   ├── meshes_00000.tar.gz
    │   └── ...
    ├── images/             # 82.5k Multi-view images
    │   │                   # (27.5k sets: Neutral + Pain in 3 views)
    │   ├── README.txt
    │   ├── images_00000.tar.gz
    │   └── ...
    └── annotations/        # 25k JSON annotations (AUs, PSPI scores)
        ├── README.txt
        ├── annotations_00000.tar.gz
        └── ...

### Annotations Format
Each annotation JSON contains:
- **AUs:** Intensity of specific Action Units (e.g., AU4, AU6, AU7, AU9, AU10, AU43).
- **PSPI:** The calculated Prkachin and Solomon Pain Intensity score.

## 🚀 Usage

### 1. Downloading the Data
You can download the dataset using the `huggingface_hub` python library or via git.

**Using Python:**

    from datasets import load_dataset

    dataset = load_dataset("SoroushMehraban/3D-Pain")

**Using Git:**

    git lfs install
    git clone https://huggingface.co/datasets/SoroushMehraban/3D-Pain

### 2. Training (ViTPain)
To train the Vision Transformer baseline (ViTPain) described in the paper:

    python train.py --config configs/vit_pain.yaml --batch_size 32

## 🧠 Methodology

### 1. 3DPain Generation Pipeline
Our generation pipeline consists of three stages:
1.  **3D Mesh Generation:** Utilizing the FLAME model to create geometrically diverse and anatomically plausible neutral face meshes.
2.  **Texture Synthesis:** Applying diffusion-based texturing to ensure photorealistic skin details and demographic diversity.
3.  **Neural Face Rigging:** Deforming the meshes using specific Action Unit (AU) parameters to simulate clinically grounded pain expressions, followed by multi-view rendering.

### 2. ViTPain Architecture
ViTPain is a **reference-based Vision Transformer** designed to improve generalization. 
* **Input:** It takes pairs of images—a target "pain" face and its corresponding "neutral" reference face.
* **Mechanism:** It uses a cross-attention mechanism to query the neutral face features, allowing the model to subtract identity-specific information and focus purely on the deformations caused by pain.
* **Outcome:** This approach yields higher performance on PSPI regression and pain classification compared to standard single-frame baselines.

## 🔗 Citation

If you use this dataset or code in your research, please cite our paper:

    @article{lin2025pain,
      title={Pain in 3D: Generating Controllable Synthetic Faces for Automated Pain Assessment},
      author={Lin, Xin Lei and Mehraban, Soroush and Moturu, Abhishek and Taati, Babak},
      journal={arXiv preprint arXiv:2509.16727},
      year={2025}
    }

## 📝 License

This project is licensed under the **MIT License**.