"""Configuration parsing for training scripts"""

import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration - DinoV3 with LoRA enabled by default"""
    model_size: str = "large_dinov3"  # Default to large DinoV3
    use_neutral_reference: bool = False
    multi_shot_inference: int = 1
    use_binary_classification_head: bool = False
    # Note: use_au_query_head and use_lora are now always enabled in the model


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 1e-4  # Fixed: always 1e-4
    weight_decay: float = 1e-1  # Fixed: always 1e-1
    freeze_backbone_epochs: int = 9999  # Fixed: always frozen (only train LoRA adapters)
    precision: int = 16  # Fixed: always 16-bit
    log_every_n_steps: int = 10
    random_seed: Optional[int] = 42
    lora_rank: int = 8  # LoRA is always enabled
    lora_alpha: int = 16
    synthetic_pretrained_checkpoint: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "datasets/UNBC-McMaster"
    fold: int = 0
    cv_protocol: str = "5fold"
    cross_validation: bool = False
    split_csv: Optional[str] = None
    use_entire_dataset: bool = False
    synthetic_data_dir: Optional[str] = None
    use_weighted_sampling: bool = False


@dataclass
class LossConfig:
    """Loss configuration"""
    au_loss_weight: float = 0.1  # Fixed: always 0.1
    pspi_loss_weight: float = 1.0  # Fixed: always 1.0
    use_contrastive_loss: bool = False
    contrastive_loss_weight: float = 0.1


@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str = "experiment/unbc_pspi_vit"
    resume_from_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    evaluate_only: bool = False


@dataclass
class WandbConfig:
    """Weights & Biases configuration"""
    wandb_project: str = "unbc-pspi-pain-regression"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class UNBCConfig:
    """UNBC-specific configuration"""
    dropout_rate: float = 0.5
    minority_class_weight: float = 1.0
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    use_focal_loss: bool = False
    strong_augment_prob: float = 0.0
    minority_class_oversample: bool = False
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val/regression/mae"
    calibrate_pain_threshold: bool = False
    threshold_range_min: float = 0.0
    threshold_range_max: float = 6.0
    num_thresholds: int = 200
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-1
    freeze_backbone_epochs: int = 5
    inner_validation: str = "last_epoch"
    report_best_validation: bool = False


@dataclass
class Config:
    """Main configuration container"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    unbc: UNBCConfig = field(default_factory=UNBCConfig)


def parse_args():
    """Parse command line arguments and return config object"""
    parser = argparse.ArgumentParser(description="Train pain assessment model with DinoV3 + LoRA")
    
    # Model args
    parser.add_argument("--model_size", type=str, default="large_dinov3",
                       choices=["small_dinov3", "base_dinov3", "large_dinov3"],
                       help="DinoV3 model size (default: large_dinov3)")
    parser.add_argument("--use_neutral_reference", action="store_true",
                       help="Use neutral reference images")
    parser.add_argument("--lora_rank", type=int, default=8,
                       help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha (default: 16)")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size per GPU (default: 32)")
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum training epochs (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-1,
                       help="Weight decay (default: 1e-1)")
    
    # Data args
    parser.add_argument("--data_dir", type=str, default="datasets/UNBC-McMaster",
                       help="Path to dataset directory")
    parser.add_argument("--fold", type=int, default=0,
                       help="Cross-validation fold (0-4)")
    
    # Output args
    parser.add_argument("--output_dir", type=str, default="experiment/unbc_pspi_vit",
                       help="Output directory for checkpoints and logs")
    
    # Wandb args
    parser.add_argument("--wandb_project", type=str, default="unbc-pspi-pain-regression",
                       help="Weights & Biases project name")
    
    args = parser.parse_args()
    
    # Create config from args
    cfg = Config()
    for key, value in vars(args).items():
        # Find which sub-config this belongs to
        for attr_name in ['model', 'training', 'data', 'loss', 'output', 'wandb', 'unbc']:
            sub_cfg = getattr(cfg, attr_name)
            if hasattr(sub_cfg, key):
                setattr(sub_cfg, key, value)
                break
    
    return cfg


def config_to_dict(cfg):
    """Convert config object to dictionary"""
    result = {}
    for attr_name in ['model', 'training', 'data', 'loss', 'output', 'wandb', 'unbc']:
        sub_cfg = getattr(cfg, attr_name)
        for key, value in vars(sub_cfg).items():
            result[f"{attr_name}_{key}"] = value
    return result
