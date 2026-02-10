"""Configuration parsing for training scripts."""

import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_size: str = "large_dinov3"
    use_neutral_reference: bool = False
    multi_shot_inference: int = 1


@dataclass
class TrainingConfig:
    batch_size: int = 48
    max_epochs: int = 150
    learning_rate: float = 1e-4
    weight_decay: float = 1e-1
    precision: int = 16
    # Backbone is always frozen; only LoRA adapters are trained.
    log_every_n_steps: int = 10
    random_seed: Optional[int] = 42
    lora_rank: int = 8
    lora_alpha: int = 16
    synthetic_pretrained_checkpoint: Optional[str] = None


@dataclass
class DataConfig:
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
    au_loss_weight: float = 1.0
    pspi_loss_weight: float = 1.0


@dataclass
class OutputConfig:
    output_dir: str = "experiment/unbc_pspi_vit"
    resume_from_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    evaluate_only: bool = False


@dataclass
class WandbConfig:
    wandb_project: str = "unbc-pspi-pain-regression"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class UNBCConfig:
    dropout_rate: float = 0.5


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    unbc: UNBCConfig = field(default_factory=UNBCConfig)


def parse_args():
    """Parse command line arguments and return config object."""
    parser = argparse.ArgumentParser(description="Train pain assessment model with DinoV3 + LoRA")

    # Model
    parser.add_argument("--model_size", type=str, default="large_dinov3",
                       choices=["small_dinov3", "base_dinov3", "large_dinov3"])
    parser.add_argument("--use_neutral_reference", action="store_true")
    parser.add_argument("--multi_shot_inference", type=int, default=1,
                        help="Number of neutral refs for multi-shot inference (only with --use_neutral_reference)")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # Training
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--synthetic_pretrained_checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint from synthetic training")

    # Loss
    parser.add_argument("--au_loss_weight", type=float, default=1.0)
    parser.add_argument("--pspi_loss_weight", type=float, default=1.0)

    # Data
    parser.add_argument("--data_dir", type=str, default="datasets/UNBC-McMaster")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split_csv", type=str, default=None,
                        help="Path to CSV with train/val/test splits for synthetic data")
    parser.add_argument("--use_weighted_sampling", action="store_true")

    # Output
    parser.add_argument("--output_dir", type=str, default="experiment/unbc_pspi_vit")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to Lightning checkpoint to resume training from")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="unbc-pspi-pain-regression")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    # UNBC-specific
    parser.add_argument("--dropout_rate", type=float, default=0.5)

    args = parser.parse_args()

    cfg = Config()
    for key, value in vars(args).items():
        for attr_name in ['model', 'training', 'data', 'loss', 'output', 'wandb', 'unbc']:
            sub_cfg = getattr(cfg, attr_name)
            if hasattr(sub_cfg, key):
                setattr(sub_cfg, key, value)
                break

    return cfg


def config_to_dict(cfg):
    """Convert config object to flat dictionary."""
    result = {}
    for attr_name in ['model', 'training', 'data', 'loss', 'output', 'wandb', 'unbc']:
        sub_cfg = getattr(cfg, attr_name)
        for key, value in vars(sub_cfg).items():
            result[f"{attr_name}_{key}"] = value
    return result
