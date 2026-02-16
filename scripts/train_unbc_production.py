#!/usr/bin/env python3
"""
Train a production-ready ViTPain model on the entire UNBC dataset.

This script loads pretrained weights from Pain3D synthetic data training,
then fine-tunes on ALL UNBC data (no cross-validation splits).

Use this to create a final model for production deployment.

Usage:
  python scripts/train_unbc_production.py \
    --synthetic_pretrained_checkpoint experiment/vitpain_pretrain_neutralref_v3/checkpoints/best.ckpt \
    --output_dir experiment/unbc_production \
    --max_epochs 50
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from data.unbc_loader import UNBCDataModule
from lib.models import create_vitpain_model
from lib.models.vitpain import load_pretrained_synthetic_data_model
from configs import parse_args, config_to_dict


def train_unbc_production(cfg):
    """
    Train PSPI ViT model on entire UNBC dataset (production mode).
    
    This is similar to train_unbc() but uses fold=None to train on ALL data.
    """
    if cfg.training.random_seed is not None:
        pl.seed_everything(cfg.training.random_seed, workers=True)

    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    os.makedirs(cfg.output.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  UNBC Production Training (ALL DATA)")
    print(f"{'='*80}")
    print(f"  Using {gpus} GPU(s)")
    print(f"  Output dir: {cfg.output.output_dir}")
    print(f"  Pretrained checkpoint: {cfg.training.synthetic_pretrained_checkpoint}")
    print(f"{'='*80}\n")

    num_workers = min(4, os.cpu_count() or 1)
    data_module = UNBCDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
        image_size=224,
        fold=None,  # IMPORTANT: None = use ALL data for production
        cv_protocol=cfg.data.cv_protocol,
        return_aus=True,
        pin_memory=torch.cuda.is_available(),
        use_neutral_reference=cfg.model.use_neutral_reference,
        multi_shot_inference=cfg.model.multi_shot_inference,
        use_weighted_sampling=cfg.data.use_weighted_sampling,
    )
    data_module.setup()

    print(f"Dataset statistics:")
    print(f"  Total training samples: {len(data_module.train_dataloader().dataset)}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Neutral reference: {cfg.model.use_neutral_reference}")
    print(f"  Weighted sampling: {cfg.data.use_weighted_sampling}\n")

    model_kwargs = dict(
        model_size=cfg.model.model_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
        au_loss_weight=cfg.loss.au_loss_weight,
        pspi_loss_weight=cfg.loss.pspi_loss_weight,
        lora_rank=cfg.training.lora_rank,
        lora_alpha=cfg.training.lora_alpha,
        use_neutral_reference=cfg.model.use_neutral_reference,
        dropout_rate=cfg.unbc.dropout_rate,
    )

    if cfg.training.synthetic_pretrained_checkpoint:
        print(f"Loading pretrained weights from: {cfg.training.synthetic_pretrained_checkpoint}")
        model = load_pretrained_synthetic_data_model(
            checkpoint_path=cfg.training.synthetic_pretrained_checkpoint,
            **model_kwargs,
        )
    else:
        model = create_vitpain_model(**model_kwargs)

    checkpoint_dir = os.path.join(cfg.output.output_dir, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="pspi-vit-production-{epoch:02d}-{val_regression_corr:.3f}",
            monitor="val/regression/corr",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="pspi-vit-production-{epoch:02d}-{val_regression_mae:.3f}-mae",
            monitor="val/regression/mae",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    config_dict = config_to_dict(cfg)
    config_dict["effective_batch_size"] = cfg.training.batch_size * max(1, gpus)
    config_dict["gpu_count"] = gpus
    config_dict["training_mode"] = "production (all data)"

    run_name = cfg.wandb.run_name or "unbc_production"
    logger = WandbLogger(
        project=cfg.wandb.wandb_project,
        entity=cfg.wandb.wandb_entity,
        name=run_name,
        group=cfg.wandb.wandb_group,
        save_dir=cfg.output.output_dir,
        log_model=False,
        config=config_dict,
        settings=wandb.Settings(save_code=False, _disable_meta=True),
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=gpus if gpus > 0 else 1,
        strategy="ddp_find_unused_parameters_true" if gpus > 1 else "auto",
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
    )

    print(f"Starting training for {cfg.training.max_epochs} epochs...\n")

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=cfg.output.resume_from_checkpoint,
    )

    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        print(f"\nTesting best model: {trainer.checkpoint_callback.best_model_path}")
        trainer.test(model, datamodule=data_module, ckpt_path=trainer.checkpoint_callback.best_model_path)

    print(f"\n{'='*80}")
    print(f"  Production model training complete!")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    print(f"  Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"{'='*80}")

    return trainer, model


def main():
    cfg = parse_args()
    train_unbc_production(cfg)


if __name__ == "__main__":
    main()
