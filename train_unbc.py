#!/usr/bin/env python3
"""
Simplified training script for UNBC-McMaster dataset.
This is a minimal version focusing on core training functionality.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

# Optimize for Tensor Cores
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Import our modules
from data.unbc_loader import UNBCDataModule
from lib.models import create_pspi_vit_model
from configs import parse_args, config_to_dict


def train_unbc(cfg):
    """
    Train PSPI ViT model on UNBC-McMaster dataset.
    
    Args:
        cfg: Configuration object from parse_args()
    """
    # Set random seed
    if cfg.training.random_seed is not None:
        pl.seed_everything(cfg.training.random_seed, workers=True)
        print(f"Random seed: {cfg.training.random_seed}")
    
    # Auto-detect GPUs
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Auto-detected {gpus} GPU(s)")
    
    # Create output directory
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    fold_output_dir = os.path.join(cfg.output.output_dir, f"fold_{cfg.data.fold}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # Create data module
    print(f"Loading UNBC-McMaster dataset (fold {cfg.data.fold})...")
    num_workers = min(8, os.cpu_count() or 1)
    data_module = UNBCDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.training.batch_size,
        num_workers=num_workers,
        image_size=224,
        fold=cfg.data.fold,
        cv_protocol=cfg.data.cv_protocol,
        return_aus=True,
        pin_memory=torch.cuda.is_available(),
        use_neutral_reference=cfg.model.use_neutral_reference,
        multi_shot_inference=cfg.model.multi_shot_inference,
        use_weighted_sampling=cfg.data.use_weighted_sampling,
        inner_validation=cfg.unbc.inner_validation,
    )
    
    # Setup data
    data_module.setup()
    print(f"Train set size: {len(data_module.train_dataset)}")
    print(f"Val set size: {len(data_module.val_dataset)}")
    print(f"Test set size: {len(data_module.test_dataset)}")
    
    # Create model (DinoV3 + LoRA + AU query head are always enabled)
    print(f"Creating ViTPain model with DinoV3 + LoRA (size: {cfg.model.model_size})...")
    model = create_pspi_vit_model(
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
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup callbacks
    checkpoint_dir = os.path.join(fold_output_dir, "checkpoints")
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"pspi-vit-fold{cfg.data.fold}-{{epoch:02d}}-{{val_regression_corr:.3f}}",
            monitor="val/regression/corr",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    # Setup logger
    config_dict = config_to_dict(cfg)
    config_dict["effective_batch_size"] = cfg.training.batch_size * max(1, gpus)
    config_dict["gpu_count"] = gpus
    
    run_name = cfg.wandb.run_name or f"unbc_fold{cfg.data.fold}"
    logger = WandbLogger(
        project=cfg.wandb.wandb_project,
        entity=cfg.wandb.wandb_entity,
        name=run_name,
        group=cfg.wandb.wandb_group,
        save_dir=fold_output_dir,
        log_model=False,
        config=config_dict,
        settings=wandb.Settings(save_code=False, _disable_meta=True),
    )
    
    # Create trainer
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
        enable_progress_bar=False,
        enable_model_summary=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0 if cfg.unbc.inner_validation == "last_epoch" else 2,
    )
    
    # Train
    print(f"\n{'='*50}")
    print(f"Training PSPI ViT on UNBC-McMaster Dataset")
    print(f"Fold: {cfg.data.fold}")
    print(f"Model: {cfg.model.model_size}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Max epochs: {cfg.training.max_epochs}")
    print(f"{'='*50}\n")
    
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=cfg.output.resume_from_checkpoint,
    )
    
    # Test on best checkpoint
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        print(f"\nTesting best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        test_results = trainer.test(model, datamodule=data_module, ckpt_path=trainer.checkpoint_callback.best_model_path)
        print(f"Test results: {test_results}")
    
    print(f"\nTraining complete! Results saved to: {fold_output_dir}")
    
    return trainer, model


def main():
    """Main entry point"""
    cfg = parse_args()
    train_unbc(cfg)


if __name__ == "__main__":
    main()
