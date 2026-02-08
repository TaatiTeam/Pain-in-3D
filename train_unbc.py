#!/usr/bin/env python3
"""Training script for UNBC-McMaster dataset."""

import os
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
from configs import parse_args, config_to_dict


def train_unbc(cfg):
    """Train PSPI ViT model on UNBC-McMaster dataset."""
    if cfg.training.random_seed is not None:
        pl.seed_everything(cfg.training.random_seed, workers=True)

    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    os.makedirs(cfg.output.output_dir, exist_ok=True)
    fold_output_dir = os.path.join(cfg.output.output_dir, f"fold_{cfg.data.fold}")
    os.makedirs(fold_output_dir, exist_ok=True)

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
    )
    data_module.setup()

    model = create_vitpain_model(
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
        num_sanity_val_steps=2,
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=cfg.output.resume_from_checkpoint,
    )

    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        trainer.test(model, datamodule=data_module, ckpt_path=trainer.checkpoint_callback.best_model_path)

    return trainer, model


def main():
    cfg = parse_args()
    train_unbc(cfg)


if __name__ == "__main__":
    main()
