#!/usr/bin/env python3
"""
Run 5-fold cross-validation on UNBC dataset.

This script runs training on all 5 UNBC folds with proper subject-based splitting.
All folds are grouped together in wandb for easy comparison.

Can be run with or without a pretrained model checkpoint.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_unbc_5fold_cv(
    data_dir="datasets/UNBC-McMaster",
    model_size="large_dinov3",  # DinoV3 model (always with LoRA + AU query head)
    batch_size=32,  # Same as synthetic training
    max_epochs=100,  # Same as synthetic training
    output_dir="experiment/unbc_5fold_cv",
    wandb_group="unbc_5fold_cv",
    log_every_n_steps=10,
    pretrained_checkpoint=None,
    lora_rank=8,  # LoRA rank (default: 8)
    lora_alpha=16,  # LoRA alpha (default: 16)
    use_neutral_reference=False,
    multi_shot_inference=1,
    use_weighted_sampling=False,
    folds=None,
):
    """
    Run 5-fold cross-validation on UNBC dataset.
    
    Fixed hyperparameters (hard-coded):
    - learning_rate: 1e-4
    - weight_decay: 1e-1
    - precision: 16-bit
    - image_size: 224
    - backbone: always frozen (only LoRA adapters trained)
    - au_loss_weight: 0.1
    - pspi_loss_weight: 1.0
    - wandb_project: "unbc-5fold-cv"
    
    Selects best validation checkpoint for testing (standard practice).
    
    Args:
        data_dir: Path to UNBC-McMaster dataset (default: "datasets/UNBC-McMaster")
        model_size: DinoV3 model size (default: "large_dinov3")
        batch_size: Training batch size (default: 32, same as synthetic training)
        max_epochs: Maximum number of epochs (default: 100, same as synthetic training)
        output_dir: Base output directory (default: "experiment/unbc_5fold_cv")
        wandb_group: Wandb group name for all folds (default: "unbc_5fold_cv")
        log_every_n_steps: Logging frequency (default: 10)
        pretrained_checkpoint: Optional path to pretrained checkpoint from synthetic training
        lora_rank: LoRA rank (default: 8)
        lora_alpha: LoRA alpha (default: 16)
        use_neutral_reference: Whether to use neutral reference images (default: False)
        multi_shot_inference: Number of neutral references for multi-shot inference (default: 1)
        use_weighted_sampling: Whether to use weighted sampling for class imbalance (default: False)
        folds: List of specific folds to run (default: None runs all 5 folds)
    """
    
    # Validate checkpoint path if provided
    if pretrained_checkpoint is not None:
        if not os.path.exists(pretrained_checkpoint):
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_checkpoint}\n"
                f"Please provide a valid path to the checkpoint file."
            )
    
    print("=" * 80)
    if pretrained_checkpoint:
        print("UNBC Dataset 5-Fold Cross-Validation (Pretrained Model)")
    else:
        print("UNBC Dataset 5-Fold Cross-Validation (Baseline)")
    print("=" * 80)
    if pretrained_checkpoint:
        print(f"Pretrained checkpoint: {pretrained_checkpoint}")
    # Fixed hyperparameters
    learning_rate = 1e-4
    weight_decay = 1e-1
    precision = 16
    image_size = 224
    # Backbone is always frozen (only LoRA adapters are trained)
    au_loss_weight = 0.1
    pspi_loss_weight = 1.0
    wandb_project = "unbc-5fold-cv"
    wandb_entity = None
    
    print(f"Dataset: {data_dir}")
    print(f"Model size: {model_size}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {max_epochs}")
    print(f"Learning rate: {learning_rate} (fixed)")
    print(f"Weight decay: {weight_decay} (fixed)")
    print(f"Precision: {precision}-bit (fixed)")
    print(f"Backbone: Always frozen (only LoRA adapters are trained)")
    print(f"Output directory: {output_dir}")
    print(f"Wandb project: {wandb_project}")
    print(f"Wandb group: {wandb_group}")
    print(f"Use weighted sampling: {use_weighted_sampling}")
    print()
    print("Note: Using last epoch checkpoint (no validation-based model selection)")
    print("This will run 5 separate training jobs, one for each fold (0-4):")
    print("All folds will be grouped together in wandb for easy comparison.")
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine run name prefix
    if pretrained_checkpoint:
        run_name_prefix = "unbc_pretrained_fold_"
    else:
        run_name_prefix = "unbc_baseline_fold_"

    print(f"Use neutral reference: {use_neutral_reference}")
    print(f"Multi-shot inference: {multi_shot_inference}")
    
    # Determine which folds to run
    if folds is None:
        folds_to_run = list(range(5))
    else:
        folds_to_run = list(folds)

    # Validate folds
    valid_folds = set(range(5))
    invalid = [f for f in folds_to_run if f not in valid_folds]
    if invalid:
        raise ValueError(f"Invalid fold(s) {invalid}. Valid folds are 0-4.")

    # De-duplicate while preserving order
    seen = set()
    folds_to_run = [f for f in folds_to_run if not (f in seen or seen.add(f))]

    # Run each fold sequentially
    for fold in folds_to_run:  # UNBC uses folds 0-4
        print()
        print("=" * 80)
        print(f"Starting Fold {fold}")
        print("=" * 80)
        
        # Build command
        # Note: train_unbc.py will create fold_{fold} subdirectory inside output_dir
        cmd = [
            sys.executable,
            str(project_root / "train_unbc.py"),
            "--data_dir", data_dir,
            "--model_size", model_size,
            "--batch_size", str(batch_size),
            "--max_epochs", str(max_epochs),
            "--learning_rate", str(learning_rate),
            "--weight_decay", str(weight_decay),
            "--precision", str(precision),
            "--output_dir", output_dir,  # Base directory - train_unbc.py will create fold_{fold} inside
            "--wandb_project", wandb_project,
            "--wandb_group", wandb_group,  # Always pass group (will be set to default if not provided)
            "--fold", str(fold),
            "--log_every_n_steps", str(log_every_n_steps),
            "--run_name", f"{run_name_prefix}{fold}",
        ]
        
        # AU query head is always enabled by default, no need to pass
        
        if pretrained_checkpoint:
            cmd.extend(["--synthetic_pretrained_checkpoint", pretrained_checkpoint])
        
        # LoRA is always enabled, pass rank and alpha (defaults: 8 and 16)
        cmd.extend(["--lora_rank", str(lora_rank)])
        cmd.extend(["--lora_alpha", str(lora_alpha)])
            
        if use_neutral_reference:
            cmd.append("--use_neutral_reference")
            
        if use_weighted_sampling:
            cmd.append("--use_weighted_sampling")

        # Multi-shot inference (only meaningful with neutral reference)
        if use_neutral_reference and int(multi_shot_inference) > 1:
            cmd.extend(["--multi_shot_inference", str(int(multi_shot_inference))])
        
        # Always use last epoch for cross-validation (no validation-based model selection)
        cmd.extend(["--inner_validation", "last_epoch"])
        
        # Run training for this fold
        print(f"Running command: {' '.join(cmd)}")
        print(f"DEBUG: wandb_group value being passed: '{wandb_group}'")
        print()
        
        try:
            result = subprocess.run(cmd, check=True, cwd=project_root)
            print(f"✓ Completed Fold {fold}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Fold {fold} failed with return code {e.returncode}")
            print(f"Continuing with remaining folds...")
        
        print()
    
    print("=" * 80)
    print("5-Fold Cross-Validation Complete!")
    print("=" * 80)
    print(f"Results saved in: {output_dir}/fold_*/")
    print(f"All runs grouped in wandb under project: {wandb_project}, group: {wandb_group}")


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run 5-fold cross-validation on UNBC dataset (with or without pretrained model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Optional path to pretrained checkpoint from synthetic training. "
             "If not provided, training starts from scratch. "
             "Example: experiment/vitpain/checkpoints/vitpain-epoch=99-val_regression_mae=1.234.ckpt"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/UNBC-McMaster",
        help="Path to UNBC-McMaster dataset"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="large_dinov3",
        choices=["small_dinov3", "base_dinov3", "large_dinov3"],
        help="DinoV3 model size (LoRA + AU query head + binary classification always enabled)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32, same as synthetic training)"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of epochs (default: 100, same as synthetic training)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory (defaults to 'experiment/unbc_5fold_cv_baseline' or 'experiment/unbc_5fold_cv_pretrained')"
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Wandb group name (defaults to 'unbc_5fold_cv_baseline' or 'unbc_5fold_cv_pretrained')"
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="How often to log metrics (default: 10)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--use_neutral_reference",
        action="store_true",
        help="Use a neutral reference image (PSPI=0) for the same subject (default: False)"
    )

    parser.add_argument(
        "--multi_shot_inference",
        type=int,
        default=1,
        help=(
            "Number of neutral reference images to sample per example for multi-shot inference. "
            "Only used when --use_neutral_reference is set (default: 1)."
        ),
    )
    parser.add_argument(
        "--use_weighted_sampling",
        action="store_true",
        help="Use weighted sampling to handle class imbalance (default: False)"
    )

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help=(
            "Which UNBC folds to run. Provide a space-separated list, e.g. --folds 0 2 4. "
            "Default: 0 1 2 3 4."
        ),
    )

    args = parser.parse_args()
    
    # Set defaults based on whether pretrained checkpoint is provided
    if args.pretrained_checkpoint:
        default_output_dir = args.output_dir or "experiment/unbc_5fold_cv_pretrained"
        default_wandb_group = args.wandb_group or "unbc_5fold_cv_pretrained"
    else:
        default_output_dir = args.output_dir or "experiment/unbc_5fold_cv_baseline"
        default_wandb_group = args.wandb_group or "unbc_5fold_cv_baseline"
    
    run_unbc_5fold_cv(
        data_dir=args.data_dir,
        model_size=args.model_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        output_dir=default_output_dir,
        wandb_group=default_wandb_group,
        log_every_n_steps=args.log_every_n_steps,
        pretrained_checkpoint=args.pretrained_checkpoint,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_neutral_reference=args.use_neutral_reference,
        multi_shot_inference=args.multi_shot_inference,
        use_weighted_sampling=args.use_weighted_sampling,
        folds=args.folds,
    )


if __name__ == "__main__":
    main()
