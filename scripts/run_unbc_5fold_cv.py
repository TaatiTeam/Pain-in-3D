#!/usr/bin/env python3
"""
Run 5-fold cross-validation on UNBC dataset.

Groups all folds together in wandb for easy comparison.
Can run with or without a pretrained model checkpoint.
"""

import os
import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_unbc_5fold_cv(
    data_dir="datasets/UNBC-McMaster",
    model_size="large_dinov3",
    batch_size=100,
    max_epochs=50,
    output_dir="experiment/unbc_5fold_cv",
    wandb_group="unbc_5fold_cv",
    log_every_n_steps=10,
    pretrained_checkpoint=None,
    lora_rank=8,
    lora_alpha=16,
    use_neutral_reference=False,
    multi_shot_inference=1,
    use_weighted_sampling=False,
    wandb_project="unbc-5fold-cv",
    folds=None,
):
    """
    Run 5-fold cross-validation on UNBC dataset.

    Fixed hyperparameters (hard-coded):
    - learning_rate: 1e-4
    - weight_decay: 1e-1
    - precision: 16-bit
    - image_size: 224
    - au_loss_weight: 0.1 (down-weighted for UNBC fine-tuning)
    - pspi_loss_weight: 1.0
    """
    if pretrained_checkpoint is not None:
        if not os.path.exists(pretrained_checkpoint):
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_checkpoint}\n"
                f"Please provide a valid path to the checkpoint file."
            )

    learning_rate = 1e-4
    weight_decay = 1e-1
    precision = 16
    image_size = 224
    au_loss_weight = 0.1
    pspi_loss_weight = 1.0
    wandb_entity = None

    os.makedirs(output_dir, exist_ok=True)

    if pretrained_checkpoint:
        run_name_prefix = "unbc_pretrained_fold_"
    else:
        run_name_prefix = "unbc_baseline_fold_"

    if folds is None:
        folds_to_run = list(range(5))
    else:
        folds_to_run = list(folds)

    valid_folds = set(range(5))
    invalid = [f for f in folds_to_run if f not in valid_folds]
    if invalid:
        raise ValueError(f"Invalid fold(s) {invalid}. Valid folds are 0-4.")

    seen = set()
    folds_to_run = [f for f in folds_to_run if not (f in seen or seen.add(f))]

    for fold in folds_to_run:
        # train_unbc.py creates fold_{fold} subdirectory inside output_dir
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
            "--output_dir", output_dir,
            "--wandb_project", wandb_project,
            "--wandb_group", wandb_group,
            "--fold", str(fold),
            "--log_every_n_steps", str(log_every_n_steps),
            "--run_name", f"{run_name_prefix}{fold}",
            "--au_loss_weight", str(au_loss_weight),
            "--pspi_loss_weight", str(pspi_loss_weight),
        ]

        if pretrained_checkpoint:
            cmd.extend(["--synthetic_pretrained_checkpoint", pretrained_checkpoint])

        cmd.extend(["--lora_rank", str(lora_rank)])
        cmd.extend(["--lora_alpha", str(lora_alpha)])

        if use_neutral_reference:
            cmd.append("--use_neutral_reference")

        if use_weighted_sampling:
            cmd.append("--use_weighted_sampling")

        if use_neutral_reference and int(multi_shot_inference) > 1:
            cmd.extend(["--multi_shot_inference", str(int(multi_shot_inference))])

        try:
            subprocess.run(cmd, check=True, cwd=project_root)
        except subprocess.CalledProcessError as e:
            print(f"Fold {fold} failed with return code {e.returncode}. Continuing...")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run 5-fold cross-validation on UNBC dataset (with or without pretrained model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pretrained_checkpoint", type=str, default=None,
        help="Optional path to pretrained checkpoint from synthetic training."
    )
    parser.add_argument("--data_dir", type=str, default="datasets/UNBC-McMaster",
                        help="Path to UNBC-McMaster dataset")
    parser.add_argument(
        "--model_size", type=str, default="large_dinov3",
        choices=["small_dinov3", "base_dinov3", "large_dinov3"],
        help="DinoV3 model size"
    )
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Base output directory")
    parser.add_argument("--wandb_project", type=str, default="UNBC-EVALUATION",
                        help="Wandb project name")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--use_neutral_reference", action="store_true")
    parser.add_argument("--multi_shot_inference", type=int, default=1,
                        help="Number of neutral refs for multi-shot inference (only with --use_neutral_reference)")
    parser.add_argument("--use_weighted_sampling", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Which folds to run (e.g. --folds 0 2 4)")

    args = parser.parse_args()

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
        wandb_project=args.wandb_project,
        folds=args.folds,
    )


if __name__ == "__main__":
    main()
