#!/bin/bash
# Train ViTPain model on UNBC dataset using 5-fold cross-validation

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH}:."
export PL_DISABLE_FORK_WARNING=1

PRETRAINED_CHECKPOINT="experiment/vitpain_pretrain/checkpoints/best.ckpt"

python -u scripts/run_unbc_5fold_cv.py \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --data_dir datasets/UNBC-McMaster \
    --model_size large_dinov3 \
    --batch_size 100 \
    --max_epochs 50 \
    --au_loss_weight 0.1 \
    --use_weighted_sampling \
    --use_neutral_reference \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir experiment/unbc_5fold_cv \
    --wandb_project UNBC-EVALUATION \
    --wandb_group unbc_5fold_cv \
    --log_every_n_steps 10 \
    --folds 0 1 2 3 4
