#!/bin/bash
# Train production ViTPain model on entire UNBC dataset (all data, no CV)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH}:."
export PL_DISABLE_FORK_WARNING=1

PRETRAINED_CHECKPOINT="experiment/vitpain_pretrain/checkpoints/best.ckpt"

python -u scripts/train_unbc_production.py \
    --synthetic_pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --data_dir datasets/UNBC-McMaster \
    --model_size large_dinov3 \
    --batch_size 100 \
    --max_epochs 50 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --au_loss_weight 0.1 \
    --pspi_loss_weight 1.0 \
    --use_neutral_reference \
    --use_weighted_sampling \
    --output_dir experiment/unbc_production \
    --wandb_project UNBC-PRODUCTION \
    --wandb_group unbc_production \
    --run_name unbc_production \
    --log_every_n_steps 10
