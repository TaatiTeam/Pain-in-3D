#!/bin/bash
# Train ViTPain model on synthetic pain faces dataset

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH}:."
export PL_DISABLE_FORK_WARNING=1

python -u train_vitpain.py \
    --data_dir datasets/pain_faces \
    --split_csv data/splits/uniform_data_70_20_10_split.csv \
    --model_size large_dinov3 \
    --batch_size 48 \
    --max_epochs 150 \
    --learning_rate 1e-4 \
    --weight_decay 1e-1 \
    --au_loss_weight 1.0 \
    --pspi_loss_weight 1.0 \
    --precision 16 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --use_neutral_reference \
    --log_every_n_steps 10 \
    --random_seed 42 \
    --output_dir experiment/vitpain_pretrain \
    --wandb_project vitpain-synthetic-training \
    --wandb_group pretrain \
    --run_name vitpain_pretrain
