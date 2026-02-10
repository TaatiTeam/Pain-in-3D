#!/bin/bash
#SBATCH --job-name=vitpain_pretrain_neutralref_v3
#SBATCH --account=aip-brudno
#SBATCH --time=02-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --output=/home/linxin67/projects/aip-btaati/linxin67/PainGeneration/PainGeneration_clean/logs/slurm/vitpain_pretrain_neutralref_v3_%x-%j.out

module load python
module load StdEnv/2023
module load gcc cuda arrow
module load opencv
module load rust
module load scipy-stack/2024a
module load blender
module load cmake
module load python-build-bundle/2023b
module list

PROJECT_ROOT=/home/linxin67/projects/aip-btaati/linxin67/PainGeneration
REPO_ROOT=${PROJECT_ROOT}/PainGeneration_clean

cd "$REPO_ROOT"

source "$PROJECT_ROOT/pain_env/bin/activate"
pip install --no-index platformdirs 2>/dev/null || pip install platformdirs

# sanity checks
nvidia-smi
free -h

export PYTHONPATH="${PYTHONPATH}:."
export PL_DISABLE_FORK_WARNING=1

echo "Running ViTPain pretraining with neutral reference (v3 - matched to original hyperparams)..."
echo "Fixes from v2: lr=1e-4 (was 5e-5), weight_decay=1e-1 (was 1e-2), au_loss_weight=1.0 (was 0.1)"

srun python -u train_vitpain.py \
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
    --output_dir experiment/vitpain_pretrain_neutralref_v3 \
    --wandb_project vitpain-synthetic-training \
    --wandb_group pretrain_neutralref_v3 \
    --run_name vitpain_neutralref_v3

echo "Pretraining complete."
