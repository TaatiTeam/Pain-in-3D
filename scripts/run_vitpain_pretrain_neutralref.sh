#!/bin/bash
#SBATCH --job-name=vitpain_pretrain_neutralref_v2
#SBATCH --account=aip-brudno
#SBATCH --time=02-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --output=/home/linxin67/projects/aip-btaati/linxin67/PainGeneration/PainGeneration_clean/logs/slurm/vitpain_pretrain_neutralref_v2_%x-%j.out

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

echo "Running ViTPain pretraining with neutral reference (v2 - fixed hyperparams)..."
echo "Changes from v1: lr=5e-5 (was 1e-4), weight_decay=1e-2 (was 1e-1), batch_size=48 (was 32), max_epochs=150 (was 100)"

srun python -u train_vitpain.py \
    --data_dir datasets/pain_faces \
    --split_csv data/splits/uniform_data_70_20_10_split.csv \
    --model_size large_dinov3 \
    --batch_size 48 \
    --max_epochs 150 \
    --learning_rate 5e-5 \
    --weight_decay 1e-2 \
    --precision 16 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --use_neutral_reference \
    --log_every_n_steps 10 \
    --random_seed 42 \
    --output_dir experiment/vitpain_pretrain_neutralref_v2 \
    --wandb_project vitpain-synthetic-training \
    --wandb_group pretrain_neutralref_v2 \
    --run_name vitpain_neutralref_v2

echo "Pretraining complete."
