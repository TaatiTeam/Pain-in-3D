#!/bin/bash
#SBATCH --job-name=unbc_5fold_pretrained_neutralref_clean
#SBATCH --account=aip-brudno
#SBATCH --time=02-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --output=/home/linxin67/projects/aip-btaati/linxin67/PainGeneration/PainGeneration_clean/logs/slurm/unbc_5fold_pretrained_neutralref_clean_%x-%j.out

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

# Use the latest checkpoint from the neutral-ref pretraining run
PRETRAINED_CHECKPOINT="experiment/vitpain_pretrain_neutralref/checkpoints/vitpain-epoch=138-val_regression_mae=2.018.ckpt"

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $PRETRAINED_CHECKPOINT" >&2
    exit 1
fi

echo "Running UNBC 5-fold cross-validation with pretrained neutral-ref init (PainGeneration_clean)..."
echo "Pretrained checkpoint: $PRETRAINED_CHECKPOINT"

srun python -u scripts/run_unbc_5fold_cv.py \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --data_dir datasets/UNBC-McMaster \
    --model_size large_dinov3 \
    --batch_size 100 \
    --max_epochs 50 \
    --use_weighted_sampling \
    --use_neutral_reference \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir experiment/unbc_5fold_cv_pretrained_neutralref \
    --wandb_group unbc_5fold_cv_pretrained_neutralref_clean \
    --log_every_n_steps 10 \
    --folds 0 1 2 3 4

echo "UNBC 5-fold CV (pretrained neutral-ref, clean) complete."
