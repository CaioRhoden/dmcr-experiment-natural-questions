#!/bin/bash
#SBATCH --job-name=collections_small_window

#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_collections_binary_classification.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_collections_binary_classification.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=20G
#SBATCH --time=04:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0
#SBATCH --partition=l40s,rtx8000
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


SEEDS=(1 4 54 61 73)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}

echo "RUNNING COLLECTIONS TRAIN"
python run_datamodels.py \
    --seed $S \
    --run_type collections \
    --start_idx 0 \
    --end_idx 5000 \
    --checkpoint 5000 \
    --num_subprocesses 1 \
    --format_input ALT1 \
    --mode train

    
python run_datamodels.py \
    --seed $S \
    --run_type collections \
    --start_idx 0 \
    --end_idx 5000 \
    --checkpoint 5000 \
    --batch_size 5000 \
    --num_subprocesses 1 \
    --format_input ALT1 \
    --mode test

    