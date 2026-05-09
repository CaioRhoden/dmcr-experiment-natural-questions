#!/bin/bash
#SBATCH --job-name=runs_generation_reference_based
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_runs_generation_reference_based.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_runs_generation_reference_based.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=30G
#SBATCH --time=2:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=a5000,l40s,rtx8000
#SBATCH --array=1-4

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODELS=("em" "f1"  "f1_binary" "rougel" "rougel_binary")

python run_datamodels.py \
        --run_type generation \
        --model_run_id ${MODELS[$SLURM_ARRAY_TASK_ID]}
