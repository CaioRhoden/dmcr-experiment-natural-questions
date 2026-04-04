#!/bin/bash
#SBATCH --job-name=generation_alternative_inputs_binary_classification
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_generation_alternative_inputs_binary_classification.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_generation_alternative_inputs_binary_classification.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=44G
#SBATCH --time=2:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=a5000,rtx5000,rtx8000
#SBATCH --array=0-3

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


SEEDS=(1 4 54 61 73)
S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}

python run_zero.py --seed $S

# python run_datamodels.py \
#         --seed $S \
#         --run_type generation \
#         --batch_size 500 \
#         --model_run_id binary_judge

# python run_datamodels.py \
#         --seed $S \
#         --run_type generation \
#         --batch_size 500 \
#         --model_run_id voting

# python run_datamodels.py \
#         --seed $S \
#         --run_type generation \
#         --batch_size 500 \
#         --model_run_id rougel



python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_run_id opt_voting