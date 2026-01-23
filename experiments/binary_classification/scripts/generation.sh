#!/bin/bash
#SBATCH --job-name=generation_model_analysis
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_generation_model_analysis.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_generation_model_analysis.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=44G
#SBATCH --time=2:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-4

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include

SEEDS=(1 4 54 61 73)
S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}

# python run_datamodels.py \
#         --seed $S \
#         --run_type generation \
#         --batch_size 500 \
#         --model_tag lr_groundtruth

# python run_datamodels.py \
#         --seed $S \
#         --run_type generation \
#         --batch_size 500 \
#         --model_tag lr_judge

# python run_datamodels.py \
#         --seed $S \
#         --run_type generation \
#         --batch_size 500 \
#         --model_tag lr_balanced_groundtruth

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag lr_balanced_judge