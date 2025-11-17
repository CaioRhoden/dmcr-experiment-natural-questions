#!/bin/bash
#SBATCH --job-name=generating_rag_debbuging_validation
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_generating_rag_debbuging_validation.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_generating_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=45G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=a5000
#SBATCH --array=2-4


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include

SEEDS=(1 4 54 61 73)
INSTRUCTIONS=(0 1 2)

python run_datamodels.py \
        --seed 1 \
        --instruction_idx 0 \
        --evaluator Judge \
        --run_type generation \
        --batch_size 500
