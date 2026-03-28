#!/bin/bash
#SBATCH --job-name=judge_experimetn_run_rag
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_judge_experimetn_run_rag.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_judge_experimetn_run_rag.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=50G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include


python run_rag.py