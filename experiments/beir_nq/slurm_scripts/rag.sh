#!/bin/bash
#SBATCH --job-name=rag_beir_nq
#SBATCH --output=/home/caio.rhoden/slurm/%j_rag_nq_open.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_rag_nq_open.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=10:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --partition=a5000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python run_rag.py
