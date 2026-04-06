#!/bin/bash
#SBATCH --job-name=zeroshot
#SBATCH --output=/home/caio.rhoden/slurm/%j_zeroshot.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_zeroshot.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=25G
#SBATCH --time=03:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=a5000,l40s,rtx8000,h100

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


python run_zero.py