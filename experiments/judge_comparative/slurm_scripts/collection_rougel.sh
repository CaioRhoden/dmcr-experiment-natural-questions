#!/bin/bash
#SBATCH --job-name=rougel_collections
#SBATCH --output=/home/caio.rhoden/slurm/%j_rougel_collections.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_rougel_collections.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=rtx5000,rtx8000,a5000,l40s,h100

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"


echo "-----------------------------------------------"
python run_datamodels.py \
    --run_type collections \
    --start_idx 0 \
    --end_idx 20000 \
    --checkpoint 5000 \
    --mode train \
    --num_subprocesses 4

echo "-----------------------------------------------"
python run_datamodels.py \
    --run_type collections \
    --start_idx 0 \
    --end_idx 2000 \
    --checkpoint 2000 \
    --mode test \
    --num_subprocesses 1
