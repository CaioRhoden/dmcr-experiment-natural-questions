#!/bin/bash
#SBATCH --job-name=preview_50_L2
#SBATCH --output=/home/caio.rhoden/slurm/preview_50_L2_%j.out
#SBATCH --error=/home/caio.rhoden/slurm/preview_50_L2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=15G
#SBATCH --mail-user=c214129@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"

# CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s setup
# CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s get_rag_retrieval
# CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s get_rag_generations
# CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s create_datamodels_datasets
# CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s run_pre_collections
# CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s run_collections
CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s train_datamodels
CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s evaluate_datamodels
CUDA_VISIBLE_DEVICES=0 python3 pipeline.py -s get_datamodels_generations