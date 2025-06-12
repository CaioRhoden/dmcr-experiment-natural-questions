#!/bin/bash
#SBATCH --job-name=rag_25
#SBATCH --output=/home/caio.rhoden/slurm/rag_25_%j.out
#SBATCH --error=/home/caio.rhoden/slurm/rag_25_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=40G
#SBATCH --mail-user=c214129@dac.unicamp.br
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"

# python3 pipeline.py --step setup
# python3 pipeline.py --step get_rag_retrieval
# python3 pipeline.py --step get_rag_generations
# python3 pipeline.py --step create_datamodels_datasets
# python3 pipeline.py --step run_pre_collections
# python3 pipeline.py --step run_collections
#  python3 pipeline.py --step train_datamodels
#  python3 pipeline.py --step evaluate_datamodels
 python3 pipeline.py --step get_datamodels_generations