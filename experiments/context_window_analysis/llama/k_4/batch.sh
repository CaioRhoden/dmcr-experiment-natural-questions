#!/bin/bash
#SBATCH --job-name=llama_k4
#SBATCH --output=/home/caio.rhoden/slurm/llama_k4_%j.out
#SBATCH --error=/home/caio.rhoden/slurm/llama_k4_%j.err
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

# python3 pipeline.py -s setup
# python3 pipeline.py -s get_rag_retrieval
# python3 pipeline.py -s get_rag_generations
# python3 pipeline.py -s create_datamodels_datasets
# python3 pipeline.py -s run_pre_collections
#  python3 pipeline.py -s run_collections
#  python3 pipeline.py -s train_datamodels
#  python3 pipeline.py -s evaluate_datamodels
python3 pipeline.py -s get_datamodels_generations