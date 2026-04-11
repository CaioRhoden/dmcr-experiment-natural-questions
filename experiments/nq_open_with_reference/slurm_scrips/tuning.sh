#!/bin/bash
#SBATCH --job-name=em_grid_search
#SBATCH --output=/home/caio.rhoden/slurm/%j_em_grid_search.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_em_grid_search.err
#SBATCH --mem=25G
#SBATCH --cpus-per-task=22
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=p5000,rtx5000,rtx8000,a5000,l40s,h100

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


echo "RUNNING PARAM GRID SEARCH"
cd hyperparameter_tuning
# python tuning.py --subfolder em_collection
# python tuning.py --subfolder f1_binary_collection
# python tuning.py --subfolder rougel_binary_collection

python gridsearch_tuning.py --subfolder em_collection
# python gridsearch_tuning.py --subfolder f1_binary_collection
# python gridsearch_tuning.py --subfolder rougel_binary_collection
