#!/bin/bash
#SBATCH --job-name=binary_classification_optuna_search
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_binary_classification_optuna_search.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_binary_classification_optuna_search.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=11
#SBATCH --mem-per-gpu=15G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=1
#SBATCH --partition=p5000,rtx5000,a5000,rtx8000,l40s

source ~/miniconda3/bin/activate
conda activate nq

IDS=(0 1 2 3 4)
SUBFOLDER=(groundtruth judge)

IDX=$((SLURM_ARRAY_TASK_ID % 5))
S_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SUB=${SUBFOLDER[$S_IDX]}


python run_logreg_gridsearch_holdout.py \
    --exp-index ${IDS[$IDX]} \
    --subfolder $SUB