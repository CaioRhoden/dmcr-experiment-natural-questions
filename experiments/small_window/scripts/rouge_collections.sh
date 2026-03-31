#!/bin/bash
#SBATCH --job-name=small_window_rouge_collections
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_small_window_rouge_collections.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_small_window_rouge_collections.err
#SBATCH --cpus-per-task=7
#SBATCH --mem=15G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=p5000,rtx5000,a5000,rtx8000
#SBATCH --array=0-4

SEEDS=(1 4 54 61 73)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"


echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --seed $S \
    --run_type collections \
    --start_idx 0 \
    --end_idx 1000000 \
    --checkpoint 50000 \
    --num_subprocesses 4 \
    --evaluator Rouge-L \
    --mode train


echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --seed $S \
    --run_type collections \
    --start_idx 0 \
    --end_idx 100000 \
    --checkpoint 50000 \
    --num_subprocesses 2\
    --evaluator Rouge-L \
    --mode test
