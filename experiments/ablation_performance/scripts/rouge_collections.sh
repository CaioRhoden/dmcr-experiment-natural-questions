#!/bin/bash
#SBATCH --job-name=1M_ablation_rouge_collections
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_1M_ablation_rouge_collections.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_1M_ablation_rouge_collections.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=rtx5000,rtx8000,a5000,l40s,h100
#SBATCH --array=4-5


ARR_START_INDEX=(200000 1000000 2000000 3000000 4000000 5600000 6000000 7000000 8000000 9000000)
ARR_END_INDEX=(1000000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000)

START_INDEX=${ARR_START_INDEX[$SLURM_ARRAY_TASK_ID]}
END_INDEX=${ARR_END_INDEX[$SLURM_ARRAY_TASK_ID]}

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"


echo "-----------------------------------------------"
python run_datamodels.py \
    --seed 1\
    --run_type collections \
    --start_idx $START_INDEX \
    --end_idx $END_INDEX \
    --checkpoint 100000 \
    --mode train \
    --evaluator Rouge-L


