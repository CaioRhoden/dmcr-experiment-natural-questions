#!/bin/bash
#SBATCH --job-name=1M_ablation_rouge_collections
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_1M_ablation_rouge_collections.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_1M_ablation_rouge_collections.err
#SBATCH --cpus-per-task=13
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=p5000,rtx5000,a5000,rtx8000,l40s,h100
#SBATCH --array=0,2-4


SEEDS=(1 4 54 61 73)
START_INDEX=0
END_INDEX=1000000



source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"


echo "-----------------------------------------------"
python run_datamodels.py \
    --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]} \
    --run_type collections \
    --start_idx $START_INDEX \
    --end_idx $END_INDEX \
    --checkpoint 100000 \
    --num_subprocesses 5 \
    --mode train \
    --evaluator Rouge-L


