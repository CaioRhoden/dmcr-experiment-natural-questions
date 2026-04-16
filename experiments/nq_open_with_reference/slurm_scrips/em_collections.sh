#!/bin/bash
#SBATCH --job-name=em_nq_open_collections
#SBATCH --output=/home/caio.rhoden/slurm/%j_em_nq_open_collections.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_em_nq_open_collections.err
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --partition=p5000,rtx5000,a5000,rtx8000,l40s,h100
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


echo "RUNNING PRE_COLLECTIONS TEST"
python run_datamodels.py \
    --run_type collections \
    --start_idx 0 \
    --end_idx 7220000 \
    --checkpoint 100000 \
    --evaluator SquadV2-EM \
    --num_subprocesses 1 \
    --collection_id f1_collection \
    --root_path runs_default \
    --mode train

echo "RUNNING PRE_COLLECTIONS TEST"
python run_datamodels.py \
    --run_type collections \
    --start_idx 0 \
    --end_idx 722000 \
    --checkpoint 100000 \
    --evaluator SquadV2-EM \
    --collection_id f1_collection \
    --root_path runs_default \
    --mode test

