#!/bin/bash
#SBATCH --job-name=f1_nq_open_collections
#SBATCH --output=/home/caio.rhoden/slurm/%j_f1_nq_open_collections.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_f1_nq_open_collections.err
#SBATCH --mem=17G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --partition=p5000
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
# export VLLM_WORKER_MULTIPROC_METHOD=spawn

# INIT=(0 1000000 2000000 3000000 4000000 5000000 6000000 7000000)
# END=(1000000 2000000 3000000 4000000 5000000 6000000 7000000 7220000)




# echo "RUNNING COLLECTIONS TRAIN"
# python run_datamodels.py \
#     --run_type collections \
#     --start_idx ${INIT[$SLURM_ARRAY_TASK_ID]} \
#     --end_idx ${END[$SLURM_ARRAY_TASK_ID]} \
#     --checkpoint 50000 \
#     --num_subprocesses 1 \
#     --evaluator SquadV2 \
#     --collection_id f1_collection \
#     --mode train

echo "RUNNING PRE_COLLECTIONS TEST"
python run_datamodels.py \
    --run_type collections \
    --start_idx 0 \
    --end_idx 722000 \
    --checkpoint 50000 \
    --evaluator SquadV2 \
    --num_subprocesses 1 \
    --collection_id f1_collection \
    --mode test

echo "RUNNING PRE_COLLECTIONS TEST"
python run_datamodels.py \
    --run_type collections \
    --start_idx 0 \
    --end_idx 722000 \
    --checkpoint 50000 \
    --evaluator SquadV2-EM \
    --collection_id em_collection \
    --mode test

