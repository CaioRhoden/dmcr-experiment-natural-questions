#!/bin/bash
#SBATCH --job-name=em_nq_open_collections
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_em_nq_open_collections.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_em_nq_open_collections.err
#SBATCH --mem=17G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-7
#SBATCH --partition=rtx5000,rtx8000
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

INIT=(0 1000000 2000000 3000000 4000000 5000000 6000000 7000000)
END=(1000000 2000000 3000000 4000000 5000000 6000000 7000000 7220000)

    
# echo "-----------------------------------------------"
# echo "RUNNING SETUP"
# python run_datamodels.py \
#     --run_type setup

# echo "-----------------------------------------------"
# echo "RUNNING PRE_COLLECTIONS TRAIN "
# python run_datamodels.py \
#     --run_type pre_collections \
#     --start_idx 0 \
#     --end_idx 2000 \
#     --checkpoint 250 \
#     --mode train


# echo "RUNNING PRE_COLLECTIONS TEST"
# python run_datamodels.py \
#     --run_type pre_collections \
#     --start_idx 0 \
#     --end_idx 200 \
#     --checkpoint 200 \
#     --mode test


echo "RUNNING COLLECTIONS TRAIN"
python run_datamodels.py \
    --run_type collections \
    --start_idx ${INIT[$SLURM_ARRAY_TASK_ID]} \
    --end_idx ${END[$SLURM_ARRAY_TASK_ID]} \
    --checkpoint 50000 \
    --num_subprocesses 1 \
    --evaluator SquadV2-EM \
    --collection_id em_collection \
    --mode train

    
# python run_datamodels.py \
#     --run_type collections \
#     --start_idx 0 \
#     --end_idx 100000 \
#     --checkpoint 50000 \
#     --batch_size 50000 \
#     --num_subprocesses 2 \
#     --evaluator Rouge-L \
#     --mode test

# python run_datamodels.py \
#     --run_type collections \
#     --start_idx 10 \
#     --end_idx 110 \
#     --checkpoint 100 \
#     --num_subprocesses 1 \
#     --evaluator Rouge-L \
#     --mode train
