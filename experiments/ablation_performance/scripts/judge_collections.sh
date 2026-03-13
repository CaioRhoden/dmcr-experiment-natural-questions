#!/bin/bash
#SBATCH --job-name=ablatation_judge_collections
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_ablation_judge_collections.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_ablation_judge_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=40G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-3
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include

POOL_START=(550000 3050000 5600000 8050000)
POOL_END=(2500000 5000000 7500000 10000000)
STARTING_IDX=${POOL_START[SLURM_ARRAY_TASK_ID]}
ENDING_IDX=${POOL_END[SLURM_ARRAY_TASK_ID]}
echo "RUNNING SETUP"
python run_datamodels.py \
    --seed 1 \
    --run_type setup

echo "RUNNING COLLECTIONS TRAIN"
python run_datamodels.py \
    --seed 1 \
    --run_type collections \
    --start_idx $STARTING_IDX \
    --end_idx $ENDING_IDX \
    --checkpoint 50000 \
    --num_subprocesses 1 \
    --evaluator VotingBinaryJudge \
    --batch_size 50000 \
    --mode train

    
python run_datamodels.py \
    --seed 1 \
    --run_type collections \
    --start_idx $STARTING_IDX \
    --end_idx $ENDING_IDX \
    --checkpoint 50000 \
    --num_subprocesses 1 \
    --evaluator VotingBinaryJudge \
    --batch_size 50000 \
    --mode test

    
