#!/bin/bash
#SBATCH --job-name=qwen_instruct_collections_binary_classification
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_qwen_instruct_collections_binary_classification.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_qwen_instruct_collections_binary_classification.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=20G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=4
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include

SEEDS=(1 4 54 61 73)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}

# echo "RUNNING SETUP"
# python run_datamodels.py \
#     --seed $S \
#     -root_path runs_qwen \
#     --run_type setup

echo "RUNNING COLLECTIONS TRAIN"
python run_datamodels.py \
    --seed $S \
    --run_type collections \
    --start_idx 0 \
    --end_idx 1000000 \
    --checkpoint 50000 \
    --batch_size 50000 \
    --num_subprocesses 1 \
    --evaluator VotingBinaryJudge \
    --language_model_path models/Qwen3-4B-Instruct-2507 \
    --root_path runs_qwen_instruct \
    --mode train

    
python run_datamodels.py \
    --seed $S \
    --run_type collections \
    --start_idx 0 \
    --end_idx 100000 \
    --checkpoint 50000 \
    --batch_size 50000 \
    --num_subprocesses 1 \
    --evaluator VotingBinaryJudge \
    --language_model_path models/Qwen3-4B-Instruct-2507 \
    --root_path runs_qwen_instruct \
    --mode test

    
