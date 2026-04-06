#!/bin/bash
#SBATCH --job-name=qwen_collections
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_qwen.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_qwen_err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=20G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-4%3
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


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
    --language_model_path models/Qwen3-8B \
    --root_path runs_qwen \
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
    --language_model_path models/Qwen3-8B \
    --root_path runs_qwen \
    --mode test

    
