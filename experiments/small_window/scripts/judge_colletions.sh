#!/bin/bash
#SBATCH --job-name=collections_small_window
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_collections_binary_classification.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_collections_binary_classification.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=20G
#SBATCH --time=04:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-4
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


SEEDS=(1 4 54 61 73)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}

# echo "RUNNING COLLECTIONS TRAIN"
# python run_datamodels.py \
#     --seed $S \
#     --run_type collections \
#     --start_idx 0 \
#     --end_idx 5000 \
#     --checkpoint 5000 \
#     --num_subprocesses 1 \
#     --format_input ALT1 \
#     --mode train

    
# python run_datamodels.py \
#     --seed $S \
#     --run_type collections \
#     --start_idx 0 \
#     --end_idx 5000 \
#     --checkpoint 5000 \
#     --batch_size 5000 \
#     --num_subprocesses 1 \
#     --format_input ALT1 \
#     --mode test


INST="Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to a question displayed below. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please classify the response as 1 for RESPONDS QUESTION and 0 for NOT RESPONDES QUESTION by strictly following this format: \"[[classification]]\", for example: \"Classification: [[1]]\""

echo "RUNNING COLLECTIONS TRAIN"
python run_datamodels.py \
    --seed $S \
    --run_type collections \
    --start_idx 850000 \
    --end_idx 1000000 \
    --checkpoint 50000 \
    --batch_size 50000 \
    --num_subprocesses 1 \
    --format_input recall \
    --instruction "$INST" \
    --root_path runs_opt \
    --mode train

    
python run_datamodels.py \
    --seed $S \
    --run_type collections \
    --start_idx 0 \
    --end_idx 1000000 \
    --checkpoint 50000 \
    --batch_size 50000 \
    --num_subprocesses 1 \
    --format_input recall \
    --instruction "$INST" \
    --root_path runs_opt \
    --mode test
