#!/bin/bash
#SBATCH --job-name=part1_qwen_default_nq_open_gold_pre_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%j_part1_qwen_train_default_nq_open_gold_pre_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%j_part1_qwen_train_default_nq_open_gold_pre_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline



echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --run_type pre_collections \
    --start_idx 1000 \
    --end_idx 2000 \
    --checkpoint 250 \
    --instruction extraction \
    --lm_configs.max_new_tokens 200 \
    --root_path runs_qwen \
    --language_model_path models/Qwen3-4B-Instruct-2507 \
    --mode train


# echo "RUNNING PRE_COLLECTIONS TEST"
# python run_datamodels.py \
#     --run_type pre_collections \
#     --start_idx 0 \
#     --end_idx 200 \
#     --checkpoint 200 \
#     --lm_configs.max_new_tokens 200 \
#     --instruction extraction \
#     --mode test


