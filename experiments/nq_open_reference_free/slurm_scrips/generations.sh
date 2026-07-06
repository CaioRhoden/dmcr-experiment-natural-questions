#!/bin/bash
#SBATCH --job-name=runs_generation_reference_free
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_runs_generation_reference_free.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_runs_generation_reference_free.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=30G
#SBATCH --time=3:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODELS=("naive")

python run_datamodels.py \
        --run_type generation \
        --model_run_id naive \
        --root_path runs/llama \
        --lm_configs.max_new_tokens 200 \
        --instruction extraction

# python run_datamodels.py \
#         --run_type generation \
#         --model_run_id ${MODELS[$SLURM_ARRAY_TASK_ID]} \
#         --root_path runs/qwen \
#         --lm_configs.max_new_tokens 200 \
#         --instruction extraction \
#         --language_model_path models/Qwen3-4B-Instruct-2507 


# python run_datamodels.py \
#         --run_type generation \
#         --model_run_id ${MODELS[$SLURM_ARRAY_TASK_ID]} \
#         --root_path runs/qwen_default \
#         --language_model_path models/Qwen3-4B-Instruct-2507 

# python run_datamodels.py \
#         --run_type generation \
#         --model_run_id ${MODELS[$SLURM_ARRAY_TASK_ID]} \
#         --root_path runs/llama_default