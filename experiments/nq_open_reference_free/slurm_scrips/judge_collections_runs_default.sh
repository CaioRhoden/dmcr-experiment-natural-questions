#!/bin/bash
#SBATCH --job-name=test_rag_qwen_default_prompt_judge_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_test_rag_qwen_default_prompt_judge_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_test_rag_qwen_default_prompt_judge_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate nq
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline

python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs/qwen_default --saving_dir judge_collections/qwen_default/test/pairwise_rag_judge --pairwise_rag --model_path models/Qwen3-4B-Instruct-2507 --mode test
