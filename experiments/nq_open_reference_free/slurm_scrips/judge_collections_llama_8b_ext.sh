#!/bin/bash
#SBATCH --job-name=zs_qwen_prompt_judge_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_zs_qwen_prompt_judge_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_zs_qwen_prompt_judge_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate nq
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline


##Train
python run_judge_collections.py --judge_type PromptJudge --prompt_extraction naive_judge --runs_path runs/llama_8b_extraction --saving_dir judge_collections/llama_8b_extraction/naive --model_path models/Llama-3.1-8B-Instruct 
python run_judge_collections.py --judge_type PromptJudge --prompt_extraction naive_judge --runs_path runs/llama_8b_extraction --saving_dir judge_collections/llama_8b_extraction/voting_naive --model_path models/Llama-3.1-8B-Instruct --n_generations 3

##Test
python run_judge_collections.py --judge_type PromptJudge --prompt_extraction naive_judge --runs_path runs/llama_8b_extraction --saving_dir judge_collections/llama_8b_extraction/naive --model_path models/Llama-3.1-8B-Instruct --mode test --batch_size 200
python run_judge_collections.py --judge_type PromptJudge --prompt_extraction naive_judge --runs_path runs/llama_8b_extraction --saving_dir judge_collections/llama_8b_extraction/voting_naive --model_path models/Llama-3.1-8B-Instruct --n_generations 3 --mode test --batch_size 200
