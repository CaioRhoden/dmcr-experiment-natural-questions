#!/bin/bash
#SBATCH --job-name=judge_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%j_test_judge_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%j_test_judge_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=24:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu03

source ~/miniconda3/bin/activate
conda activate nq
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline



# Naive
# python run_judge_collections.py --judge_type PromptJudge --runs_path runs/llama --prompt_instruction naive_judge --saving_dir judge_collections/llama/test/naive --mode test --batch_size 200
# python run_judge_collections.py --judge_type PromptJudge --runs_path runs/llama_default --prompt_instruction naive_judge --saving_dir judge_collections/llama_default/test/naive --mode test --batch_size 200

# python run_judge_collections.py --judge_type PromptJudge --runs_path runs/qwen --prompt_instruction naive_judge --saving_dir judge_collections/qwen/test/naive --mode test --model_path models/Qwen3-4B-Instruct-2507 --batch_size 200
# python run_judge_collections.py --judge_type PromptJudge --runs_path runs/qwen_default --prompt_instruction naive_judge --saving_dir judge_collections/qwen_default/test/naive --mode test --model_path models/Qwen3-4B-Instruct-2507 --batch_size 200

#RAG Pairwise
# python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs/llama --saving_dir judge_collections/llama/test/pairwise_rag_judge --pairwise_rag --mode test
# python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs/llama_default --saving_dir judge_collections/llama_default/test/pairwise_rag_judge --pairwise_rag --mode test

# python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs/qwen --saving_dir judge_collections/qwen/test/pairwise_rag_judge --pairwise_rag --model_path models/Qwen3-4B-Instruct-2507 --mode test
# python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs/qwen_default --saving_dir judge_collections/qwen_default/test/pairwise_rag_judge --pairwise_rag --model_path models/Qwen3-4B-Instruct-2507 --mode test


#Zeroshot Pairwise
# python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs/llama --saving_dir judge_collections/llama/test/pairwise_zeroshot_judge --mode test
# python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs/llama_default --saving_dir judge_collections/llama_default/test/pairwise_zeroshot_judge --mode test

python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs/qwen --saving_dir judge_collections/qwen/test/pairwise_zeroshot_judge --pairwise_rag --model_path models/Qwen3-4B-Instruct-2507 --mode test
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs/qwen_default --saving_dir judge_collections/qwen_default/test/pairwise_zeroshot_judge --pairwise_rag --model_path models/Qwen3-4B-Instruct-2507 --mode test


#Votinhg Naive
# python run_judge_collections.py --judge_type PromptJudge --runs_path runs/llama --prompt_instruction naive_judge --saving_dir judge_collections/llama/test/voting_naive --mode test --batch_size 200 --n_generations 3
# python run_judge_collections.py --judge_type PromptJudge --runs_path runs/llama_default --prompt_instruction naive_judge --saving_dir judge_collections/llama_default/test/voting_naive --mode test --batch_size 200 --n_generations 3

# python run_judge_collections.py --judge_type PromptJudge --runs_path runs/qwen --prompt_instruction naive_judge --saving_dir judge_collections/qwen/test/voting_naive --mode test --model_path models/Qwen3-4B-Instruct-2507 --batch_size 200 --n_generations 3
# python run_judge_collections.py --judge_type PromptJudge --runs_path runs/qwen_default --prompt_instruction naive_judge --saving_dir judge_collections/qwen_default/test/voting_naive --mode test --model_path models/Qwen3-4B-Instruct-2507 --batch_size 200 --n_generations 3



