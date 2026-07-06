#!/bin/bash
#SBATCH --job-name=ext_llama_judge_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_ext_llama_judge_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_ext_llama_judge_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=2


source ~/miniconda3/bin/activate
conda activate nq
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline
START_IDX=(0 500 1000 1500)
END_IDX=(499 999 1499 1999)

# Get start and end indices for this array task
START=${START_IDX[$SLURM_ARRAY_TASK_ID]}
END=${END_IDX[$SLURM_ARRAY_TASK_ID]}


##Train
python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --runs_path runs/llama_8b_extraction --saving_dir judge_collections/llama_8b_extraction/naive --model_path models/Llama-3.1-8B-Instruct --start_collection_idx $START --end_collection_idx $END
python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --runs_path runs/llama_8b_extraction --saving_dir judge_collections/llama_8b_extraction/voting_naive --model_path models/Llama-3.1-8B-Instruct --n_generations 3 --start_collection_idx $START --end_collection_idx $END

##Test - only run for array task 0
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --runs_path runs/llama_8b_extraction --saving_dir judge_collections/llama_8b_extraction/test/naive --model_path models/Llama-3.1-8B-Instruct --mode test --batch_size 200
    python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --runs_path runs/llama_8b_extraction --saving_dir judge_collections/llama_8b_extraction/test/voting_naive --model_path models/Llama-3.1-8B-Instruct --n_generations 3 --mode test --batch_size 200
fi
