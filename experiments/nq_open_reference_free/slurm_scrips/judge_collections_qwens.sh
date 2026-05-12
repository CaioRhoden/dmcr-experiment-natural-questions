#!/bin/bash
#SBATCH --job-name=pairwise_qwen_prompt_judge_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_pairwise_qwen_prompt_judge_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_pairwise_qwen_prompt_judge_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu03
#SBATCH --array=0-1


source ~/miniconda3/bin/activate
conda activate nq
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline


case ${SLURM_ARRAY_TASK_ID} in
    0)
        # Naive Judge
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction pairwise_judge --runs_path runs/qwen --saving_dir judge_collections/qwen/pairwise_rag_judge --model_path models/Qwen3-4B-Instruct-2507 --pairwise_rag
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction pairwise_judge --runs_path runs/qwen-default --saving_dir judge_collections/qwen_default/pairwise_rag_judge --model_path models/Qwen3-4B-Instruct-2507 --pairwise_rag
        ;;

    1)
        # Pairwise Judge RAG
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction pairwise_judge --runs_path runs/qwen --saving_dir judge_collections/qwen/pairwise_zeroshot_judge --model_path models/Qwen3-4B-Instruct-2507
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction pairwise_judge --runs_path runs/qwen-default --saving_dir judge_collections/qwen_default/pairwise_zeroshot_judge --model_path models/Qwen3-4B-Instruct-2507
        ;;
    *)
        echo "Error: Unmapped SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac
