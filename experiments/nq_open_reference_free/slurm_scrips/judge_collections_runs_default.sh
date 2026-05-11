#!/bin/bash
#SBATCH --job-name=qwen_prompt_judge_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_qwen_prompt_judge_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_qwen_prompt_judge_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu03
#SBATCH --array=0-3


source ~/miniconda3/bin/activate
conda activate nq
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline


case ${SLURM_ARRAY_TASK_ID} in
    0)
        # Naive Judge
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --runs_path runs/qwen --saving_dir judge_collections/qwen/naive --model_path models/Qwen3-4B-Instruct-2507
        ;;

    1)
        # Pairwise Judge RAG
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --runs_path runs/qwen_default --saving_dir judge_collections/qwen_default/naive --model_path models/Qwen3-4B-Instruct-2507
        ;;

    2)
        # Pairwise Judge Zero-Shot
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --runs_path runs/qwen --saving_dir judge_collections/qwen/voting_naive  --n_generations 3 --model_path models/Qwen3-4B-Instruct-2507
        ;;
    # 1)
    #     # Pairwise Judge RAG
    #     python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs_default --saving_dir judge_collections/runs_default/pairwise_rag_judge --pairwise_rag
    #     ;;

    # 2)
    #     # Pairwise Judge Zero-Shot
    #     python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --runs_path runs_default --saving_dir judge_collections/runs_default/pairwise_zeroshot_judge
    #     ;;
    3)
        # Voting Naive Judge
         python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --runs_path runs/qwen_default --saving_dir judge_collections/qwen_default/voting_naive --n_generations 3 --model_path models/Qwen3-4B-Instruct-2507
        ;;
    *)
        echo "Error: Unmapped SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

    # 3)
    #     # Voting Pairwise Judge RAG
    #     python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_dir judge_collections/voting_pairwise_rag_judge/train --pairwise_rag --n_generations 3
    #     python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_dir judge_collections/voting_pairwise_rag_judge/test --pairwise_rag --mode test --n_generations 3
    #     ;;

        # 5)
    #     # Voting Pairwise Judge Zero-Shot
    #     python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_dir judge_collections/voting_pairwise_zeroshot_judge/train --n_generations 3
    #     python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_dir judge_collections/voting_pairwise_zeroshot_judge/test --mode test --n_generations 3
    #     ;;


    #
    #  python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --runs_path runs/qwen --saving_dir judge_collections/qwen/naive --start_collection_idx 1 --end_collection_idx 5 --batch_size 2