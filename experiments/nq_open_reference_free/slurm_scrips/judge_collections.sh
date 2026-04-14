#!/bin/bash
#SBATCH --job-name=judge_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_judge_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_judge_collections.err
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


case ${SLURM_ARRAY_TASK_ID} in
    0)
        # Naive Judge
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --saving_dir judge_collections/naive/train
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --saving_dir judge_collections/naive/test --mode test
        ;;
    1)
        # Voting Naive Judge
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --saving_dir judge_collections/voting_naive/train --n_generations 3
        python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --saving_dir judge_collections/voting_naive/test --mode test --n_generations 3
        ;;
    2)
        # Pairwise Judge RAG
        python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_dir judge_collections/pairwise_rag_judge/train2 --pairwise_rag --start_idx 2166000 --end_idx 4332000
        ;;

    3)
        # Pairwise Judge Zero-Shot
        python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_dir judge_collections/pairwise_rag_judge/train3 --pairwise_rag --start_idx 7039500 --batch_size 180500
        python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_dir judge_collections/pairwise_zeroshot_judge/train3 --start_idx 7039500 --batch_size 180500

        ;;
    4)
        # Pairwise Judge RAG
        python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_dir judge_collections/pairwise_rag_judge/train1 --pairwise_rag --end_idx 2166000
        ;;

    5)
        # Pairwise Judge RAG
        python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_dir judge_collections/pairwise_zeroshot_judge/train1  --end_idx 3249000
        ;;

    6)
        # Pairwise Judge Zero-Shot
        python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_dir judge_collections/pairwise_zeroshot_judge/train2 --start_idx 5415000 --end_idx 7039500
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