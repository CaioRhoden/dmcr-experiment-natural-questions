#!/bin/bash
#SBATCH --job-name=judge_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%j_judge_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%j_judge_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=50G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#NAive Judge
python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --saving_path judge_collections/naive/train.feather
python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --saving_path judge_collections/naive/test.feather --mode test

#Voting Naive Judge
python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --saving_path judge_collections/voting_naive/train.feather --n_generations 3
python run_judge_collections.py --judge_type PromptJudge --prompt_instruction naive_judge --saving_path judge_collections/voting_naive/test.feather --mode test --n_generations 3

#Pairwise Judge RAG
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/pairwise_rag_judge/train.feather --pairwise_rag 
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/pairwise_rag_judge/test.feather --pairwise_rag --mode test

#Voting Pairwise Judge RAG
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/voting_pairwise_rag_judge/train.feather --pairwise_rag --n_generations 3
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/voting_pairwise_rag_judge/test.feather --pairwise_rag --mode test --n_generations 3

#Pairwise Judge Zero-Shot
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/pairwise_zeroshot_judge/train.feather
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/pairwise_zeroshot_judge/test.feather --mode test

#Voting Pairwise Judge Zero-Shot
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/voting_pairwise_zeroshot_judge/train.feather --n_generations 3
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/voting_pairwise_zeroshot_judge/test.feather --mode test --n_generations 3


#Pairwise Judge RAG recall prompt
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction recall_pairwise_judge --saving_path judge_collections/recall_pairwise_rag_judge/train.feather --pairwise_rag 
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction recall_pairwise_judge --saving_path judge_collections/recall_pairwise_rag_judge/test.feather --pairwise_rag --mode test

#Voting Pairwise Judge RAG recall prompt
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction recall_pairwise_judge --saving_path judge_collections/voting_recall_pairwise_rag_judge/train.feather --pairwise_rag  --n_generations 3
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction recall_pairwise_judge --saving_path judge_collections/voting_recall_pairwise_rag_judge/test.feather --pairwise_rag --mode test --n_generations 3

#Pairwise Judge Zero-Shot recall prompt
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction recall_pairwise_judge --saving_path judge_collections/recall_pairwise_zeroshot_judge/train.feather
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction recall_pairwise_judge --saving_path judge_collections/recall_pairwise_zeroshot_judge/test.feather --mode test

#Voting Pairwise Judge Zero-Shot recall prompt
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction recall_pairwise_judge --saving_path judge_collections/voting_recall_pairwise_zeroshot_judge/train.feather --n_generations 3
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction recall_pairwise_judge --saving_path judge_collections/voting_recall_pairwise_zeroshot_judge/test.feather --mode test --n_generations 3