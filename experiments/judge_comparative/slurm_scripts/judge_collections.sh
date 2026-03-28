#!/bin/bash
#SBATCH --job-name=judge_collections
#SBATCH --output=/home/caio.rhoden/slurm/%j_judge_collections.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_judge_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=03:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/train/pairwise_rag_judge.feather --pairwise_rag 
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/train/pairwise_zeroshot_judge.feather
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/train_opt/pairwise_zeroshot_judge.feather --pre_collections_path experiment_81_opt/datamodels/pre_collections 
python run_judge_collections.py --judge_type PairwiseJudge --prompt_instruction pairwise_judge --saving_path judge_collections/train_opt/pairwise_rag_judge.feather --pairwise_rag --pre_collections_path experiment_81_opt/datamodels/pre_collections 