#!/bin/bash
#SBATCH --job-name=train_extraction_nq_open_gold_pre_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%j_train_extraction_nq_open_gold_pre_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%j_train_extraction_nq_open_gold_pre_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline



echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --run_type pre_collections \
    --start_idx 1000 \
    --end_idx 2000 \
    --checkpoint 250 \
    --root_path runs_default \
    --instruction default \
    --mode train


# echo "RUNNING PRE_COLLECTIONS TEST"
# python run_datamodels.py \
#     --run_type pre_collections \
#     --start_idx 0 \
#     --end_idx 200 \
#     --checkpoint 200 \
#     --root_path runs_default \
#     --instruction default \
#     --mode test


# echo "RUNNING COLLECTIONS TRAIN"
# python run_datamodels.py \
#     --seed $S \
#     --run_type collections \
#     --start_idx 0 \
#     --end_idx 1000000 \
#     --checkpoint 50000 \
#     --batch_size 50000 \
#     --num_subprocesses 4 \
#     --root_path runs_opt \
#     --evaluator Rouge-L \
#     --mode train

    
# python run_datamodels.py \
#     --seed $S \
#     --run_type collections \
#     --start_idx 0 \
#     --end_idx 100000 \
#     --checkpoint 50000 \
#     --batch_size 50000 \
#     --num_subprocesses 2 \
#     --evaluator Rouge-L \
#     --root_path runs_opt \
#     --mode test
