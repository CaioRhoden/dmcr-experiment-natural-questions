#!/bin/bash
#SBATCH --job-name=pre_collections
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_pre_collections.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_pre_collections.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=25G
#SBATCH --time=24:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-4
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include



SEEDS=(1 4 54 61 73)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}

    
# echo "-----------------------------------------------"
# echo "RUNNING SETUP"
# python run_datamodels.py \
#     --seed $S \
#     --run_type setup

# echo "-----------------------------------------------"
# echo "RUNNING PRE_COLLECTIONS TRAIN "
# python run_datamodels.py \
#     --seed $S \
#     --run_type pre_collections \
#     --start_idx 0 \
#     --end_idx 2000 \
#     --checkpoint 250 \
#     --mode train


# echo "RUNNING PRE_COLLECTIONS TEST"
# python run_datamodels.py \
#     --seed $S \
#     --run_type pre_collections \
#     --start_idx 0 \
#     --end_idx 200 \
#     --checkpoint 200 \
#     --mode test



# INST="You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."

    
# echo "-----------------------------------------------"
# echo "RUNNING SETUP"
# python run_datamodels.py \
#     --seed $S \
#     --root_path runs_opt \
#     --run_type setup

# echo "-----------------------------------------------"
# echo "RUNNING PRE_COLLECTIONS TRAIN "
# python run_datamodels.py \
#     --seed $S \
#     --run_type pre_collections \
#     --start_idx 0 \
#     --root_path runs_opt \
#     --end_idx 2000 \
#     --checkpoint 250 \
#     --instruction "$INST" \
#     --mode train


# echo "RUNNING PRE_COLLECTIONS TEST"
# python run_datamodels.py \
#     --seed $S \
#     --run_type pre_collections \
#     --root_path runs_opt \
#     --start_idx 0 \
#     --end_idx 200 \
#     --checkpoint 200 \
#     --instruction "$INST" \
#     --mode test

    
INST="You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."

    
echo "-----------------------------------------------"
echo "RUNNING SETUP"
python run_datamodels.py \
    --seed $S \
    --root_path datamodels_runs/extraction \
    --run_type setup

echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --seed $S \
    --run_type pre_collections \
    --start_idx 0 \
    --root_path datamodels_runs/extraction \
    --end_idx 2000 \
    --checkpoint 250 \
    --instruction extraction \
    --mode train


echo "RUNNING PRE_COLLECTIONS TEST"
python run_datamodels.py \
    --seed $S \
    --run_type pre_collections \
    --root_path datamodels_runs/extraction \
    --start_idx 0 \
    --end_idx 200 \
    --checkpoint 200 \
    --instruction extraction \
    --mode test
