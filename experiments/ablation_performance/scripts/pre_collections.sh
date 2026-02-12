#!/bin/bash
#SBATCH --job-name=pc_ablation
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_pc_ablation.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_pc_ablation.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=139G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=1
#SBATCH --exclude=gpu03
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
    
echo "-----------------------------------------------"
echo "RUNNING SETUP"
python run_datamodels.py \
    --seed $S \
    --run_type setup

echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --seed $S \
    --run_type pre_collections \
    --start_idx 18000 \
    --end_idx 20000 \
    --checkpoint 1000 \
    --mode train

# echo "RUNNING PRE_COLLECTIONS TEST"
# python run_datamodels.py \
#     --seed $S \
#     --run_type pre_collections \
#     --start_idx 0 \
#     --end_idx 1000 \
#     --checkpoint 1000 \
#     --mode test


            
