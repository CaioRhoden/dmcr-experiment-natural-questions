#!/bin/bash
#SBATCH --job-name=pre_collection_qwen
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_pre_collection_qwen.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_pre_collection_qwen.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=50G
#SBATCH --time=48:00:00
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
S=${SEEDS[SLURM_ARRAY_TASK_ID]}
    
echo "-----------------------------------------------"
echo "RUNNING SETUP"
python run_datamodels.py \
    --seed $S \
    --run_type setup \
    --root_path runs_qwen

echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --seed $S \
    --run_type pre_collections \
    --start_idx 0 \
    --end_idx 2000 \
    --checkpoint 250 \
    --language_model_path models/Qwen3-8B \
    --root_path runs_qwen \
    --mode train

echo "RUNNING PRE_COLLECTIONS TEST"
python run_datamodels.py \
    --seed $S \
    --run_type pre_collections \
    --start_idx 0 \
    --end_idx 200 \
    --checkpoint 200 \
    --language_model_path models/Qwen3-8B \
    --root_path runs_qwen \
    --mode test


            
