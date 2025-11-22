#!/bin/bash
#SBATCH --job-name=pc_rag_debbuging_validation
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_pc_rag_debbuging_validation.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_pc_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=43G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=1-4
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include

SEEDS=(4 54 61 73)
INSTUCTIONS=(0)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

for INSTRUCTION_IDX in "${INSTUCTIONS[@]}"; do
    
    echo "Running setup for seed $S and instruction index $INSTRUCTION_IDX"
    echo "-----------------------------------------------"
    echo "RUNNING SETUP"
    python run_datamodels.py \
        --seed $S \
        --instruction_idx $INSTRUCTION_IDX \
        --run_type setup

    echo "-----------------------------------------------"
    echo "RUNNING PRE_COLLECTIONS TRAIN "
    python run_datamodels.py \
        --seed $S \
        --instruction_idx $INSTRUCTION_IDX \
        --run_type pre_collections \
        --start_idx 0 \
        --end_idx 2000 \
        --checkpoint 200 \
        --mode train

    echo "RUNNING PRE_COLLECTIONS TEST"
    python run_datamodels.py \
        --seed $S \
        --instruction_idx $INSTRUCTION_IDX \
        --run_type pre_collections \
        --start_idx 0 \
        --end_idx 200 \
        --checkpoint 200 \
        --mode test

    
done
        
