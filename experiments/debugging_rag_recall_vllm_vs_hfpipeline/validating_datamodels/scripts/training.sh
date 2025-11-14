#!/bin/bash
#SBATCH --job-name=rag_debbuging_validation
#SBATCH --output=/home/users/caio.rhoden/slurm/%j_rag_debbuging_validation.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%j_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=138G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate nq
# export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include

SEEDS=(1 4 54 61 73)
INSTUCTIONS=(0)

for S in "${SEEDS[@]}"; do
    for 
     
    echo "Running setup for seed $S and instruction index $INSTRUCTION_IDX"
    echo "-----------------------------------------------"
    echo "RUNNING SETUP"
    python run_datamodels.py \
        --seed $S \
        --instruction_idx 0 \
        --run_type setup
    echo "-----------------------------------------------"
        # echo "TRAINING DATAMODELS"

    python run_datamodels.py \
        --seed $S \
        --instruction_idx 0 \
        --run_type training \
        --num_subprocesses 1 \
        --evaluator Judge
done