#!/bin/bash
#SBATCH --job-name=generating_rag_debbuging_validation
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_generating_rag_debbuging_validation.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_generating_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=22G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=rtx5000
#SBATCH --array=0-4


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"

SEEDS=(1 4 54 61 73)
INSTRUCTIONS=(0 1 2)

STARTING_IDX=$((SLURM_ARRAY_TASK_ID*200000))
ENDING_IDX=$((STARTING_IDX + 200000))
for SEED in "${SEEDS[@]}"; do
    for INSTRUCTION_IDX in "${INSTRUCTIONS[@]}"; do
        echo "-----------------------------------------------"
        echo "RUNNING PRE_COLLECTIONS TRAIN "
        python run_datamodels.py \
            --seed  $SEED\
            --instruction_idx $INSTRUCTION_IDX \
            --run_type collections \
            --start_idx $STARTING_IDX \
            --end_idx $ENDING_IDX \
            --checkpoint 25000 \
            --num_subprocesses 4 \
            --mode train
    done
done

