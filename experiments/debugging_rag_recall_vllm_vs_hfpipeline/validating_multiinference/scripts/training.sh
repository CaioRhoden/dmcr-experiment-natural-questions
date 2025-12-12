#!/bin/bash
#SBATCH --job-name=instruction_2_training_rag_debbuging_validation
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_instruction_2_training_rag_debbuging_validation.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_instruction_2_training_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-4
#SBATCH --partition=rtx8000


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"



SEEDS=(1)
INSTRUCTIONS=(2)

STARTING_IDX=$((SLURM_ARRAY_TASK_ID*100))
ENDING_IDX=$((STARTING_IDX + 100))


echo "Running setup for seed $SEED and instruction index 1"

for SEED in "${SEEDS[@]}"; do
    for INSTRUCTION_IDX in "${INSTRUCTIONS[@]}"; do
        echo "-----------------------------------------------"
        echo "RUNNING TRAINING FOR INSTRUCTION INDEX $INSTRUCTION_IDX"
        python run_datamodels.py \
            --seed $SEED \
            --instruction_idx $INSTRUCTION_IDX \
            --start_idx $STARTING_IDX \
            --end_idx $ENDING_IDX \
            --run_type training \
            --num_subprocesses 4 \
            --checkpoint 25 \
            --evaluator Judge
    done
done


