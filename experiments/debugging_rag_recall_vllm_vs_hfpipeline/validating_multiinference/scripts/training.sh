#!/bin/bash
#SBATCH --job-name=MI_training_rag_debbuging_validation
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_iMItraining_rag_debbuging_validation.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_iMItraining_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-4,10-14



source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"



SEEDS=(1 4 54 61 73)
INSTUCTIONS=(0 1 2)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}
TMP_INSTRUCTION=$((SLURM_ARRAY_TASK_ID / 5))
INSTRUCTION_IDX=${INSTUCTIONS[$TMP_INSTRUCTION]}


echo "Running setup for seed $S and instruction index $INSTRUCTION_IDX"


echo "-----------------------------------------------"
echo "RUNNING TRAINING FOR INSTRUCTION INDEX $INSTRUCTION_IDX"
python run_datamodels.py \
    --seed $SEED \
    --instruction_idx $INSTRUCTION_IDX \
    --run_type training \
    --num_subprocesses 5 \
    --checkpoint 50 \
    --evaluator Judge


