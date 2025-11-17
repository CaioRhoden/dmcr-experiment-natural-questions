#!/bin/bash
#SBATCH --job-name=training_rag_debbuging_validation
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_training_rag_debbuging_validation.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_training_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-4
#SBATCH --partition=rtx8000


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"


SEEDS=(1 4 54 61 73)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}


echo "Running setup for seed $S and instruction index 1"
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
    --instruction_idx 2 \
    --run_type training \
    --num_subprocesses 1 \
    --evaluator Judge
