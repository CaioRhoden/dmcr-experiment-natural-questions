#!/bin/bash
#SBATCH --job-name=MI_training_rag_debbuging_validation
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_iMItraining_rag_debbuging_validation.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_iMItraining_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=7,9
#SBATCH --partition=rtx5000,a5000,l40s



source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"



SEEDS=(1 4 54 61 73)
NINFERENCES=(3 5)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
SEED=${SEEDS[$S_ID]}
TMP_INFERENCE=$((SLURM_ARRAY_TASK_ID / 5))
NINFERENCE=${NINFERENCES[$TMP_INFERENCE]}




echo "-----------------------------------------------"
echo "RUNNING TRAINING FOR SEED $S AND NUM INFERENCES $NINFERENCE"
python run_datamodels.py \
    --seed $SEED \
    --num_sentences $NINFERENCE \
    --run_type training \
    --num_subprocesses 5 \
    --checkpoint 50 \
    --evaluator Judge


