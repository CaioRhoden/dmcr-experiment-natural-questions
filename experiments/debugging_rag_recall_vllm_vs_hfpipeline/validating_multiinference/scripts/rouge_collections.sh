#!/bin/bash
#SBATCH --job-name=MI_rc_rag_debbuging_validation
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_MI_rc_rag_debbuging_validation.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_MI_rc_rag_debbuging_validation.err
#SBATCH --cpus-per-task=7
#SBATCH --mem=15G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=rtx5000,a5000
#SBATCH --array=0-9


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"

SEEDS=(1 4 54 61 73)
NUM_SENTENCES=(3 5)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
SEED=${SEEDS[$S_ID]}
NS_ID=$((SLURM_ARRAY_TASK_ID / 5))
NUM_SENTENCES_IDX=${NUM_SENTENCES[$NS_ID]}

echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN FOR SEED $S AND NUM SENTENCES $NUM_SENTENCES_IDX"
python run_datamodels.py \
    --seed  $SEED\
    --num_sentences $NUM_SENTENCES_IDX \
    --run_type collections \
    --start_idx 0 \
    --end_idx 1000000 \
    --checkpoint 50000 \
    --num_subprocesses 5 \
    --mode train


