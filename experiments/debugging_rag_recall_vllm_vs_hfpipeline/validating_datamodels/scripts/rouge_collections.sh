#!/bin/bash
#SBATCH --job-name=rg_rag_debbuging_validation
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_rg_rag_debbuging_validation.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_rg_rag_debbuging_validation.err
#SBATCH --cpus-per-task=7
#SBATCH --mem=15G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=rtx5000,a5000,l40s
#SBATCH --array=0-9


SEEDS=(1 4 54 61 73)
INSTRUCTIONS=(0 1 2)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}
INST_ID=$((SLURM_ARRAY_TASK_ID / 5))
INST=${INSTRUCTIONS[$INST_ID]}





source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"


echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --seed $S \
    --instruction_idx $INST \
    --run_type collections \
    --start_idx 0 \
    --end_idx 1000000 \
    --checkpoint 50000 \
    --num_subprocesses 4 \
    --mode train


