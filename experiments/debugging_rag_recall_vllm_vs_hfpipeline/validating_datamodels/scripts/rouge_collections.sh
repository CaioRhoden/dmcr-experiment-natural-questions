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
#SBATCH --partition=a5000
#SBATCH --array=0

START_IDX=620000
END_IDX=700000



source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"


echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --seed 1 \
    --instruction_idx 0 \
    --run_type collections \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    --checkpoint 20000 \
    --mode train


