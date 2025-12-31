#!/bin/bash
#SBATCH --job-name=MI_rag_debbuging_validation
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_MI_rag_debbuging_validation.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_MI_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=23G
#SBATCH --time=3:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-4
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SEEDS=(1 4 54 61 73)
SENTENCES=(3 5)


S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

for SENTENCE in "${SENTENCES[@]}"; do
     python run_rag.py \
         --seed $S \
         --num_sentences $SENTENCE
done
    
