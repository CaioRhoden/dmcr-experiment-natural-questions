#!/bin/bash
#SBATCH --job-name=rag_debbuging_validation
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_rag_debbuging_validation.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=45G
#SBATCH --time=02:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-2
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


GEN=("vllm1" "vllm3" "vllm5")
NUM_SENTENCES=(1 3 5)

EXP=${GEN[$SLURM_ARRAY_TASK_ID]}
NUM_SENT=${NUM_SENTENCES[$SLURM_ARRAY_TASK_ID]}


python run_rag.py \
    --exp_name $EXP \
    --num_sentences $NUM_SENT

    

    
