#!/bin/bash
#SBATCH --job-name=run_rag
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_run_rag.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_run_rag.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=50G
#SBATCH --time=01:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-4
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include


SEEDS=(1 4 54 61 73)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}
INST="You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."

# python run_rag.py --seed $S

python run_rag.py --seed $S --root_path runs_opt --instruction "$INST"