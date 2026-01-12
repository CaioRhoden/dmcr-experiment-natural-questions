#!/bin/bash
#SBATCH --job-name=multi_judge
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_multi_judge.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_multi_judge.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=40G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=5-9
#SBATCH --exclude=gpu03
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include

SEEDS=(1 4 54 61 73)
INSTUCTIONS=(0 1 2)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}
TMP_INSTRUCTION=$((SLURM_ARRAY_TASK_ID / 5))
INSTRUCTION_IDX=${INSTUCTIONS[$TMP_INSTRUCTION]}

echo "RUNNING SETUP"
python run_datamodels.py \
    --seed $S \
    --instruction_idx $INSTRUCTION_IDX \
    --run_type setup



echo "RUNNING COLLECTIONS TRAIN"
python run_datamodels.py \
    --seed $S \
    --instruction_idx $INSTRUCTION_IDX \
    --run_type collections \
    --batch_size 1000000 \
    --start_idx 0 \
    --end_idx 1000000 \
    --checkpoint 25000 \
    --num_subprocesses 1 \
    --evaluator Judge \
    --mode train \
    --multiple_grading


echo "RUNNING COLLECTIONS TRAIN"
python run_datamodels.py \
    --seed $S \
    --instruction_idx $INSTRUCTION_IDX \
    --run_type collections \
    --batch_size 100000\
    --start_idx 0 \
    --end_idx 100000 \
    --checkpoint 25000 \
    --num_subprocesses 1 \
    --evaluator Judge \
    --mode test \
    --multiple_grading
        
