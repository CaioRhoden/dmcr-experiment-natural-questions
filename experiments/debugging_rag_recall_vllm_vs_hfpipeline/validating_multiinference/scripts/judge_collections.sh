#!/bin/bash
#SBATCH --job-name=judge_collectiong_multi_sentences
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_judge_collectiong_multi_sentences.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_judge_collectiong_multi_sentences.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=139G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-9
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclude=gpu03


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include

SEEDS=(1 4 54 61 73)
NUM_SENTENCES=(3 5)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}
NS_ID=$((SLURM_ARRAY_TASK_ID / 5))
NUM_SENTENCES_IDX=${NUM_SENTENCES[$NS_ID]}


echo "RUNNING SETUP"
python run_datamodels.py \
    --seed $S \
    --num_sentences $NUM_SENTENCES_IDX \
    --run_type setup

echo "RUNNING COLLECTIONS TRAIN"
python run_datamodels.py \
    --seed $S \
    --num_sentences $NUM_SENTENCES_IDX \
    --run_type collections \
    --start_idx 0 \
    --end_idx 1000000 \
    --checkpoint 20000 \
    --num_subprocesses 1 \
    --batch_size 1000000 \
    --evaluator Judge \
    --mode train

    

        
