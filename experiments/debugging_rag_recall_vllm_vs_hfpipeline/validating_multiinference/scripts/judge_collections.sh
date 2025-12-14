#!/bin/bash
#SBATCH --job-name=judge_collectiong_multi_sentences
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_judge_collectiong_multi_sentences.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_judge_collectiong_multi_sentences.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=139G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-4
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
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

for NUM_SENTENCES_IDX in "${NUM_SENTENCES[@]}"; do
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

    
done
        
