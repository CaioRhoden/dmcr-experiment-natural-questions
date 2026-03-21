#!/bin/bash
#SBATCH --job-name=generation_model_analysis
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_generation_model_analysis.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_generation_model_analysis.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=44G
#SBATCH --time=2:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include

SEEDS=(1 4 54 61 73)
S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag groundtruth_500

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag judge_500

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag groundtruth_1k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag judge_1k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag groundtruth_2k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag judge_2k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag groundtruth_3k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag judge_3k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag groundtruth_5k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag judge_5k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag groundtruth_10k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag judge_10k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag groundtruth_15k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag judge_15k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag groundtruth_19k

python run_datamodels.py \
        --seed $S \
        --run_type generation \
        --batch_size 500 \
        --model_tag judge_19k
