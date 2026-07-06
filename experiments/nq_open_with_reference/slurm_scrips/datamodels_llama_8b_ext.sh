#!/bin/bash
#SBATCH --job-name=llama_8b_ext_pre_coll
#SBATCH --output=/home/caio.rhoden/slurm/%j_llama_8b_ext_pre_coll.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_llama_8b_ext_pre_coll.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn


echo "-----------------------------------------------"
echo "RUNNING SETUP"
python run_datamodels.py \
    --run_type setup

echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --run_type pre_collections \
    --start_idx 0 \
    --end_idx 2000 \
    --checkpoint 250 \
    --mode train