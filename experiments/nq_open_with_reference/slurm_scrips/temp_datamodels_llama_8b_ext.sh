#!/bin/bash
#SBATCH --job-name=ext_llama_8b_pre_coll
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_llama_8b_ext_pre_coll.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_llama_8b_ext_pre_coll.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-3


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
START_IDX=(0 500 1000 1500)
END_IDX=(500 1000 1500 2000)

echo "-----------------------------------------------"
echo "RUNNING SETUP"
python run_datamodels.py \
    --run_type setup \
    --root_path runs/llama_8b_extraction

echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --run_type pre_collections \
    --start_idx ${START_IDX[$SLURM_ARRAY_TASK_ID]} \
    --end_idx ${END_IDX[$SLURM_ARRAY_TASK_ID]} \
    --checkpoint 250 \
    --instruction extraction  \
    --lm_configs.max_new_tokens 200 \
    --root_path runs/llama_8b_extraction \
    --language_model_path models/Llama-3.1-8B-Instruct \
    --mode train

## if SLURTM_ARRAY_TASK_ID is 0, run test too
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "-----------------------------------------------"
    echo "RUNNING PRE_COLLECTIONS TEST "
    python run_datamodels.py \
        --run_type pre_collections \
        --start_idx 0 \
        --end_idx 200 \
        --checkpoint 200 \
        --instruction extraction  \
        --lm_configs.max_new_tokens 200 \
        --root_path runs/llama_8b_extraction \
        --language_model_path models/Llama-3.1-8B-Instruct \
        --mode test
fi