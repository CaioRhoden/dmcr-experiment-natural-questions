#!/bin/bash
#SBATCH --job-name=pc_rag_debbuging_validation
#SBATCH --output=/home/users/caio.rhoden/slurm/%A_%a_pc_rag_debbuging_validation.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%A_%a_pc_rag_debbuging_validation.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=139G
#SBATCH --time=48:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=10-14
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
INST=${INSTUCTIONS[$TMP_INSTRUCTION]}

    
echo "Running setup for seed $S and instruction index $INST"
echo "-----------------------------------------------"
echo "RUNNING SETUP"
python run_datamodels.py \
    --seed $S \
    --instruction_idx $INST \
    --run_type setup

echo "-----------------------------------------------"
echo "RUNNING PRE_COLLECTIONS TRAIN "
python run_datamodels.py \
    --seed $S \
    --instruction_idx $INST \
    --run_type pre_collections \
    --start_idx 0 \
    --end_idx 2000 \
    --checkpoint 200 \
    --mode train

# echo "RUNNING PRE_COLLECTIONS TEST"
# python run_datamodels.py \
#     --seed $S \
#     --instruction_idx $INST \
#     --run_type pre_collections \
#     --start_idx 0 \
#     --end_idx 200 \
#     --checkpoint 200 \
#     --mode test

    
done
        
