#!/bin/bash
#SBATCH --job-name=rag_nq_open
#SBATCH --output=/home/caio.rhoden/slurm/%j_rag_nq_open.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_rag_nq_open.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=06:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=l40s,a5000,rtx8000


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Running RAG"
python run_rag.py --model_run_id rag --instruction default --tags rag_default --only_generate
echo "Running RAG with extraction instruction"
python run_rag.py --model_run_id rag_extraction --instruction extraction --model_run_id rag_extraction --tags rag_extraction --only_generate --lm_configs.max_new_tokens 200
