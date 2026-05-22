#!/bin/bash
#SBATCH --job-name=rag_nq_open
#SBATCH --output=/home/users/caio.rhoden/slurm/%j_rag_nq_open.out
#SBATCH --error=/home/users/caio.rhoden/slurm/%j_rag_nq_open.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=02:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --mail-type=BEGIN,END,FAIL


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# echo "Running RAG"
# python run_rag.py --model_run_id rag --instruction default --tags rag_default --only_generate
# echo "Running RAG with extraction instruction"
# python run_rag.py --model_run_id rag_extraction --instruction extraction --model_run_id rag_extraction --tags rag_extraction --only_generate --lm_configs.max_new_tokens 200



echo "Running RAG"
# python run_rag.py --model_run_id rag --instruction default --tags rag_default --only_generate --root_path runs/qwen_default --language_model_path models/Qwen3-4B-Instruct-2507
# echo "Running RAG with extraction instruction"
# python run_rag.py --model_run_id rag_extraction --instruction extraction --model_run_id rag_extraction --tags rag_extraction --only_generate --lm_configs.max_new_tokens 200 --root_path runs/qwen --language_model_path models/Qwen3-4B-Instruct-2507


## LLama 8B
python run_rag.py --model_run_id rag_extraction --instruction extraction --tags rag_extraction --lm_configs.max_new_tokens 200  --root_path runs/llama_8b_extraction --language_model_path models/Llama-3.1-8B-Instruct
python run_rag.py --model_run_id rag_instruction --instruction default --tags rag_extraction  --root_path runs/llama_8b_instruction --language_model_path models/Llama-3.1-8B-Instruct