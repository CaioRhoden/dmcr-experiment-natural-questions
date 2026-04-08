#!/bin/bash
#SBATCH --job-name=run_rag_extraction
#SBATCH --output=/home/caio.rhoden/slurm/%A_%a_run_rag_extraction.out
#SBATCH --error=/home/caio.rhoden/slurm/%A_%a_run_rag_extraction.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=20G
#SBATCH --time=01:00:00
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --array=0-4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBARCH --partition=a5000


source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn



SEEDS=(1 4 54 61 73)

S_ID=$((SLURM_ARRAY_TASK_ID % 5))
S=${SEEDS[$S_ID]}
# INST="You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."

# python run_rag.py --seed $S

# python run_rag.py --seed $S --root_path runs_opt --instruction "$INST"

EXTRACTION_INSTRUCTION="You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES. Begin your answer by providing a very short explanation. Be as objective as possible. After providing your explanation, please generate your response by strictly following this format: \"RESPONSE: [[<response>]]\"."
python run_rag.py --seed $S --root_path datamodels_runs/extraction --mmodel_run_id rag_extraction --instruction "$EXTRACTION_INSTRUCTION" --only_generate --lm_configs.max_new_tokens 200

REASONING_INSTRUCTION="You are given a question and you MUST respond giving a answer answer (max 5 tokens), respond with NO-RES if you cannot answer. Begin your answer by providing a very short explanation. Be as objective as possible. After providing your explanation, please generate your response by strictly following this format: \"RESPONSE: [[<response>]]\"."
python run_rag.py --seed $S --root_path datamodels_runs/extraction --model_run_id rag_reasoning --instruction "$REASONING_INSTRUCTION" --only_generate --lm_configs.max_new_tokens 200
22

DEFAULT_INSTRUCTION="You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens"
python run_rag.py --seed $S --root_path datamodels_runs/extraction --model_run_id rag --instruction "$DEFAULT_INSTRUCTION" --only_generate --lm_configs.max_new_tokens 15
