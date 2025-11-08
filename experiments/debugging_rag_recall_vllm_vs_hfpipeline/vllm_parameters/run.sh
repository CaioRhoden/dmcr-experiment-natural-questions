source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"


python3 generate_random_samples.py --n_samples 2500 --seed 42 --partition dev


## Run all RAG
# sh scripts/run_rag.sh

## DATAMODELS
### 