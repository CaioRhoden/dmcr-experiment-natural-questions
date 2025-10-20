source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"


python3 generate_random_samples.py --n_samples 500 --seed 10 --partition dev

# Run all zero shots
sh scripts/run_zero_shot.sh

## Run all RAG
sh scripts/run_rag.sh

## DATAMODELS
### 