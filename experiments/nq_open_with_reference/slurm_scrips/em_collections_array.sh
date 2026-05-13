#!/bin/bash
#SBATCH --job-name=em_nq_open_collections_array
#SBATCH --output=/home/caio.rhoden/slurm/%j_em_nq_open_collections_array_%a.out
#SBATCH --error=/home/caio.rhoden/slurm/%j_em_nq_open_collections_array_%a.err
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --mail-user="c214129@dac.unicamp.br"
#SBATCH --partition=p5000,rtx5000,a5000,rtx8000,l40s,h100
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export HF_DATASETS_CACHE="/tmp/huggingface_${SLURM_ARRAY_TASK_ID}/datasets"
export HF_METRICS_CACHE="/tmp/huggingface_${SLURM_ARRAY_TASK_ID}/metrics"
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

# Copy pre-cached metric to job's temp cache
mkdir -p "$HF_METRICS_CACHE"
cp -r /home/caio.rhoden/.cache/huggingface/metrics/squad_v2 "$HF_METRICS_CACHE/" 2>/dev/null || true

# Calculate start and end indices based on array task ID
# Intervals: 1M-2M, 2M-3M, 3M-4M, 4M-5M, 5M-6M, 6M-7M, 7M-7.22M
START_IDX=$((0 + SLURM_ARRAY_TASK_ID * 1000000))
END_IDX=$((START_IDX + 1000000))
CHECKPOINT_INTERVAL=1000000

# Cap at 7220000
if [ $END_IDX -gt 7220000 ]; then
    END_IDX=7220000
    CHECKPOINT_INTERVAL=220000
fi

echo "Task $SLURM_ARRAY_TASK_ID: Processing interval START_IDX=$START_IDX, END_IDX=$END_IDX"

# Array of root paths to iterate over
ROOT_PATHS=("runs/llama_default" "runs/qwen" "runs/qwen_default")

# For each root path, run the datamodels training
for root_path in "${ROOT_PATHS[@]}"; do
    echo "=========================================="
    echo "Running EM collection for $root_path with interval START_IDX=$START_IDX, END_IDX=$END_IDX"
    echo "=========================================="
    
    python run_datamodels.py \
        --run_type collections \
        --start_idx $START_IDX \
        --end_idx $END_IDX \
        --checkpoint $CHECKPOINT_INTERVAL \
        --evaluator SquadV2-EM \
        --num_subprocesses 1 \
        --collection_id em_collection \
        --root_path $root_path \
        --mode train
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to run datamodels for $root_path"
        exit 1
    fi
done

echo "Task $SLURM_ARRAY_TASK_ID completed successfully"
