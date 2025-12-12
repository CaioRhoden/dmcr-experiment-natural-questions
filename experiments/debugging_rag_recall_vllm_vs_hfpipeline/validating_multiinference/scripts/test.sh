source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"



SEEDS=(54 61)
INSTRUCTIONS=(0)


echo "Running setup for seed $SEED and instruction index 1"

for SEED in "${SEEDS[@]}"; do
    for INSTRUCTION_IDX in "${INSTRUCTIONS[@]}"; do
        echo "-----------------------------------------------"
        echo "RUNNING TRAINING FOR INSTRUCTION INDEX $INSTRUCTION_IDX"
        python run_datamodels.py \
            --seed $SEED \
            --instruction_idx $INSTRUCTION_IDX \
            --start_idx 400 \
            --end_idx 410 \
            --run_type training \
            --num_subprocesses 5 \
            --evaluator Judge
    done
done
