

source ~/miniconda3/bin/activate
conda activate nq
export WANDB_MODE="offline"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$CONDA_PREFIX/include
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include

SEEDS=(1 4 54 61 73)
INSTUCTIONS=(0 1 2)

for S in "${SEEDS[@]}"; do
    for INSTRUCTION_IDX in "${INSTUCTIONS[@]}"; do
        
        echo "Running setup for seed $S and instruction index $INSTRUCTION_IDX"
        echo "-----------------------------------------------"
        echo "RUNNING SETUP"
        python run_datamodels.py \
            --seed $S \
            --instruction_idx $INSTRUCTION_IDX \
            --run_type setup

        echo "-----------------------------------------------"
        echo "RUNNING PRE_COLLECTIONS TRAIN "
        python run_datamodels.py \
            --seed $S \
            --instruction_idx $INSTRUCTION_IDX \
            --run_type pre_collections \
            --start_idx 1000 \
            --end_idx 2000 \
            --checkpoint 200 \
            --mode train

        echo "RUNNING PRE_COLLECTIONS TEST"
        python run_datamodels.py \
            --seed $S \
            --instruction_idx $INSTRUCTION_IDX \
            --run_type pre_collections \
            --start_idx 0 \
            --end_idx 200 \
            --checkpoint 200 \
            --mode test

                echo "RUNNING PRE_COLLECTIONS TRAIN "
        python run_datamodels.py \
            --seed $S \
            --instruction_idx $INSTRUCTION_IDX \
            --run_type pre_collections \
            --start_idx 1000 \
            --end_idx 2000 \
            --checkpoint 200 \
            --evaluator Judge \
            --mode train

        echo "RUNNING PRE_COLLECTIONS TEST"
        python run_datamodels.py \
            --seed $S \
            --instruction_idx $INSTRUCTION_IDX \
            --run_type pre_collections \
            --start_idx 0 \
            --end_idx 200 \
            --checkpoint 200 \
            --evaluator Judge \
            --mode test


        echo "-----------------------------------------------"
        echo "RUNNING COLLECTIONS TRAIN"
        python run_datamodels.py \
            --seed $S \
            --instruction_idx $INSTRUCTION_IDX \
            --run_type collections \
            --start_idx 0 \
            --end_idx 1000000 \
            --checkpoint 20000 \
            --num_subprocesses 5 \
            --mode train

        echo "RUNNING COLLECTIONS TEST"
        python run_datamodels.py \
            --seed $S \
            --instruction_idx $INSTRUCTION_IDX \
            --run_type collections \
            --start_idx 0 \
            --end_idx 200000 \
            --checkpoint 20000 \
            --num_subprocesses 5 \
            --mode test

        
        echo "-----------------------------------------------"
        echo "TRAINING DATAMODELS"

        python run_datamodels.py \
            --seed $S \
            --instruction_idx $INSTRUCTION_IDX \
            --run_type training

        python run_datamodels.py \
            --seed $S \
            --instruction_idx $INSTRUCTION_IDX \
            --run_type training \
            --num_subprocesses 1 \
            --evaluator Judge


        echo "-----------------------------------------------"
        echo "GENERATING USING DATAMODELS"

        python run_datamodels.py \
            --seed $S \
            --instruction_idx $INSTRUCTION_IDX \
            --run_type generations

    done
        
done