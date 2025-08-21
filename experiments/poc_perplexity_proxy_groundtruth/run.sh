#!/bin/bash

# This script runs the perplexity baseline experiment for the specified seeds.

SEEDS=(860 5191 5390 5734 7270)


for seed in "${SEEDS[@]}"; do
    python save_perplexity_collections.py --seed $seed --start_idx 0 --end_idx 50 --optional_instruction "" --saving_prefix non_normalized_perplexity_baseline
    python save_perplexity_collections.py --seed $seed --start_idx 0 --end_idx 50 --saving_prefix non_normalized_perplexity_collections_with_instructions
    python normalize_perplexity.py --seed $seed --target_prefix non_normalized_perplexity_collections_with_instructions --saving_prefix perplexity_with_instructions
     python normalize_perplexity.py --seed $seed --target_prefix non_normalized_perplexity_collections --saving_prefix perplexity_baseline
    python run_pipeline.py --seed $seed --run_type datamodels_training  --model_run_id perplexity_with_instruction --train_collection_id perplexity_with_instruction
    python run_pipeline.py --seed $seed --run_type datamodels_generations  --model_run_id perplexity_with_instruction --train_collection_id perplexity_with_instruction
    python run_pipeline.py --seed $seed --run_type datamodels_training --model_run_id perplexity_baseline --train_collection_id perplexity_baseline
    python run_pipeline.py --seed $seed --run_type datamodels_generations --model_run_id perplexity_baseline --train_collection_id perplexity_baseline

done

echo "All seeds processed."
