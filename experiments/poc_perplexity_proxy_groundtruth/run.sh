#!/bin/bash

# This script runs the perplexity baseline experiment for the specified seeds.

SEEDS=(860 5191 5390 5734 7270)
TEST_SAMPLES=400
BATCH_SIZE=50

for seed in "${SEEDS[@]}"; do
  echo "Processing seed: $seed"

  # Step 1: Save perplexity collections in batches
  for (( i=0; i<$TEST_SAMPLES; i+=$BATCH_SIZE )); do
    start=$i
    end=$((i + BATCH_SIZE))
    echo "  Running save_perplexity_collections for range $start-$end"
    python save_perplexity_collections.py --seed "$seed" --start_idx "$start" --end_idx "$end"
  done

  # Step 2: Normalize the perplexity collections
  echo "  Normalizing perplexity collections"
  python normalize_perplexity.py --seed "$seed" --target_prefix "non_normalized_perplexity_collections" --saving_prefix "normalized_perplexity_baseline"

done

echo "All seeds processed."
