import os
import polars as pl
import gc

def main():
    
    for exp in ["experiment_1", "experiment_4", "experiment_54", "experiment_61", "experiment_73"]:
        file_path = f"{exp}/train.feather"
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping")
            continue
        
        print(f"Processing {exp}...")
        
        try:
            # Use lazy evaluation to avoid loading entire dataset into memory
            train_lazy = pl.scan_ipc(file_path)
            
            # Split into train and test using lazy evaluation
            test = train_lazy.filter(pl.col("collection_idx") >= 1900).collect()
            train = train_lazy.filter(pl.col("collection_idx") < 1900).collect()
            
            print(f"  Train: {train.shape[0]} rows, Test: {test.shape[0]} rows")
            
            # Write with memory-efficient settings
            train.write_ipc(f"{exp}/train.feather")
            test.write_ipc(f"{exp}/test.feather")
            
            print(f"  ✓ {exp} completed")
            
            # Explicit cleanup
            del train, test
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {exp}: {e}")
            gc.collect()
            continue

if __name__ == "__main__":
    main()