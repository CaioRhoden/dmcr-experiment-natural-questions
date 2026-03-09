import os
import pandas as pd

def main():
    root_dir = '/home/users/caio.rhoden/work/dmcr-experiment-natural-questions/experiments/binary_classification/binary_collections'
    results = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.feather'):
                filepath = os.path.join(dirpath, filename)
                try:
                    df = pd.read_feather(filepath)
                    length = len(df)
                    # Extract experiment name from path
                    parts = filepath.replace(root_dir, '').strip('/').split('/')
                    experiment = parts[1] if len(parts) > 1 else 'unknown'
                    results.append((experiment, filepath, length))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    # Sort by experiment name
    results.sort(key=lambda x: x[0])
    
    for experiment, filepath, length in results:
        print(f"{filepath}: {length}")

if __name__ == "__main__":
    main()