from utils.get_random_nq_dataset import create_n_datasets
from pathlib import Path
import argparse


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent.parent

    parser = argparse.ArgumentParser(description="Generate random subsets of the NQ dataset")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation")
    args = parser.parse_args()
    create_n_datasets(
        root_path=str(root),
        n_datasets=5,
        n_samples=500,
        partition="dev",
        seed=args.seed,
        path_prefix="experiment",
        save_file_name="questions.feather",
    )
