import argparse
from utils.get_random_nq_dataset import get_random_nq_dataset
from utils.set_random_seed import set_random_seed
import random

set_random_seed(42)

if __name__ == "__main__":

    seed = random.randint(1,1000)
    args = argparse.ArgumentParser()
    args.add_argument("--n_samples", type=int, default=500)
    args.add_argument("--seed", type=int, default=seed)
    args.add_argument("--partition", type=str, default="dev")

    args = args.parse_args()

    get_random_nq_dataset(
        root_path="../../",
        n_samples=args.n_samples,
        save_path=f"questions_{args.n_samples}_{args.seed}_{args.partition}.feather",
        partition=args.partition,
        seed=args.seed
    )

    