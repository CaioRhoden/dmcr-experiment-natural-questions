
import faiss
import argparse
from utils.create_index import create_flag_embedder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--saving_path", type=str, default="../../data/wiki_dump2018_nq_open/processed/wiki.index")
    args = parser.parse_args()

    if args.metric == "ip":
        args.metric = faiss.METRIC_INNER_PRODUCT
    elif args.metric == "l2":
        args.metric = faiss.METRIC_L2
    elif args.metric == "cosine":
        pass
    else:
        raise ValueError(f"Unknown metric: {args.metric}")


    create_flag_embedder(
        metric=args.metric
    )