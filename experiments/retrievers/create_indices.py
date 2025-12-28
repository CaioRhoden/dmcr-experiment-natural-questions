from utils.create_index import create_hf_embedder, create_flag_embedder, create_bm25_corpus
import argparse
import pathlib
import faiss
PATH =  pathlib.Path(__file__).parent.parent.parent



def main(indice_name: str):
    
    if indice_name == "nv2":
        print("Creating NV-Embed-v2 index...")
        print(f"{PATH}/models/NV-Embed-v2")
        create_hf_embedder(
            embedder_path=f"{PATH}/models/NV-Embed-v2",
            saving_path=f"{PATH}/data/indices/nv2_index.faiss",
            embedding_size=4096,
            wiki_path=f"{PATH}/data/wiki_dump2018_nq_open/processed/wiki.feather",
            metric="cosine",
        )
    elif indice_name == "qwen":
        create_hf_embedder(
            embedder_path=f"{PATH}/models/Qwen3-Embedding-8B",
            saving_path=f"{PATH}/data/indices/qwen_index.faiss",
            wiki_path=f"{PATH}/data/wiki_dump2018_nq_open/processed/wiki.feather",
            embedding_size=2048,
            metric="cosine",
        )
    elif indice_name == "bge":
        create_flag_embedder(
            embedder_path=f"{PATH}/models/bge-base-en-v1.5",
            saving_path=f"{PATH}/data/indices/bge_index.faiss",
            metric=faiss.METRIC_INNER_PRODUCT,
        )
    elif indice_name == "bm25":
        create_bm25_corpus(
            saving_path=f"{PATH}/data/indices/nq_corpus/bm25_corpus",
        )
    else:
        raise ValueError(f"Unknown indice name: {indice_name}")


if __name__ == "__main__":
    print(f"PATH: {PATH}")
    print("==========================================================")
    parser = argparse.ArgumentParser(description="Create indices for different retrievers.")
    parser.add_argument(
        "--indice_name", 
        type=str, 
        required=True, 
        choices=["nv2", "qwen", "bge", "bm25"], 
        help="Name of the indice to create."
    )
    
    args = parser.parse_args()
    
    main(args.indice_name)