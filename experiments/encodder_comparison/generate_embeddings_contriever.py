from utils.create_index import create_hf_embedder


if __name__ == "__main__":
    create_hf_embedder(
        wiki_path="../../data/wiki_dump2018_nq_open/processed/wiki.feather",
        embedder_path="../../models/contriever",
        saving_path="../../data/wiki_dump2018_nq_open/processed/wiki_contriever.index",
        embedding_size=768,
        batch_size=80000,
        nlist=100
    )