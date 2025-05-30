from utils.create_index import create_hf_embedder


if __name__ == "__main__":
    create_hf_embedder(
        wiki_path="../../data/wiki_dump2018_nq_open/processed/wiki.feather",
        embedder_path="../../models/llms/jina-embeddings-v3",
        saving_path="../../data/wiki_dump2018_nq_open/processed/wiki_jina.index",
        embedding_size=512,
        batch_size=80000,
        nlist=100,
        encode_kwargs={"task": "retrieval.passage", "truncate_dim": 512}
    )