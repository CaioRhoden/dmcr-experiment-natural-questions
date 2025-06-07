import faiss

index = faiss.read_index("data/wiki_dump2018_nq_open/processed/wiki_contriever.index")
print(index.ntotal)