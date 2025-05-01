import polars as pl
import torch
import os
import random
import numpy as np
from FlagEmbedding import FlagModel
import faiss
import gc
from tqdm import tqdm  


torch.backends.cudnn.enabled = False
# NumPy
seed = 42
np.random.seed(seed)
random.seed(seed)
# PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


EMBERDDER_PATH = "../../models/llms/bge-base-en-v1.5"
embedder = FlagModel(EMBERDDER_PATH, devices=["cuda:0"], use_fp16=True)
index = faiss.IndexScalarQuantizer(768, 8)

# index = faiss.read_index("wiki.index")
WIKI_PATH = "../../data/wiki_dump2018_nq_open/processed/wiki.feather"
wiki = pl.read_ipc(WIKI_PATH).with_row_index("idx")
total_size = len(wiki)
batch_size = 80000
torch.backends.cuda.enable_cudnn_sdp(False)


for start in range(0, total_size, batch_size):
    
    end = min(start + batch_size, total_size)
    print(f"End: {end}")
    
    batch_texts = wiki[start:end].select("text").to_numpy().flatten().tolist()
    
    # Encode the current batch
    batch_embeddings = embedder.encode(
        batch_texts,
        convert_to_numpy=True,
    )
    
    # Add to index
    index.add(batch_embeddings.astype('float32'))

    faiss.write_index(index, "wiki.index")

    print(f"Index size: {index.ntotal}")
    
    # Optional: Clear memory if needed
    del batch_texts, batch_embeddings
    torch.cuda.empty_cache()
    gc.collect()
    




