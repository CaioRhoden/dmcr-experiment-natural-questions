import polars as pl
import torch
import os
import random
import numpy as np
from FlagEmbedding import FlagModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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


def create_cosine_flag_embeeder():

    


    EMBERDDER_PATH = "../models/llms/bge-base-en-v1.5"
    embedder = FlagModel(EMBERDDER_PATH, devices=["cuda:0"], use_fp16=True)
    nlist = 100  # Number of IVF clusters
    quantizer = faiss.IndexFlatIP(768)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, 768, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT
    )

    # index = faiss.read_index("wiki.index")
    WIKI_PATH = "../data/wiki_dump2018_nq_open/processed/wiki.feather"
    wiki = pl.read_ipc(WIKI_PATH).with_row_index("idx")
    wiki = wiki.sample(fraction=1.0, shuffle=True, seed=42)
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
        faiss.normalize_L2(batch_embeddings.astype('float32'))

        if start == 0:
            index.train(batch_embeddings)


        # Add to index
        index.add(batch_embeddings)

        faiss.write_index(index, "wiki_cosine.index")

        print(f"Index size: {index.ntotal}")
        
        # Optional: Clear memory if needed
        del batch_texts, batch_embeddings
        torch.cuda.empty_cache()
        gc.collect()
        

def create_hf_embedder(wiki_path: str, embedder_path: str, saving_path: str, embedding_size: int, batch_size=80000, nlist:int = 100):

    
    embedder = HuggingFaceEmbedding(model_name=embedder_path)
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, embedding_size, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT
    )

    # index = faiss.read_index("wiki.index")
    WIKI_PATH = wiki_path
    wiki = pl.read_ipc(WIKI_PATH).with_row_index("idx")
    wiki = wiki.sample(fraction=1.0, shuffle=True, seed=42)
    total_size = len(wiki)
    torch.backends.cuda.enable_cudnn_sdp(False)


    for start in range(0, total_size, batch_size):
        
        end = min(start + batch_size, total_size)
        print(f"End: {end}")
        
        batch_texts = wiki[start:end].select("text").to_numpy().flatten().tolist()
        
        
        # Encode the current batch
        batch_embeddings_list = embedder.get_text_embedding_batch (batch_texts)
        batch_embeddings = np.array(batch_embeddings_list, dtype=np.float16)

        if start == 0:
            index.train(batch_embeddings)


        # Add to index
        index.add(batch_embeddings)

        faiss.write_index(index, saving_path)

        print(f"Index size: {index.ntotal}")
        
        # Optional: Clear memory if needed
        del batch_texts, batch_embeddings
        torch.cuda.empty_cache()
        gc.collect()
    




