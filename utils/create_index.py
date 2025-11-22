import polars as pl
import torch
import numpy as np
from FlagEmbedding import FlagModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
import faiss
import gc
import math

from utils.set_random_seed import set_random_seed


def create_flag_embedder(
    saving_path: str,
    metric: str = faiss.METRIC_INNER_PRODUCT,
    embedder_path: str = "../../models/bge-base-en-v1.5",
    wiki_path: str = "../../data/wiki_dump2018_nq_open/processed/wiki.feather",
):
    set_random_seed(42)
    
    # Load data sequentially to preserve ID mapping
    wiki = pl.read_ipc(wiki_path).with_row_index("idx")
    total_size = len(wiki)
    
    # Dynamic nlist calculation (heuristic: 4 * sqrt(N))
    nlist = int(4 * math.sqrt(total_size))
    print(f"Dataset size: {total_size}, using nlist={nlist}")

    # Initialize Model
    embedder = FlagModel(embedder_path, devices=["cuda:0"])
    
    # Initialize Index
    d = 768
    quantizer = faiss.IndexFlatIP(d)
    
    # Handle Metric Selection
    faiss_metric = faiss.METRIC_INNER_PRODUCT # Default for cosine if normalized
    if metric != "cosine" and metric != faiss.METRIC_INNER_PRODUCT:
        faiss_metric = metric 

    index = faiss.index_factory(d, "IVF16384,SQ8", faiss_metric)

    # TRAINING PHASE
    # Sample random subset for training to ensure balanced clusters
    # Training on 30 * nlist is a safe minimum
    train_size = max(30 * nlist, 650000) 
    if train_size > total_size:
        train_size = total_size

    print(f"Training index on {train_size} samples...")
    train_data = wiki.sample(n=train_size, shuffle=True, seed=42)
    train_texts = train_data.select("text").to_numpy().flatten().tolist()
    
    train_embeddings = embedder.encode(train_texts, convert_to_numpy=True)
    train_embeddings = np.array(train_embeddings, dtype=np.float32)
    
    if metric == "cosine":
        faiss.normalize_L2(train_embeddings)
        
    index.train(train_embeddings)
    
    # Cleanup training memory
    del train_texts, train_embeddings, train_data
    gc.collect()

    # ADDING PHASE
    batch_size = 80000
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        print(f"Processing batch: {start} to {end}")
        
        # Slice sequentially
        batch_texts = wiki[start:end].select("text").to_numpy().flatten().tolist()
        
        batch_embeddings = embedder.encode(batch_texts, convert_to_numpy=True)
        batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
        
        if metric == "cosine":
            faiss.normalize_L2(batch_embeddings)

        index.add(batch_embeddings)

    # SAVE ONCE AT THE END
    print(f"Writing index to {saving_path}...")
    faiss.write_index(index, saving_path)
        

def create_llamaindex_hf_embedder(
        wiki_path: str, 
        embedder_path: str, 
        saving_path: str, 
        embedding_size: int, 
        batch_size=80000, 
        nlist:int = 100,
        model_kwargs: dict = {},
    ):

    
    embedder = HuggingFaceEmbedding(model_name=embedder_path, model_kwargs=model_kwargs)
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
        batch_embeddings_list = embedder.get_text_embedding_batch(batch_texts)
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
    


def create_hf_embedder(
        wiki_path: str, 
        embedder_path: str, 
        saving_path: str, 
        embedding_size: int, 
        batch_size=80000, 
        nlist:int = 100,
        metric = faiss.METRIC_INNER_PRODUCT,
        encode_kwargs: dict = {},
    ):

    
    embedder = SentenceTransformer(
        embedder_path,
        trust_remote_code=True,
    )

    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, embedding_size, nlist, faiss.ScalarQuantizer.QT_fp16, metric
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
        batch_embeddings_list = embedder.encode(batch_texts, **encode_kwargs)
        batch_embeddings = normalize_embeddings(np.array(batch_embeddings_list, dtype=np.float16))

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
    
def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        return embeddings / norms
