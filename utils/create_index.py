import polars as pl
import torch
import numpy as np
import joblib
import re
from FlagEmbedding import FlagModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import gc
import math

from utils.set_random_seed import set_random_seed
import os


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
    
    # Normalize for cosine similarity (IP with normalized vectors = cosine)
    if metric == "cosine" or metric == faiss.METRIC_INNER_PRODUCT:
        faiss.normalize_L2(train_embeddings)
        
    index.train(train_embeddings)
    
    # Cleanup training memory
    del train_texts, train_embeddings, train_data
    gc.collect()

    # ADDING PHASE
    batch_size = 80000
    for start in tqdm(range(0, total_size, batch_size), desc="Adding batches to index", unit="batch"):
        end = min(start + batch_size, total_size)
        print(f"Processing batch: {start} to {end}")
        
        # Slice sequentially
        batch_texts = wiki[start:end].select("text").to_numpy().flatten().tolist()
        
        batch_embeddings = embedder.encode(batch_texts, convert_to_numpy=True)
        batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity (IP with normalized vectors = cosine)
        if metric == "cosine" or metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(batch_embeddings)

        index.add(batch_embeddings)

    # SAVE ONCE AT THE END
    print(f"Writing index to {saving_path}...")
    faiss.write_index(index, saving_path)
        



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
    set_random_seed(42)

    # Load data sequentially to preserve ID mapping
    wiki = pl.read_ipc(wiki_path).with_row_index("idx")
    total_size = len(wiki)
    
    # Dynamic nlist calculation (heuristic: 4 * sqrt(N))
    nlist = int(4 * math.sqrt(total_size))
    print(f"Dataset size: {total_size}, using nlist={nlist}")
    
    # Initialize Model
    embedder = SentenceTransformer(
        embedder_path,
        trust_remote_code=True,
    )
    torch.backends.cuda.enable_cudnn_sdp(False)
    
    # Handle Metric Selection
    faiss_metric = faiss.METRIC_INNER_PRODUCT # Default for cosine if normalized
    if metric != "cosine" and metric != faiss.METRIC_INNER_PRODUCT:
        faiss_metric = metric 

    index = faiss.index_factory(4096, "IVF16384,SQ8", faiss_metric)

    # TRAINING PHASE
    # Sample random subset for training to ensure balanced clusters
    # Training on 30 * nlist is a safe minimum
    train_size = max(30 * nlist, 650000) 
    if train_size > total_size:
        train_size = total_size

    print(f"Training index on {train_size} samples...")
    train_data = wiki.sample(n=train_size, shuffle=True, seed=42)
    train_texts = train_data.select("text").to_numpy().flatten().tolist()
    
    train_embeddings_list = embedder.encode(train_texts, **encode_kwargs)
    train_embeddings = np.array(train_embeddings_list, dtype=np.float32)
    
    # Normalize for cosine similarity (IP with normalized vectors = cosine)
    if metric == "cosine" or metric == faiss.METRIC_INNER_PRODUCT:
        faiss.normalize_L2(train_embeddings)
        
    print("Training FAISS index...")
    index.train(train_embeddings)
    print("FAISS index trained.")
    
    # Cleanup training memory
    del train_texts, train_embeddings, train_embeddings_list, train_data
    gc.collect()

    # ADDING PHASE
    for start in tqdm(range(0, total_size, batch_size), desc="Adding batches to index", unit="batch"):
        end = min(start + batch_size, total_size)
        print(f"Processing batch: {start} to {end}")
        
        # Slice sequentially
        batch_texts = wiki[start:end].select("text").to_numpy().flatten().tolist()
        
        # Encode the current batch
        batch_embeddings_list = embedder.encode(batch_texts, **encode_kwargs)
        batch_embeddings = np.array(batch_embeddings_list, dtype=np.float32)
        
        # Normalize for cosine similarity (IP with normalized vectors = cosine)
        if metric == "cosine" or metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(batch_embeddings)

        index.add(batch_embeddings)

        print(f"Index size: {index.ntotal}")
        
        # Optional: Clear memory if needed
        del batch_texts, batch_embeddings, batch_embeddings_list
        torch.cuda.empty_cache()
        gc.collect()

    # SAVE ONCE AT THE END
    print(f"Writing index to {saving_path}...")
    faiss.write_index(index, saving_path)
    
def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        return embeddings / norms


def _simple_word_tokenizer(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize text into alphanumeric words; keeps BM25 corpus deterministic."""
    if lowercase:
        text = text.lower()
    return re.findall(r"[A-Za-z0-9]+", text)


def create_bm25_corpus(
    wiki_path: str = "../../data/wiki_dump2018_nq_open/processed/wiki.feather",
    saving_path: str = "../../data/wiki_dump2018_nq_open/processed/wiki_bm25_corpus.joblib",
    text_column: str = "text",
    batch_size: int = 100000,
    lowercase: bool = True,
    compress: int | str | None = 3,
):
    """Create a BM25-ready token corpus in shards and persist with joblib.

    The full corpus (22M docs) is streamed in batches to keep memory bounded.
    We write token shards alongside a manifest saved at `saving_path`.
    """

    set_random_seed(42)

    # Count documents without loading them all at once
    total_size = (
        pl.scan_ipc(wiki_path)
        .select(pl.len())
        .collect(streaming=True)
        .item()
    )
    print(f"Dataset size: {total_size}")

    shard_paths: list[str] = []
    n_shards = math.ceil(total_size / batch_size)

    for shard_idx, start in enumerate(
        tqdm(
            range(0, total_size, batch_size),
            total=n_shards,
            desc="Tokenizing batches",
            unit="batch",
        )
    ):
        end = min(start + batch_size, total_size)

        batch_df = (
            pl.scan_ipc(wiki_path)
            .select(pl.col(text_column))
            .slice(start, batch_size)
            .collect(streaming=True)
        )

        texts = batch_df[text_column].to_list()
        tokens_batch = [
            _simple_word_tokenizer(text or "", lowercase=lowercase) for text in texts
        ]

        shard_path = f"{saving_path}.shard{shard_idx:05d}.joblib"
        joblib.dump(tokens_batch, shard_path, compress=compress)
        shard_paths.append(shard_path)

        del batch_df, texts, tokens_batch
        gc.collect()

    manifest = {
        "corpus_shards": shard_paths,
        "doc_count": total_size,
        "text_column": text_column,
        "batch_size": batch_size,
        "tokenizer": "simple_word_tokenizer_lower" if lowercase else "simple_word_tokenizer",
    }

    joblib.dump(manifest, saving_path, compress=compress)
    print(f"Wrote manifest to {saving_path} with {len(shard_paths)} shards")
