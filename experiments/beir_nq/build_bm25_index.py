"""
Build a BM25 index from a feather corpus file.

This script reads a corpus from a feather file and builds a BM25 index
that can be used for retrieval tasks.
"""

import pickle
import os
from pathlib import Path
from rank_bm25 import BM25Okapi
import polars as pl


def build_bm25_index(corpus_path: str, output_path: str = None) -> str:
    """
    Build a BM25 index from a corpus feather file.
    
    Parameters:
    -----------
    corpus_path : str
        Path to the corpus.feather file
    output_path : str, optional
        Path where to save the BM25 index pickle file.
        If None, saves as 'bm25_index.pkl' in the same directory as corpus.
        
    Returns:
    --------
    str
        Path to the saved BM25 index file
    """
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    print(f"Loading corpus from: {corpus_path}")
    
    # Read the corpus feather file
    df = pl.read_ipc(corpus_path)
    print(f"Loaded corpus with {len(df)} documents")
    
    # Ensure we have required columns (usually 'title' and 'text' or 'contents')
    if 'text' in df.columns:
        text_col = 'text'
    else:
        # Use the first non-index column
        cols = df.columns
        text_col = cols[1] if len(cols) > 1 else cols[0]
        print(f"Using column '{text_col}' as text content")
    
    # Tokenize and build BM25 index
    print("Tokenizing corpus and building BM25 index...")
    corpus_tokens = []
    for i, row in enumerate(df.iter_rows(named=True)):
        text = row[text_col]
        if isinstance(text, str):
            tokens = text.split()
            corpus_tokens.append(tokens)
        else:
            corpus_tokens.append([])
        
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} documents...")
    
    # Create BM25 index
    print("Creating BM25 index...")
    bm25_index = BM25Okapi(corpus_tokens)
    
    # Determine output path
    if output_path is None:
        corpus_dir = os.path.dirname(corpus_path)
        output_path = os.path.join(corpus_dir, 'bm25_index.pkl')
    
    # Save the index
    print(f"Saving BM25 index to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(bm25_index, f)
    
    print(f"✓ BM25 index successfully built and saved")
    print(f"  Index size: {len(corpus_tokens)} documents")
    print(f"  Output file: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.parent.parent
    corpus_path = script_dir / "data"/ "beir_nq" / "processed" / "corpus.feather"
    
    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = None
    
    try:
        build_bm25_index(str(corpus_path), output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
