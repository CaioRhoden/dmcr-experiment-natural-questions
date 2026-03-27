import json
import os
import numpy as np
import polars as pl

REQUIRED_COLUMNS = {'text', 'title'}
VALID_INDEX_VALUES = {0, 1}


def _validate_parameters(path_documents: str, path_retrievals: str, retrieval_key: str, indeces) -> None:
    """Validate input parameter types."""
    if not isinstance(path_documents, str):
        raise TypeError(f"path_documents must be str, got {type(path_documents)}")
    if not isinstance(path_retrievals, str):
        raise TypeError(f"path_retrievals must be str, got {type(path_retrievals)}")
    if not isinstance(retrieval_key, str):
        raise TypeError(f"retrieval_key must be str, got {type(retrieval_key)}")
    if not isinstance(indeces, (np.ndarray, list)):
        raise TypeError(f"indeces must be np.ndarray or list, got {type(indeces)}")


def _validate_and_load_files(path_documents: str, path_retrievals: str) -> tuple[pl.DataFrame, dict]:
    """Validate file paths and load documents and retrievals."""
    # Validate file existence and format
    for path, name, ext in [(path_documents, "Documents", ".feather"), (path_retrievals, "Retrievals", ".json")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")
        if not path.endswith(ext):
            raise ValueError(f"{name} file must be {ext} format, got: {path}")
    
    # Load documents
    try:
        documents = pl.read_ipc(path_documents)
    except Exception as e:
        raise ValueError(f"Failed to read .feather file: {e}")
    
    # Validate required columns
    missing_columns = REQUIRED_COLUMNS - set(documents.columns)
    if missing_columns:
        raise ValueError(f"Documents DataFrame missing required columns: {missing_columns}. Available: {set(documents.columns)}")
    
    # Load retrievals
    try:
        with open(path_retrievals, 'r') as f:
            retrievals = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON retrievals file: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read retrievals file: {e}")
    
    return documents, retrievals


def _validate_retrievals(retrievals: dict, retrieval_key: str) -> list:
    """Validate and extract document indices from retrievals."""
    if retrieval_key not in retrievals:
        raise ValueError(f"retrieval_key '{retrieval_key}' not found. Available keys: {list(retrievals.keys())}")
    
    doc_indices = retrievals[retrieval_key]
    if not isinstance(doc_indices, list):
        raise ValueError(f"Expected doc_indices to be list, got {type(doc_indices)}")
    
    return doc_indices


def _validate_indices_array(indeces: np.ndarray, doc_indices: list, num_documents: int) -> np.ndarray:
    """Validate indices array values and bounds."""
    if len(indeces) == 0:
        raise ValueError("indeces array is empty")
    if len(indeces) != len(doc_indices):
        raise ValueError(f"Length mismatch: indeces has {len(indeces)}, doc_indices has {len(doc_indices)}")
    
    # Check for invalid values
    invalid_values = set(np.unique(indeces)) - VALID_INDEX_VALUES
    if invalid_values:
        raise ValueError(f"indeces contains invalid values {invalid_values}. Only 0 or 1 allowed")
    
    # Validate bounds for selected indices
    for i, is_selected in enumerate(indeces):
        if is_selected == 1:
            doc_idx = doc_indices[i]
            if not isinstance(doc_idx, int):
                raise ValueError(f"Document index at position {i} is not an integer: {doc_idx}")
            if not 0 <= doc_idx < num_documents:
                raise ValueError(f"Document index {doc_idx} at position {i} out of bounds [0, {num_documents-1}]")
    
    return indeces if isinstance(indeces, np.ndarray) else np.array(indeces)


def _build_context(documents: pl.DataFrame, doc_indices: list, indeces: np.ndarray) -> str:
    """Build context string from selected documents."""
    context = ""
    count = 1
    
    for i, is_selected in enumerate(indeces):
        if is_selected == 1:
            doc_idx = doc_indices[i]
            title = documents[doc_idx]["title"].to_numpy().flatten()[0]
            text = documents[doc_idx]["text"].to_numpy().flatten()[0]
            context += f"Document[{count}](Title: {title}){text}\n\n"
            count += 1
    
    return context


def get_wiki_context(
        path_documents: str,
        path_retrievals: str,
        retrieval_key: str,
        indeces: np.array,

) -> str:
    """
    Retrieves context for selected document indices from a JSON retrieval index and .feather documents.

    Args:
        path_documents (str): Path to the .feather file containing documents with 'text' and 'title' columns.
        path_retrievals (str): Path to the JSON file containing retrieval indices mapped by key.
        retrieval_key (str): Key in the JSON file to use for retrieving document indices.
        indeces (np.array): Array of floats where 1 indicates which indices to include in the context.

    Returns:
        str: Concatenated context string with selected documents formatted as Document[{count}](Title: {title}){text}.
    
    Raises:
        TypeError: If parameters have incorrect types.
        FileNotFoundError: If files do not exist.
        ValueError: If file content is invalid or missing required fields.
    """
    _validate_parameters(path_documents, path_retrievals, retrieval_key, indeces)
    documents, retrievals = _validate_and_load_files(path_documents, path_retrievals)
    doc_indices = _validate_retrievals(retrievals, retrieval_key)
    indeces = _validate_indices_array(indeces, doc_indices, len(documents))
    
    return _build_context(documents, doc_indices, indeces)

