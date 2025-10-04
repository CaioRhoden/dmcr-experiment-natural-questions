import os
import re
import torch
import json
import numpy as np
from pathlib import Path


def get_sort_key(file_path: Path) -> int:
    """Extracts the 'CORRECT_NUMBER' from a filename for sorting."""
    # This regex finds a pattern like '{ANY_NUMBER}_{KEY_NUMBER}_weights.pt'
    # and captures the KEY_NUMBER.
    match = re.search(r'\d+_(\d+)_weights\.pt$', file_path.name)
    if match:
        # Return the captured number as an integer for correct numerical sorting.
        return int(match.group(1))
    # Return a large number for files that don't match, placing them last.
    return float('inf')

def concat_sorted_tensors(directory_path: str, concat_dim: int = 0) -> torch.Tensor:
    """
    Finds, sorts, and concatenates tensors from a directory.

    Args:
        directory_path (str): The path to the directory containing the .pt files.
        concat_dim (int): The dimension along which to concatenate the tensors.

    Returns:
        torch.Tensor: A single tensor containing all concatenated data.
    """
    # 1. Define the directory and find all relevant files
    p = Path(directory_path)
    if not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    file_list = list(p.glob('*_weights.pt'))

    # 2. Sort the files based on the extracted 'CORRECT_NUMBER'
    sorted_files = sorted(file_list, key=get_sort_key)
    
    if not sorted_files:
        print("Warning: No files matching the pattern were found.")
        return torch.empty(0)

    print("Files will be loaded and concatenated in this order:")
    for f in sorted_files:
        print(f"  - {f.name}")

    # 3. Load tensors from the sorted file list
    tensors_to_concat = [torch.load(f) for f in sorted_files]

    # 4. Concatenate all tensors into a single tensor
    final_tensor = torch.cat(tensors_to_concat, dim=concat_dim)

    return final_tensor


def weights_to_dict(weights: torch.Tensor, subset_documents: dict[str, list[int]], num_models=50) -> list[dict[str, list[int]]]:

    """
    Convert a tensor of weights to a dictionary of the top k values for each row.
    
    Parameters
    ----------
    weights : torch.Tensor
        The tensor of weights
    subset_documents : dict[str, list[int]]
        The dictionary of subset documents
    
    Returns
    -------
    list[dict[str, list[int]]]
        A list of two dictionaries, where the first dictionary contains the reversed ordered values of 
        and the second dictionary contains the weights for each of the top k values.
    """
    weights = weights.cpu().detach().numpy()
    new_dict1 = {}
    new_dict2 = {}

    for i in range(num_models):
        key = str(i)
        orig_list = subset_documents[key]
        row = weights[i]
        sorted_indices = np.argsort(row)

        
        # Create new_dict1: reorder the original list based on descending order of the tensor row
        new_list = [orig_list[index] for index in sorted_indices]
        new_dict1[key] = new_list
        
        # Create new_dict2: reverse the tensor row and convert to list
        reversed_row = row[sorted_indices]
        new_dict2[key] = reversed_row.tolist()
    
    
    return [new_dict1, new_dict2]


def load_weights_to_json(
        weights_dir: str,
        subset_documents_path: str,
        saving_path: str,
        saving_id: str,
        num_models=50
    ):


    """
    Load the weights from a saved torch tensor and the subset documents from a json file,
    and then converts the weights to a dictionary of the top k values for each row, and
    saves the result to a json file.
    
    Parameters
    ----------

    subset_documents_path : str
        The path to the json file containing the subset documents
    k : int
        The number of top values to select
    saving_path : str
        The path to save the result
    """

    ### Flag exists "weights.pt"
    exists_weights = os.path.exists(f"{weights_dir}/weights.pt")
    # Flag exists more than on file ending with "_weights.pt"
    existes_more_weights = len([f for f in os.listdir(weights_dir) if f.endswith("_weights.pt")]) >= 1
    
    ##Loading weights
    if exists_weights:
        weights = torch.load(f"{weights_dir}/weights.pt", weights_only=True)
    elif not exists_weights and existes_more_weights:
        weights = concat_sorted_tensors(weights_dir)
        assert weights.shape[0] == num_models, f"Number of models in the weights tensor ({weights.shape[0]}) does not match the expected number of models ({num_models})."
    else:
        raise FileNotFoundError(f"No weights file found in {weights_dir}. Please ensure that 'weights.pt' or files ending with '_weights.pt' exist.")
    
    subset_documents = json.load(open(subset_documents_path, "r"))
    result = weights_to_dict(weights, subset_documents, num_models=num_models)

    print("Saving weights to json in: ", saving_path)
    json.dump(result[0], open(f"{saving_path}/{saving_id}_indexes.json", "w"))
    json.dump(result[1], open(f"{saving_path}/{saving_id}_weights.json", "w"))
