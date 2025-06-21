import torch
import json
import numpy as np

def weights_to_dict(weights: torch.Tensor, subset_documents: dict[str, list[int]]) -> list[dict[str, list[int]]]:

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

    for i in range(50):
        key = str(i)
        orig_list = subset_documents[key]
        row = weights[i]
        sorted_indices = np.argsort(row)[::-1]

        
        # Create new_dict1: reorder the original list based on descending order of the tensor row
        new_list = [orig_list[index] for index in sorted_indices]
        new_dict1[key] = new_list
        
        # Create new_dict2: reverse the tensor row and convert to list
        reversed_row = row[::-1]
        new_dict2[key] = reversed_row.tolist()
    
    
    return [new_dict1, new_dict2]


def load_weights_to_json(
        weights_path: str,
        subset_documents_path: str,
        saving_path: str,
        saving_id: str,
    ):


    """
    Load the weights from a saved torch tensor and the subset documents from a json file,
    and then converts the weights to a dictionary of the top k values for each row, and
    saves the result to a json file.
    
    Parameters
    ----------
    weights_path : str
        The path to the saved torch tensor
    subset_documents_path : str
        The path to the json file containing the subset documents
    k : int
        The number of top values to select
    saving_path : str
        The path to save the result
    """
    
    weights = torch.load(weights_path, weights_only=True)
    subset_documents = json.load(open(subset_documents_path, "r"))
    result = weights_to_dict(weights, subset_documents)

    print("Saving weights to json in: ", saving_path)
    json.dump(result[0], open(f"{saving_path}/{saving_id}_indexes.json", "w"))
    json.dump(result[1], open(f"{saving_path}/{saving_id}_weights.json", "w"))
