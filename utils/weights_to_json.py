import torch
import json

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
        A list of two dictionaries, where the first dictionary contains the top k values for each row,
        and the second dictionary contains the weights for each of the top k values.
    """
    sorted_indices = torch.argsort(weights, dim=1, descending=True).tolist()
    new_json1 = {}
    new_json2 = {}
    
    
    return [new_json1, new_json2]


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

    print("Saving wieghts to json in: ", saving_path)
    json.dump(result[0], open(f"{saving_path}/{saving_id}_indexes.json", "w"))
    json.dump(result[1], open(f"{saving_path}/{saving_id}_weights.json", "w"))
