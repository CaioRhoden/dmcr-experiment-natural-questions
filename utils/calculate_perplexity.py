import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_perplexity(text, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, context=None, device="cpu"):
    """
    Calculate perplexity of a given text using a causal language model.

    Args:
        text (str): The text to calculate the perplexity of.
        model (AutoModelForCausalLM): The model to use for calculation.
        tokenizer (AutoTokenizer): The tokenizer to use.
        context (str, optional): The context to use. Defaults to None.

    Returns:
        float: The calculated perplexity.
    """
    full_text = context + text if context else text
    
    inputs = tokenizer(
        full_text, 
        return_tensors="pt", 
        truncation=True
    ).to(device)
    
    if context:
        context_tokens = tokenizer(
            context, 
            return_tensors="pt", 
            truncation=True
        )
        context_len = context_tokens.input_ids.shape[1]
        labels = inputs.input_ids.clone()
        labels[:, :context_len] = -100
    else:
        labels = inputs.input_ids
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
    
    return torch.exp(loss).item()


