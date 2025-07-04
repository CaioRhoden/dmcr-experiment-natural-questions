from pyparsing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import polars as pl
from wandb import config

def calculate_perplexity(text, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, context=None, device="cpu", configs=None):
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
    if configs is None:
        configs = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_length": 2048
        }

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
        outputs = model(**inputs, labels=labels, **configs)
        loss = outputs.loss
    
    return torch.exp(loss).item()

def calculate_batch_perplexity(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    contexts: list[str],
    device: str = "cpu",
    max_length: int|None = None,
    stride: int = 512
) -> torch.Tensor:
    """
    Calculate perplexity for a batch of texts given optional contexts.

    Args:
        texts (List[str]): List of target texts.
        model (AutoModelForCausalLM): The language model.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the model.
        contexts (List[str], optional): Optional list of context strings.
        device (str): Device to run on.

    Returns:
        torch.Tensor: Perplexity values for each input in the batch.
    """
    
    assert len(texts) == len(contexts), "texts and contexts must be the same length"

    model.eval()
    model.to(device)

    if max_length is None:
        max_length = model.config.n_positions if hasattr(model.config, "n_positions") else 2048

    perplexities = []

    for context, text in tqdm(zip(contexts, texts), total=len(texts), desc="Calculating perplexities"):
        full_text = context + text
        encodings = tokenizer(full_text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)

        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids_slice = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_slice.clone()
            target_ids[:, :-trg_len] = -100  # mask all tokens except trg_len

            with torch.no_grad():
                outputs = model(input_ids_slice, labels=target_ids)
                neg_log_likelihood = outputs.loss

            num_valid_tokens = (target_ids != -100).sum().item()
            num_loss_tokens = max(num_valid_tokens - target_ids.size(0), 1)  # account for shift

            nll_sum += neg_log_likelihood.item() * num_loss_tokens
            n_tokens += num_loss_tokens

            if end_loc == seq_len:
                break
            prev_end_loc = end_loc

        avg_nll = nll_sum / n_tokens
        ppl = torch.exp(torch.tensor(avg_nll))
        perplexities.append(ppl)

    return torch.stack(perplexities).to(device)

