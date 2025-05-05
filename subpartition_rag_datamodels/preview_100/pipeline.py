######################################################################################################
## Create each datamodel folder -> partition target and random data for each -> run datamodels steps
#######################################################################################################
import polars as pl
import os
import argparse
import torch
import numpy as np
import random
from dmcr.datamodels.setter import StratifiedSetter
from dmcr.datamodels.pipeline import DatamodelsNQPipeline
from dmcr.datamodels.config import DatamodelConfig, LogConfig
from dmcr.models import GenericInstructModelHF
from dmcr.evaluators import Rouge_L_evaluator
import datetime
import faiss
import json
from FlagEmbedding import FlagModel
import tqdm

seed = 42
# NumPy
np.random.seed(seed)
random.seed(seed)
pl.set_random_seed(seed)

# PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def setup():

    """
    Setup for the experiment.

    This function downloads the 100 question golden dataset, writes it to questions.feather and creates the retrieval, generations, results and datamodels folders.
    """
    GOLDEN_PATH = "../../data/nq_open_gold/processed/train.feather"
    train  = pl.read_ipc(GOLDEN_PATH).head(2)
    train.write_ipc("questions.feather")

    ## Create structure
    os.mkdir("retrieval")
    os.mkdir("generations")
    os.mkdir("results")
    os.mkdir("datamodels")

def get_rag_retrieval():

    ## Setup variables
    """
    Load the faiss indices and iterate questions to get the l2 and ip retrieval data for each question.
    This function writes the retrieval data into retrieval_data.json in the retrieval folder.

    Parameters:
    None

    Returns:
    None
    """
    IP_FAISS_INDEX_PATH = "../../data/wiki_dump2018_nq_open/wiki_ip.index"
    L2_FAISS_INDEX_PATH = "../../data/wiki_dump2018_nq_open/wiki2.index"


    retrieval_data = {
        "l2": {},
        "ip": {}
    }

    df = pl.read_ipc("questions.feather")

    ### Load faiss indices
    l2_index = faiss.read_index(L2_FAISS_INDEX_PATH)
    ip_index = faiss.read_index(IP_FAISS_INDEX_PATH)
    EMBERDDER_PATH = "../../models/llms/bge-base-en-v1.5"
    embedder = FlagModel(EMBERDDER_PATH, devices=["cuda:0"], use_fp16=True)



    ### Iterate questions
    for idx in range(len(df)):

        question = df[idx]["question"].to_numpy().flatten()[0]
        query_embedding = embedder.encode(
            [question],
            convert_to_numpy=True,
        )
        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        ### Get l2 and ip neighbors
        l2_ids, l2_scores = l2_index.search(query_embedding, 100)
        ip_ids, ip_scores = ip_index.search(query_embedding, 100)

        retrieval_data["l2"][idx] = (l2_ids.tolist()[0], l2_scores.tolist()[0])
        retrieval_data["ip"][idx] = (ip_ids.tolist()[0], ip_scores.tolist()[0])

    ## Save into json
    with open("retrieval/rag_retrieval.json", "w") as f:
        json.dump(retrieval_data, f)

    

def get_generations():
    

    ## Setup variables
    K = 4
    RETRIEVAL_REFERENCE_PATH = "retrieval/rag_retrieval.json"
    WIKI_PATH = "../../data/wiki_dump2018_nq_open/processed/wiki.feather"
    wiki = pl.read_ipc(WIKI_PATH).with_row_index("idx")
    QUESTIONS_PATH = "questions.feather"
    questions = pl.read_ipc(QUESTIONS_PATH)

    model = GenericInstructModelHF("../../models/llms/Llama-3.2-3B-Instruct", quantization=True)
    model_configs = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_length": 2048,
            "max_new_tokens": 10,
            "num_return_sequences": 5
    }

    generations = {}

    ## Load retrieval data
    with open(RETRIEVAL_REFERENCE_PATH, "r") as f:
        retrieval_data = json.load(f)
    retrieval_data = retrieval_data["l2"]

    ## Iterate questions
    for r_idx in range(len(retrieval_data)):

        top_k = retrieval_data[f"{r_idx}"][1][0:K]
        docs = wiki.filter(pl.col("idx").is_in(top_k))

        ## Generate prompt
        prompt = "Documents: \n"
        for doc_idx in range(len(top_k)-1, -1, -1):
            prompt += f"Document[{K-doc_idx}](Title: {docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}){docs.filter(pl.col('idx')==top_k[doc_idx])['title'].to_numpy().flatten()[0]}\n\n"
        prompt += f"Question: {questions[r_idx]['question'].to_numpy().flatten()[0]}\nAnswer: "

        ## Generate output
        outputs = model.run(
            prompt, 
            instruction="You are given a question and you MUST try to give a real SHORT ANSWER in 5 tokens, you can use the available documents ", 
            config_params=model_configs
        )

        generations[f"{r_idx}"] = [str(out["generated_text"]) for out in outputs]

        with open("generations/generations.json", "w") as f:
            json.dump(generations, f)

    ## Save into json
    
        




    




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", "-s", type=str, required=True)
    args = parser.parse_args()

    step = args.step

    match step:
        case "setup":
            setup()

        case "get_rag_retrieval":
            get_rag_retrieval()

        case "get_generations":
            get_generations()