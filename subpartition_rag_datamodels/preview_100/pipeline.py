######################################################################################################
## Create each datamodel folder -> partition target and random data for each -> run datamodels steps
#######################################################################################################
import polars as pl
import os
import argparse
import torch
import numpy as np
import random
from dmcr.datamodels.setter.IndexBasedSetter import IndexBasedSetter
from dmcr.datamodels.setter.SetterConfig import IndexBasedSetterConfig
from dmcr.datamodels.pipeline import DatamodelsIndexBasedNQPipeline
from dmcr.datamodels.config import DatamodelIndexBasedConfig, LogConfig
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


### Global variables
RETRIEVAL_REFERENCE_PATH = "retrieval/rag_retrieval_indexes.json"
WIKI_PATH = "../../data/wiki_dump2018_nq_open/processed/wiki.feather"
EMBERDDER_PATH = "../../models/llms/bge-base-en-v1.5"
GOLDEN_PATH = "../../data/nq_open_gold/processed/train.feather"
VECTOR_DB_PATH = "../../data/wiki_dump2018_nq_open/wiki2.index"
QUESTIONS_PATH = "questions.feather"
MODEL_LANGUAGE_PATH = "../../models/llms/Llama-3.2-3B-Instruct"

def setup():

    """
    Setup for the experiment.

    This function downloads the 100 question golden dataset, writes it to questions.feather and creates the retrieval, generations, results and datamodels folders.
    """
    train  = pl.read_ipc(GOLDEN_PATH).head(2)
    train.write_ipc("questions.feather")

    ## Create structure
    os.mkdir("retrieval")
    os.mkdir("generations")
    os.mkdir("results")
    os.mkdir("datamodels")

    ## Create Datamodels Structure
    os.mkdir("datamodels/datasets")
    os.mkdir("datamodels/pre_collections")
    os.mkdir("datamodels/collections")
    os.mkdir("datamodels/models")


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
    


    retrieval_indexes = {}
    retrieval_distances = {}

    df = pl.read_ipc("questions.feather")

    ### Load faiss indices
    index = faiss.read_index(VECTOR_DB_PATH)
    # ip_index = faiss.read_index(IP_FAISS_INDEX_PATH)
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
        scores, ids = index.search(query_embedding, 100)
        # ip_ids, ip_scores = ip_index.search(query_embedding, 100)

        retrieval_indexes[idx] = ids.tolist()[0]
        retrieval_distances[idx] = scores.tolist()[0]
        # retrieval_data["ip"][idx] = (ip_ids.tolist()[0], ip_scores.tolist()[0])

    ## Save into json
    with open("retrieval/rag_retrieval_indexes.json", "w") as f:
        json.dump(retrieval_indexes, f)

    with open("retrieval/rag_retrieval_distances.json", "w") as f:
        json.dump(retrieval_distances, f)

def get_generations():


    ## Setup variables
    K = 4
    wiki = pl.read_ipc(WIKI_PATH).with_row_index("idx")
    questions = pl.read_ipc(QUESTIONS_PATH)

    model = GenericInstructModelHF(MODEL_LANGUAGE_PATH)
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

    print(retrieval_data)
    ## Iterate questions
    for r_idx in range(len(retrieval_data)):

        top_k = retrieval_data[f"{r_idx}"][0:K]
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
    
        
def create_datamodels_datasets():
    """
    This function creates two .h5 files, training and testing, with respective sizes train_samples and test_samples
    Each element of the dataset corresponds in array of k samples going from [0, size_index)
    These elements represents the position on the RAG dict, as the index for each sample may vary the position in the relative top-size_indez retrieved samples will be
    the same
    """

    DATASET_PATH = "datamodels"
    setter_config = IndexBasedSetterConfig(
        save_path=DATASET_PATH,
        size_index=100,
        k=4,
        train_samples=20,
        test_samples=4
    )

    setter = IndexBasedSetter(config=setter_config)
    setter.set()
    
def run_pre_colections():


    model = GenericInstructModelHF(MODEL_LANGUAGE_PATH)

    model_configs = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_length": 2048,
            "max_new_tokens": 10
    }

    config = DatamodelIndexBasedConfig(
        k = 4,
        num_models= 2,
        datamodels_path = "datamodels",
        train_set_path=WIKI_PATH,
        test_set_path=QUESTIONS_PATH
    )

    train_log_config = LogConfig(
        project="subpartition-datamodels-rag",
        dir="logs",
        id=f"train_test_{str(datetime.datetime.now)}",
        name=f"train_test",
        config={
            "llm": "Llama-3.2-3B-Instruct",
            "gpu": f"{torch.cuda.get_device_name(0)}",
            "index": "FAISS_L2",
            "size_index": 100,
            "model_configs": model_configs,
            "datamodel_configs": repr(config)
        },
        tags=["test", "pre_collections", "FAISS_L2", "top_100"]
    )

    test_log_config = LogConfig(
        project="subpartition-datamodels-rag",
        dir="logs",
        id=f"test_test_{str(datetime.datetime.now)}",
        name=f"test_test",
        config={
            "llm": "Llama-3.2-3B-Instruct",
            "gpu": f"{torch.cuda.get_device_name(0)}",
            "index": "FAISS_L2",
            "size_index": 100,
            "model_configs": model_configs,
            "datamodel_configs": repr(config)
        },
        tags=["test", "pre_collections", "FAISS_L2", "top_100"]
    )



    datamodel = DatamodelsIndexBasedNQPipeline(config=config)

    print("Start Creating Train Pre Collection")
    datamodel.create_pre_collection(
        instruction= "You are given a question and you MUST respond in 5 tokens, use the provided documents to try to answer the question",
        llm = model,
        start_idx = 0, 
        end_idx = 20, 
        mode = "train", 
        log = True, 
        log_config = train_log_config, 
        checkpoint = 2, 
        output_column = "answers",
        model_configs = model_configs,
        rag_indexes_path="retrieval/rag_retrieval_indexes.json"
    )

    print("Start Creating Test Pre Collection")
    datamodel.create_pre_collection(
        instruction= "You are given a question and you MUST respond in 5 tokensuse the provided documents to try to answer the question",
        llm = model,
        start_idx = 0, 
        end_idx = 4, 
        mode = "test", 
        log = True, 
        log_config = test_log_config, 
        checkpoint = 2, 
        output_column = "answers",
        model_configs = model_configs,
        rag_indexes_path="retrieval/rag_retrieval_indexes.json"
    )


def run_collections():



    config = DatamodelIndexBasedConfig(
        k = 4,
        num_models= 2,
        datamodels_path = "datamodels",
        train_set_path=WIKI_PATH,
        test_set_path=QUESTIONS_PATH
    )


    evaluator = Rouge_L_evaluator()

    datamodel = DatamodelsIndexBasedNQPipeline(config)

    test_log_config = LogConfig(
        project="subpartition-datamodels-rag",
        dir="logs",
        id=f"test_collections_{str(datetime.datetime.now)}",
        name=f"test_test_collections",
        config={
            "evaluator": "Rouge-L",
            "gpu": f"{torch.cuda.get_device_name(0)}",
            "index": "FAISS_L2",
            "size_index": 100,
            "datamodel_configs": repr(config)
        },
        tags=["test", "collections", "FAISS_L2", "top_100"]
    )

    train_log_config = LogConfig(
        project="subpartition-datamodels-rag",
        dir="logs",
        id=f"train_collections_{str(datetime.datetime.now)}",
        name=f"test_train_collections",
        config={
            "evaluator": "Rouge-L",
            "gpu": f"{torch.cuda.get_device_name(0)}",
            "index": "FAISS_L2",
            "size_index": 100,
            "datamodel_configs": repr(config)
        },
        tags=["test", "collections", "FAISS_L2", "top_100"]
    )

    print("Start Creating Train Collection")
    datamodel.create_collection(
        evaluator = evaluator,
        mode = "train",
        collection_name = "collection_datamodel_L2_train",
        log = True,
        log_config = train_log_config
    )


    print("Start Creating Test Collection")
    datamodel.create_collection(
        evaluator = evaluator,
        mode = "test",
        collection_name ="collection_datamodel_L2_test",
        log = True,
        log_config = test_log_config
    )







    




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

        case "create_datamodels_datasets":
            create_datamodels_datasets()

        case "run_pre_collections":
            run_pre_colections()
        
        case "run_collections":
            run_collections()