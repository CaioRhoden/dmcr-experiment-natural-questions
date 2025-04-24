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
    Sets up the datamodels pipeline by partitioning the data for each example

    For each example, this function creates a folder with the name question_{i}_datamodels
    and partitions the data into target and random data. The target data is the data that
    is in the golden corpus, and the random data is the data that is not in the golden
    corpus. The target data is saved in a file called target.feather and the random data
    is saved in a file called random.feather.

    The function also saves the test data for each example in a file called test.csv.

    The data is partitioned using the following steps:
    1. Filter the data to only include the example with the given id.
    2. Explode the answers column to create a separate row for each answer.
    3. Save the test data to a file called test.csv.
    4. Filter the data to only include the rows that are in the golden corpus.
    5. Filter the data to only include the rows that are not in the golden corpus.
    6. Sample the data to create a random sample of 15 rows.
    7. Save the target data to a file called target.feather.
    8. Save the random data to a file called random.feather.
    """
    
    GOLDEN_PATH = "../data/nq_open_gold/processed/train.feather"
    WIKI_PATH = "../data/wiki_dump2018_nq_open/processed/wiki.feather"
    train  = pl.read_ipc(GOLDEN_PATH)
    wiki = pl.read_ipc(WIKI_PATH).with_row_index("idx")

    
    selected_ids = [
        4393532674001821363	,
        -4144729966148354479,
        1317425899790858647,
        824576888464737344,
        -1245842872065838644
    ]


    train.filter(pl.col("example_id").is_in(selected_ids))

    for i in range(len(selected_ids)):
        os.mkdir(f"question_{i}_datamodels")
        test = train.filter(pl.col("example_id") == selected_ids[i])
        test.explode("answers").write_csv(f"question_{i}_datamodels/test_set.csv")
    

        ### Partition
        gold_ids_in_corpus = (
            train
            .filter(pl.col("example_id") == selected_ids[i])
            .select(pl.col("idx_gold_in_corpus").alias("idx"))
        )

        filter_wiki_titles= (
            wiki
            .join(gold_ids_in_corpus, on="idx", how="inner")
            .select("idx", "title")
        )

        target = (
            wiki
            .join(filter_wiki_titles, on="title", how="inner")
            .join(filter_wiki_titles, on="idx", how="anti")
            .drop("idx")
            .with_row_index("idx")
        )

        random = (
            wiki
            .join(filter_wiki_titles, on="idx", how="anti")
            .join(filter_wiki_titles, on="title", how="anti")
            .with_columns(
                pl.lit(None).alias("idx_right")
            )
            .sample(n=15, shuffle=True, seed=42)
            .drop("idx")
            .with_row_index("idx")
        )

        target.write_ipc(f"question_{i}_datamodels/target.feather")
        random.write_ipc(f"question_{i}_datamodels/random.feather")

def setter(question: int):
    DATAMODEL_PATH = f"question_{question}_datamodels"
    setter = StratifiedSetter(
            load_path_target=f"{DATAMODEL_PATH}/target.feather",
            load_path_random=f"{DATAMODEL_PATH}/random.feather",
            save_path=f"{DATAMODEL_PATH}",
            k=4,
            n_samples_target=50,
            n_test_target=5,
            n_samples_mix=50,
            n_test_mix=5,
            n_samples_random=50,
            n_test_random=5,
            index_col="idx",
            seed=42
        )

    setter.set()

def create_pre_collections(question: int):

    DATAMODEL_PATH = f"question_{question}_datamodels"
    model = GenericInstructModelHF("../models/llms/Llama-3.2-3B-Instruct")

    model_configs = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_length": 2048,
            "max_new_tokens": 10
    }

    config = DatamodelConfig(
        k = 4,
        num_models= 1,
        datamodels_path = f"{DATAMODEL_PATH}",
    )

    log_config = LogConfig(
        project="nq_stratified_datamodels",
        dir="../logs",
        id=f"pre_collections_question_{question}_{str(datetime.datetime.now)}",
        name=f"pre_collection_question_{question}",
        config={
            "k": 4,
            "num_models": 1,
            "evaluator": "Rouge_L_evaluator",
            "llm": "Llama-3.2-3B-Instruct",
            "gpu": f"{torch.cuda.get_device_name(0)}",
        },
        tags=[f"question_{question}", "pre_collections"]
    )



    datamodel = DatamodelsNQPipeline(config)

    print("Start Creating Train Pre Collection")
    datamodel.create_pre_collection(
        instruction= "You are given a question and you MUST respond in 5 tokens, use the provided documents to try to answer the question",
        llm = model,
        start_idx = 0, 
        end_idx = -1, 
        mode = "train", 
        log = True, 
        log_config = log_config, 
        checkpoint = 150, 
        output_column = "answers",
        model_configs = model_configs 
    )

    print("Start Creating Test Pre Collection")
    datamodel.create_pre_collection(
        instruction= "You are given a question and you MUST respond in 5 tokensuse the provided documents to try to answer the question",
        llm = model,
        start_idx = 0, 
        end_idx = 15, 
        mode = "test", 
        log = True, 
        log_config = log_config, 
        checkpoint = 150, 
        output_column = "answers",
        model_configs = model_configs 
    )

def create_collections(question: int):

    DATAMODEL_PATH = f"question_{question}_datamodels"

    config = DatamodelConfig(
        k = 4,
        num_models= 1,
        datamodels_path = f"{DATAMODEL_PATH}",
    )

    log_config = LogConfig(
        project="nq_stratified_datamodels",
        dir="../logs",
        id=f"instruction_variation_1_collections_question_{question}_{str(datetime.datetime.now)}",
        name=f"instruction_variation_1_collection_question_{question}",
        config={
            "k": 4,
            "num_models": 1,
            "evaluator": "Rouge_L_evaluator",
            "llm": "Llama-3.2-3B-Instruct",
            "gpu": f"{torch.cuda.get_device_name(0)}",
        },
        tags=[f"question_{question}", "collections", "instruction_variation_1"]
    )

    evaluator = Rouge_L_evaluator()

    datamodel = DatamodelsNQPipeline(config)

    print("Start Creating Train Collection")
    datamodel.create_collection(
        evaluator = evaluator,
        mode = "train",
        collection_name = f"collection_question_{question}",
    )


    print("Start Creating Test Collection")
    datamodel.create_collection(
        evaluator = evaluator,
        mode = "test",
        collection_name = f"collection_question_{question}",
    )

def train_datamodel(question: int):

    DATAMODEL_PATH = f"question_{question}_datamodels"
    config = DatamodelConfig(
        k = 4,
        num_models= 1,
        datamodels_path = f"{DATAMODEL_PATH}",
    )

    log_config = LogConfig(
        project="nq_stratified_datamodels",
        dir="../logs",
        id=f"training_question_{question}_{str(datetime.datetime.now)}",
        name=f"training_question_{question}",
        config={
            "k": 4,
            "num_models": 1,
            "epochs": 100,
            "train_batches": 3,
            "val_batches": 1,
            "lr": 1e-4,
            "patience": 10,
            "gpu": f"{torch.cuda.get_device_name(0)}",
        },
        tags=[f"question_{question}", "training", "instruction_variation_1"]
    )

    datamodel = DatamodelsNQPipeline(config)

    datamodel.train_datamodels(
        collection_name=f"collection_question_{question}",
        epochs=100,
        train_batches=3,
        val_batches=1,
        val_size=0.1,
        lr=1e-4,
        patience=10,
        log=True,
        log_config=log_config,
        log_epochs=10,
        run_id=f"regression_question_{question}",
    )

def evaluate_datamodel(question: int):

    DATAMODEL_PATH = f"question_{question}_datamodels"
    config = DatamodelConfig(
        k = 4,
        num_models= 1,
        datamodels_path = f"{DATAMODEL_PATH}",
    )

    log_config = LogConfig(
        project="nq_stratified_datamodels",
        dir="../logs",
        id=f"evaluation_question_{question}_{str(datetime.datetime.now)}",
        name=f"evaluation_question_{question}",
        config={
            "k": 4,
            "num_models": 1,
            "metrics": "mse"
        },
        tags=[f"question_{question}", "evaluation", "instruction_variation_1"]
    )

    datamodel = DatamodelsNQPipeline(config)
    datamodel.evaluate_test_collections(
        evaluation_id=f"evaluation_question_{question}",
        collection_name=f"collection_question_{question}",
        model_id=f"regression_question_{question}",
        log=True,
        log_config=log_config
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", "-s", type=str, required=True)
    parser.add_argument("--id", "-i", type=int, required=False)
    args = parser.parse_args()

    step = args.step

    match step:
        case "setup":
            setup()

        case "setter":
            setter(args.id)

        case "pre_collections":
            create_pre_collections(args.id)

        case "collections":
            create_collections(args.id)

        case "train":
            train_datamodel(args.id)

        case "evaluate":
            evaluate_datamodel(args.id)