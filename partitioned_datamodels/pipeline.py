######################################################################################################
## Create each datamodel folder -> partition target and random data for each -> run datamodels steps
#######################################################################################################
import polars as pl
import os
import argparse
from dmcr.datamodels.setter import StratifiedSetter
from dmcr.datamodels.pipeline import DatamodelsNQPipeline
from dmcr.datamodels.config import DatamodelConfig, LogConfig
from dmcr.models import GenericInstructModelHF
from dmcr.datamodels.evaluator import Rouge_L_evaluator


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
        test.explode("answers").write_csv(f"question_{i}_datamodels/test.csv")
    

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
        )

        random = (
            wiki
            .join(filter_wiki_titles, on="idx", how="anti")
            .join(filter_wiki_titles, on="title", how="anti")
            .sample(n=15, shuffle=True, seed=42)
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
    model = GenericInstructModelHF("../models/Llama-3.2-3B-Instruct")

    config = DatamodelConfig(
        k = 4,
        num_models= 1,
        datamodels_path = f"{DATAMODEL_PATH}",
        train_collections_idx = None,
        test_collections_idx = None,
        test_set = None,
        train_set = None,
        instructions= None,
        llm = model,
        evaluator=Rouge_L_evaluator(),
    )

    log_config = LogConfig(
        project="bbh_pre_collection",
        dir="log",
        id="bbh_pre_collection",
        name="bbh_pre_collection",
        config={
            "k": 8,
            "num_models": 40,
            "evaluator": "GleuEvaluator",
            "llm": "Llama-3.1-8B-Instruct",
            "gpu": "Quadro RTX500",
        },
        tags=["bbh", "dl-28", "pre_collections"],
    )



    datamodel = DatamodelPipeline(config)
    datamodel.set_collections_index()
    datamodel.set_dataframes()
    datamodel.set_instructions_from_path()

    print("Start Creating Pre Collection")
    datamodel.create_pre_collection(start_idx = start_idx, end_idx = end_idx, type=type, log=True, log_config=log_config, checkpoint=25)



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

