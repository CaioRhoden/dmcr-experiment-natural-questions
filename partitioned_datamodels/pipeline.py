######################################################################################################
## Create each datamodel folder -> partition target and random data for each -> run datamodels steps
#######################################################################################################
import polars as pl
import os
import argparse



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

def setter(step: str, question: int):
    pass

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
            setter(step, args.id)

