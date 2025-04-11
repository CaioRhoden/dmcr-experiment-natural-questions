######################################################################################################
## Create each datamodel folder -> partition target and random data for each -> run datamodels steps
#######################################################################################################
import polars as pl
import os



def setup():

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
        train.select(pl.col("example_id") == selected_ids[i]).write_csv(f"question_{i}_datamodels/test.csv")
    

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




def run_datamodels(step: str, question: int):
    pass

if __name__ == "__main__":
    setup()
    run_datamodels("0", 0)