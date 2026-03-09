from utils.ebm_classifier import DataLoader, EBMPipeline, EBMTrainer, RocAucEvaluator
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
 
EXPERIMENTS = ["experiment_1", "experiment_4", "experiment_54", "experiment_61", "experiment_73"]

def saving_plot_results(df: pl.DataFrame):

    df = df.with_columns(pl.lit("voting").alias("type"))
    fig, axes = plt.subplots(1, e=(20, 4), sharex=True, sharey=True)
    color_map = {"voting": "#87ebda"}  # Pastel green
    for col, exp in enumerate(EXPERIMENTS):
        subset = df.filter((pl.col("type") == t) & (pl.col("experiment") == exp))
        sns.histplot(data=subset.to_pandas(), x="auc", bins=20, color=color_map[t], ax=axes[col])
        axes[col].set_title(f"{t.capitalize()} - {exp}")
        axes[col].set_xlim(0, 1)
        if col == 0:
            axes[col].set_ylabel("Count")
        else:
            axes[col].set_ylabel("")
        axes[col].set_xlabel("AUC")

    plt.tight_layout()
    plt.savefig("ebm_auc_histograms.png")


def run_ebm_pipeline():
    voting_data_loader = DataLoader(load_path="../binary_collections/voting_alt1")

    ##

    configs = {
        "n_jobs": 9,
        "early_stopping_rounds": 50,
        "max_interaction_bins": 32,
        
    }
    trainer = EBMTrainer(model_configs=configs)
    evaluator = RocAucEvaluator()

    voting_pipeline = EBMPipeline(data_loader=voting_data_loader, trainer=trainer, evaluator=evaluator)
    
    print("Running EBM Classifier Pipeline for Voting supervised experiment...")
    v_results = voting_pipeline.run(saving_model="ebm_voting_models.pkl")

    print("Combining evaluation results...")
    v_results.write_ipc("ebm_voting_evaluation_results.feather", compression="zstd")
    saving_plot_results(v_results)

if __name__ == "__main__":
    run_ebm_pipeline()


    