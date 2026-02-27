from utils.ebm_classifier import DataLoader, EBMPipeline, EBMTrainer, RocAucEvaluator
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
 
EXPERIMENTS = ["experiment_1", "experiment_4", "experiment_54", "experiment_61", "experiment_73"]

def saving_plot_results(df: pl.DataFrame):

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    color_map = {"groundtruth": "#87CEEB", "judge": "#F8B88B"}  # Pastel blue, pastel orange

    for row, t in enumerate(["groundtruth", "judge"]):
        for col, exp in enumerate(EXPERIMENTS):
            subset = df.filter((pl.col("type") == t) & (pl.col("experiment") == exp))
            sns.histplot(data=subset.to_pandas(), x="auc", bins=20, color=color_map[t], ax=axes[row, col])
            axes[row, col].set_title(f"{t.capitalize()} - {exp}")
            axes[row, col].set_xlim(0, 1)
            if col == 0:
                axes[row, col].set_ylabel("Count")
            else:
                axes[row, col].set_ylabel("")
            axes[row, col].set_xlabel("AUC")

    plt.tight_layout()
    plt.savefig("ebm_auc_histograms.png")


def run_ebm_pipeline():
    groundtruth_data_loader = DataLoader(load_path="../binary_collections/groundtruth")
    judge_data_loader = DataLoader(load_path="../binary_collections/judge")

    ##

    configs = {
        "n_jobs": 9,
        "early_stopping_rounds": 50,
        "max_interaction_bins": 32,
        
    }
    trainer = EBMTrainer(model_configs=configs)
    evaluator = RocAucEvaluator()

    g_pipeline = EBMPipeline(data_loader=groundtruth_data_loader, trainer=trainer, evaluator=evaluator)
    j_pipeline = EBMPipeline(data_loader=judge_data_loader, trainer=trainer, evaluator=evaluator)
    
    print("Running EBM Classifier Pipeline for Groundtruth supervised experiment...")
    g_results = g_pipeline.run(saving_model="ebm_groundtruth_models.pkl")
    
    print("Running EBM Classifier Pipeline for Judge supervised experiment...")
    j_results = j_pipeline.run(saving_model="ebm_judge_models.pkl")

    print("Combining evaluation results...")
    g_results = g_results.with_columns(pl.lit("groundtruth").alias("type"))
    j_results = j_results.with_columns(pl.lit("judge").alias("type"))
    combined_results = pl.concat([g_results, j_results])


    combined_results.write_ipc("ebm_evaluation_results.feather", compression="zstd")

    print("Saving evaluation results plot...")
    saving_plot_results(combined_results)

if __name__ == "__main__":
    run_ebm_pipeline()


    