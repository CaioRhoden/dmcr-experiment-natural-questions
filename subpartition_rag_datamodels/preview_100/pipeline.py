import argparse
from utils.RAGBasedExperimentPipeline import RAGBasedExperimentPipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", "-s", type=str, required=True)
    args = parser.parse_args()

    step = args.step

    pipeline = RAGBasedExperimentPipeline(config_path="config.yaml")
    pipeline.set_random_seed()
    pipeline.invoke_pipeline_stpe(step)