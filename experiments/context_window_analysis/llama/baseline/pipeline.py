import argparse
from utils.BaselinePipeline import BaselinePipeline
from utils.set_random_seed import set_random_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", "-s", type=str, required=True)
    args = parser.parse_args()

    step = args.step

    pipeline = BaselinePipeline(config_path="config.yaml")
    set_random_seed(42)
    pipeline.invoke_pipeline_step(step)