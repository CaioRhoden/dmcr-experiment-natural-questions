from utils.Runner import InferenceRunner
import yaml
import json
import argparse

def main(config_path: str, experiment_setting_path: str):

    config = yaml.safe_load(open(config_path))
    experiment_setting = json.load(open(experiment_setting_path))

    runner = InferenceRunner(config, experiment_setting)
    runner()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str, required=True)
    parser.add_argument("--experiment_setting_path", "-s", type=str, required=True)
    args = parser.parse_args()

    main(args.config_path, args.experiment_setting_path)

