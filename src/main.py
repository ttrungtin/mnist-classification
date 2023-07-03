import argparse
import yaml

from src.config import ProjectConfig
from model.model_solver import ModelSolver
from data.data_solver import RawDataProcessor


if __name__ == "__main__":
    # arguments collect
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-dir", type=str, required=True)
    args = parser.parse_args()

    # arguments init
    config_dir = args.config_dir
    with open(config_dir, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    project_conf = ProjectConfig()
    project_conf.config_yaml = config

    #
    print(project_conf.config_yaml)

    #
    model = ModelSolver(project_conf)
    data = RawDataProcessor(project_conf)
    model.do_train(data_solver=data)
