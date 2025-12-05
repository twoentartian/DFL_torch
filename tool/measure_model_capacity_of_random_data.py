import os, sys, argparse, logging
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, util
from py_src import ml_setup


logger = logging.getLogger("measure_model_capacity_of_random_data")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure the model capacity in terms of how many random samples can be memorized')
    parser.add_argument("-m", "--model", type=str, required=True, help="specify the model type")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="specify the dataset type")

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset

    util.set_logging(logger, "main")

    current_ml_setup = ml_setup.get_ml_setup_from_config(model_name, dataset_type=dataset_name)

    random_dataset_type = ml_setup.dataset_type_to_random(current_ml_setup.dataset_type)
    print(f'{random_dataset_type.name}')
