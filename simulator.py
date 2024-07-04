import argparse
import os
import logging
import importlib.util
import shutil
import sys
from datetime import datetime
from py_src import internal_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DFL simulator (torch version)')
    parser.add_argument('--config', type=str, default="./simulator_config.py", help='path to config file, default: "./simulator_config.py')
    args = parser.parse_args()

    # create output dir
    output_folder_path = os.path.join(os.curdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
    os.mkdir(output_folder_path)
    backup_path = os.path.join(output_folder_path, internal_names.default_backup_folder_name)
    os.mkdir(backup_path)

    # logging
    logging.basicConfig(filename=os.path.join(output_folder_path, internal_names.log_file_name), level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    simulator_base_logger = logging.getLogger("SimulatorBase")

    # config file
    config_file_path = args.config
    if not os.path.exists(config_file_path):
        logging.info(f"config file ({config_file_path}) does not exists, create the default config file.")
        shutil.copyfile(os.path.join(os.curdir, internal_names.py_src_dir_name, internal_names.default_config_file_name), config_file_path)
    shutil.copy2(config_file_path, backup_path)  # backup config file
    spec = importlib.util.spec_from_file_location(internal_names.config_module_name, config_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[internal_names.config_module_name] = module
    spec.loader.exec_module(module)

    simulator_base_logger.info(f"config file path: ({config_file_path}), name: ({module.config_name})")

