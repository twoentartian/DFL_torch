import argparse
import logging
import os
import sys
import shutil

from datetime import datetime

from py_src import internal_names, configuration_file

simulator_base_logger = logging.getLogger(internal_names.logger_simulator_base_name)


def set_logging(log_file_path: str):
    class ExitOnExceptionHandler(logging.StreamHandler):
        def emit(self, record):
            if record.levelno == logging.CRITICAL:
                raise SystemExit(-1)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] [%(name)s] --- %(message)s (%(filename)s:%(lineno)s)")

    file = logging.FileHandler(log_file_path)
    file.setLevel(logging.DEBUG)
    file.setFormatter(formatter)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    simulator_base_logger.setLevel(logging.DEBUG)
    simulator_base_logger.addHandler(file)
    simulator_base_logger.addHandler(console)
    simulator_base_logger.addHandler(ExitOnExceptionHandler())
    simulator_base_logger.info("logging setup complete")

    del file, console, formatter

def main():
    parser = argparse.ArgumentParser(description='DFL simulator (torch version)')
    parser.add_argument('--config', type=str, default="./simulator_config.py", help='path to config file, default: "./simulator_config.py')
    args = parser.parse_args()

    # create output dir
    output_folder_path = os.path.join(os.curdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
    os.mkdir(output_folder_path)
    backup_path = os.path.join(output_folder_path, internal_names.default_backup_folder_name)
    os.mkdir(backup_path)

    # init config file
    config_file_path = args.config
    config_file = configuration_file.load_configuration(config_file_path)
    shutil.copy2(config_file_path, backup_path)  # backup config file
    simulator_base_logger.info(f"config file path: ({config_file_path}), name: ({config_file.config_name}).")
    config_ml_setup = config_file.get_ml_setup()
    config_ml_setup.self_validate()

