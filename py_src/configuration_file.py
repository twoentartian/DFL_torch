import os
import importlib.util
import shutil
import sys
import logging

from py_src import internal_names, util

logger = logging.getLogger(f"{internal_names.logger_simulator_base_name}.{util.basename_without_extension(__file__)}")


def load_configuration(config_file_path: str):
    if not os.path.exists(config_file_path):
        logger.info(f"config file ({config_file_path}) does not exists, create the default config file.")
        shutil.copyfile(os.path.join(os.curdir, internal_names.py_src_dir_name, internal_names.default_config_file_name), config_file_path)

    spec = importlib.util.spec_from_file_location(internal_names.config_module_name, config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[internal_names.config_module_name] = config_module
    spec.loader.exec_module(config_module)
    return config_module
