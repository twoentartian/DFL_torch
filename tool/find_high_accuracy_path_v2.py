import os
import argparse
import logging
import shutil
import sys
import json
from enum import Enum, auto
from typing import Final, Optional, List
import torch
import re
import torch.optim as optim
import concurrent.futures
import copy
from dataclasses import dataclass, field
from datetime import datetime
from torch.utils.data import DataLoader

from find_high_accuracy_path_v2.runtime_parameters import RuntimeParameters, WorkMode
from find_high_accuracy_path import set_logging, get_files_to_process

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, model_average, model_variance_correct, special_torch_layers, cuda, util, configuration_file
from py_src.service import record_weights_difference, record_test_accuracy_loss, record_variance, record_model_stat, record_training_loss

logger = logging.getLogger("find_high_accuracy_path_v2")



def process_file_func(index, path, runtime_parameter: RuntimeParameters):
    start_point, end_point = runtime_parameter.start_and_end_point_for_paths[index]
    start_file_name = os.path.basename(start_point).replace('.model.pt', '')
    if end_point == "origin":
        assert runtime_parameter.work_mode == WorkMode.to_origin
        end_file_name = "origin"
    elif end_point == "inf":
        assert runtime_parameter.work_mode == WorkMode.to_inf
        end_file_name = "inf"
    else:
        end_file_name = os.path.basename(end_point).replace('.model.pt', '')

    # logger
    task_name = f"{start_file_name}-{end_file_name}"
    child_logger = logging.getLogger(f"find_high_accuracy_path.{task_name}")
    set_logging(child_logger, task_name, log_file_path=os.path.join(runtime_parameter.output_folder_path, "info.log"))
    child_logger.info("logging setup complete")

    if runtime_parameter.use_cpu:
        device = torch.device("cpu")
        gpu = None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu = cuda.CudaDevice(0)

    thread_per_process = runtime_parameter.total_cpu_count // runtime_parameter.worker_count
    torch.set_num_threads(thread_per_process)

    # output folders
    arg_output_folder_path = os.path.join(runtime_parameter.output_folder_path, f"{start_file_name}-{end_file_name}")
    if os.path.exists(arg_output_folder_path):
        child_logger.warning(f"{arg_output_folder_path} already exists")
    else:
        os.makedirs(arg_output_folder_path)

    """load models"""
    cpu_device = torch.device("cpu")
    start_model_stat_dict, start_model_name = util.load_model_state_file(start_point)
    current_ml_setup = ml_setup.get_ml_setup_from_config(start_model_name)
    child_logger.info(f"find model type is {start_model_name}")

    start_model = copy.deepcopy(current_ml_setup.model)
    initial_model_stat = start_model.state_dict()
    if runtime_parameter.work_mode == WorkMode.to_origin:
        end_model_stat_dict = {k: torch.zeros_like(v) for k, v in start_model_stat_dict.items()}
    elif runtime_parameter.work_mode == WorkMode.to_inf:
        end_model_stat_dict = {k: v * 100 for k, v in start_model_stat_dict.items()}
    elif runtime_parameter.work_mode == WorkMode.to_certain_model:
        end_model_stat_dict, end_model_name = util.load_model_state_file(end_point)
        assert end_model_name == start_model_name


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Move the model towards certain direction and keep its accuracy')
    parser.add_argument("start_folder", type=str, help="folder containing starting models")

    parser.add_argument("end_folder", type=str, help="folder containing destination models, or 'inf', 'origin' ")
    parser.add_argument("--mapping_mode", type=str, default='auto', choices=['auto', 'all_to_all', 'each_to_each', 'one_to_all', 'all_to_one'])

    parser.add_argument("-c", '--core', type=int, default=os.cpu_count(), help='specify the number of CPU cores to use')
    parser.add_argument("-w", "--worker", type=int, default=1, help='specify how many models to train in parallel')

    parser.add_argument("--save_ticks", type=str, help='specify when to record the models (e.g. [1,2,3,5-10]), only works when --save_format is set to work.')
    parser.add_argument("--save_interval", type=int, default=1, help='specify the saving interval')
    parser.add_argument("--save_format", type=str, default='none', choices=['none', 'file', 'lmdb'])
    parser.add_argument("--cpu", action='store_true', help='force using CPU for training')
    parser.add_argument("--amp", action='store_true', help='enable auto mixed precision')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')

    config_file = configuration_file.load_configuration("find_high_accuracy_path_v2_config.py")

    args = parser.parse_args()

    set_logging(logger, "main")
    logger.info("logging setup complete")

    runtime_parameter = RuntimeParameters()

    runtime_parameter.use_cpu = args.cpu
    runtime_parameter.use_amp = args.amp
    runtime_parameter.save_ticks = args.save_ticks
    runtime_parameter.save_interval = args.save_interval
    runtime_parameter.save_format = args.save_format

    # find all paths to process
    start_folder = args.start_folder
    if args.end_folder == "origin":
        runtime_parameter.work_mode = WorkMode.to_origin
        assert args.mapping_mode == "auto", "mapping mode has to be 'auto' for move to origin"
        files_in_start_folder = sorted(set(temp_file for temp_file in os.listdir(start_folder) if temp_file.endswith('model.pt')))
        paths_to_find = [(os.path.join(start_folder, i), "origin") for i in files_in_start_folder]
    elif args.end_folder == "inf":
        runtime_parameter.work_mode = WorkMode.to_inf
        assert args.mapping_mode == "auto", "mapping mode has to be 'auto' for move to inf"
        files_in_start_folder = sorted(set(temp_file for temp_file in os.listdir(start_folder) if temp_file.endswith('model.pt')))
        paths_to_find = [(os.path.join(start_folder, i), "inf") for i in files_in_start_folder]
    else:
        runtime_parameter.work_mode = WorkMode.to_certain_model
        paths_to_find = get_files_to_process(args.start_folder, args.end_folder, args.mapping_mode)
    paths_to_find_count = len(paths_to_find)
    runtime_parameter.start_and_end_point_for_paths = paths_to_find
    logger.info(f"totally {paths_to_find_count} paths to process: {paths_to_find}")

    # create output folder
    if args.output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.mkdir(output_folder_path)
    runtime_parameter.output_folder_path = output_folder_path
    shutil.copyfile(__file__, os.path.join(output_folder_path, os.path.basename(__file__)))
    info_file = open(os.path.join(output_folder_path, "arguments.txt"), 'x')
    info_file.write(f"{args}")
    info_file.flush()
    info_file.close()

    # worker and cpu cores setting
    runtime_parameter.total_cpu_count = args.core
    runtime_parameter.worker_count = args.worker
    if runtime_parameter.worker_count > paths_to_find_count:
        runtime_parameter.worker_count = paths_to_find_count
    logger.info(f"worker: {runtime_parameter.worker_count}")

    logger.info(f"final runtime parameters: {runtime_parameter.print()}")

    # start process
    with concurrent.futures.ProcessPoolExecutor(max_workers=runtime_parameter.worker_count) as executor:
        futures = [executor.submit(process_file_func, index, path, runtime_parameter) for index, path in enumerate(paths_to_find)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()





