import json
import os
import argparse
import io
import lmdb
import torch
import logging
import sys
from pathlib import Path
import pandas as pd
from typing import Optional
from datetime import datetime
from sklearn.decomposition import IncrementalPCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import util

ignore_layers_with_keywords = ["running_mean", "running_var", "num_batches_tracked"]
lmdb_folder_names = ["0.lmdb", "model_stat.lmdb"] # these folder names will be assumed to be lmdb folder
folder_name = ["model_stat"] # these folders will be assumed to have model state files

logger = logging.getLogger("ipca_all_path")

def set_logging(target_logger, task_name, log_file_path=None):
    class ExitOnExceptionHandler(logging.StreamHandler):
        def emit(self, record):
            if record.levelno == logging.CRITICAL:
                raise SystemExit(-1)

    formatter = logging.Formatter(f"[%(asctime)s] [%(levelname)8s] [{task_name}] --- %(message)s (%(filename)s:%(lineno)s)")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    target_logger.setLevel(logging.DEBUG)
    target_logger.addHandler(console)
    target_logger.addHandler(ExitOnExceptionHandler())

    if log_file_path is not None:
        file = logging.FileHandler(log_file_path)
        file.setLevel(logging.DEBUG)
        file.setFormatter(formatter)
        target_logger.addHandler(file)

    del console, formatter

def load_models_from_lmdb(lmdb_path, arg_node_name, desired_length:Optional[int]=None, model_state_cache=None):
    if model_state_cache is not None:
        if lmdb_path in model_state_cache:
            logger.info(f"use cached lmdb {lmdb_path}")
            return model_state_cache[lmdb_path]

    lmdb_env = lmdb.open(lmdb_path, readonly=True)
    tick_and_models = {}
    with lmdb_env.begin() as txn:
        read_sample_ratio = 1
        if desired_length is not None:
            length = txn.stat()['entries']
            read_sample_ratio = length // desired_length

        cursor = txn.cursor()
        count = 0
        for key, value in cursor:
            count += 1
            if count == read_sample_ratio:
                key = key.decode("utf-8")
                key = key.replace('.model.pt', '')
                items = key.split('/')
                current_node_name = int(items[0])
                if current_node_name != arg_node_name:
                    continue
                tick = int(items[1])
                buffer = io.BytesIO(value)
                state_dict = torch.load(buffer, map_location=torch.device('cpu'))
                tick_and_models[tick] = state_dict
                count = 0
    if model_state_cache is not None:
        logger.info(f"write lmdb {lmdb_path} to cache")
        model_state_cache[lmdb_path] = tick_and_models
    return tick_and_models

def load_models_from_file(folder_path, arg_node_name, desired_length:Optional[int]=None, model_state_cache=None):
    if model_state_cache is not None:
        if folder_path in model_state_cache:
            logger.info(f"use cached model state folder {folder_path}")
            return model_state_cache[folder_path]
    tick_and_models = {}
    subfolder_path = Path(folder_path) / str(arg_node_name)
    for file_path in subfolder_path.glob("*.model.pt"):
        filename = file_path.stem
        tick_str = filename.replace(".model", "")
        try:
            tick = int(tick_str)
            model_state_dict,_,_ = util.load_model_state_file(file_path)
            tick_and_models[tick] = model_state_dict
        except ValueError:
            logger.error(f"Skipping file '{file_path.name}' - invalid tick format")
        except Exception as e:
            logger.error(f"Error loading '{file_path.name}': {e}")
    tick_and_models = dict(sorted(tick_and_models.items()))

    if model_state_cache is not None:
        logger.info(f"write model state folder {folder_path} to cache")
        model_state_cache[folder_path] = tick_and_models
    return tick_and_models

def extract_weights(model_stat, layer_name):
    weights = model_stat[layer_name].numpy()
    return weights.flatten()

def incremental_pca_all_path(arg_path_folder, arg_output_folder, arg_node_name: int, dimension, only_layers=None, sample_points=None, enable_cache=False):
    generated_targets = {}
    all_sub_folders = []
    assert len(arg_path_folder) > 0
    for folder in arg_path_folder:
        assert os.path.exists(folder)
        sub_folders = [f.path for f in os.scandir(folder) if f.is_dir()]
        all_sub_folders = all_sub_folders + sub_folders
        generated_targets[folder] = sub_folders
    assert len(all_sub_folders) > 0
    all_sub_folders = sorted(all_sub_folders)

    if enable_cache:
        logger.info("enable lmdb cache")
        model_state_cache = {}
    else:
        logger.info("disable lmdb cache")
        model_state_cache = None

    layer_and_ipca = []
    dimension_to_index = {}
    for index, d in enumerate(dimension):
        layer_and_ipca.append({})
        dimension_to_index[d] = index

    for single_sub_folder in all_sub_folders:
        for model_state_file_path, model_state_file_type in zip(lmdb_folder_names+folder_name, ["lmdb" for _ in range(len(lmdb_folder_names))] + ["file" for _ in range(len(folder_name))]):
            model_state_file_path = os.path.join(single_sub_folder, model_state_file_path)
            if not os.path.exists(model_state_file_path):
                continue
            if model_state_file_type == "lmdb":
                logger.info(f"loading lmdb: {model_state_file_path}")
                tick_and_models = load_models_from_lmdb(model_state_file_path, arg_node_name, desired_length=sample_points, model_state_cache=model_state_cache)
            elif model_state_file_type == "file":
                tick_and_models = load_models_from_file(model_state_file_path, arg_node_name, desired_length=sample_points, model_state_cache=model_state_cache)
            else:
                raise NotImplementedError

            ticks_ordered = sorted(tick_and_models.keys())
            sample_model = tick_and_models[next(iter(tick_and_models))]
            for layer_name in sample_model.keys():
                ignore = False
                for k in ignore_layers_with_keywords:
                    if k in layer_name:
                        ignore = True
                        break
                if ignore:
                    continue
                if only_layers is not None and (layer_name not in only_layers):
                    continue
                logger.info(f"processing layer {layer_name}")
                weights_list = [extract_weights(tick_and_models[tick], layer_name) for tick in ticks_ordered]
                for d in dimension:
                    current_dimension_ipca = layer_and_ipca[dimension_to_index[d]]
                    if layer_name not in current_dimension_ipca:
                        current_dimension_ipca[layer_name] = IncrementalPCA(n_components=d)
                    current_dimension_ipca[layer_name].partial_fit(weights_list)

    generated_layer_names = set()
    for folder in arg_path_folder:
        os.mkdir(os.path.join(arg_output_folder, folder))
        sub_folders = [f.name for f in os.scandir(folder) if f.is_dir()]
        sub_folder_path = [f.path for f in os.scandir(folder) if f.is_dir()]
        for index, name in enumerate(sub_folders):
            for model_state_file_path, model_state_file_type in zip(lmdb_folder_names + folder_name, ["lmdb" for _ in range(len(lmdb_folder_names))] + ["file" for _ in range(len(folder_name))]):
                model_state_file_path = os.path.join(sub_folder_path[index], model_state_file_path)
                if not os.path.exists(model_state_file_path):
                    continue
                if model_state_file_type == "lmdb":
                    logger.info(f"loading lmdb: {model_state_file_path}")
                    tick_and_models = load_models_from_lmdb(model_state_file_path, arg_node_name, desired_length=sample_points, model_state_cache=model_state_cache)
                elif model_state_file_type == "file":
                    tick_and_models = load_models_from_file(model_state_file_path, arg_node_name, desired_length=sample_points, model_state_cache=model_state_cache)
                else:
                    raise NotImplementedError
                ticks_ordered = sorted(tick_and_models.keys())
                sample_model = tick_and_models[next(iter(tick_and_models))]
                for layer_name in sample_model.keys():
                    ignore = False
                    for k in ignore_layers_with_keywords:
                        if k in layer_name:
                            ignore = True
                            break
                    if ignore:
                        continue
                    if only_layers is not None and (layer_name not in only_layers):
                        continue
                    generated_layer_names.add(layer_name)
                    weights_list = [extract_weights(tick_and_models[tick], layer_name) for tick in ticks_ordered]
                    for d in dimension:
                        output_file_path = os.path.join(arg_output_folder, folder, f"{name}_{layer_name}_{d}d.csv")
                        logger.info(f"processing output {output_file_path}")
                        ipca = layer_and_ipca[dimension_to_index[d]][layer_name]
                        result = ipca.transform(weights_list)
                        column_names = [f"PCA Dimension {i}" for i in range(d)]
                        df = pd.DataFrame(result, columns=[column_names])
                        df.insert(0, "tick", pd.Series(ticks_ordered))
                        df.to_csv(str(output_file_path))

    info_file = "info.json"
    info_file_path = os.path.join(arg_output_folder, info_file)
    info_target = {"targets": generated_targets, "layer_names": list(generated_layer_names), "dimension": dimension}
    with open(info_file_path, "w") as f:
        json.dump(info_target, f, indent=4)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Ues PCA to reduce dimension for high accuracy paths')
    parser.add_argument("path_folder", type=str, nargs='+', help="high accuracy path data folder")
    parser.add_argument("-p", "--points", type=int, help="number of sample points per path to visualize", default=0)
    parser.add_argument("-l", "--layer", type=str, nargs='+', help="only plot these layers, default: plot all layers")
    parser.add_argument("--node_name", type=int, default=0)
    parser.add_argument("--disable_3d", action='store_true')
    parser.add_argument("--disable_2d", action='store_true')
    parser.add_argument('--cache', default=False, action=argparse.BooleanOptionalAction, help="enable/disable cache lmdb in memory")

    args = parser.parse_args()
    path_folder = args.path_folder
    node_name = args.node_name
    enable_cache = args.cache
    points = None if args.points == 0 else args.points
    plot_dimensions = [2, 3]
    if args.disable_3d:
        plot_dimensions.remove(3)
    if args.disable_2d:
        plot_dimensions.remove(2)
    assert len(plot_dimensions) > 0

    only_layers = args.layer

    # create output folder
    output_folder_path = os.path.join(os.curdir, f"{__file__}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")}")
    os.mkdir(output_folder_path)
    set_logging(logger, "main", log_file_path=os.path.join(output_folder_path, "log.txt"))

    info_file_path = incremental_pca_all_path(path_folder, output_folder_path, node_name, sample_points=points, only_layers=only_layers, dimension=plot_dimensions, enable_cache=enable_cache)