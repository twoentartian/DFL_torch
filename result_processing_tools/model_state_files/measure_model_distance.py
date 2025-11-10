import argparse

import lmdb
import os
import sys
import logging
import io
from collections import OrderedDict
from typing import Dict, Iterable, Callable

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import lmdb_pack, util

ModelWeightsState = Dict[str, torch.Tensor]

logger = logging.getLogger("measure_model_distance")

def pair_distance(a: ModelWeightsState, b: ModelWeightsState, norm_layers) -> (Dict[str, float], float):
    """
    Distance between two models: sum of L2 norms over keys shared by both.
    (You can switch to MSE or cosine if needed.)
    """
    all_dist = {}
    total_dist = 0.0
    # Only compare overlapping layers; skip shapes that don't match
    for k in a.keys() & b.keys():
        if a[k].dtype is torch.long or b[k].dtype is torch.long:
            continue
        if k in norm_layers:
            continue
        ta = a[k].view(-1)
        tb = b[k].view(-1)
        if ta.numel() != tb.numel():
            raise RuntimeError("model shape mismatch")
        diff = ta - tb
        dist = float(torch.linalg.vector_norm(diff))
        all_dist[k] = dist
        total_dist += dist
    return all_dist, total_dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Move the model towards certain direction and keep its accuracy')
    parser.add_argument("model_a", type=str, help="path to the first model")
    parser.add_argument("model_b", type=str, help="path to the second model")
    parser.add_argument("--norm_layer_keywords", nargs='*', default=['norm', 'bn', 'num_batches_tracked', 'running_var', 'running_mean'], help="specify the keyword for normalization layers")

    args = parser.parse_args()

    model_a_path = args.model_a
    model_b_path = args.model_b
    norm_layer_keywords = args.norm_layer_keywords

    util.set_logging(logger, "main")
    logger.info("logging setup complete")

    model_state_a, model_name_a, _ = util.load_model_state_file(model_a_path)
    model_state_b, model_name_b, _ = util.load_model_state_file(model_b_path)
    assert model_name_a == model_name_b, f"different model names, {model_name_a} != {model_name_b}"

    norm_layers = set()
    for layer_name, _ in model_state_a.items():
        for k in norm_layer_keywords:
            if k in layer_name:
                norm_layers.add(layer_name)
    logger.info(f"norm_layers = {norm_layers}")
    non_norm_layers = set()
    for layer_name, _ in model_state_a.items():
        if layer_name not in norm_layers:
            non_norm_layers.add(layer_name)
    logger.info(f"non_norm_layers = {non_norm_layers}")

    layerwise_distance, total_distance = pair_distance(model_state_a, model_state_b, norm_layers)
    logger.info(f"total distance = {total_distance} \n layerwise_distance = {layerwise_distance}")

