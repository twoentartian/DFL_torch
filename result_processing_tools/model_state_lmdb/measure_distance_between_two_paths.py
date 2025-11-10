import argparse

import lmdb
import os
import sys
import logging
import io
from datetime import datetime
from collections import OrderedDict
from typing import Dict, Iterable, Callable

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import lmdb_pack, util, ml_setup

logger = logging.getLogger("measure_distance_between_two_paths")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ModelWeightsState = Dict[str, torch.Tensor]
Loader = Callable[[int], ModelWeightsState]

def get_tick_interval(lmdb_env, txn=None) -> (str, int):
    if txn is None:
        txn = lmdb_env.begin()
    with txn.cursor() as cur:
        cur.first()
        # assume tick 0 is recorded
        major_node, first_tick = lmdb_pack.get_node_name_and_tick_from_lmdb_index(cur.key())
        assert first_tick == 0
        cur.next()
        node, second_tick = lmdb_pack.get_node_name_and_tick_from_lmdb_index(cur.key())
        assert major_node == node
        interval = second_tick - first_tick
        for i in range(1, interval):
            name = lmdb_pack.generate_lmdb_index_from_node_name_and_tick(major_node, i)
            if txn.get(name.encode()) is not None:
                interval = i
        entry_count = txn.stat()["entries"]
    return major_node, interval, entry_count

def load_model_state_from_lmdb(lmdb_env, node_name:int, tick:int, txn=None) -> ModelWeightsState:
    if txn is None:
        txn = lmdb_env.begin()
    with txn.cursor() as cur:
        key = lmdb_pack.generate_lmdb_index_from_node_name_and_tick(node_name, tick)
        value = txn.get(key.encode())
        buffer = io.BytesIO(value)
        state_dict = torch.load(buffer, map_location=device)
    return state_dict

class LRUCache:
    """A tiny index→model cache with fixed capacity."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._d: "OrderedDict[int, ModelWeightsState]" = OrderedDict()

    def get(self, idx: int, loader: Loader) -> ModelWeightsState:
        if idx in self._d:
            self._d.move_to_end(idx)
            return self._d[idx]
        obj = loader(idx)
        self._d[idx] = obj
        self._evict_if_needed()
        return obj

    def prefetch(self, idxs: Iterable[int], loader: Loader) -> None:
        for i in idxs:
            if i in self._d:
                self._d.move_to_end(i)
                continue
            self._d[i] = loader(i)
            self._evict_if_needed()

    def _evict_if_needed(self):
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)  # evict LRU

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
    parser.add_argument("lmdb_path_a", type=str, help="path to the first lmdb folder")
    parser.add_argument("lmdb_path_b", type=str, help="path to the second lmdb folder")
    parser.add_argument("--search_length", type=int, default=10, help="search nearby models to find the shortest length")
    parser.add_argument("--norm_layer_keywords", nargs='*', default=['norm', 'bn', 'num_batches_tracked', 'running_var', 'running_mean'], help="specify the keyword for normalization layers")
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')

    args = parser.parse_args()

    path_a = args.lmdb_path_a
    path_b = args.lmdb_path_b
    search_length = args.search_length
    norm_layer_keywords = args.norm_layer_keywords

    if args.output_folder_name is None:
        time_now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_folder_path = os.path.join(os.curdir, f"{__file__}_{time_now_str}")
    else:
        output_folder_path = os.path.join(os.curdir, args.output_folder_name)
    os.mkdir(output_folder_path)

    util.set_logging(logger, "main", log_file_path=os.path.join(output_folder_path, "info.log"))
    logger.info("logging setup complete")

    info_file = open(os.path.join(output_folder_path, "arguments.txt"), 'x')
    info_file.write(f"{args}")
    info_file.flush()
    info_file.close()

    env_a = lmdb.open(path_a, readonly=True, lock=False, readahead=True, max_readers=1024)
    env_b = lmdb.open(path_b, readonly=True, lock=False, readahead=True, max_readers=1024)
    txn_a = env_a.begin()
    txn_b = env_b.begin()
    node_in_a, interval_a, entry_count_a = get_tick_interval(env_a, txn=txn_a)
    node_in_b, interval_b, entry_count_b = get_tick_interval(env_b, txn=txn_b)
    assert node_in_a == node_in_b == 0, f"there should only be 1 node called '0' in lmdb database, get a-{node_in_a} and b-{node_in_b}"
    node_name = node_in_a
    assert entry_count_a == entry_count_b
    assert interval_a == interval_b
    interval = interval_a
    entry_count = entry_count_a
    logger.info(f"interval = {interval}")
    logger.info(f"entry_count = {entry_count}")

    model_b_cache = LRUCache(search_length*2+1)
    norm_layers = None
    non_norm_layers = None
    distance_csv = None
    layer_order = None
    for index_a in range(entry_count):
        tick_a = index_a * interval
        model_a = load_model_state_from_lmdb(env_a, node_name, tick_a, txn=txn_a)

        # init
        if norm_layers is None:
            norm_layers = []
            for layer_name, _ in model_a.items():
                for k in norm_layer_keywords:
                    if k in layer_name:
                        norm_layers.append(layer_name)
            logger.info(f"norm_layers = {norm_layers}")
            non_norm_layers = []
            for layer_name, _ in model_a.items():
                if layer_name not in norm_layers:
                    non_norm_layers.append(layer_name)
            logger.info(f"non_norm_layers = {non_norm_layers}")
        if distance_csv is None:
            distance_csv = open(os.path.join(output_folder_path, "distance.csv"), "w+")
            layer_order = [k for k in non_norm_layers]
            header = ",".join(["a_tick", "shortest_distance", "b_tick", *layer_order])
            distance_csv.write(header + "\n")
            distance_csv.flush()

        shortest_distance = float("inf")
        shortest_layerwise_distance = None
        shortest_index = 0
        for i in range(index_a-search_length, index_a+search_length+1):
            if i < 0:
                continue
            if i >= entry_count:
                continue
            model_b = model_b_cache.get(i, lambda index: load_model_state_from_lmdb(env_b, node_name, index*interval, txn=txn_b))
            layerwise_distance, total_distance = pair_distance(model_a, model_b, norm_layers)
            if total_distance < shortest_distance:
                shortest_distance = total_distance
                shortest_layerwise_distance = layerwise_distance
                shortest_index = i

        distance_per_row = [f"{shortest_layerwise_distance[layer_name]:.4e}" for layer_name in layer_order]
        row = ",".join([f"{index_a * interval_a}", f"{shortest_distance:.4e}", f"{shortest_index * interval_b}", *distance_per_row])
        distance_csv.write(row + "\n")
        distance_csv.flush()

        logger.info(f"for index_a: {index_a}, the shortest distance is {shortest_distance} at index_b: {shortest_index}")

