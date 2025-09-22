import os
import torch
import io
import re
import lmdb
from typing import Optional, List
from py_src import util
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


class ModelStatRecorder(Service):
    def __init__(self, interval, model_name, dataset_name, phase=SimulationPhase.END_OF_TICK, record_node=None, record_at_tick: Optional[List[int]] = None) -> None:
        super().__init__()
        if record_at_tick is None:
            self.record_at_tick = []
        self.save_path = None
        self.save_path_for_each_node = None
        self.record_node = record_node
        self.known_nodes_to_record = set()
        self.interval = interval
        self.record_phase = phase
        self.save_format = None
        self.save_lmdb = None
        self.write_count = 0
        self.model_name = model_name
        self.dataset_name = dataset_name

    @staticmethod
    def get_service_name() -> str:
        return "model_stat_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, save_format="lmdb", *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING

        node_names = []
        for node_name, target_node in parameters.node_container.items():
            if self._is_current_node_recorded(node_name):
                self.known_nodes_to_record.add(node_name)
                node_names.append(node_name)
        self.initialize_without_runtime_parameters(node_names, output_path, save_format)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if (parameters.current_tick % self.interval != 0) and (parameters.current_tick not in self.record_at_tick):
            return      # skip is not time yet

        if parameters.phase == self.record_phase:
            node_names = []
            model_stats = []
            for node_name in self.known_nodes_to_record:
                model_stat = parameters.node_container[node_name].get_model_stat()
                node_names.append(node_name)
                model_stats.append(model_stat)
            self.trigger_without_runtime_parameters(parameters.current_tick, node_names, model_stats)

    def initialize_without_runtime_parameters(self, node_names, output_path, save_format="lmdb", lmdb_db_name=None):
        assert save_format in ["lmdb", "file"], "save_format must be one of 'lmdb', 'file'"
        self.save_format = save_format
        if self.save_format == "lmdb":
            if lmdb_db_name is None:
                self.save_path = os.path.join(output_path, "model_stat.lmdb")
            else:
                self.save_path = os.path.join(output_path, f"{lmdb_db_name}.lmdb")
        elif self.save_format == "file":
            self.save_path = os.path.join(output_path, "model_stat")
        os.mkdir(self.save_path)
        if self.save_format == "lmdb":
            lmdb_inst = lmdb.open(self.save_path, map_size=4 * 1024**4) # 4TB
            self.save_lmdb = lmdb_inst
        elif self.save_format == "file":
            self.save_path_for_each_node = {}
            for node_name in node_names:
                save_path_for_this_node = os.path.join(self.save_path, str(node_name))
                self.save_path_for_each_node[node_name] = save_path_for_this_node
                os.mkdir(save_path_for_this_node)

    def trigger_without_runtime_parameters(self, tick, node_names, model_stats):
        assert len(node_names) == len(model_stats)
        self.write_count += 1
        if self.save_format == "lmdb":
            lmdb_inst = self.save_lmdb
            with lmdb_inst.begin(write=True) as txn:
                for index, node_name in enumerate(node_names):
                    lmdb_tx_name = f"{node_name}/{tick}.model.pt"
                    model_stat = model_stats[index]
                    buffer = io.BytesIO()
                    torch.save(model_stat, buffer)
                    txn.put(lmdb_tx_name.encode(), buffer.getvalue())
            if self.write_count >= 1000:
                self._monitor_and_adjust_map_size()
                self.write_count = 0
        elif self.save_format == "file":
            for index, node_name in enumerate(node_names):
                assert node_name in self.save_path_for_each_node.keys()
                save_path_for_this_node = self.save_path_for_each_node[node_name]
                model_stat = model_stats[index]
                current_node_output_path = os.path.join(save_path_for_this_node, f"{tick}.model.pt")
                util.save_model_state(current_node_output_path, model_stat, self.model_name, self.dataset_name)

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, lmdb_db_name=None, *args, **kwargs):
        if self.save_format is None:
            return
        elif self.save_format == "lmdb":
            lmdb_name = "model_stat" if lmdb_db_name is None else lmdb_db_name
            existing_lmdb_path = os.path.join(checkpoint_folder_path, f"{lmdb_name}.lmdb")
            existing_lmdb = lmdb.open(existing_lmdb_path, readonly=True, lock=False)
            with self.save_lmdb.begin(write=True) as write_txn:
                with existing_lmdb.begin() as read_txn:
                    cursor = read_txn.cursor()
                    for key, value in cursor:
                        match = re.match(rb"(\d+)/(\d+)\.model\.pt", key)
                        tick = int(match.group(2))
                        if tick < restore_until_tick:
                            write_txn.put(key, value)
        elif self.save_format == "file":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _is_current_node_recorded(self, node_name) -> bool:
        record_current_node = True
        if (self.record_node is not None) and (node_name not in self.record_node):
            record_current_node = False
        return record_current_node

    def _get_lmdb_size(self):
        data_file = os.path.join(self.save_path, 'data.mdb')
        lock_file = os.path.join(self.save_path, 'lock.mdb')
        # Check if the files exist and get their combined size
        total_size = 0
        if os.path.exists(data_file):
            total_size += os.path.getsize(data_file)
        if os.path.exists(lock_file):
            total_size += os.path.getsize(lock_file)
        return total_size

    def _monitor_and_adjust_map_size(self, threshold=0.8):
        current_size = self._get_lmdb_size()
        current_mapsize = self.save_lmdb.info()['map_size']

        if current_size > current_mapsize * threshold:
            # Increase map size (e.g., double the current size)
            new_mapsize = current_mapsize * 2
            self.save_lmdb.set_mapsize(new_mapsize)

    def __del__(self):
        if self.save_format == "lmdb":
            self.save_lmdb.close()
