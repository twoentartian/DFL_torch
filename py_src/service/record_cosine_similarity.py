import os
import torch
import torch.nn.functional as F
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


class ServiceVarianceRecorder(Service):
    def __init__(self, interval, phase=[SimulationPhase.END_OF_TICK], record_node=None) -> None:
        super().__init__()
        self.save_path = None
        self.save_files = {}
        self.header_order = None
        self.record_node = record_node
        self.known_nodes_to_record = set()
        self.reference_model_state = {}
        self.header = None
        self.interval = interval
        self.record_phase = phase
        self.logger = None

    @staticmethod
    def get_service_name() -> str:
        return "cosine_similarity_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING

        node_names = []
        model_stats = []
        for node_name, target_node in parameters.node_container.items():
            if self._is_current_node_recorded(node_name):
                self.known_nodes_to_record.add(node_name)
                node_names.append(node_name)
                model_stats.append(target_node.get_model_stat())
        self.initialize_without_runtime_parameters(node_names, model_stats, output_path)

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.current_tick % self.interval != 0:
            return      # skip is not time yet

        if parameters.phase in self.record_phase:
            node_names = []
            model_stats = []
            for node_name in self.known_nodes_to_record:
                model_stat = parameters.node_container[node_name].get_model_stat()
                node_names.append(node_name)
                model_stats.append(model_stat)
            self.trigger_without_runtime_parameters(parameters.current_tick, node_names, model_stats, phase_str=parameters.phase.name)

    def initialize_without_runtime_parameters(self, node_names_and_model_stats, output_path, logger=None):
        self.logger = logger
        self.save_path = os.path.join(output_path, "cosine_similarity")
        os.mkdir(self.save_path)
        for node_name, model_stat in node_names_and_model_stats.items():
            file = open(os.path.join(self.save_path, f"{node_name}.csv"), "w+")
            self._write_header(model_stat, file)
            self.save_files[node_name] = file
            self.set_reference_model_state(node_name, model_stat)

    def trigger_without_runtime_parameters(self, tick, node_names_and_model_stats, phase_str=None):
        for node_name, model_stat in node_names_and_model_stats.items():
            file = self.save_files[node_name]
            model_state_cpu = {k: v.detach().cpu().clone() for k, v in model_stat.items()}
            self._write_row(tick, phase_str, node_name, model_state_cpu, file)

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        for node_name in self.known_nodes_to_record:
            infile_path = os.path.join(checkpoint_folder_path, "cosine_similarity", f"{node_name}.csv")
            with open(infile_path, 'r', newline='') as infile:
                next(infile)
                output_file = self.save_files[node_name]
                for line in infile:
                    row_tick = int(line.split(",", 1)[0])
                    if row_tick < restore_until_tick:
                        output_file.write(line)
                output_file.flush()

    def set_reference_model_state(self, node_name, model_stat):
        self.reference_model_state[node_name] = {k: v.detach().cpu().clone() for k, v in model_stat.items()}

    @staticmethod
    def _calculate_layerwise_similarity(state_dict1, state_dict2):
        """
        Computes cosine similarity for each parameter layer individually.
        """
        similarities = {}
        assert state_dict1.keys() == state_dict2.keys()

        # We use state_dict1 keys as the reference
        for key in state_dict1.keys():
            if key not in state_dict2:
                continue

            v1 = state_dict1[key].detach().float().view(-1)
            v2 = state_dict2[key].detach().float().view(-1)

            # Cosine similarity is undefined for scalars or zero-vectors
            if v1.numel() > 1:
                sim = F.cosine_similarity(v1, v2, dim=0)
                similarities[key] = sim.item()
            else:
                # For single scalars (like some batchnorm scalars),
                # we just check if they are identical
                similarities[key] = 1.0 if torch.equal(v1, v2) else 0.0

        return similarities

    def _is_current_node_recorded(self, node_name) -> bool:
        record_current_node = True
        if (self.record_node is not None) and (node_name not in self.record_node):
            record_current_node = False
        return record_current_node

    def _write_header(self, model_stat, file):
        all_names = []
        if self.header is None:
            for name, module in model_stat.items():
                if 'weight' in name:
                    all_names.append(name)
            header = ",".join(["tick", "phase", *all_names])
            self.header_order = all_names
            file.write(header + "\n")
        else:
            file.write(self.header + "\n")
        file.flush()

    def _write_row(self, tick, phase_str, node_name, model_stat, file):
        ref_model_state = self.reference_model_state[node_name]
        similarity_values = self._calculate_layerwise_similarity(model_stat, ref_model_state)

        row_value = [str(tick), str(phase_str)]
        for single_layer_name in self.header_order:
            v = similarity_values[single_layer_name]
            row_value.append(f'{v:.7E}')
        row = ",".join(row_value)
        file.write(f"{row}\n")
        file.flush()

    def __del__(self):
        for node_name, file in self.save_files.items():
            file.flush()
            file.close()
