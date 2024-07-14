import os
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase


class ServiceVarianceRecorder(Service):
    def __init__(self, interval, phase=SimulationPhase.END_OF_TICK, record_node=None) -> None:
        import os
        super().__init__()
        self.save_path = None
        self.save_files = {}
        self.header_order = None
        self.record_node = record_node
        self.known_nodes_to_record = set()
        self.header = None
        self.interval = interval
        self.record_phase = phase

    @staticmethod
    def get_service_name() -> str:
        return "variance_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        self.save_path = os.path.join(output_path, "variance")
        os.mkdir(self.save_path)
        for node_name, target_node in parameters.node_container.items():
            if self._is_current_node_recorded(node_name):
                self.known_nodes_to_record.add(node_name)
                file = open(os.path.join(self.save_path, f"{node_name}.csv"), "w+")
                self._write_header(target_node.get_model_stat(), file)
                self.save_files[node_name] = file

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if parameters.current_tick % self.interval != 0:
            return      # skip is not time yet

        if parameters.phase == self.record_phase:
            for node_name in self.known_nodes_to_record:
                file = self.save_files[node_name]
                model_stat = parameters.node_container[node_name].get_model_stat()
                self._write_row(parameters.current_tick, model_stat, file)

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
            header = ",".join(["tick", *all_names])
            self.header_order = all_names
            file.write(header + "\n")
        else:
            file.write(self.header + "\n")
        file.flush()

    def _write_row(self, tick, model_stat, file):
        import torch
        row_value = [str(tick)]
        for single_layer_name in self.header_order:
            weights = model_stat[single_layer_name]
            variance = torch.var(weights).item()
            row_value.append(f'{variance:e}')
        row = ",".join(row_value)
        file.write(f"{row}\n")
        file.flush()

    def __del__(self):
        for node_name, file in self.save_files.items():
            file.flush()
            file.close()
