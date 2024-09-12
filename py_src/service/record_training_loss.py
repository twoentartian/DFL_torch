import os
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase

class ServiceTrainingLossRecorder(Service):
    def __init__(self, interval, save_file_name="training_loss.csv"):
        super().__init__()
        self.save_file = None
        self.node_order = None
        self.save_file_name = save_file_name
        self.interval = interval

    @staticmethod
    def get_service_name() -> str:
        return "training_loss_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        node_order = []
        for node_name, target_node in parameters.node_container.items():
            node_order.append(node_name)
        self.initialize_without_runtime_parameters(output_path, node_order)

    def initialize_without_runtime_parameters(self, output_path, node_order):
        self.save_file = open(os.path.join(output_path, f"{self.save_file_name}"), "w+")
        self.node_order = node_order
        node_order_str = [str(i) for i in self.node_order]
        header = ",".join(["tick", *node_order_str])
        self.save_file.write(header + "\n")

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if (parameters.phase == SimulationPhase.AFTER_TRAINING) and (parameters.current_tick % self.interval == 0):
            node_name_and_loss = {}
            for node_name in self.node_order:
                node_loss = parameters.node_container[node_name].most_recent_loss
                node_name_and_loss[node_name] = node_loss
            self.trigger_without_runtime_parameters(parameters.current_tick, node_name_and_loss)

    def trigger_without_runtime_parameters(self, tick, node_name_and_loss):
        loss_row = []
        for node_name in self.node_order:
            node_loss = node_name_and_loss[node_name]
            loss_row.append('%.4f' % node_loss)
        row = ",".join([str(tick), *loss_row])
        self.save_file.write(row + "\n")
        self.save_file.flush()

    def __del__(self):
        self.save_file.flush()
        self.save_file.close()
