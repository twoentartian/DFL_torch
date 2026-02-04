import os
from py_src.service_base import Service
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase

class ServiceTrainingLossAccuracyRecorder(Service):
    def __init__(self, interval, loss_file_name="training_loss.csv", accuracy_file_name="training_accuracy.csv"):
        super().__init__()
        self.loss_file = None
        self.accuracy_file = None
        self.node_order = None
        self.loss_file_name = loss_file_name
        self.accuracy_file_name = accuracy_file_name
        self.interval = interval
        self.logger = None

    @staticmethod
    def get_service_name() -> str:
        return "training_loss_recorder"

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        assert parameters.phase == SimulationPhase.INITIALIZING
        node_order = []
        for node_name, target_node in parameters.node_container.items():
            node_order.append(node_name)
        self.initialize_without_runtime_parameters(output_path, node_order)

    def initialize_without_runtime_parameters(self, output_path, node_order, logger=None):
        self.logger = logger
        self.loss_file = open(os.path.join(output_path, f"{self.loss_file_name}"), "w+")
        self.accuracy_file = open(os.path.join(output_path, f"{self.accuracy_file_name}"), "w+")
        self.node_order = node_order
        node_order_str = [str(i) for i in self.node_order]
        header = ",".join(["tick", *node_order_str])
        self.loss_file.write(header + "\n")
        self.accuracy_file.write(header + "\n")

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        if (parameters.phase == SimulationPhase.AFTER_TRAINING) and (parameters.current_tick % self.interval == 0):
            node_name_and_loss = {}
            node_name_and_accuracy = {}
            for node_name in self.node_order:
                node_loss = parameters.node_container[node_name].most_recent_loss
                node_accuracy = parameters.node_container[node_name].most_recent_accuracy
                node_name_and_loss[node_name] = node_loss
                node_name_and_accuracy[node_name] = node_accuracy
            self.trigger_without_runtime_parameters(parameters.current_tick, node_name_and_loss, node_name_and_accuracy)

    def trigger_without_runtime_parameters(self, tick, node_name_and_loss, node_name_and_accuracy):
        loss_row = []
        for node_name in self.node_order:
            node_loss = node_name_and_loss[node_name]
            loss_row.append('%.4f' % node_loss)
        row = ",".join([str(tick), *loss_row])
        self.loss_file.write(row + "\n")
        self.loss_file.flush()

        accuracy_row = []
        for node_name in self.node_order:
            node_accuracy = node_name_and_accuracy[node_name]
            accuracy_row.append('%.4f' % node_accuracy)
        row = ",".join([str(tick), *accuracy_row])
        self.accuracy_file.write(row + "\n")
        self.accuracy_file.flush()

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        infile_path = os.path.join(checkpoint_folder_path, self.loss_file_name)
        with open(infile_path, 'r', newline='') as infile:
            next(infile)
            for line in infile:
                row_tick = int(line.split(",", 1)[0])
                if row_tick < restore_until_tick:
                    self.loss_file.write(line)
        self.loss_file.flush()

        infile_path = os.path.join(checkpoint_folder_path, self.accuracy_file_name)
        with open(infile_path, 'r', newline='') as infile:
            next(infile)
            for line in infile:
                row_tick = int(line.split(",", 1)[0])
                if row_tick < restore_until_tick:
                    self.accuracy_file.write(line)
        self.accuracy_file.flush()

    def __del__(self):
        self.loss_file.flush()
        self.loss_file.close()
        self.accuracy_file.flush()
        self.accuracy_file.close()
