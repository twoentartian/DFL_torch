import argparse
import os
import logging
import shutil
import sys

from datetime import datetime
from py_src import configuration_file, internal_names, initial_checking, cuda, node, dataset
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
simulator_base_logger = logging.getLogger(internal_names.logger_simulator_base_name)


def set_logging(log_file_path: str):
    class ExitOnExceptionHandler(logging.StreamHandler):
        def emit(self, record):
            if record.levelno == logging.CRITICAL:
                raise SystemExit(-1)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)8s] [%(name)s] --- %(message)s (%(filename)s:%(lineno)s)")

    file = logging.FileHandler(log_file_path)
    file.setLevel(logging.DEBUG)
    file.setFormatter(formatter)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    simulator_base_logger.setLevel(logging.DEBUG)
    simulator_base_logger.addHandler(file)
    simulator_base_logger.addHandler(console)
    simulator_base_logger.addHandler(ExitOnExceptionHandler())
    simulator_base_logger.info("logging setup complete")

    del file, console, formatter


def train_single_node(target_node: node.Node, rt_para: RuntimeParameters):
    pass


def begin_simulation(runtime_parameters: RuntimeParameters, config_file: configuration_file, training_dataset: dataset.DatasetWithFastLabelSelection):
    # begin simulation
    while runtime_parameters.current_tick <= config_file.max_tick:
        """start of tick"""
        runtime_parameters.phase = SimulationPhase.START_OF_TICK

        """before training"""
        runtime_parameters.phase = SimulationPhase.BEFORE_TRAINING

        """training"""
        runtime_parameters.phase = SimulationPhase.TRAINING

        simulator_base_logger.info(f"current tick: {runtime_parameters.current_tick}")
        node_target: node.Node
        node_name: str
        for node_name, node_target in runtime_parameters.node_container.items():
            if node_target.next_training_tick == runtime_parameters.current_tick:
                # perform training
                for data, label in node_target.train_loader:
                    loss = cuda.current_env_info.submit_training_job(node_target.model, node_target.optimizer, config_file.ml_setup.criterion, data, label)
                    simulator_base_logger.info(f"training on node {node_target.name}, loss = {loss:.2f}")
                    node_target.next_training_tick = config_file.get_next_training_time(node_target, runtime_parameters)

                    break

        """after training"""
        runtime_parameters.phase = SimulationPhase.AFTER_TRAINING

        """before averaging"""
        runtime_parameters.phase = SimulationPhase.BEFORE_AVERAGING

        """averaging"""
        runtime_parameters.phase = SimulationPhase.AVERAGING

        """after averaging"""
        runtime_parameters.phase = SimulationPhase.AFTER_AVERAGING

        """end of tick"""
        runtime_parameters.phase = SimulationPhase.END_OF_TICK

        runtime_parameters.current_tick += 1


def main():
    parser = argparse.ArgumentParser(description='DFL simulator (torch version)')
    parser.add_argument('--config', type=str, default="./simulator_config.py", help='path to config file, default: "./simulator_config.py')
    args = parser.parse_args()

    # create output dir
    output_folder_path = os.path.join(os.curdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
    os.mkdir(output_folder_path)
    backup_path = os.path.join(output_folder_path, internal_names.default_backup_folder_name)
    os.mkdir(backup_path)

    # init logging
    set_logging(os.path.join(output_folder_path, internal_names.log_file_name))

    # init config file
    config_file_path = args.config
    config_file = configuration_file.load_configuration(config_file_path)
    shutil.copy2(config_file_path, backup_path)  # backup config file
    simulator_base_logger.info(f"config file path: ({config_file_path}), name: ({config_file.config_name}).")

    # init cuda
    cuda.current_env_info.measure_memory_consumption_for_performing_ml(config_file.ml_setup)
    cuda.current_env_info.allocate_executors()
    cuda.current_env_info.print_ml_info()
    cuda.current_env_info.print_gpu_info()

    # set up runtime_parameters
    runtime_parameters = RuntimeParameters()
    runtime_parameters.max_tick = config_file.max_tick
    runtime_parameters.current_tick = 0
    runtime_parameters.dataset_label = config_file.ml_setup.dataset_label

    # check and create topology
    nodes_set = initial_checking.check_consistent_nodes(config_file.get_topology, config_file.max_tick)
    runtime_parameters.topology = config_file.get_topology(runtime_parameters)

    # create dataset
    training_dataset = dataset.DatasetWithFastLabelSelection(config_file.ml_setup.training_data)

    # create nodes
    runtime_parameters.node_container = {}
    for single_node in nodes_set:
        temp_node = node.Node(single_node, config_file.ml_setup.model)
        temp_node.set_ml_setup(config_file.ml_setup)
        # next training tick
        next_training_time = config_file.get_next_training_time(temp_node, runtime_parameters)
        temp_node.set_next_training_tick(next_training_time)
        # label distribution
        label_distribution = config_file.get_label_distribution(temp_node, runtime_parameters)
        assert label_distribution is not None
        temp_node.set_label_distribution(label_distribution, training_dataset)
        # optimizer
        optimizer = config_file.get_optimizer(temp_node, runtime_parameters)
        assert optimizer is not None
        temp_node.set_optimizer(optimizer)
        # add node to container
        runtime_parameters.node_container[single_node] = temp_node

    # begin simulation
    begin_simulation(runtime_parameters, config_file, training_dataset)

    exit(0)


if __name__ == "__main__":
    main()
