import argparse
import os
import logging
import shutil
import torch

from datetime import datetime
from py_src import configuration_file, internal_names, initial_checking, cuda, node, dfl_logging, simulator_common, cpu
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.ml_setup_base import dataset_intermediate_layer as dataset_il

simulator_base_logger = logging.getLogger(internal_names.logger_simulator_base_name)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main(config_file_path, output_folder_name):
    # create output dir
    if output_folder_name is None:
        output_folder_path = os.path.join(os.curdir, f"lr_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")}")
    else:
        output_folder_path = os.path.join(os.curdir, output_folder_name)
    os.mkdir(output_folder_path)
    backup_path = os.path.join(output_folder_path, internal_names.default_backup_folder_name)
    os.mkdir(backup_path)

    # init logging
    dfl_logging.set_logging(os.path.join(output_folder_path, internal_names.log_file_name), simulator_base_logger)

    # init config file
    config_file = configuration_file.load_configuration(config_file_path)
    shutil.copy2(config_file_path, backup_path)  # backup config file
    simulator_base_logger.info(f"config file path: ({config_file_path}), name: ({config_file.config_name}).")
    config_ml_setup = config_file.get_ml_setup()
    config_ml_setup.self_validate()

    # set up runtime_parameters
    runtime_parameters = RuntimeParameters()
    runtime_parameters.max_tick = config_file.max_tick
    runtime_parameters.current_tick = 0
    runtime_parameters.dataset_label = config_ml_setup.dataset_label
    runtime_parameters.phase = SimulationPhase.INITIALIZING
    runtime_parameters.output_path = output_folder_path

    # check and create topology
    nodes_set = initial_checking.check_consistent_nodes(config_file.get_topology, config_file.max_tick)
    topology = config_file.get_topology(runtime_parameters)
    runtime_parameters.topology = topology
    simulator_base_logger.info(f"topology is updated at tick {runtime_parameters.current_tick}")
    simulator_common.save_topology_to_file(topology, runtime_parameters.current_tick, runtime_parameters.output_path, mpi_enabled=True)

    # create dataset
    training_dataset = dataset_il.DatasetWithFastLabelSelection(config_ml_setup.training_data)

    # create nodes
    runtime_parameters.node_container = {}
    for single_node in nodes_set:
        temp_node = node.Node(single_node, config_ml_setup, use_cpu=True)
        # create optimizer for this node if using cpu
        optimizer, lr_scheduler = config_file.get_optimizer(temp_node, temp_node.model, runtime_parameters, config_ml_setup)
        temp_node.set_optimizer(optimizer)
        if lr_scheduler is not None:
            temp_node.set_lr_scheduler(lr_scheduler)

        # setup ml config(dataset label distribution, etc)
        temp_node.set_ml_setup(config_ml_setup)

        # next training tick
        next_training_time = config_file.get_next_training_time(temp_node, runtime_parameters)
        temp_node.set_next_training_tick(next_training_time)
        # average_algorithm
        average_algorithm = config_file.get_average_algorithm(temp_node, runtime_parameters)
        temp_node.set_average_algorithm(average_algorithm)
        # average buffer size
        average_buffer_size = config_file.get_average_buffer_size(temp_node, runtime_parameters)
        temp_node.set_average_buffer_size(average_buffer_size)
        # label distribution
        label_distribution = config_file.get_label_distribution(temp_node, runtime_parameters)
        temp_node.set_label_distribution(label_distribution, dataset_with_fast_label=training_dataset)
        # add node to container
        runtime_parameters.node_container[single_node] = temp_node

    # init nodes
    config_file.node_behavior_control(runtime_parameters)

    # begin simulation
    runtime_parameters.mpi_enabled = False

    data_trained = 0
    while runtime_parameters.current_tick <= config_file.max_tick:
        runtime_parameters.phase = SimulationPhase.START_OF_TICK
        node_target: node.Node
        node_name: str

        training_node_names = []
        for node_name, node_target in runtime_parameters.node_container.items():
            if node_target.next_training_tick == runtime_parameters.current_tick:
                node_target.is_training_this_tick = True
                training_node_names.append(node_name)
                optimizer = node_target.optimizer
                lr_scheduler = node_target.lr_scheduler
                data_trained += node_target.ml_setup.training_batch_size
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                lr = get_lr(optimizer)
                simulator_base_logger.info(f"tick: {runtime_parameters.current_tick}, epoch: {data_trained/len(node_target.train_loader.dataset)}, training node: {node_target.name}, lr={lr}")

        """update next training tick"""
        for index, node_name in enumerate(training_node_names):
            node_target = runtime_parameters.node_container[node_name]
            node_target.next_training_tick = config_file.get_next_training_time(node_target, runtime_parameters)

        runtime_parameters.current_tick += 1
    exit(0)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='DFL simulator (torch version)')
    parser.add_argument('--config', type=str, default="./simulator_config.py", help='path to config file, default: "./simulator_config.py')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    args = parser.parse_args()

    main(args.config, args.output_folder_name)
