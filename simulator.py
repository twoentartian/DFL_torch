import argparse
import os
import logging
import shutil
import sys
import time
import torch

from typing import Final
from datetime import datetime
from py_src import configuration_file, internal_names, initial_checking, cuda, node, dataset
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.ml_setup import MlSetup

simulator_base_logger = logging.getLogger(internal_names.logger_simulator_base_name)

REPORT_FINISH_TIME_PER_TICK: Final[int] = 100

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


def submit_training_job_cpu(training_node, criterion: torch.nn.CrossEntropyLoss, training_data: torch.Tensor, training_label: torch.Tensor):
    model = training_node.model
    optimizer = training_node.optimizer
    optimizer.zero_grad(set_to_none=True)
    output = model(training_data)
    loss = criterion(output, training_label)
    loss.backward()
    optimizer.step()
    return loss


def begin_simulation(runtime_parameters: RuntimeParameters, config_file, ml_config: MlSetup, current_cuda_env):
    # begin simulation
    timer = time.time()
    while runtime_parameters.current_tick <= config_file.max_tick:
        # report remaining time
        if runtime_parameters.current_tick % REPORT_FINISH_TIME_PER_TICK == 0 and runtime_parameters.current_tick != 0:
            time_elapsed = time.time() - timer
            timer = time.time()
            remaining = (config_file.max_tick - runtime_parameters.current_tick) // REPORT_FINISH_TIME_PER_TICK
            time_to_finish = remaining * time_elapsed
            finish_time = timer + time_to_finish
            simulator_base_logger.info(f"time taken for {REPORT_FINISH_TIME_PER_TICK} ticks: {time_elapsed:.2f}s ,expected to finish at {datetime.fromtimestamp(finish_time)}")

        """"""""" start of tick """""""""
        runtime_parameters.phase = SimulationPhase.START_OF_TICK
        for service_name, service_inst in runtime_parameters.service_container.items():
            service_inst.trigger(runtime_parameters)

        # reset all status flags
        for node_name, node_target in runtime_parameters.node_container.items():
            node_target.reset_statu_flags()

        """"""""" before training """""""""
        runtime_parameters.phase = SimulationPhase.BEFORE_TRAINING
        for service_name, service_inst in runtime_parameters.service_container.items():
            service_inst.trigger(runtime_parameters)

        """"""""" training """""""""
        runtime_parameters.phase = SimulationPhase.TRAINING
        for service_name, service_inst in runtime_parameters.service_container.items():
            service_inst.trigger(runtime_parameters)

        simulator_base_logger.info(f"current tick: {runtime_parameters.current_tick}/{runtime_parameters.max_tick}")
        node_target: node.Node
        node_name: str

        training_node_names = []
        for node_name, node_target in runtime_parameters.node_container.items():
            if node_target.next_training_tick == runtime_parameters.current_tick:
                node_target.is_training_this_tick = True
                training_node_names.append(node_name)
                for data, label in node_target.train_loader:
                    if config_file.force_use_cpu:
                        loss = submit_training_job_cpu(node_target, ml_config.criterion, data, label)
                        node_target.most_recent_loss = loss
                        simulator_base_logger.info(f"tick: {runtime_parameters.current_tick}, training node: {node_target.name}, loss={node_target.most_recent_loss:.2f}")
                    else:
                        loss = current_cuda_env.submit_training_job(node_target, ml_config.criterion, data, label)
                        node_target.most_recent_loss = loss
                        simulator_base_logger.info(f"tick: {runtime_parameters.current_tick}, training node: {node_target.name}, loss={node_target.most_recent_loss:.2f}")
                    break

        """update next training tick"""
        for index, node_name in enumerate(training_node_names):
            node_target = runtime_parameters.node_container[node_name]
            node_target.next_training_tick = config_file.get_next_training_time(node_target, runtime_parameters)

        """"""""" after training """""""""
        runtime_parameters.phase = SimulationPhase.AFTER_TRAINING
        for service_name, service_inst in runtime_parameters.service_container.items():
            service_inst.trigger(runtime_parameters)

        """"""""" before averaging """""""""
        runtime_parameters.phase = SimulationPhase.BEFORE_AVERAGING
        for service_name, service_inst in runtime_parameters.service_container.items():
            service_inst.trigger(runtime_parameters)

        """"""""" averaging """""""""
        runtime_parameters.phase = SimulationPhase.AVERAGING
        for service_name, service_inst in runtime_parameters.service_container.items():
            service_inst.trigger(runtime_parameters)

        nodes_averaged = set()
        for node_name, node_target in runtime_parameters.node_container.items():
            if node_target.is_training_this_tick:
                send_model = node_target.is_sending_model()
                if not send_model:
                    continue

                # get model stat
                model_stat = node_target.get_model_stat()
                for k, v in model_stat.items():
                    model_stat[k] = v.cpu()
                # send model to peers
                neighbors = list(runtime_parameters.topology.neighbors(node_target.name))
                for neighbor in neighbors:
                    neighbor_node: node.Node = runtime_parameters.node_container[neighbor]
                    neighbor_node.model_averager.add_model(model_stat)
                    if neighbor_node.model_buffer_size <= neighbor_node.model_averager.get_model_count():
                        # performing average!
                        averaged_model = neighbor_node.model_averager.get_model(self_model=neighbor_node.get_model_stat())
                        neighbor_node.set_model_stat(averaged_model)
                        neighbor_node.is_averaging_this_tick = True
                        nodes_averaged.add(neighbor_node.name)
        if len(nodes_averaged) > 0:
            simulator_base_logger.info(f"tick: {runtime_parameters.current_tick}, averaging on {len(nodes_averaged)} nodes: {nodes_averaged}")

        """"""""" after averaging """""""""
        runtime_parameters.phase = SimulationPhase.AFTER_AVERAGING
        for service_name, service_inst in runtime_parameters.service_container.items():
            service_inst.trigger(runtime_parameters)

        """"""""" end of tick """""""""
        runtime_parameters.phase = SimulationPhase.END_OF_TICK
        for service_name, service_inst in runtime_parameters.service_container.items():
            service_inst.trigger(runtime_parameters)

        runtime_parameters.current_tick += 1


def main():
    current_cuda_env = cuda.CudaEnv()

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
    config_ml_setup = config_file.get_ml_setup()
    config_ml_setup.self_validate()

    # set up runtime_parameters
    runtime_parameters = RuntimeParameters()
    runtime_parameters.max_tick = config_file.max_tick
    runtime_parameters.current_tick = 0
    runtime_parameters.dataset_label = config_ml_setup.dataset_label
    runtime_parameters.phase = SimulationPhase.INITIALIZING

    # check and create topology
    nodes_set = initial_checking.check_consistent_nodes(config_file.get_topology, config_file.max_tick)
    runtime_parameters.topology = config_file.get_topology(runtime_parameters)

    # init cuda
    if not config_file.force_use_cpu:
        current_cuda_env.measure_memory_consumption_for_performing_ml(config_ml_setup)
        current_cuda_env.generate_execution_strategy(config_ml_setup.model, config_file, config_ml_setup, len(nodes_set), override_use_model_stat=config_file.override_use_model_stat)
        current_cuda_env.print_ml_info()
        current_cuda_env.print_gpu_info()

    # create dataset
    training_dataset = dataset.DatasetWithFastLabelSelection(config_ml_setup.training_data)

    # create nodes
    runtime_parameters.node_container = {}
    for single_node in nodes_set:
        if config_file.force_use_cpu:
            temp_node = node.Node(single_node, config_ml_setup, use_cpu=True)
            # create optimizer for this node if using cpu
            optimizer = config_file.get_optimizer(temp_node, temp_node.model, runtime_parameters, config_ml_setup)
            temp_node.set_optimizer(optimizer)
        else:
            # find allocated gpu
            allocated_gpu = None
            for gpu in current_cuda_env.cuda_device_list:
                if single_node in gpu.nodes_allocated:
                    allocated_gpu = gpu
                    break
            assert allocated_gpu is not None
            if current_cuda_env.use_model_stat:
                temp_node = node.Node(single_node, config_ml_setup, use_model_stat=True, allocated_gpu=allocated_gpu, optimizer=allocated_gpu.optimizer)
            else:
                temp_node = node.Node(single_node, config_ml_setup, use_model_stat=False, allocated_gpu=allocated_gpu)
                # create optimizer for this node if using model stat
                optimizer = config_file.get_optimizer(temp_node, temp_node.model, runtime_parameters, config_ml_setup)
                temp_node.set_optimizer(optimizer)
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
        assert label_distribution is not None
        temp_node.set_label_distribution(label_distribution, training_dataset)
        # add node to container
        runtime_parameters.node_container[single_node] = temp_node

    # init nodes
    config_file.node_behavior_control(runtime_parameters)

    # init service
    service_list = config_file.get_service_list()
    for service_inst in service_list:
        service_inst.initialize(runtime_parameters, output_folder_path, config_file=config_file, ml_setup=config_ml_setup, cuda_env=current_cuda_env)
        runtime_parameters.service_container[service_inst.get_service_name()] = service_inst

    # begin simulation
    begin_simulation(runtime_parameters, config_file, config_ml_setup, current_cuda_env)

    exit(0)


if __name__ == "__main__":
    # global initialization
    torch.multiprocessing.set_start_method('spawn')
    main()
