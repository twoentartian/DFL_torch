import argparse
import os
import logging
import shutil
import torch

from datetime import datetime
from py_src import configuration_file, internal_names, initial_checking, cuda, node, dataset, dfl_logging, simulator_common
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.service.print_memory_consumption import PrintMemoryConsumption

simulator_base_logger = logging.getLogger(internal_names.logger_simulator_base_name)

ENABLE_MEMORY_RECORD = True

def main(config_file_path, output_folder_name):
    current_cuda_env = cuda.CudaEnv()

    # create output dir
    if output_folder_name is None:
        output_folder_path = os.path.join(os.curdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
    else:
        output_folder_path = os.path.join(os.curdir, output_folder_name)
    os.mkdir(output_folder_path)
    backup_path = os.path.join(output_folder_path, internal_names.default_backup_folder_name)
    os.mkdir(backup_path)

    if ENABLE_MEMORY_RECORD:
        memory_service = PrintMemoryConsumption(100, save_file_name="base_memory_profiler.txt")
        memory_service.initialize_without_runtime_parameters(output_folder_path)

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

    # init cuda
    if not config_file.force_use_cpu:
        current_cuda_env.measure_memory_consumption_for_performing_ml(config_ml_setup)
        current_cuda_env.measure_memory_consumption_for_performing_ml(config_ml_setup)
        current_cuda_env.generate_execution_strategy(nodes_set,
                                                     override_use_model_stat=config_file.override_use_model_stat,
                                                     override_allocate_all_models=config_file.override_allocate_all_models)
        current_cuda_env.prepare_gpu_memory(config_ml_setup.model, config_file, config_ml_setup, nodes_set)
        current_cuda_env.print_ml_info()
        current_cuda_env.print_gpu_info()

    # create dataset
    training_dataset = dataset.DatasetWithFastLabelSelection(config_ml_setup.training_data)

    # create nodes
    if ENABLE_MEMORY_RECORD:
        memory_service.trigger_without_runtime_parameters(0, "BEFORE_CREATE_NODES")

    runtime_parameters.node_container = {}
    for single_node in nodes_set:
        if config_file.force_use_cpu:
            temp_node = node.Node(single_node, config_ml_setup, use_cpu=True)
            # create optimizer for this node if using cpu
            optimizer, lr_scheduler = config_file.get_optimizer(temp_node, temp_node.model, runtime_parameters, config_ml_setup)
            temp_node.set_optimizer(optimizer)
            if lr_scheduler is not None:
                temp_node.set_lr_scheduler(lr_scheduler)
        else:
            # find allocated gpu
            allocated_gpu = None
            for gpu in current_cuda_env.cuda_device_list:
                if single_node in gpu.nodes_allocated:
                    allocated_gpu = gpu
                    break
            assert allocated_gpu is not None, f"node cannot find a suitable GPU, is there enough GPUs?"
            if current_cuda_env.use_model_stat:
                temp_node = node.Node(single_node, config_ml_setup, use_model_stat=True, allocated_gpu=allocated_gpu, optimizer=allocated_gpu.optimizer)
            else:
                temp_node = node.Node(single_node, config_ml_setup, use_model_stat=False, allocated_gpu=allocated_gpu)
                # create optimizer for this node if using model stat
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
        assert label_distribution is not None
        temp_node.set_label_distribution(label_distribution, dataset_with_fast_label=training_dataset)
        # add node to container
        runtime_parameters.node_container[single_node] = temp_node

        if ENABLE_MEMORY_RECORD:
            memory_service.trigger_without_runtime_parameters(0, f"AFTER_CREATE_NODE_{temp_node.name}")

    if ENABLE_MEMORY_RECORD:
        memory_service.trigger_without_runtime_parameters(0, "AFTER_CREATE_NODES")

    # init nodes
    config_file.node_behavior_control(runtime_parameters)

    # init service
    service_list = config_file.get_service_list()
    for service_inst in service_list:
        service_inst.initialize(runtime_parameters, output_folder_path, config_file=config_file, ml_setup=config_ml_setup, cuda_env=current_cuda_env)
        runtime_parameters.service_container[service_inst.get_service_name()] = service_inst

    # begin simulation
    runtime_parameters.mpi_enabled = False
    if ENABLE_MEMORY_RECORD:
        memory_service.trigger_without_runtime_parameters(0, "BEFORE_SIMULATION")
    simulator_common.begin_simulation(runtime_parameters, config_file, config_ml_setup, current_cuda_env, simulator_base_logger)

    exit(0)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='DFL simulator (torch version)')
    parser.add_argument('--config', type=str, default="./simulator_config.py", help='path to config file, default: "./simulator_config.py')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    args = parser.parse_args()

    main(args.config, args.output_folder_name)
