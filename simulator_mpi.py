import argparse
import logging
import os
import shutil
import torch

from datetime import datetime
from mpi4py import MPI

from py_src import internal_names, configuration_file, dfl_logging, nx_lib, initial_checking, cuda, mpi_util, dataset, node, simulator_common
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.service.print_memory_consumption import PrintMemoryConsumption

simulator_base_logger = logging.getLogger(internal_names.logger_simulator_base_name)


MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

ENABLE_MEMORY_RECORD = False

def main(config_file_path, output_folder_name=None):
    # read config
    config_file = configuration_file.load_configuration(config_file_path)
    output_folder_path = None
    if hasattr(config_file, 'save_name'):
        output_folder_path = config_file.save_name

    # create output dir
    if MPI_rank == 0:
        if output_folder_path is None:
            if output_folder_name is None:
                output_folder_path = os.path.join(os.curdir, datetime.now().strftime("MPI_%Y-%m-%d_%H-%M-%S_%f"))
            else:
                output_folder_path = os.path.join(os.curdir, output_folder_name)
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
        else:
            print(f"{output_folder_path} exists.")
            MPI.COMM_WORLD.Abort()
        backup_path = os.path.join(output_folder_path, internal_names.default_backup_folder_name)
        if not os.path.exists(backup_path):
            os.mkdir(backup_path)
        else:
            print(f"{backup_path} exists.")
            MPI.COMM_WORLD.Abort()
    else:
        output_folder_path = None
        backup_path = None
    output_folder_path = MPI_comm.bcast(output_folder_path, root=0)
    backup_path = MPI_comm.bcast(backup_path, root=0)

    output_folder_path = os.path.join(output_folder_path, f"rank_{MPI_rank}")
    os.mkdir(output_folder_path)

    if ENABLE_MEMORY_RECORD:
        memory_service = PrintMemoryConsumption(100, save_file_name="base_memory_profiler.txt")
        memory_service.initialize_without_runtime_parameters(output_folder_path)

    # init logging
    dfl_logging.set_logging(os.path.join(output_folder_path, internal_names.log_file_name), simulator_base_logger)

    # process config
    simulator_base_logger.info(f"config file path: ({config_file_path}), name: ({config_file.config_name}).")
    if MPI_rank == 0:
        shutil.copy2(config_file_path, backup_path)  # backup config file
        config_ml_setup = config_file.get_ml_setup()
        config_ml_setup.self_validate()
    else:
        config_ml_setup = None
    config_ml_setup = MPI_comm.bcast(config_ml_setup, root=0)

    # set up runtime_parameters
    runtime_parameters = RuntimeParameters()
    runtime_parameters.max_tick = config_file.max_tick
    runtime_parameters.current_tick = 0
    runtime_parameters.dataset_label = config_ml_setup.dataset_label
    runtime_parameters.phase = SimulationPhase.INITIALIZING
    runtime_parameters.output_path = output_folder_path
    runtime_parameters.mpi_enabled = True

    # check, create topology and split nodes
    if MPI_rank == 0:
        nodes_set = initial_checking.check_consistent_nodes(config_file.get_topology, config_file.max_tick)
        topology, communities, inter_community_edges = distribut_computing_workload(args.config, MPI_size)
        simulator_common.save_topology_to_file(topology, runtime_parameters.current_tick, runtime_parameters.output_path, mpi_enabled=True)
        simulator_base_logger.info(f"split topology({topology.number_of_nodes()} nodes) to {len(communities)} communities: {[len(community) for community in communities]}, inter community edges counts: {len(inter_community_edges)}")
    else:
        nodes_set = None
        communities = None
        topology = None
    nodes_set = MPI_comm.bcast(nodes_set, root=0)
    communities = MPI_comm.bcast(communities, root=0)
    topology = MPI_comm.bcast(topology, root=0)
    simulator_base_logger.info(f"topology is updated at tick {runtime_parameters.current_tick}")
    runtime_parameters.topology = topology
    nodes_map = {}
    for temp_rank in range(MPI_size):
        nodes_map[temp_rank] = communities[temp_rank]
    self_nodes = nodes_map[MPI_rank]

    # cuda
    current_cuda_env = cuda.CudaEnv()
    if MPI_rank == 0:
        current_cuda_env.measure_memory_consumption_for_performing_ml(config_ml_setup, measure_in_new_process=False)
        current_cuda_env.print_ml_info()
    MPI_comm.barrier()
    sys_cuda_info = mpi_util.collect_netbios_cuda_info()

    all_mpi_info = MPI_comm.gather(sys_cuda_info, root=0)
    mpi_world = mpi_util.MpiWorld()
    if MPI_rank == 0:
        for mpi_process_rank, sys_cuda_info in enumerate(all_mpi_info):
            hostname, gpus = sys_cuda_info
            if hostname not in mpi_world.all_hosts:
                temp_host = mpi_util.MpiHost(hostname)
                for gpu_index, gpu in enumerate(gpus):
                    gpu_name, total_mem, used_mem, free_mem = gpu
                    temp_host.add_gpu(gpu_index, gpu_name, total_mem, free_mem)
                mpi_world.add_mpi_host(temp_host)
            else:
                # check gpu names match
                assert len(gpus) == len(mpi_world.all_hosts[hostname].gpus)
                for index, gpu in mpi_world.all_hosts[hostname].gpus.items():
                    gpu_name, total_mem, used_mem, free_mem = gpus[index]
                    assert gpu.name == gpu_name
            mpi_world.all_hosts[hostname].add_mpi_process_rank(mpi_process_rank, nodes_map[mpi_process_rank])
        mpi_world.allocate_nodes_to_gpu()
        # print all MPI host info
        mpi_world.print_info()

    if MPI_rank == 0:
        mpi_world.determine_mem_strategy(current_cuda_env.memory_consumption_model_MB, current_cuda_env.memory_consumption_dataset_MB,
                                         override_use_model_stat=config_file.override_use_model_stat,
                                         override_allocate_all_models=config_file.override_allocate_all_models)
        strategy = mpi_world.gpu_mem_strategy
    else:
        strategy = None
    mpi_world = MPI_comm.bcast(mpi_world, root=0)
    strategy = MPI_comm.bcast(strategy, root=0)

    # self information
    self_hostname, _ = sys_cuda_info
    self_host = mpi_world.all_hosts[self_hostname]
    self_mpi_process = self_host.mpi_process[MPI_rank]
    self_gpu = self_mpi_process.allocated_gpu
    self_nodes = self_mpi_process.nodes

    # allocate model memory
    if strategy == mpi_util.MpiGpuMemStrategy.AllocateAllModels:
        current_cuda_env.use_model_stat = False
    elif strategy == mpi_util.MpiGpuMemStrategy.ShareSingleModel:
        current_cuda_env.use_model_stat = True
    else:
        raise NotImplementedError
    simulator_base_logger.info(f"MPI Process {self_mpi_process.rank} allocates {len(self_nodes)} nodes on GPU {self_gpu.gpu_index} of HOST {self_hostname} ({self_gpu.name})")
    current_cuda_env.mpi_prepare_gpu_memory(config_ml_setup.model, config_file, config_ml_setup, self_nodes, self_mpi_process.allocated_gpu.gpu_index)

    # create dataset
    training_dataset = dataset.DatasetWithFastLabelSelection(config_ml_setup.training_data)

    # create nodes
    if ENABLE_MEMORY_RECORD:
        memory_service.trigger_without_runtime_parameters(0, "BEFORE_CREATE_NODES")

    runtime_parameters.node_container = {}
    current_allocated_gpu = None
    for single_node in self_nodes:
        if config_file.force_use_cpu:
            temp_node = node.Node(single_node, config_ml_setup, use_cpu=True)
            # create optimizer for this node if using cpu
            optimizer, lr_scheduler = config_file.get_optimizer(temp_node, temp_node.model, runtime_parameters, config_ml_setup)
            temp_node.set_optimizer(optimizer)
            if lr_scheduler is not None:
                temp_node.set_lr_scheduler(lr_scheduler)
        else:
            gpu = current_cuda_env.cuda_device_list[self_gpu.gpu_index]
            current_allocated_gpu = gpu
            if current_cuda_env.use_model_stat:
                temp_node = node.Node(single_node, config_ml_setup, use_model_stat=True, allocated_gpu=gpu, optimizer=gpu.optimizer)
            else:
                temp_node = node.Node(single_node, config_ml_setup, use_model_stat=False, allocated_gpu=gpu)
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
    config_file.node_behavior_control(runtime_parameters, mpi_world=mpi_world)

    # init service
    service_list = config_file.get_service_list()
    for service_inst in service_list:
        service_inst.initialize(runtime_parameters, output_folder_path, config_file=config_file, ml_setup=config_ml_setup,
                                cuda_env=current_cuda_env, gpu=current_allocated_gpu)
        runtime_parameters.service_container[service_inst.get_service_name()] = service_inst

    # begin simulation
    if ENABLE_MEMORY_RECORD:
        memory_service.trigger_without_runtime_parameters(0, "BEFORE_SIMULATION")
    simulator_common.begin_simulation(runtime_parameters, config_file, config_ml_setup, current_cuda_env, simulator_base_logger, mpi_world)

    exit(0)

def distribut_computing_workload(config_file_path, num_of_communities):
    config_file = configuration_file.load_configuration(config_file_path)
    runtime_parameters = RuntimeParameters()
    runtime_parameters.max_tick = config_file.max_tick
    runtime_parameters.current_tick = 0
    runtime_parameters.phase = SimulationPhase.INITIALIZING
    topology = config_file.get_topology(runtime_parameters)
    communities = nx_lib.split_to_equal_size_communities(topology, num_of_communities)
    inter_community_edges = nx_lib.get_inter_community_edges(topology, communities)
    return topology, communities, inter_community_edges


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='MPI-based DFL simulator (torch version)')
    parser.add_argument('--config', type=str, default="./simulator_config.py", help='path to config file, default: "./simulator_config.py')
    parser.add_argument("-o", "--output_folder_name", default=None, help='specify the output folder name')
    parser.add_argument("-T", "--thread", default=1, type=int, help='specify the number of thread for pytorch')
    args = parser.parse_args()

    torch.set_num_threads(args.thread)
    main(args.config, args.output_folder_name)
