import argparse
import logging
import os
import sys
import shutil
import networkx as nx
import torch
from datetime import datetime

from mpi4py import MPI

from py_src import internal_names, configuration_file, dfl_logging, nx_lib, initial_checking, cuda, mpi_util
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase

simulator_base_logger = logging.getLogger(internal_names.logger_simulator_base_name)

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

def main(config_file_path):
    # create output dir
    if MPI_rank == 0:
        output_folder_path = os.path.join(os.curdir, datetime.now().strftime("MPI_%Y-%m-%d_%H-%M-%S_%f"))
        os.mkdir(output_folder_path)
        backup_path = os.path.join(output_folder_path, internal_names.default_backup_folder_name)
        os.mkdir(backup_path)
    else:
        output_folder_path = None
        backup_path = None
    output_folder_path = MPI_comm.bcast(output_folder_path, root=0)
    backup_path = MPI_comm.bcast(backup_path, root=0)

    output_folder_path = os.path.join(output_folder_path, f"rank_{MPI_rank}")
    os.mkdir(output_folder_path)


    # init logging
    dfl_logging.set_logging(os.path.join(output_folder_path, internal_names.log_file_name), simulator_base_logger)

    # read config
    config_file = configuration_file.load_configuration(config_file_path)
    simulator_base_logger.info(f"config file path: ({config_file_path}), name: ({config_file.config_name}).")
    if MPI_rank == 0:
        shutil.copy2(config_file_path, backup_path)  # backup config file
        config_ml_setup = config_file.get_ml_setup()
        config_ml_setup.self_validate()
    else:
        config_ml_setup = None
    config_ml_setup = MPI_comm.bcast(config_ml_setup, root=0)

    # split nodes
    if MPI_rank == 0:
        topology, communities, inter_community_edges = distribut_computing_workload(args.config, MPI_size)
        simulator_base_logger.info(f"split topology({topology.number_of_nodes()} nodes) to {len(communities)} communities: {[len(community) for community in communities]}, inter community edges counts: {len(inter_community_edges)}")
    else:
        communities = None
    communities = MPI_comm.bcast(communities, root=0)
    nodes_map = {}
    for temp_rank in range(MPI_size):
        nodes_map[temp_rank] = communities[temp_rank]
    self_nodes = nodes_map[MPI_rank]

    # set up runtime_parameters
    runtime_parameters = RuntimeParameters()
    runtime_parameters.max_tick = config_file.max_tick
    runtime_parameters.current_tick = 0
    runtime_parameters.dataset_label = config_ml_setup.dataset_label
    runtime_parameters.phase = SimulationPhase.INITIALIZING

    # check and create topology
    if MPI_rank == 0:
        nodes_set = initial_checking.check_consistent_nodes(config_file.get_topology, config_file.max_tick)
        topology = config_file.get_topology(runtime_parameters)
    else:
        nodes_set = None
        topology = None
    nodes_set = MPI_comm.bcast(nodes_set, root=0)
    topology = MPI_comm.bcast(topology, root=0)
    runtime_parameters.topology = topology

    # cuda
    current_cuda_env = cuda.CudaEnv()
    if MPI_rank == 0:
        current_cuda_env.measure_memory_consumption_for_performing_ml(config_ml_setup, measure_in_new_process=False)
        current_cuda_env.print_ml_info()
    MPI_comm.barrier()
    cuda_info = mpi_util.collect_sys_cuda_info()
    all_mpi_info = MPI_comm.gather(cuda_info, root=0)
    mpi_world = mpi_util.MpiWorld()
    if MPI_rank == 0:
        for mpi_process_rank, cuda_info in enumerate(all_mpi_info):
            hostname, gpus = cuda_info
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

        # print all MPI host info
        mpi_world.print_info()
    if MPI_rank == 0:
        mpi_world.determine_mem_strategy(current_cuda_env.memory_consumption_model_MB, current_cuda_env.memory_consumption_dataset_MB, override_use_model_stat=config_file.override_use_model_stat)
        strategy = mpi_world.gpu_mem_strategy
    else:
        strategy = None
    strategy = MPI_comm.bcast(strategy, root=0)




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
    args = parser.parse_args()

    main(args.config)
