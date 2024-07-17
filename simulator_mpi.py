import argparse
import logging
import os
import sys
import shutil
import networkx as nx
import torch
from datetime import datetime

from mpi4py import MPI

from py_src import internal_names, configuration_file, dfl_logging, nx_lib, initial_checking, cuda
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
    self_nodes = None
    for temp_rank in range(MPI_size):
        if temp_rank == MPI_rank:
            self_nodes = communities[temp_rank]
        else:
            nodes_map[temp_rank] = communities[temp_rank]
    simulator_base_logger.info(f"MPI RANK {MPI_rank} has {len(self_nodes)} nodes: {self_nodes}")

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
        current_cuda_env.measure_memory_consumption_for_performing_ml(config_ml_setup)
        current_cuda_env.generate_execution_strategy(config_ml_setup.model, config_file, config_ml_setup, len(nodes_set), override_use_model_stat=config_file.override_use_model_stat)



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
