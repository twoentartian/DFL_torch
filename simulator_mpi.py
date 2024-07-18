import argparse
import logging
import os
import shutil
import time
import torch

from typing import Final
from datetime import datetime
from mpi4py import MPI
from mpi4py.util import pkl5

from py_src import internal_names, configuration_file, dfl_logging, nx_lib, initial_checking, cuda, mpi_util, dataset, node, cpu, mpi_data_payload, simulator_common
from py_src.simulation_runtime_parameters import RuntimeParameters, SimulationPhase
from py_src.ml_setup import MlSetup

simulator_base_logger = logging.getLogger(internal_names.logger_simulator_base_name)

REPORT_FINISH_TIME_PER_TICK: Final[int] = 100

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()




def begin_simulation(runtime_parameters: RuntimeParameters, config_file, ml_config: MlSetup, current_cuda_env, mpi_world: mpi_util.MpiWorld):
    large_comm = pkl5.Intracomm(MPI.COMM_WORLD)

    # self information
    self_hostname = mpi_util.collect_netbios_name()
    self_host = mpi_world.all_hosts[self_hostname]
    self_mpi_process = self_host.mpi_process[MPI_rank]
    self_gpu = self_mpi_process.allocated_gpu
    self_nodes = self_mpi_process.nodes
    nodes_map_to_rank = {}
    for host in mpi_world.all_hosts.values():
        for mpi_process in host.mpi_process.values():
            for single_node in mpi_process.nodes:
                nodes_map_to_rank[single_node] = mpi_process.rank



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
                        loss = cpu.submit_training_job_cpu(node_target, ml_config.criterion, data, label)
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
        mpi_data_pack_and_dst = {}
        for dst_mpi_rank in range(MPI_size):
            if dst_mpi_rank != MPI_rank:
                mpi_data_pack_and_dst[dst_mpi_rank] = mpi_data_payload.MpiDataPack(MPI_rank)
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
                    if neighbor in self_nodes:
                        # the target node is in my MPI process
                        averaged = simulator_common.send_model_stat_to_receiver(runtime_parameters, neighbor, model_stat)
                        if averaged:
                            nodes_averaged.add(neighbor)
                    else:
                        # the target node is other MPI processes
                        dst_mpi_rank = nodes_map_to_rank[neighbor]
                        mpi_data_pack_and_dst[dst_mpi_rank].add_mpi_data(node_name, model_stat, neighbor)

        MPI_comm.barrier()

        # share models in MPI world
        assert len(mpi_data_pack_and_dst.keys()) == (MPI_size - 1)
        send_reqs = []
        for dst_mpi_rank, mpi_data_pack in mpi_data_pack_and_dst.items():
            all_sent_models = mpi_data_pack.get_mpi_data()
            req = large_comm.isend(mpi_data_pack, dst_mpi_rank, tag=mpi_data_payload.MpiMessageTag.ModelStateData.value)
            send_reqs.append(req)

        received_data = {}
        while len(received_data.keys()) < (MPI_size - 1):
            status = MPI.Status()
            rmsg = large_comm.mprobe(status=status)
            tag = status.Get_tag()
            sender = status.Get_source()
            assert tag == mpi_data_payload.MpiMessageTag.ModelStateData.value
            rreq = rmsg.irecv()
            robj = rreq.wait()
            received_data[sender] = robj

        # add models from MPI to average buffer
        for sender_mpi_rank, mpi_data_pack in received_data.items():
            model_stat_list = mpi_data_pack.get_mpi_data()
            for (src_node, model_stat, dst_node) in model_stat_list:
                averaged = simulator_common.send_model_stat_to_receiver(runtime_parameters, dst_node, model_stat)
                if averaged:
                    nodes_averaged.add(dst_node)

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

    # set up runtime_parameters
    runtime_parameters = RuntimeParameters()
    runtime_parameters.max_tick = config_file.max_tick
    runtime_parameters.current_tick = 0
    runtime_parameters.dataset_label = config_ml_setup.dataset_label
    runtime_parameters.phase = SimulationPhase.INITIALIZING

    # check, create topology and split nodes
    if MPI_rank == 0:
        nodes_set = initial_checking.check_consistent_nodes(config_file.get_topology, config_file.max_tick)
        topology, communities, inter_community_edges = distribut_computing_workload(args.config, MPI_size)
        simulator_base_logger.info(f"split topology({topology.number_of_nodes()} nodes) to {len(communities)} communities: {[len(community) for community in communities]}, inter community edges counts: {len(inter_community_edges)}")
    else:
        nodes_set = None
        communities = None
        topology = None
    nodes_set = MPI_comm.bcast(nodes_set, root=0)
    communities = MPI_comm.bcast(communities, root=0)
    topology = MPI_comm.bcast(topology, root=0)
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
        mpi_world.determine_mem_strategy(current_cuda_env.memory_consumption_model_MB, current_cuda_env.memory_consumption_dataset_MB, override_use_model_stat=config_file.override_use_model_stat)
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
    runtime_parameters.node_container = {}
    for single_node in self_nodes:
        if config_file.force_use_cpu:
            temp_node = node.Node(single_node, config_ml_setup, use_cpu=True)
            # create optimizer for this node if using cpu
            optimizer = config_file.get_optimizer(temp_node, temp_node.model, runtime_parameters, config_ml_setup)
            temp_node.set_optimizer(optimizer)
        else:
            gpu = current_cuda_env.cuda_device_list[self_gpu.gpu_index]
            if current_cuda_env.use_model_stat:
                temp_node = node.Node(single_node, config_ml_setup, use_model_stat=True, allocated_gpu=gpu, optimizer=gpu.optimizer)
            else:
                temp_node = node.Node(single_node, config_ml_setup, use_model_stat=False, allocated_gpu=gpu)
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
    runtime_parameters.mpi_enabled = True
    begin_simulation(runtime_parameters, config_file, config_ml_setup, current_cuda_env, mpi_world)

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
    args = parser.parse_args()

    main(args.config)
