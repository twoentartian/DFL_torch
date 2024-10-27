import os
import time
import pickle
from datetime import datetime
from typing import Final
from mpi4py import MPI
from mpi4py.util import pkl5

from py_src import node, cpu, mpi_data_payload, mpi_util, ml_setup
from py_src.simulation_runtime_parameters import SimulationPhase, RuntimeParameters
from simulator_mpi import MPI_rank

REPORT_FINISH_TIME_PER_TICK: Final[int] = 100

def send_model_stat_to_receiver(runtime_parameters, dst_node, model_stat) -> bool:
    """return whether the node is averaged"""
    neighbor_node: node.Node = runtime_parameters.node_container[dst_node]
    neighbor_node.model_averager.add_model(model_stat)
    if neighbor_node.model_buffer_size <= neighbor_node.model_averager.get_model_count():
        # performing average!
        averaged_model = neighbor_node.model_averager.get_model(self_model=neighbor_node.get_model_stat())
        neighbor_node.set_model_stat(averaged_model)
        neighbor_node.is_averaging_this_tick = True
        return True
    return False



def simulation_phase_start_of_tick(runtime_parameters: RuntimeParameters, logger):
    runtime_parameters.phase = SimulationPhase.START_OF_TICK
    for service_name, service_inst in runtime_parameters.service_container.items():
        service_inst.trigger(runtime_parameters)

    # reset all status flags
    for node_name, node_target in runtime_parameters.node_container.items():
        node_target.reset_statu_flags()

def simulation_phase_before_training(runtime_parameters: RuntimeParameters, logger):
    runtime_parameters.phase = SimulationPhase.BEFORE_TRAINING
    for service_name, service_inst in runtime_parameters.service_container.items():
        service_inst.trigger(runtime_parameters)


def simulation_phase_training(runtime_parameters: RuntimeParameters, logger, config_file, ml_config, current_cuda_env):
    runtime_parameters.phase = SimulationPhase.TRAINING
    for service_name, service_inst in runtime_parameters.service_container.items():
        service_inst.trigger(runtime_parameters)

    logger.info(f"current tick: {runtime_parameters.current_tick}/{runtime_parameters.max_tick}")
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
                    logger.info(f"tick: {runtime_parameters.current_tick}, training node: {node_target.name}, loss={node_target.most_recent_loss:.2f}")
                else:
                    loss = current_cuda_env.submit_training_job(node_target, ml_config.criterion, data, label)
                    node_target.most_recent_loss = loss
                    logger.info(f"tick: {runtime_parameters.current_tick}, training node: {node_target.name}, loss={node_target.most_recent_loss:.2f}")
                break

    """update next training tick"""
    for index, node_name in enumerate(training_node_names):
        node_target = runtime_parameters.node_container[node_name]
        node_target.next_training_tick = config_file.get_next_training_time(node_target, runtime_parameters)


def simulation_phase_after_training(runtime_parameters: RuntimeParameters, logger):
    runtime_parameters.phase = SimulationPhase.AFTER_TRAINING
    for service_name, service_inst in runtime_parameters.service_container.items():
        service_inst.trigger(runtime_parameters)


def simulation_phase_before_averaging(runtime_parameters: RuntimeParameters, logger):
    runtime_parameters.phase = SimulationPhase.BEFORE_AVERAGING
    for service_name, service_inst in runtime_parameters.service_container.items():
        service_inst.trigger(runtime_parameters)

def simulation_phase_averaging(runtime_parameters: RuntimeParameters, logger, mpi_world=None):
    runtime_parameters.phase = SimulationPhase.AVERAGING
    for service_name, service_inst in runtime_parameters.service_container.items():
        service_inst.trigger(runtime_parameters)

    use_mpi = mpi_world is not None

    if use_mpi:
        MPI_comm = MPI.COMM_WORLD
        MPI_rank = MPI_comm.Get_rank()
        MPI_size = MPI_comm.Get_size()
        large_comm = pkl5.Intracomm(MPI_comm)
        mpi_data_pack_and_dst = {}
        for dst_mpi_rank in range(MPI_size):
            if dst_mpi_rank != MPI_rank:
                mpi_data_pack_and_dst[dst_mpi_rank] = mpi_data_payload.MpiDataPack(MPI_rank)
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

    nodes_averaged = set()
    for node_name, node_target in runtime_parameters.node_container.items():
        if node_target.is_training_this_tick:
            send_model = node_target.is_sending_model()
            if not send_model:
                continue

            # get model stat
            model_stat = node_target.get_model_stat()
            model_stat_serialized = None
            for k, v in model_stat.items():
                model_stat[k] = v.cpu()
            # send model to peers
            neighbors = list(runtime_parameters.topology.neighbors(node_target.name))

            for neighbor in neighbors:
                if use_mpi:
                    if neighbor in self_nodes:
                        # the target node is in my MPI process
                        averaged = send_model_stat_to_receiver(runtime_parameters, neighbor, model_stat)
                        if averaged:
                            nodes_averaged.add(neighbor)
                    else:
                        # the target node is in other MPI processes
                        dst_mpi_rank = nodes_map_to_rank[neighbor]
                        if model_stat_serialized is None:
                            model_stat_serialized = mpi_data_payload.serialize_model_stat(model_stat)
                        mpi_data_pack_and_dst[dst_mpi_rank].add_mpi_data(node_name, model_stat_serialized, neighbor)
                else:
                    averaged = send_model_stat_to_receiver(runtime_parameters, neighbor, model_stat)
                    if averaged:
                        nodes_averaged.add(neighbor)

    # share in MPI world
    if use_mpi:
        MPI_comm.barrier()

        # share models in MPI world
        assert len(mpi_data_pack_and_dst.keys()) == (MPI_size - 1)
        send_reqs = []
        for dst_mpi_rank, mpi_data_pack in mpi_data_pack_and_dst.items():
            all_sent_models = mpi_data_pack.get_mpi_data()
            req = large_comm.isend(mpi_data_pack, dst_mpi_rank, tag=mpi_data_payload.MpiMessageTag.ModelStateData.value)
            send_reqs.append(req)

        received_data = {}
        for sender_rank in range(MPI_size):
            if sender_rank == MPI_rank:
                continue
            robj = large_comm.recv(None, sender_rank, tag=mpi_data_payload.MpiMessageTag.ModelStateData.value)
            received_data[sender_rank] = robj

        # add models from MPI to average buffer
        for sender_mpi_rank, mpi_data_pack in received_data.items():
            model_stat_list = mpi_data_pack.get_mpi_data()
            for (src_node, model_stat, dst_node) in model_stat_list:
                averaged = send_model_stat_to_receiver(runtime_parameters, dst_node, model_stat)
                if averaged:
                    nodes_averaged.add(dst_node)

        MPI_comm.barrier()

    # logger
    if len(nodes_averaged) > 0:
        logger.info(f"tick: {runtime_parameters.current_tick}, averaging on {len(nodes_averaged)} nodes: {nodes_averaged}")

def simulation_phase_after_averaging(runtime_parameters: RuntimeParameters, logger):
    runtime_parameters.phase = SimulationPhase.AFTER_AVERAGING
    for service_name, service_inst in runtime_parameters.service_container.items():
        service_inst.trigger(runtime_parameters)

def simulation_phase_end_of_tick(runtime_parameters: RuntimeParameters, logger):
    runtime_parameters.phase = SimulationPhase.END_OF_TICK
    for service_name, service_inst in runtime_parameters.service_container.items():
        service_inst.trigger(runtime_parameters)

def begin_simulation(runtime_parameters: RuntimeParameters, config_file, ml_config: ml_setup.MlSetup, current_cuda_env, logger, mpi_world: mpi_util.MpiWorld=None):
    # begin simulation
    timer = time.time()

    while runtime_parameters.current_tick <= config_file.max_tick:
        # update topology?
        if mpi_world is not None:
            MPI_comm = MPI.COMM_WORLD
            new_topology = config_file.get_topology(runtime_parameters)
            new_topology = MPI_comm.bcast(new_topology, root=0)
            if new_topology is not None:
                if MPI_rank == 0:
                    topology_folder = os.path.join(runtime_parameters.output_path, "topology")
                    os.makedirs(topology_folder, exist_ok=True)
                    topology_file = open(os.path.join(topology_folder, f"{runtime_parameters.current_tick}.pickle"), "wb")
                    pickle.dump(new_topology, topology_file)
                    topology_file.close()
                runtime_parameters.topology = new_topology
                logger.info(f"topology is updated at tick {runtime_parameters.current_tick}")

        # update label distribution?
        for _, single_node in runtime_parameters.node_container.items():
            new_label_distribution = config_file.get_label_distribution(single_node, runtime_parameters)
            if new_label_distribution is not None:
                single_node.set_label_distribution(new_label_distribution)
                logger.info(f"update label distribution to {new_label_distribution} for {single_node.name}.")

        # report remaining time
        if runtime_parameters.current_tick % REPORT_FINISH_TIME_PER_TICK == 0 and runtime_parameters.current_tick != 0:
            time_elapsed = time.time() - timer
            timer = time.time()
            remaining = (config_file.max_tick - runtime_parameters.current_tick) // REPORT_FINISH_TIME_PER_TICK
            time_to_finish = remaining * time_elapsed
            finish_time = timer + time_to_finish
            logger.info(f"time taken for {REPORT_FINISH_TIME_PER_TICK} ticks: {time_elapsed:.2f}s ,expected to finish at {datetime.fromtimestamp(finish_time)}")

        """"""""" start of tick """""""""
        simulation_phase_start_of_tick(runtime_parameters, logger)

        """"""""" before training """""""""
        simulation_phase_before_training(runtime_parameters, logger)

        """"""""" training """""""""
        simulation_phase_training(runtime_parameters, logger, config_file, ml_config, current_cuda_env)

        """"""""" after training """""""""
        simulation_phase_after_training(runtime_parameters, logger)

        """"""""" before averaging """""""""
        simulation_phase_before_averaging(runtime_parameters, logger)

        """"""""" averaging """""""""
        simulation_phase_averaging(runtime_parameters, logger, mpi_world)

        """"""""" after averaging """""""""
        simulation_phase_after_averaging(runtime_parameters, logger)

        """"""""" end of tick """""""""
        simulation_phase_end_of_tick(runtime_parameters, logger)

        runtime_parameters.current_tick += 1


