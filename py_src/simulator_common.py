from py_src import node, cpu
from py_src.simulation_runtime_parameters import SimulationPhase, RuntimeParameters

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