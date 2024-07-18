from py_src import node

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

