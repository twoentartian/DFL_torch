from py_src.node import Node
from py_src.simulation_runtime_parameters import RuntimeParameters

def global_broadcast(runtime_parameters: RuntimeParameters, src_node_name: int):
    assert src_node_name in runtime_parameters.node_container.keys(), f"node {src_node_name} does not exist in the network: {runtime_parameters.node_container.keys()}"
    stc_model_stat = runtime_parameters.node_container[src_node_name].get_model_stat()
    for node_name, node_target in runtime_parameters.node_container.items():
        node_target: Node
        if src_node_name == node_name:
            continue
        node_target.set_model_stat(stc_model_stat)
