import networkx as nx
from py_src import simulation_runtime_parameters

config_name = "default_config"

max_tick = 10000        # total simulation ticks


def get_topology(parameters: simulation_runtime_parameters.RuntimeParameters) -> nx.Graph:
    get_topology.topology = None
    if get_topology.topology is None:
        get_topology.topology = nx.complete_graph(10)
    return get_topology.topology



