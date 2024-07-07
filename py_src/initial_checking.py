import os
import logging
import networkx as nx

from py_src import simulation_runtime_parameters, util, internal_names

logger = logging.getLogger(f"{internal_names.logger_simulator_base_name}.{util.basename_without_extension(__file__)}")


def check_consistent_nodes(topology_generation_function, total_tick: int) -> set[int]:
    previous_nodes = None
    max_edge_count = 0
    initial_edge = None
    parameters = simulation_runtime_parameters.RuntimeParameters()
    parameters.max_tick = total_tick
    for tick in range(total_tick+1):
        parameters.current_tick = tick
        topology = topology_generation_function(parameters)
        current_nodes = set(topology.nodes())
        edges = topology.edges()
        if len(edges) > max_edge_count:
            max_edge_count = len(edges)
        if initial_edge is None:
            initial_edge = len(edges)
        if previous_nodes is None:
            previous_nodes = current_nodes
        else:
            if previous_nodes != current_nodes:
                extra_nodes = current_nodes - previous_nodes
                missing_nodes = previous_nodes - current_nodes
                logger.critical(f"nodes (count:{len(current_nodes)}) at tick {tick} is different from previous nodes (count:{len(previous_nodes)}). Extra: {extra_nodes}, missing: {missing_nodes}")
    logger.info(f"total nodes: {len(previous_nodes)}, initial edges: {initial_edge}, max edge count: {max_edge_count}")
    return previous_nodes