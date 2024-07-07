import networkx as nx
import numpy as np
import torch
from py_src import simulation_runtime_parameters, ml_setup, node
from py_src.simulation_runtime_parameters import SimulationPhase
from py_src.config_file_util import label_distribution

config_name = "default_config"

max_tick = 10000        # total simulation ticks

""""""""" Machine learning related parameters """""""""""
""" predefined: """
ml_setup = ml_setup.resnet18_cifar10

""" or self defined: """
# ml_parameters = ml_setup.MlSetup()
# ml_parameters.model = None
# ml_parameters.training_data = None
# ml_parameters.testing_data = None


""""""""""" Topology related parameters """""""""""
def get_topology(parameters: simulation_runtime_parameters.RuntimeParameters) -> nx.Graph:
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        get_topology.current_topology = nx.complete_graph(10)
        return get_topology.current_topology
    return None


""""""""""" Training time related parameters """""""""""
"""this function will be called after training to get the next training tick"""
def get_next_training_time(target_node: node.Node, parameters: simulation_runtime_parameters.RuntimeParameters) -> int:
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        return 0
    return target_node.next_training_tick + 10


""""""""""" Dataset related parameters """""""""""
def get_label_distribution(target_node: node.Node, parameters: simulation_runtime_parameters.RuntimeParameters):
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        return label_distribution.label_distribution_non_iid_dirichlet(target_node, parameters, 0.5)
    return None


""""""""""" Dataset related parameters """""""""""
def get_optimizer(target_node: node.Node, parameters: simulation_runtime_parameters.RuntimeParameters):
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        return torch.optim.Adam(target_node.model.parameters(), lr=0.001)
    return None
