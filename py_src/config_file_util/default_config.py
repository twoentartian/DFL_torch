import networkx as nx
import torch
from py_src import ml_setup, node
from py_src.ml_setup import MlSetup
from py_src.simulation_runtime_parameters import SimulationPhase, RuntimeParameters
from py_src.config_file_util import label_distribution

config_name = "default_config"

max_tick = 1000        # total simulation ticks

"""do you want to put all models in GPU or only keep model stat in memory and share a model in gpu?"""
"""None: let simulator decide"""
override_use_model_stat = None


""""""""" Machine learning related parameters """""""""""
""" predefined: """
def get_ml_setup():
    get_ml_setup.__ml_setup = None
    if get_ml_setup.__ml_setup is None:
        get_ml_setup.__ml_setup = ml_setup.resnet18_cifar10()
    return get_ml_setup.__ml_setup

""" or self defined: """
# ml_parameters = ml_setup.MlSetup()
# ml_parameters.model = None
# ml_parameters.training_data = None
# ml_parameters.testing_data = None


""""""""""" Topology related parameters """""""""""
def get_topology(parameters: RuntimeParameters) -> nx.Graph:
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        get_topology.current_topology = nx.complete_graph(10)
        return get_topology.current_topology
    return None


""""""""""" Training time related parameters """""""""""
"""this function will be called after training to get the next training tick"""
def get_next_training_time(target_node: node.Node, parameters: RuntimeParameters) -> int:
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        return 0
    return target_node.next_training_tick + 10


""""""""""" Dataset related parameters """""""""""
def get_label_distribution(target_node: node.Node, parameters: RuntimeParameters):
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        # return label_distribution.label_distribution_non_iid_dirichlet(target_node, parameters, 0.5)
        return label_distribution.label_distribution_iid(target_node, parameters)
    return None


""""""""""" Dataset related parameters """""""""""
def get_optimizer(target_node: node.Node, model: torch.nn.Module, parameters: RuntimeParameters, ml_setup: MlSetup):
    assert model is not None
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        return torch.optim.Adam(model.parameters(), lr=0.001)
    return None