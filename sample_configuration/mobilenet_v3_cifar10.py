import networkx as nx
import torch
import math

from py_src import ml_setup, node, model_average, nx_lib
from py_src.ml_setup import MlSetup
from py_src.service.record_weights_difference import ServiceWeightsDifferenceRecorder
from py_src.simulation_runtime_parameters import SimulationPhase, RuntimeParameters
from py_src.config_file_util import label_distribution
from py_src.service.record_variance import ServiceVarianceRecorder
from py_src.service.record_training_loss import ServiceTrainingLossRecorder
from py_src.service.record_test_accuracy_loss import ServiceTestAccuracyLossRecorder
from py_src.model_variance_correct import VarianceCorrector, VarianceCorrectionType

config_name = "default_config"

max_tick = 600000        # total simulation ticks

force_use_cpu = False

"""do you want to put all models in GPU or only keep model stat in memory and share a model in gpu?"""
"""None | False: let simulator decide"""
override_use_model_stat = None
override_allocate_all_models = None

""""""""" Global Machine learning related parameters """""""""""
""" predefined: """
def get_ml_setup():
    get_ml_setup.__ml_setup = None
    if get_ml_setup.__ml_setup is None:
        # get_ml_setup.__ml_setup = ml_setup.resnet18_cifar10()
        # get_ml_setup.__ml_setup = ml_setup.lenet5_mnist()
        # get_ml_setup.__ml_setup = ml_setup.cct7_cifar10()
        get_ml_setup.__ml_setup = ml_setup.mobilenet_v3_small_cifar10()
    return get_ml_setup.__ml_setup

""" or self defined: """
# ml_parameters = ml_setup.MlSetup()
# ml_parameters.model = None
# ml_parameters.training_data = None
# ml_parameters.testing_data = None


def get_average_algorithm(target_node: node.Node, parameters: RuntimeParameters):
    # variance_correction = VarianceCorrector(VarianceCorrectionType.FollowOthers)
    # return model_average.StandardModelAverager(variance_corrector=variance_correction)
    return model_average.StandardModelAverager()

def get_average_buffer_size(target_node: node.Node, parameters: RuntimeParameters):
    neighbors = list(parameters.topology.neighbors(target_node.name))
    R = 1
    return R * len(neighbors)


""""""""""" Global Topology related parameters """""""""""
"""topology will be updated every tick if not None"""
def get_topology(parameters: RuntimeParameters) -> nx.Graph:
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        # get_topology.current_topology = nx.complete_graph(50)
        # G: nx.Graph = nx.random_regular_graph(8, 20)
        G = nx.Graph()
        G.add_node(0)
        get_topology.current = G
        # get_topology.current = nx_lib.load_topology_from_edge_list_file(f"topology/ca-netscience.data")
        # G = nx.Graph()
        # # # G.add_edge(0,1)
        # G.add_node(0)
        # get_topology.current = G
        return get_topology.current

    return None

""""""""""" Node related parameters """""""""""
"""parameters here will only be initialized once at beginning"""
def node_behavior_control(parameters: RuntimeParameters):
    if parameters.phase == SimulationPhase.INITIALIZING:
        for node_name, node_target in parameters.node_container.items():
            node_target.send_model_after_P_training = 1


""""""""""" Training time related parameters """""""""""
"""this function will be called after training to get the next training tick"""
def get_next_training_time(target_node: node.Node, parameters: RuntimeParameters) -> int:
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        return 0
    return target_node.next_training_tick + 10


""""""""""" Dataset related parameters """""""""""
"""label distribution will be updated every tick if not None"""
def get_label_distribution(target_node: node.Node, parameters: RuntimeParameters):
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        # return label_distribution.label_distribution_non_iid_dirichlet(target_node, parameters, 0.5)
        get_label_distribution.current = label_distribution.label_distribution_iid(target_node, parameters)
        return get_label_distribution.current

    return None



""""""""""" Dataset related parameters """""""""""
def get_optimizer(target_node: node.Node, model: torch.nn.Module, parameters: RuntimeParameters, ml_setup: MlSetup):
    """warning: you are not allowed to change optimizer during simulation when use_model_stat == True"""
    assert model is not None
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        lr_scheduler = None
        return optimizer, lr_scheduler
    return None, None


""""""""""" Service related parameters """""""""""
def get_service_list():
    service_list = []

    service_list.append(ServiceVarianceRecorder(100, phase=[SimulationPhase.START_OF_TICK, SimulationPhase.AFTER_TRAINING, SimulationPhase.AFTER_AVERAGING]))
    service_list.append(ServiceTrainingLossRecorder(100))
    service_list.append(ServiceTestAccuracyLossRecorder(100, 100))
    service_list.append(ServiceWeightsDifferenceRecorder(100))

    return service_list

