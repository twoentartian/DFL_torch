import networkx as nx
import torch
from py_src import ml_setup, node, model_average
from py_src.ml_setup import MlSetup
from py_src.simulation_runtime_parameters import SimulationPhase, RuntimeParameters
from py_src.config_file_util import label_distribution
from py_src.service.record_variance import ServiceVarianceRecorder
from py_src.service.record_training_loss import ServiceTrainingLossRecorder
from py_src.service.record_test_accuracy_loss import ServiceTestAccuracyLossRecorder
from py_src.model_variance_correct import VarianceCorrector, VarianceCorrectionType

config_name = "default_config"

max_tick = 10000        # total simulation ticks

force_use_cpu = False

"""do you want to put all models in GPU or only keep model stat in memory and share a model in gpu?"""
"""None | False: let simulator decide"""
override_use_model_stat = None
override_allocate_all_models = True

""""""""" Global Machine learning related parameters """""""""""
""" predefined: """
def get_ml_setup():
    get_ml_setup.__ml_setup = None
    if get_ml_setup.__ml_setup is None:
        # get_ml_setup.__ml_setup = ml_setup.resnet18_cifar10()
        get_ml_setup.__ml_setup = ml_setup.lenet5_mnist()
    return get_ml_setup.__ml_setup

""" or self defined: """
# ml_parameters = ml_setup.MlSetup()
# ml_parameters.model = None
# ml_parameters.training_data = None
# ml_parameters.testing_data = None


def get_average_algorithm(target_node: node.Node, parameters: RuntimeParameters):
    variance_correction = VarianceCorrector(VarianceCorrectionType.FollowOthers)
    return model_average.StandardModelAverager(variance_corrector=variance_correction)
    # return model_average.StandardModelAverager()

def get_average_buffer_size(target_node: node.Node, parameters: RuntimeParameters):
    neighbors = list(parameters.topology.neighbors(target_node.name))
    R = 1
    return R * len(neighbors)


""""""""""" Global Topology related parameters """""""""""
def get_topology(parameters: RuntimeParameters) -> nx.Graph:
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        # get_topology.current_topology = nx.complete_graph(50)
        get_topology.current_topology = nx.random_regular_graph(8, 50)
        return get_topology.current_topology
    return None

""""""""""" Node related parameters """""""""""
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
def get_label_distribution(target_node: node.Node, parameters: RuntimeParameters):
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        # return label_distribution.label_distribution_non_iid_dirichlet(target_node, parameters, 0.5)
        return label_distribution.label_distribution_iid(target_node, parameters)
    return None


""""""""""" Dataset related parameters """""""""""
def get_optimizer(target_node: node.Node, model: torch.nn.Module, parameters: RuntimeParameters, ml_setup: MlSetup):
    """warning: you are not allowed to change optimizer during simulation when use_model_stat == True"""
    assert model is not None
    if parameters.phase == SimulationPhase.INITIALIZING:    # init
        # return torch.optim.Adam(model.parameters(), lr=0.001)
        return torch.optim.SGD(model.parameters(), lr=0.01)
    return None


""""""""""" Service related parameters """""""""""
def get_service_list():
    service_list = []

    service_list.append(ServiceVarianceRecorder(20))
    service_list.append(ServiceTrainingLossRecorder(20))
    service_list.append(ServiceTestAccuracyLossRecorder(20, 100))

    return service_list

