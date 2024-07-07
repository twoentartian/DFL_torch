import numpy as np
from py_src import simulation_runtime_parameters, ml_setup, node


def label_distribution_iid(target_node: node.Node, parameters: simulation_runtime_parameters.RuntimeParameters) -> np.ndarray:
    return np.repeat(1, len(parameters.dataset_label))

def label_distribution_non_iid_dirichlet(target_node: node.Node, parameters: simulation_runtime_parameters.RuntimeParameters, alpha) -> np.ndarray:
    return np.random.dirichlet(np.repeat(alpha, len(parameters.dataset_label)))

def label_distribution_default(target_node: node.Node, parameters: simulation_runtime_parameters.RuntimeParameters) -> np.ndarray:
    return np.random.dirichlet(np.repeat(0, len(parameters.dataset_label)))