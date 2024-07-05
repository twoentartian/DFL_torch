import networkx as nx
import torchvision
from py_src import simulation_runtime_parameters

config_name = "default_config"

max_tick = 10000        # total simulation ticks
model = torchvision.models.resnet18(progress=False, num_classes=10, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None)


def get_topology(parameters: simulation_runtime_parameters.RuntimeParameters) -> nx.Graph:
    get_topology.topology = None
    if get_topology.topology is None:
        get_topology.topology = nx.complete_graph(10)
    return get_topology.topology



