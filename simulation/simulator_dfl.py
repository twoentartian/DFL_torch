from point import Point
import os, sys, argparse

import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import nx_lib

class Config(object):
    def __init__(self):
        self.BETA = 0.5
        self.TICK = 10000
        self.NN_DIMENSION = 500
        def get_train_interval(node: Node, G: nx.Graph):
            return 10
        self.TRAIN_INTERVAL_FUNC = get_train_interval


class Node(object):
    def __init__(self, id: int, config: Config):
        self.point = Point(D=config.NN_DIMENSION)
        self.id = id
        self.train_interval_func = config.TRAIN_INTERVAL_FUNC

    def get_peers(self, G: nx.Graph):
        peers = list(G.neighbors(self.id))
        return peers

    def get_neural_network(self):
        return self.point

    def __repr__(self) -> str:
        return self.point.__repr__()


def point_location_to_accuracy():
    pass


def average_step(target_node: Node, G: nx.Graph, config: Config):
    peers = target_node.get_peers(G)
    averaged_nn = None
    for p in peers:
        if averaged_nn is None:
            averaged_nn = p.get_neural_network()
        else:
            averaged_nn = averaged_nn + p.get_neural_network()
    averaged_nn = averaged_nn / len(peers)
    output_nn = target_node.get_neural_network() * config.BETA + averaged_nn * (1-config.BETA)
    return output_nn


class RuntimeValue(object):
    def __init__(self, config: Config):
        self.raw_config = config
        self.current_tick = 0
        self.max_tick = config.TICK


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate fully decentralized training')
    parser.add_argument('--topology', '-t', type=str, required=True, help='path to topology file')

    args = parser.parse_args()

    topology_file_path = args.topology
    nn_dimension = args.nn_dimension

    config = Config()

    G = nx_lib.load_topology_from_edge_list_file(topology_file_path)
    all_nodes = {}
    for n in G.nodes:
        all_nodes[n] = Node(n, nn_dimension)

    ### begin simulation ###
    runtime_var = RuntimeValue(config)
    while runtime_var.current_tick < config.TICK:


        runtime_var.current_tick += 1

    nx_lib.display_graph(G, node_size=100)