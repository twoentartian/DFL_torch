from point import Point
import os, sys, argparse

import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import nx_lib

class Node(object):
    def __init__(self, id: int, nn_dimension: int):
        self.point = Point(D=nn_dimension)
        self.id = id
        self.id = id

    def get_peers(self, G: nx.Graph):
        peers = list(G.neighbors(self.id))

def point_location_to_accuracy():
    pass

def average_step():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate fully decentralized training')
    parser.add_argument('--topology', '-t', type=str, required=True, help='path to topology file')
    parser.add_argument('--nn-dimension', '-d', type=int, default=500, help='default neural network dimension')

    args = parser.parse_args()

    topology_file_path = args.topology
    nn_dimension = args.nn_dimension

    G = nx_lib.load_topology_from_edge_list_file(topology_file_path)
    all_nodes = {}
    for n in G.nodes:
        all_nodes[n] = Node(n, nn_dimension)



    nx_lib.display_graph(G, node_size=100)