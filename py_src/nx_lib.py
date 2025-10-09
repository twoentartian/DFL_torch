import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import kernighan_lin_bisection

def split_to_equal_size_communities(topology: nx.Graph, num_of_communities):
    n = topology.number_of_nodes()

    communities = [list(c) for c in kernighan_lin_bisection(topology)]

    # Split further until we get the desired number of communities
    while len(communities) < num_of_communities:
        new_communities = []
        for community in communities:
            subgraph = topology.subgraph(community)
            if len(subgraph) > 1:
                # Split the community into two
                split = [list(c) for c in kernighan_lin_bisection(subgraph)]
                new_communities.extend(split)
            else:
                # If a community has only one node, just keep it as is
                new_communities.append(community)
        communities = new_communities

    # If the number of communities is more than needed, merge the smallest communities
    while len(communities) > num_of_communities:
        # Find the smallest communities to merge
        communities = sorted(communities, key=len)
        merged_community = communities[0] + communities[1]
        communities = [merged_community] + communities[2:]

    # Balance the community sizes if they are not exactly equal
    total_nodes = sum(len(community) for community in communities)
    target_size = total_nodes // num_of_communities

    # Adjust community sizes to be more balanced
    while True:
        largest = max(communities, key=len)
        smallest = min(communities, key=len)

        if len(largest) - len(smallest) <= 1:
            break

        # Move a node from the largest to the smallest
        node_to_move = largest.pop()
        smallest.append(node_to_move)

    return communities


def get_inter_community_edges(topology: nx.Graph, communities):
    # Create a dictionary to map each node to its community
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    # Initialize the count of inter-community edges
    inter_community_edges = []

    # Iterate through all edges in the graph
    for u, v in topology.edges():
        if node_to_community[u] != node_to_community[v]:
            inter_community_edges.append((u, v))

    return inter_community_edges


def load_topology_from_edge_list_file(file_path):
    assert os.path.exists(file_path)
    graph = nx.Graph()
    with open(file_path, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.split())
            graph.add_edge(node1, node2)
    return graph


def display_graph(G, layout=None, node_color='lightblue',
                  node_size=500, with_labels=True, title='Network Graph',
                  figsize=(10, 8), edge_color='gray', font_size=10):
    """
    Display a NetworkX graph using matplotlib.

    Parameters:
    -----------
    G : networkx.Graph
        The graph to display
    layout : str
        Layout algorithm: 'spring', 'circular', 'random', 'kamada_kawai', 'shell'
    node_color : str or list
        Color(s) for nodes
    node_size : int or list
        Size(s) of nodes
    with_labels : bool
        Whether to show node labels
    title : str
        Title of the plot
    figsize : tuple
        Figure size (width, height)
    edge_color : str
        Color of edges
    font_size : int
        Font size for labels
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Choose layout
    layouts = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'random': nx.random_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'shell': nx.shell_layout
    }
    if layout is None:
        pos = nx.nx_agraph.graphviz_layout(G)
    elif layout in layouts:
        pos = layouts[layout](G, seed=42) if layout == 'spring' else layouts[layout](G)
    else:
        raise ValueError('Invalid layout')
    # Draw the graph
    nx.draw(G, pos,
            with_labels=with_labels,
            node_color=node_color,
            node_size=node_size,
            edge_color=edge_color,
            font_size=font_size,
            font_weight='bold')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()