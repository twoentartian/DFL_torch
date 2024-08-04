import networkx as nx
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

