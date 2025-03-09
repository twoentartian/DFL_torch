import sys

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import colorsys
import os
import re
import cv2
import multiprocessing

import pandas

import argparse

topology_folder = 'topology'
accuracy_file_path = 'accuracy.csv'
draw_interval = 1
fps = 2
dpi = 200
override_existing_cache = True
HSV_H_start = 40
HSV_H_end = 256

video_cache_path = "./video_cache"


def load_graph(topology_folder_path: str, verbose=False):
    import pickle

    graphs = {}
    for file_name in os.listdir(topology_folder_path):
        if file_name.endswith(".pickle") and file_name[:-7].isdigit():
            key = int(file_name[:-7])  # Extract the integer from the file name
            if verbose:
                print(f"loading graph at tick {key}.")
            file_path = os.path.join(topology_folder_path, file_name)
            with open(file_path, 'rb') as file:
                graphs[key] = pickle.load(file)
    graphs = dict(sorted(graphs.items()))
    union_graph = nx.Graph()
    for tick, G in graphs.items():
        union_graph.add_nodes_from(list(G.nodes))
        union_graph.add_edges_from(list(G.edges))

    return graphs, union_graph


def save_fig(G: nx.Graph, tick, save_name, node_accuracies, layout, node_labels, node_size, with_labels, override_existing=False, secondary_accuracies=None, secondary_node_labels=None):
    if not override_existing and os.path.exists(save_name):
        return

    if secondary_accuracies is None:
        # only one plot
        fig = plt.figure(frameon=False)
        fig.set_size_inches(12, 12)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.text(0, 0, "tick = " + str(tick))
        cmap = matplotlib.colormaps.get_cmap('viridis')
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
        node_colors = [cmap(normalize(node_accuracy)) for node_accuracy in node_accuracies]

        nx.draw(G, node_color=node_colors, with_labels=with_labels, pos=layout, font_color='k', labels=node_labels, alpha=0.7, linewidths=0.1, width=0.1, font_size=8, node_size=node_size)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
        sm.set_array([0, 1])
        fig.colorbar(sm, ax=ax, orientation='vertical', label='Values', shrink=0.4)
        fig.savefig(save_name, dpi=dpi, pad_inches=0)
        plt.close(fig)
    else:
        # two plots
        fig = plt.figure(frameon=False)
        fig.set_size_inches(18, 9)
        ax = plt.Axes(fig, [0., 0., 0.5, 1.])
        ax2 = plt.Axes(fig, [0.5, 0., 0.5, 1.])
        ax.set_axis_off()
        ax2.set_axis_off()
        fig.add_axes(ax)
        fig.add_axes(ax2)
        ax.text(0, 0, f"tick = {tick}, main accuracy")
        ax2.text(0, 0, f"tick = {tick}, 2nd accuracy")

        cmap = matplotlib.colormaps.get_cmap('viridis')
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
        node_colors = [cmap(normalize(node_accuracy)) for node_accuracy in node_accuracies]
        nx.draw(G, ax=ax, node_color=node_colors, with_labels=with_labels, pos=layout, font_color='k', labels=node_labels, alpha=0.7, linewidths=0.1, width=0.1, font_size=8, node_size=node_size)

        cmap2 = matplotlib.colormaps.get_cmap('viridis')
        normalize2 = matplotlib.colors.Normalize(vmin=0, vmax=1)
        node_colors2 = [cmap2(normalize2(node_accuracy)) for node_accuracy in secondary_accuracies]
        nx.draw(G, ax=ax2, node_color=node_colors2, with_labels=with_labels, pos=layout, font_color='k', labels=secondary_node_labels, alpha=0.7, linewidths=0.1, width=0.1, font_size=8, node_size=node_size)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
        sm.set_array([0, 1])
        fig.colorbar(sm, ax=ax, orientation='vertical', label='Values', shrink=0.4)
        fig.colorbar(sm, ax=ax2, orientation='vertical', label='Values', shrink=0.4)
        fig.savefig(save_name, dpi=dpi, pad_inches=0)
        plt.close(fig)


def save_raw_fig(G: nx.Graph, save_name, node_color, layout, node_labels, node_size, with_labels, override_existing=False):
    if not override_existing and os.path.exists(save_name):
        return

    fig = plt.figure(frameon=False)
    fig.set_size_inches(12, 12)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    nx.draw(G, node_color=node_color, with_labels=with_labels, pos=layout, font_color='k', labels=node_labels, alpha=0.7, linewidths=0.1, width=0.1, font_size=8, node_size=node_size)

    fig.savefig(save_name, dpi=dpi, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    # parser args
    parser = argparse.ArgumentParser(description="generate a video for accuracy trends, put this script in the folder of 'accuracy.csv' and 'topology' folder.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--override_cache", action='store_true', help="override images cache?")
    args = parser.parse_args()
    config = vars(args)
    override_cache = False
    if config["override_cache"]:
        override_cache = True

    graphs, union_graph = load_graph(os.path.join(os.path.curdir, topology_folder), True)

    accuracy_df = pandas.read_csv(accuracy_file_path, index_col=0, header=0)

    ## user can load second_accuracy_df here (optional)
    second_accuracy_df = None

    total_tick = len(accuracy_df.index)
    draw_counter = 0
    tick_to_draw = []
    for tick in accuracy_df.index:
        if draw_counter >= draw_interval-1:
            draw_counter = 0
            tick_to_draw.append(tick)
        else:
            draw_counter = draw_counter + 1
            continue

    # layout = nx.spring_layout(G, k=5/math.sqrt(G.order()))
    # layout = nx.circular_layout(G)
    # layout = nx.spectral_layout(G)
    # layout = nx.kamada_kawai_layout(G)
    # layout = nx.shell_layout(G)
    # layout = nx.random_layout(G)
    layout = nx.nx_agraph.graphviz_layout(union_graph)

    node_name = union_graph.nodes
    peer_change_list_index = 0
    if not os.path.isdir(video_cache_path):
        os.mkdir(video_cache_path)

    pool = multiprocessing.Pool(processes=os.cpu_count())

    N = len(union_graph.nodes)
    print(f"N={N}")
    node_size = int(50000/N)
    node_size = max(10, node_size)
    print(f"draw_node_size={node_size}")
    with_labels = True
    if N > 300:
        with_labels = False
    print(f"draw_with_labels={with_labels}")

    next_graph_change_tick, G = next(iter(graphs.items()))
    graph_change_ticks = sorted(graphs.keys())
    for tick in tick_to_draw:
        print("processing tick: " + str(tick))
        if os.path.exists(os.path.join(video_cache_path, str(tick) + ".png")) and not override_existing_cache:
            continue

        node_labels = {}

        # update G
        if tick >= next_graph_change_tick:
            G = graphs[next_graph_change_tick]
            next_key_index = graph_change_ticks.index(next_graph_change_tick) + 1
            if len(graph_change_ticks) > next_key_index:
                next_graph_change_tick = graph_change_ticks[next_key_index]
            else:
                next_graph_change_tick = sys.maxsize

        # node color
        node_accuracies = []

        for node in G.nodes:
            accuracy = accuracy_df.loc[tick, str(node)]
            node_accuracies.append(accuracy)
            node_labels[node] = str(accuracy)

        node_labels2 = None
        second_node_accuracies = None
        if second_accuracy_df is not None:
            node_labels2 = {}
            second_node_accuracies = []
            for node in G.nodes:
                v = 0
                try:
                    # Attempt to retrieve the value using iloc for integer-based indexing
                    # Use loc for label-based indexing, e.g., df.loc[row_idx, col_idx]
                    v = second_accuracy_df.loc[tick, node]
                except IndexError:
                    v = 0
                except KeyError:
                    v = 0
                second_node_accuracies.append(v)
                node_labels2[node] = str(v)

        # save to files
        pool.apply_async(save_fig, (G.copy(), tick, os.path.join(video_cache_path, str(tick) + ".png"), node_accuracies, layout, node_labels, node_size, with_labels, override_cache, second_node_accuracies, node_labels2))
        # save_fig(G.copy(), tick, os.path.join(video_cache_path, str(tick) + ".png"), node_accuracies, layout, node_labels, node_size, with_labels, override_cache, second_node_accuracies, node_labels2)

        # save the map
        if tick == tick_to_draw[0]:
            # map
            maximum_degree_node, maximum_degree = max(G.degree)
            node_color = []
            node_labels = {}
            for node in G.nodes:
                (r, g, b) = colorsys.hsv_to_rgb(HSV_H_start / 256 + (1 - HSV_H_start / 256) * G.degree[node] / maximum_degree, 0.5, 1.0)
                node_color.append([r, g, b])
                node_labels[node] = f"{node}({G.degree[node]})"
            pool.apply_async(save_raw_fig, (G.copy(), "map.pdf", node_color, layout, node_labels, node_size, True,  True))

    pool.close()
    pool.join()

    # let opencv generate video
    print("generating video")

    first_img = cv2.imread(os.path.join(video_cache_path, str(tick_to_draw[0]) + ".png"))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, channel = first_img.shape
    video = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))
    for tick in tick_to_draw:
        img = cv2.imread(os.path.join(video_cache_path, str(tick) + ".png"))
        video.write(img)
    video.release()
