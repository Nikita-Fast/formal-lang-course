from typing import Tuple

import cfpq_data
import networkx
import networkx as nx


def load_graph(name: str):
    try:
        graph_path = cfpq_data.download(name)
        return cfpq_data.graph_from_csv(graph_path)
    except FileNotFoundError as e:
        raise e


def get_graph_info(name: str):
    g = load_graph(name)

    labels = []

    for _, _, lbl in g.edges(data=True):
        labels.append(lbl["label"])

    return g.number_of_nodes(), g.number_of_edges(), labels


def create_and_save_two_cycles_graph(
    nodes_in_fst_cycle, nodes_in_snd_cycle, labels, path
):
    g = cfpq_data.labeled_two_cycles_graph(
        nodes_in_fst_cycle, nodes_in_snd_cycle, labels=labels
    )
    networkx.drawing.nx_pydot.write_dot(g, path)


def create_two_cycle_graph(
    first_vertices: int, second_vertices: int, edge_labels: Tuple[str, str]
) -> nx.MultiDiGraph:
    """
    Create two cycle graph with labels on the edges.
    Parameters
    ----------
    first_vertices: int
        Amount of vertices in the first cycle
    second_vertices: int
        Amount of vertices in the second cycle
    edge_labels: Tuple[str, str]
        Labels for the edges on the first and second cycle
    Returns
    -------
    nx.MultiDiGraph
        Generated graph with two cycles
    """
    return cfpq_data.labeled_two_cycles_graph(
        first_vertices, second_vertices, labels=edge_labels
    )
