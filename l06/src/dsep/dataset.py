"""Function for generation and visualization of PGMs."""
import networkx as nx


def _get_example_graph():
    """Example taken from Lecture 6 slide 55."""
    G = nx.DiGraph()
    G.add_edges_from([
        (1, 3), (2, 3),
        (3, 4), (3, 8),
        (4, 6), (5, 6),
        (6, 7)
    ])
    
    node_pos = {
        1: (0, 0), 2: (2, 0), 
        3: (1, -1), 
        5: (-2, -2), 4: (0, -2), 8: (2, -2),
        6: (-1, -3), 7: (-1, -5)
    }
    
    return G, node_pos


def get_graph(which, verbose=False):
    if which == 'first':
        G, npos = _get_example_graph()
    elif which == 'second':
        # Modification of example graph (Lecture 6; slide 59)
        G, npos = _get_example_graph()
        G.add_edge(1, 5)
    else:
        raise RuntimeError('Unknown graph')
    
    return G, npos


def visualize(G, C, node_pos):
    colors = ['gray' if v in C else 'red' for v in G.nodes()]
    nx.draw(G, with_labels=True, node_color=colors, pos=node_pos, node_size=2000)
