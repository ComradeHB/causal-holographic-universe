"""
Helper functions for boundary ordering and analysis.
"""

import networkx as nx
import numpy as np

def order_boundary_by_time(net):
    """Return boundary nodes sorted by creation time (guarantees a 1D chain)."""
    b = net.boundary()
    b.sort(key=lambda n: net.graph.nodes[n].get('time', 0))
    return b

def order_boundary_by_bfs(net):
    """
    Order boundary using BFS on the largest connected component.
    (Used in the paper to show the boundary self‑organizes.)
    """
    b = net.boundary()
    G_undir = net.graph.to_undirected()
    bound_sub = G_undir.subgraph(b)
    comps = list(nx.connected_components(bound_sub))
    if not comps:
        # fallback to time order
        return order_boundary_by_time(net)
    main_boundary = sorted(comps, key=len)[-1]
    start = next(iter(main_boundary))
    return list(nx.bfs_tree(bound_sub, start).nodes())

def add_spacelike_chain(graph, boundary_ordered, chi):
    """
    Add undirected edges to connect consecutive boundary nodes.
    Each edge gets capacity = log(chi). Returns a copy of the graph.
    """
    G = graph.copy()
    for i in range(len(boundary_ordered) - 1):
        u = boundary_ordered[i]
        v = boundary_ordered[i + 1]
        G.add_edge(u, v)
        G[u][v]['capacity'] = np.log(chi)
    return G
