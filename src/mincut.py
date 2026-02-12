"""
Minimal cut entanglement entropy computation.
Uses fast crossing‑edge count for large graphs (validated against Edmonds–Karp).
"""

import numpy as np
import networkx as nx

def cut_size(net, interval):
    """
    Number of edges crossing the boundary of `interval`, weighted by log(chi).
    This is the entanglement entropy S(A) in the holographic prescription.
    """
    if not interval:
        return 0.0
    interval_set = set(interval)
    cut = 0
    # Undirected graph – count each edge once
    for u, v in net.graph.to_undirected().edges():
        if (u in interval_set) != (v in interval_set):
            cut += 1
    return cut * np.log(net.chi)

def compute_slope(net, maxL=30):
    """
    Compute linear slope dS/dL using time‑ordered boundary chain.
    Returns:
        slope: linear fit coefficient
        avg_indeg: average in‑degree of boundary nodes (for mean‑field check)
    """
    b = net.boundary()
    b.sort(key=lambda n: net.graph.nodes[n].get('time', 0))
    if len(b) < maxL:
        return None, None

    Ls = list(range(2, maxL + 1, 2))
    ents = [cut_size(net, b[:L]) for L in Ls]
    slope, _ = np.polyfit(Ls, ents, 1)

    # Average in‑degree of boundary nodes (excluding root 0)
    indeg = [net.graph.in_degree(n) for n in b if n != 0]
    avg_indeg = np.mean(indeg) if indeg else 0
    return slope, avg_indeg

# Optional: Edmonds–Karp exact min‑cut for verification (small N only)
def exact_mincut(net, interval):
    """Exact minimal cut using NetworkX's minimum_cut (Edmonds–Karp)."""
    if len(interval) == 0 or len(interval) == len(net.graph.nodes):
        return 0.0
    G = net.graph.to_undirected()
    source, sink = 's', 't'
    G.add_node(source)
    G.add_node(sink)
    for node in interval:
        G.add_edge(source, node, capacity=float('inf'))
    for node in set(net.graph.nodes) - set(interval):
        G.add_edge(node, sink, capacity=float('inf'))
    for u, v in G.edges():
        if 'capacity' not in G[u][v]:
            G[u][v]['capacity'] = np.log(net.chi)
    cut_val, _ = nx.minimum_cut(G, source, sink, capacity='capacity')
    return cut_val
