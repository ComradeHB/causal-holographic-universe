"""
Causal graph growth model for de Sitter holography.
Core CTN class with ancestor cache and dynamical boundary bonds.
"""

import numpy as np
import networkx as nx

class CTN:
    """Causal Tensor Network – graph growth with ancestor bias and boundary linking."""

    def __init__(self, chi=4):
        self.chi = chi
        self.graph = nx.DiGraph()
        self._ancestors = {}
        self.time_step = 0
        self._seed()

    def _seed(self):
        """Initialize minimal causal diamond."""
        self._add_node(0, [], 0)
        self._add_node(1, [0], 1)
        self._add_edge(0, 1)
        self._add_node(2, [0], 1)
        self._add_edge(0, 2)
        self._add_edge(1, 2)

    def _add_node(self, nid, parents, time):
        """Add node with creation time and ancestor cache."""
        self.graph.add_node(nid, time=time)
        anc = set(parents)
        for p in parents:
            anc |= self._ancestors.get(p, set())
        self._ancestors[nid] = anc

    def _add_edge(self, u, v):
        """Add directed edge."""
        self.graph.add_edge(u, v)

    def boundary(self):
        """Return list of boundary nodes (out‑degree zero)."""
        return [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

    def grow(self, target, growth_prob=0.3, ancestor_prob=0.3, boundary_link_prob=0.3):
        """
        Grow graph until `target` nodes.
        - Causal edges: with probability `growth_prob`, attach child to current node.
        - Ancestor edges: with probability `ancestor_prob`, attach to random ancestor.
        - Spacelike boundary bonds: with probability `boundary_link_prob`, connect two random boundary nodes.
        """
        step = 0
        while len(self.graph) < target:
            step += 1
            # --- Causal growth ---
            for node in list(self.graph.nodes):
                if len(self.graph) >= target:
                    break
                if np.random.rand() < growth_prob:
                    parents = [node]
                    if ancestor_prob > 0 and self._ancestors[node] and np.random.rand() < ancestor_prob:
                        parents.append(np.random.choice(list(self._ancestors[node])))
                    nid = len(self.graph.nodes)
                    self._add_node(nid, parents, step)
                    for p in parents:
                        self._add_edge(p, nid)
                    if len(self.graph) >= target:
                        break
            # --- Dynamical spacelike boundary bonds ---
            if boundary_link_prob > 0 and np.random.rand() < boundary_link_prob:
                b = self.boundary()
                if len(b) > 1:
                    u, v = np.random.choice(b, 2, replace=False)
                    if not self.graph.has_edge(u, v):
                        self._add_edge(u, v)
            if step % 20 == 0:
                print(f"Step {step:4d} | Nodes = {len(self.graph):6d} / {target}", end='\r')
        print(f"Step {step:4d} | Nodes = {len(self.graph):6d} / {target}")
        self.time_step = step
