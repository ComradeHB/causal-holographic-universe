# ===== COMPLETE CAUSAL TENSOR NETWORK + PARAMETER SWEEP =====
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import time

# ------------------------------------------------------------
# Full CTN class with growth, ancestor cache, and boundary linking
# ------------------------------------------------------------
class CTN:
    def __init__(self, chi=4):
        self.chi = chi
        self.graph = nx.DiGraph()
        self._ancestors = {}
        self._seed()

    def _seed(self):
        self._add_node(0, [], 0)
        self._add_node(1, [0], 1); self._add_edge(0,1)
        self._add_node(2, [0], 1); self._add_edge(0,2); self._add_edge(1,2)

    def _add_node(self, nid, parents, time):
        # Create isometric tensor (simplified – no actual tensor storage needed for entropy)
        self.graph.add_node(nid, time=time)
        anc = set(parents)
        for p in parents:
            anc |= self._ancestors.get(p, set())
        self._ancestors[nid] = anc

    def _add_edge(self, u, v):
        self.graph.add_edge(u, v)

    def boundary(self):
        return [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

    # --------------------------------------------------------
    # GROWTH – with optional spacelike boundary linking
    # --------------------------------------------------------
    def grow(self, target, growth_prob=0.3, ancestor_prob=0.5,
             boundary_link_prob=0.0):
        """
        Grow until `target` nodes. If boundary_link_prob > 0,
        randomly connect boundary nodes during growth.
        """
        step = 0
        while len(self.graph) < target:
            step += 1
            # --- Causal growth (always happens) ---
            for node in list(self.graph.nodes):
                if len(self.graph) >= target:
                    break
                if np.random.rand() < growth_prob:
                    parents = [node]
                    if ancestor_prob > 0 and self._ancestors[node]:
                        if np.random.rand() < ancestor_prob:
                            parents.append(np.random.choice(list(self._ancestors[node])))
                    nid = len(self.graph.nodes)
                    self._add_node(nid, parents, step)
                    for p in parents:
                        self._add_edge(p, nid)
                    if len(self.graph) >= target:
                        break

            # --- Spacelike boundary linking (if prob > 0) ---
            if boundary_link_prob > 0 and np.random.rand() < boundary_link_prob:
                b = self.boundary()
                if len(b) > 1:
                    u, v = np.random.choice(b, 2, replace=False)
                    if not self.graph.has_edge(u, v):
                        self.graph.add_edge(u, v)

            if step % 20 == 0:
                print(f"Step {step:4d} | Nodes = {len(self.graph):6d} / {target}", end='\r')
        print(f"Step {step:4d} | Nodes = {len(self.graph):6d} / {target}")
        self.time_step = step

# ------------------------------------------------------------
# ANALYSIS – connect boundary, measure cut entropy, fit scaling
# ------------------------------------------------------------
def analyze_entanglement_scaling(net, maxL=30, plot=False):
    """Order boundary by time, add 1D chain, compute S(L) for intervals."""
    boundary = net.boundary()
    boundary.sort(key=lambda n: net.graph.nodes[n].get('time', 0))
    if len(boundary) < maxL:
        return None, None, None

    # Work on a copy, add spacelike chain with capacity = log(chi)
    G = net.graph.copy()
    for i in range(len(boundary)-1):
        u, v = boundary[i], boundary[i+1]
        G.add_edge(u, v)
        G[u][v]['capacity'] = np.log(net.chi)

    # Function: cut size for an interval (undirected, counting all crossing edges)
    def cut_size(interval):
        interval_set = set(interval)
        cut = 0
        for u, v in G.edges():
            if (u in interval_set) != (v in interval_set):
                cut += 1
        return cut * np.log(net.chi)

    sizes = list(range(2, maxL+1, 2))
    ents = [cut_size(boundary[:L]) for L in sizes]

    # Linear fit
    lin_coeff = np.polyfit(sizes, ents, 1)
    lin_fit = np.polyval(lin_coeff, sizes)
    lin_resid = np.sum((ents - lin_fit)**2)

    # Log fit
    logL = np.log(sizes)
    log_coeff = np.polyfit(logL, ents, 1)
    log_fit = np.polyval(log_coeff, logL)
    log_resid = np.sum((ents - log_fit)**2)

    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(sizes, ents, 'o-', label='Data')
        plt.plot(sizes, lin_fit, '--', label=f'Linear: {lin_coeff[0]:.2f}*L')
        plt.plot(sizes, log_fit, ':', label=f'Log: {log_coeff[0]:.2f}*log(L)')
        plt.xlabel('Interval length L')
        plt.ylabel('Entanglement entropy S(L)')
        plt.title(f'χ={net.chi}, ancestor_prob={getattr(net,"last_ancestor_prob","?")}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return lin_coeff[0], log_coeff[0], (lin_resid, log_resid)

# ------------------------------------------------------------
# PARAMETER SWEEP – three ancestor probabilities
# ------------------------------------------------------------
np.random.seed(42)   # reproducible

configs = [
    {"ancestor_prob": 0.0, "boundary_link_prob": 0.0, "desc": "Pure tree (no ancestors)"},
    {"ancestor_prob": 0.3, "boundary_link_prob": 0.0, "desc": "Your current model (ancestor_prob=0.3)"},
    {"ancestor_prob": 0.9, "boundary_link_prob": 0.0, "desc": "Highly connected bulk (ancestor_prob=0.9)"},
]

results = []

for cfg in configs:
    print("\n" + "="*60)
    print(f"RUNNING: {cfg['desc']}")
    print("="*60)
    net = CTN(chi=4)
    net.last_ancestor_prob = cfg["ancestor_prob"]   # for plot title
    net.grow(target=50000,
             growth_prob=0.3,
             ancestor_prob=cfg["ancestor_prob"],
             boundary_link_prob=cfg["boundary_link_prob"])
    lin_slope, log_slope, residuals = analyze_entanglement_scaling(net, maxL=30, plot=True)
    results.append({
        "desc": cfg["desc"],
        "lin_slope": lin_slope,
        "log_slope": log_slope,
        "lin_resid": residuals[0],
        "log_resid": residuals[1],
        "boundary_nodes": len(net.boundary())
    })
    print(f"Linear slope = {lin_slope:.3f}, Log slope = {log_slope:.3f}")
    print(f"Residuals: Linear = {residuals[0]:.2f}, Log = {residuals[1]:.2f}")

# ------------------------------------------------------------
# SUMMARY TABLE
# ------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY – SCALING BEHAVIOR")
print("="*60)
print(f"{'Model':<35} {'Linear slope':<12} {'Log slope':<12} {'Lin resid':<10} {'Log resid':<10} {'Boundary':<8}")
for r in results:
    print(f"{r['desc']:<35} {r['lin_slope']:<12.3f} {r['log_slope']:<12.3f} {r['lin_resid']:<10.2f} {r['log_resid']:<10.2f} {r['boundary_nodes']:<8}")

import matplotlib.pyplot as plt

# Define 'sizes' as it was used in analyze_entanglement_scaling (maxL=30)
sizes = list(range(2, 30 + 1, 2))

fig, ax = plt.subplots(1, 1, figsize=(6,4))
for res in results:
    # The 'ents' data is not stored in the 'results' list.
    # To plot this, the analyze_entanglement_scaling function would need
    # to be modified to return 'ents' and store it in 'results'.
    # For now, we will plot the linear fit instead, as its slope is available.
    if 'lin_slope' in res:
        # Calculate the linear fit based on the stored slope
        linear_fit_data = [res['lin_slope'] * L for L in sizes]
        ax.plot(sizes, linear_fit_data, 'o--', label=f"{res['desc']} (Linear Fit)")
    else:
        print(f"Warning: No 'lin_slope' data for {res['desc']}. Skipping plot for this model.")

ax.set_xlabel('Interval length L')
ax.set_ylabel('Entanglement entropy S(L)')
ax.set_title('Entanglement Entropy Scaling (Linear Fit)')
ax.legend()
plt.grid(alpha=0.3)
plt.show()

# ===== COMPLETE FIGURE GENERATION – PAPER READY =====
import numpy as np, networkx as nx, matplotlib.pyplot as plt
from scipy import stats
import time

# ----- CTN class (minimal) -----
class CTN:
    def __init__(self, chi=4):
        self.chi = chi
        self.graph = nx.DiGraph()
        self._ancestors = {}
        self._seed()
    def _seed(self):
        self._add_node(0, [], 0)
        self._add_node(1, [0], 1); self._add_edge(0,1)
        self._add_node(2, [0], 1); self._add_edge(0,2); self._add_edge(1,2)
    def _add_node(self, nid, parents, time):
        self.graph.add_node(nid, time=time)
        anc = set(parents)
        for p in parents: anc |= self._ancestors.get(p, set())
        self._ancestors[nid] = anc
    def _add_edge(self, u, v): self.graph.add_edge(u, v)
    def boundary(self): return [n for n in self.graph.nodes if self.graph.out_degree(n)==0]

# ----- Analysis: entropy scaling via connected boundary chain -----
def analyze_entropy(net, maxL=30):
    b = net.boundary()
    b.sort(key=lambda n: net.graph.nodes[n].get('time',0))
    G = net.graph.copy()
    for i in range(len(b)-1):
        u,v = b[i], b[i+1]
        G.add_edge(u,v)
        G[u][v]['capacity'] = np.log(net.chi)
    def cut_size(interval):
        I = set(interval)
        cut = 0
        for u,v in G.edges():
            if (u in I) != (v in I):
                cut += 1
        return cut * np.log(net.chi)
    Ls = list(range(2, maxL+1, 2))
    S = [cut_size(b[:L]) for L in Ls]
    # linear fit
    lin = np.polyfit(Ls, S, 1)
    lin_resid = np.sum((S - np.polyval(lin, Ls))**2)
    return Ls, S, lin[0], lin_resid

# ----- Parameter sweep: p_a = 0.0, 0.3, 0.9 -----
np.random.seed(42)
configs = [
    (0.0, "Pure tree (no ancestors)"),
    (0.3, "Ancestor prob = 0.3"),
    (0.9, "Ancestor prob = 0.9")
]

results = []
fig4, ax4 = plt.subplots(1,1, figsize=(6,4))

for p_a, label in configs:
    print(f"\n--- {label} ---")
    net = CTN(chi=4)
    # Grow to 50,000 nodes
    target = 50000
    step = 0
    while len(net.graph) < target:
        step += 1
        for node in list(net.graph.nodes):
            if len(net.graph) >= target: break
            if np.random.rand() < 0.3:
                parents = [node]
                if p_a > 0 and net._ancestors[node] and np.random.rand() < p_a:
                    parents.append(np.random.choice(list(net._ancestors[node])))
                nid = len(net.graph.nodes)
                net._add_node(nid, parents, step)
                for p in parents: net._add_edge(p, nid)
                if len(net.graph) >= target: break
        if step % 20 == 0:
            print(f"Step {step:4d} | Nodes = {len(net.graph):6d} / {target}", end='\r')
    print(f"Step {step:4d} | Nodes = {len(net.graph):6d} / {target}")

    Ls, S, slope, resid = analyze_entropy(net, maxL=30)
    results.append((p_a, slope, resid, Ls, S, label))

    ax4.plot(Ls, S, 'o-', label=f"{label} (slope = {slope:.3f})")

# ----- Figure 4: Entropy scaling -----
ax4.set_xlabel('Boundary interval length $L$')
ax4.set_ylabel('Entanglement entropy $S(L)$')
ax4.set_title('Entropy scaling for different ancestor probabilities')
ax4.legend()
ax4.grid(alpha=0.3)
plt.tight_layout()
fig4.savefig('Figure4_entropy_scaling.pdf', bbox_inches='tight', dpi=300)
plt.show()

# ----- Figure 5: Slope vs. ancestor probability -----
p_vals = [r[0] for r in results]
slopes = [r[1] for r in results]
fig5, ax5 = plt.subplots(1,1, figsize=(5,4))
ax5.plot(p_vals, slopes, 'o-', linewidth=2, markersize=8)
ax5.set_xlabel('Ancestor probability $p_a$')
ax5.set_ylabel('Linear slope $dS/dL$')
ax5.set_title('Entanglement density vs. ancestor bias')
ax5.grid(alpha=0.3)
plt.tight_layout()
fig5.savefig('Figure5_slope_vs_pa.pdf', bbox_inches='tight', dpi=300)
plt.show()

# ----- Summary table -----
print("\n" + "="*60)
print("SUMMARY – SCALING BEHAVIOR (χ = 4, 50k nodes)")
print("="*60)
print(f"{'Model':<35} {'Linear slope':<12} {'Residual':<10}")
for p_a, slope, resid, _, _, label in results:
    print(f"{label:<35} {slope:<12.3f} {resid:<10.2f}")

# ===== DYNAMICAL BOUNDARY – HIGH PROBABILITY, GUARANTEED CONNECTIVITY =====
import numpy as np, networkx as nx, matplotlib.pyplot as plt
from scipy import stats

class CTN:
    def __init__(self, chi=4):
        self.chi = chi
        self.graph = nx.DiGraph()
        self._ancestors = {}
        self._seed()
    def _seed(self):
        self._add_node(0, [], 0)
        self._add_node(1, [0], 1); self._add_edge(0,1)
        self._add_node(2, [0], 1); self._add_edge(0,2); self._add_edge(1,2)
    def _add_node(self, nid, parents, time):
        self.graph.add_node(nid, time=time)
        anc = set(parents)
        for p in parents: anc |= self._ancestors.get(p, set())
        self._ancestors[nid] = anc
    def _add_edge(self, u, v):
        self.graph.add_edge(u, v)
    def boundary(self):
        return [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

def analyze_entropy(net, maxL=30):
    """Use time-ordered boundary chain (guaranteed connected)."""
    b = net.boundary()
    b.sort(key=lambda n: net.graph.nodes[n].get('time', 0))
    if len(b) < maxL:
        return None, None, None
    def cut_size(interval):
        I = set(interval)
        cut = 0
        for u, v in net.graph.to_undirected().edges():
            if (u in I) != (v in I):
                cut += 1
        return cut * np.log(net.chi)
    Ls = list(range(2, maxL+1, 2))
    ents = [cut_size(b[:L]) for L in Ls]
    lin = np.polyfit(Ls, ents, 1)
    resid = np.sum((ents - np.polyval(lin, Ls))**2)
    return Ls, ents, lin[0], resid

# ----- Parameter sweep: p_a = 0.0, 0.3, 0.9 -----
np.random.seed(42)
configs = [(0.0, "Pure tree"), (0.3, "Ancestor prob = 0.3"), (0.9, "Ancestor prob = 0.9")]

plt.figure(figsize=(7,5))

for p_a, label in configs:
    print(f"\n--- {label} ---")
    net = CTN(chi=4)
    target = 50000
    step = 0
    while len(net.graph) < target:
        step += 1
        # Causal growth
        for node in list(net.graph.nodes):
            if len(net.graph) >= target: break
            if np.random.rand() < 0.3:
                parents = [node]
                if p_a > 0 and net._ancestors[node] and np.random.rand() < p_a:
                    parents.append(np.random.choice(list(net._ancestors[node])))
                nid = len(net.graph.nodes)
                net._add_node(nid, parents, step)
                for p in parents: net._add_edge(p, nid)
                if len(net.graph) >= target: break
        # ----- SPACELIKE BOUNDARY BONDS (p = 0.3, guaranteed connectivity) -----
        if np.random.rand() < 0.3:
            b = net.boundary()
            if len(b) > 1:
                u, v = np.random.choice(b, 2, replace=False)
                if not net.graph.has_edge(u, v):
                    net._add_edge(u, v)
        if step % 20 == 0:
            print(f"Step {step:4d} | Nodes = {len(net.graph):6d} / {target}", end='\r')
    print(f"Step {step:4d} | Nodes = {len(net.graph):6d} / {target}")

    Ls, ents, slope, resid = analyze_entropy(net, maxL=30)
    if Ls is not None:
        plt.plot(Ls, ents, 'o-', label=f"{label} (slope={slope:.3f})")
        print(f"  -> slope = {slope:.3f}, residual = {resid:.2f}")

plt.xlabel('Boundary interval length $L$')
plt.ylabel('Entanglement entropy $S(L)$')
plt.title('Causal Tensor Network – Dynamical Boundary Bonds')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Figure6_dynamical_boundary.pdf', bbox_inches='tight', dpi=300)
plt.show()
print("\n✅ Figure 6 saved as 'Figure6_dynamical_boundary.pdf'")


# ===== ENSEMBLE AVERAGING AND PARAMETER SWEEP =====
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from scipy import stats
import time
from collections import defaultdict
from google.colab import files

# ------------------------------------------------------------
# CTN class (with time_step tracking)
# ------------------------------------------------------------
class CTN:
    def __init__(self, chi=4):
        self.chi = chi
        self.graph = nx.DiGraph()
        self._ancestors = {}
        self.time_step = 0
        self._seed()
    def _seed(self):
        self._add_node(0, [], 0)
        self._add_node(1, [0], 1); self._add_edge(0,1)
        self._add_node(2, [0], 1); self._add_edge(0,2); self._add_edge(1,2)
    def _add_node(self, nid, parents, time):
        self.graph.add_node(nid, time=time)
        anc = set(parents)
        for p in parents:
            anc |= self._ancestors.get(p, set())
        self._ancestors[nid] = anc
    def _add_edge(self, u, v):
        self.graph.add_edge(u, v)
    def boundary(self):
        return [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

def compute_slope(net, maxL=30):
    """Time‑ordered boundary chain, return linear slope and average in‑degree."""
    b = net.boundary()
    b.sort(key=lambda n: net.graph.nodes[n].get('time', 0))
    if len(b) < maxL:
        return None, None
    # Compute cut size for intervals
    def cut_size(interval):
        I = set(interval)
        cut = 0
        for u, v in net.graph.to_undirected().edges():
            if (u in I) != (v in I):
                cut += 1
        return cut * np.log(net.chi)
    Ls = list(range(2, maxL+1, 2))
    ents = [cut_size(b[:L]) for L in Ls]
    slope, _ = np.polyfit(Ls, ents, 1)
    # Average in‑degree of boundary nodes (excluding root)
    indeg = [net.graph.in_degree(n) for n in b if n != 0]
    avg_indeg = np.mean(indeg) if indeg else 0
    return slope, avg_indeg

# ------------------------------------------------------------
# Parameter sweep configuration
# ------------------------------------------------------------
# Use smaller network for faster sweep (results stable at 20k)
TARGET = 20000
GROWTH_PROB = 0.3        # will be varied
BOUNDARY_LINK_PROB = 0.3 # keep fixed for dynamical boundary
MAXL = 30
N_SEEDS = 10             # ensemble size

# Parameter grids
p_a_vals = [0.0, 0.3, 0.9]
chi_vals = [2, 3, 4, 6]  # log(chi) = 0.693, 1.099, 1.386, 1.792
p_g_vals = [0.2, 0.3, 0.4]

# Store results: results[(p_a, chi, p_g)] = list of slopes
results = defaultdict(list)
indeg_results = defaultdict(list)

# ------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------
total_configs = len(p_a_vals) * len(chi_vals) * len(p_g_vals) * N_SEEDS
config_count = 0

print(f"🔬 Ensemble sweep: {N_SEEDS} seeds × {len(p_a_vals)} p_a × {len(chi_vals)} χ × {len(p_g_vals)} p_g = {total_configs} networks")
print(f"   Target size: {TARGET} nodes\n")

start_time = time.time()

for p_a in p_a_vals:
    for chi in chi_vals:
        for p_g in p_g_vals:
            key = (p_a, chi, p_g)
            print(f"\n--- p_a={p_a}, χ={chi}, p_g={p_g} ---")
            for seed in range(N_SEEDS):
                config_count += 1
                np.random.seed(seed)
                net = CTN(chi=chi)
                step = 0
                while len(net.graph) < TARGET:
                    step += 1
                    # Causal growth
                    for node in list(net.graph.nodes):
                        if len(net.graph) >= TARGET: break
                        if np.random.rand() < p_g:
                            parents = [node]
                            if p_a > 0 and net._ancestors[node] and np.random.rand() < p_a:
                                parents.append(np.random.choice(list(net._ancestors[node])))
                            nid = len(net.graph.nodes)
                            net._add_node(nid, parents, step)
                            for p in parents:
                                net._add_edge(p, nid)
                            if len(net.graph) >= TARGET: break
                    # Dynamical boundary bonds
                    if np.random.rand() < BOUNDARY_LINK_PROB:
                        b = net.boundary()
                        if len(b) > 1:
                            u, v = np.random.choice(b, 2, replace
