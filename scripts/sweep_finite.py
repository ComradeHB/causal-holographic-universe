# ===== FINITE-SIZE SCALING SWEEP =====
# Figure 4: N = 20k, 25k, 30k, 35k, 40k, 45k, 50k, 55k, 60k
# Fixed: p_a = 0.3, χ = 4, p_g = 0.3, p_b = 0.3, 8 seeds per N

import numpy as np
import networkx as nx
import pandas as pd
import time
from collections import defaultdict

# ------------------------------------------------------------
# CTN class (minimal, with dynamical boundary bonds)
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
    """Time‑ordered boundary chain → linear slope."""
    b = net.boundary()
    b.sort(key=lambda n: net.graph.nodes[n].get('time', 0))
    if len(b) < maxL:
        return None
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
    return slope

# ------------------------------------------------------------
# Fixed parameters (as in paper)
# ------------------------------------------------------------
CHI = 4
PA = 0.3
PG = 0.3
PB = 0.3
MAXL = 30
N_SEEDS = 8

N_VALS = [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]

results = defaultdict(list)   # key = N, value = list of slopes

print(f"🔬 Finite‑size sweep: {len(N_VALS)} sizes × {N_SEEDS} seeds = {len(N_VALS)*N_SEEDS} networks")
start = time.time()

for N in N_VALS:
    print(f"\n--- N = {N} ---")
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        net = CTN(chi=CHI)
        step = 0
        while len(net.graph) < N:
            step += 1
            # --- Causal growth ---
            for node in list(net.graph.nodes):
                if len(net.graph) >= N: break
                if np.random.rand() < PG:
                    parents = [node]
                    if PA > 0 and net._ancestors[node] and np.random.rand() < PA:
                        parents.append(np.random.choice(list(net._ancestors[node])))
                    nid = len(net.graph.nodes)
                    net._add_node(nid, parents, step)
                    for p in parents:
                        net._add_edge(p, nid)
                    if len(net.graph) >= N: break
            # --- Dynamical boundary bonds (p_b = 0.3) ---
            if np.random.rand() < PB:
                b = net.boundary()
                if len(b) > 1:
                    u, v = np.random.choice(b, 2, replace=False)
                    if not net.graph.has_edge(u, v):
                        net._add_edge(u, v)
            if step % 20 == 0:
                print(f"   Seed {seed+1}/{N_SEEDS} | Step {step:4d} | Nodes = {len(net.graph):6d} / {N}", end='\r')
        slope = compute_slope(net, MAXL)
        if slope is not None:
            results[N].append(slope)
        print(f"   Seed {seed+1}/{N_SEEDS} | Done | slope = {slope:.3f}")

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
rows = []
for N in N_VALS:
    slopes = results[N]
    mean_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    rows.append([N, mean_slope, std_slope, len(slopes)])

df = pd.DataFrame(rows, columns=['N', 'mean_slope', 'std_slope', 'n_seeds'])
df.to_csv('finite_size_slopes.csv', index=False)

print("\n" + "="*60)
print("✅ Finite‑size sweep complete.")
print("📁 Saved to finite_size_slopes.csv")
print("\nResults preview:")
print(df.to_string(index=False))

# ------------------------------------------------------------
# (Optional) Quick plot – matches your Figure 4
# ------------------------------------------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.errorbar(df['N'], df['mean_slope'], yerr=df['std_slope'],
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6)
plt.xlabel('Network size N')
plt.ylabel('Entanglement slope dS/dL')
plt.title(f'Finite‑size scaling (p_a={PA}, χ={CHI}, p_g={PG})')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('finite_size_plot.pdf')
plt.show()
print("📊 Quick plot saved as finite_size_plot.pdf")
