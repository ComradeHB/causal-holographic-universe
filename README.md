
# Causal Graph Models for de Sitter Holography

[![Paper 1](https://img.shields.io/badge/Paper 1-arXiv.YYMM.NNNNN-B3181B?logo=arXiv)](https://arxiv.org/abs/YYMM.NNNNN)
[![Paper 2](https://img.shields.io/badge/Paper 2-arXiv.YYMM.NNNNN-B3181B?logo=arXiv)](https://arxiv.org/abs/YYMM.NNNNN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains the complete simulation code and analysis pipeline for two companion papers:

1. **Paper 1 (Classical)** – “Ensemble Statistics of Linear Entanglement Scaling in a Causal Graph Model for de Sitter Holography”  
2. **Paper 2 (Quantum)** – “From Hubs to Holography: How Graph Regularity Unlocks Entanglement”  

Both papers introduce and explore a **causal random graph model** for de Sitter holography, first classically and then with quantum tensor networks.

- 📄 **Paper 1:** [arXiv:YYMM.NNNNN [gr-qc, hep-th]](https://arxiv.org/abs/YYMM.NNNNN) *(update when available)*  
- 📄 **Paper 2:** [arXiv:YYMM.NNNNN [gr-qc, hep-th]](https://arxiv.org/abs/YYMM.NNNNN) *(update when available)*  
- 🐙 **Code:** https://github.com/ComradeHB/causal-holographic-universe  
- 🏛️ **DOI (Paper 1):** [10.5281/zenodo.18626027](https://doi.org/10.5281/zenodo.XXXXXXX)  
- 🏛️ **DOI (Paper 2):** [10.5281/zenodo.18647441](https://doi.org/10.5281/zenodo.XXXXXXX) *(to be assigned)*

---

## 📋 Repository Structure

```

causal-holographic-universe/
├── paper1/                       # Paper 1: classical model
│   ├── data/                      # Summary CSV files
│   ├── figures/                    # Publication‑ready PDFs
│   ├── scripts/                    # Parameter sweep scripts
│   ├── src/                        # Core source code
│   ├── README.md                   # Paper 1 details (this file)
│   └── ... (other Paper 1 files)
├── paper2/                       # Paper 2: quantum extension
│   ├── Main.tex                    # LaTeX source
│   ├── refs.bib                    # Bibliography
│   ├── fig1_hub_fraction.png       # Figure 1
│   ├── fig2_chi_scaling.png        # Figure 2
│   └── README.md                   # Paper 2 details (to be added)
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md                     # This file (overview)
└── requirements.txt              # Python dependencies

```

---

## 📄 Paper 1: Classical Causal Graph Model

**Title:** Ensemble Statistics of Linear Entanglement Scaling in a Causal Graph Model for de Sitter Holography  
**Author:** Heidi Anderson  
**Status:** Submitted to Physical Review E (preprint available on arXiv)

### 🔑 Key contributions
- First tunable, numerically tractable causal graph model for dS/CFT
- **Ensemble simulations** with error bars and finite‑size scaling (10 seeds per config, 8 for finite‑size)
- Perfect **linear scaling** of boundary minimal cuts across all parameters
- Analytical mean‑field estimate \(\langle k_{\text{bulk}} \rangle = (1-p_a)^{-1}\)

### 📊 Main results
- Slope \(dS/dL\) increases monotonically with \(p_a\)
- Slope is linear in \(\log \chi\) with coefficient \(1 + p_a\)
- Slope is independent of growth probability \(p_g\)
- Pure tree limit (\(p_a = 0\)) gives slope exactly \(\log \chi\) (zero variance)

For full details, see the [`paper1/`](paper1/) folder.

---

## 📄 Paper 2: Quantum Extension – Graph Regularity vs. Perfect Tensors

**Title:** From Hubs to Holography: How Graph Regularity Unlocks Entanglement  
**Author:** Heidi Anderson  
**Status:** In preparation (to be submitted to Physical Review D)

### 🔑 Key contributions
- **Capping node degree** (max degree 6) increases the entanglement ratio \(R = S_q(1)/C_{\text{min}}(1)\) from ~0.05 to 0.138 at \(\chi=4\) – a 2.6× increase (Welch’s t‑test, \(p<10^{-15}\)).
- **Perfect tensors** (AME(4,4) on degree‑4 nodes) provide no statistically significant additional benefit (\(p=0.22\)).
- **Hub analysis** on uncapped graphs (2000 nodes) shows that every boundary interval of length \(L \ge 2\) has a hub (degree ≥8) in its causal past, explaining why capping works.
- **Bond dimension scaling** reveals a decrease in \(R\) for \(\chi \ge 5\), which control experiments show is dominated by finite‑size effects and compression artifacts – a cautionary tale for small‑graph studies.

For source files (LaTeX, figures, bibliography), see the [`paper2/`](paper2/) folder.

---

## ⚙️ Requirements & Installation

This project uses **Python 3.8+** and requires the following packages:

```

numpy
networkx
matplotlib
scipy
pandas

```

### 🔹 Using pip

```bash
git clone https://github.com/ComradeHB/causal-holographic-universe.git
cd causal-holographic-universe
pip install -r requirements.txt
```

🔹 Using Conda (recommended for reproducibility)

```bash
conda env create -f environment.yml
conda activate causal-holography
```

(If you don't have environment.yml, you can generate it with conda env export --from-history > environment.yml.)

---

🚀 Reproducing the Paper Results

📈 Paper 1 – Figures 1–3 (Ensemble sweeps)

```bash
python paper1/scripts/sweep_pa_chi_pg_ensemble.py
```

📉 Paper 1 – Figure 4 (Finite‑size scaling)

```bash
python paper1/scripts/sweep_finite.py
```

🖼️ Paper 1 – Generating the PDF figures

Once the CSV files are present (they are included in the repository), run:

```bash
python paper1/figures/generate_figures.py
```

🔬 Paper 2 – Quantum simulations

The quantum simulations were performed using the Python library Quimb and custom scripts (available upon request). The final data are embedded in the LaTeX source and figures. To reproduce the exact contraction results, please contact the author.

---

🧠 Algorithm Overview (Common to Both Papers)

1. Causal growth – Start from a causal diamond, then stochastically attach child nodes with probability p_g. With probability p_a, also attach a random ancestor (ancestor cache gives O(1) lookup).
2. Boundary bonds – With probability p_b = 0.3, add a spacelike edge between two random boundary nodes; the boundary self‑organizes into a 1D chain.
3. Entanglement entropy – For a boundary interval A, compute the minimal cut through the undirected graph. Each crossing edge contributes \log \chi to the entropy. For Paper 2, the quantum entropy is obtained by contracting random tensor networks on the same graphs.

---

🧪 Extending the Model

This repository is designed to be easily extended. Common modifications:

· Perfect tensors – Replace random isometries with perfect tensors (HaPPY‑style).
· Higher bond dimensions – Increase \chi (tested up to 6 in Paper 1; up to 8 in Paper 2).
· Alternative boundary dynamics – Adjust p_b or implement deterministic boundary chains.
· 2D boundaries – Modify the boundary‑ordering logic in src/utils.py.

Pull requests and issues are welcome!

---

🏷️ Citation

If you use this code or ideas from either paper in your own research, please cite the appropriate paper(s):

Paper 1

```bibtex
@article{Anderson2026ensemble,
  title     = {Ensemble Statistics of Linear Entanglement Scaling in a Causal Graph Model for de Sitter Holography},
  author    = {Anderson, Heidi},
  journal   = {arXiv preprint arXiv:YYMM.NNNNN},
  year      = {2026},
  note      = {Submitted to Physical Review E},
  url       = {https://arxiv.org/abs/YYMM.NNNNN},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

Paper 2

```bibtex
@article{Anderson2026hubs,
  title     = {From Hubs to Holography: How Graph Regularity Unlocks Entanglement},
  author    = {Anderson, Heidi},
  journal   = {arXiv preprint arXiv:YYMM.NNNNN},
  year      = {2026},
  note      = {In preparation},
  url       = {https://arxiv.org/abs/YYMM.NNNNN},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

A CITATION.cff file is included in this repository – GitHub will automatically show a “Cite this repository” button.

---

📄 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

🙏 Acknowledgements

· The open‑source scientific Python community (NumPy, NetworkX, Matplotlib, SciPy, Pandas, Quimb).
· The anonymous reviewers for their constructive feedback.
· This research did not receive any specific grant from funding agencies in the public, commercial, or not‑for‑profit sectors.

---

Maintained by Heidi Anderson
📧 heidilanderson0@gmail.com
🐙 https://github.com/ComradeHB
🗓️ Last updated: February 15, 2026

```

**What I changed / added:**

- Added a second paper badge and DOI placeholder.
- Restructured the repository layout to include `paper1/` and `paper2/`.
- Moved all Paper 1 details under a dedicated section, keeping the original content.
- Added a new section for Paper 2 with key results, contributions, and a pointer to the `paper2/` folder.
- Updated the **Reproducing results** section to distinguish Paper 1 scripts (now under `paper1/`) and noted that Paper 2 simulation code is available upon request (since the quantum code is not yet in the repo).
- Kept the algorithm overview generic.
- Updated citation section to include both papers.
- Updated acknowledgements to include the library and the library mention.

Once you upload the Paper 2 source files into a `paper2/` folder, this README will be complete. Good luck with your submissions!
1. **Paper 1 (Classical)** – “Ensemble Statistics of Linear Entanglement Scaling in a Causal Graph Model for de Sitter Holography”  
2. **Paper 2 (Quantum)** – “From Hubs to Holography: How Graph Regularity Unlocks Entanglement”  

Both papers introduce and explore a **causal random graph model** for de Sitter holography, first classically and then with quantum tensor networks.

- 📄 **Paper 1:** [arXiv:YYMM.NNNNN [gr-qc, hep-th]](https://arxiv.org/abs/YYMM.NNNNN) *(update when available)*  
- 📄 **Paper 2:** [arXiv:YYMM.NNNNN [gr-qc, hep-th]](https://arxiv.org/abs/YYMM.NNNNN) *(update when available)*  
- 🐙 **Code:** https://github.com/ComradeHB/causal-holographic-universe  
- 🏛️ **DOI (Paper 1):** [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)  
- 🏛️ **DOI (Paper 2):** [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX) *(to be assigned)*

---

## 📋 Repository Structure

---

## 📁 Repository Structure

```

causal-holographic-universe/
├── data/                         # Summary CSV files with means and std devs
│   ├── slope_summary.csv
│   └── finite_size_slopes.csv
├── figures/                      # Publication‑ready PDFs + figure generation script
│   ├── generate_figures.py
│   ├── slope_vs_pa_ensemble (4) (1).pdf
│   ├── slope_vs_logchi_ensemble (4) (2).pdf
│   ├── slope_vs_pg_ensemble (4) (1).pdf
│   └── Figure10_finite_size_scaling (3) (2).pdf
├── scripts/                      # Parameter sweep scripts
│   ├── sweep_pa_chi_pg_ensemble.py
│   └── sweep_finite.py
├── src/                          # Core source code
│   ├── causal_graph.py
│   ├── mincut.py
│   └── utils.py
├── .gitignore                   # Python gitignore
├── CITATION.cff                # Citation metadata
├── LICENSE                     # MIT License
├── README.md                   # This file
└── requirements.txt           # Python dependencies

```

---

## ⚙️ Requirements & Installation

This project uses **Python 3.8+** and requires the following packages:

```

numpy
networkx
matplotlib
scipy
pandas

```

### 🔹 Using pip

```bash
git clone https://github.com/ComradeHB/causal-holographic-universe.git
cd causal-holographic-universe
pip install -r requirements.txt
```

🔹 Using Conda (recommended for reproducibility)

```bash
conda env create -f environment.yml
conda activate causal-holography
```

(If you don't have environment.yml, you can generate it with conda env export --from-history > environment.yml.)

---

🚀 Reproducing the Paper Results

📈 Figures 1–3 (Ensemble sweeps)

To reproduce the main results for p_a, \chi, and p_g (10 seeds, 20k nodes):

```bash
python scripts/sweep_pa_chi_pg_ensemble.py
```

This will generate the raw slope data and save it as slope_summary.csv in the data/ folder.

📉 Figure 4 (Finite‑size scaling)

To reproduce the finite‑size scaling sweep (8 seeds, N = 20k–60k):

```bash
python scripts/sweep_finite.py
```

This will generate finite_size_slopes.csv in the data/ folder.

🖼️ Generating the PDF figures (instant)

Once the CSV files are present (they are already included in this repository), run:

```bash
python figures/generate_figures.py
```

All four publication‑ready PDFs will be created in the figures/ folder.

---

📊 Results Summary

p_a Mean slope (\chi=4) Std dev Predicted (1+p_a)\log 4
0.0 1.386 0.000 1.386
0.3 1.730 0.042 1.802
0.9 2.545 0.051 2.634

\chi \log\chi Mean slope (p_a=0.3) \langle k_{\text{bulk}}\rangle
2 0.693 0.902 ± 0.047 1.302
3 1.099 1.429 ± 0.062 1.301
4 1.386 1.730 ± 0.042 1.249
6 1.792 2.348 ± 0.078 1.310

N (size) Mean slope (p_a=0.3, \chi=4, p_g=0.3)
20000 1.760
25000 1.780
30000 1.795
35000 1.810
40000 1.830
45000 1.820
50000 1.800
55000 1.790
60000 1.785

---

🧠 Algorithm Overview

1. Causal growth

· Initialize with a causal diamond (3 nodes, 1 spacelike edge).
· At each time step, iterate over existing nodes.
· With probability p_g, attach a child node to the current node.
· With probability p_a, attach an additional edge to a random ancestor (ancestor cache gives O(1) lookup).

2. Dynamical boundary bonds

· With probability p_b = 0.3, add a spacelike (undirected) edge between two random boundary nodes.
· The boundary spontaneously self‑organizes into a one‑dimensional chain.

3. Entanglement entropy (minimal cut)

· Convert the graph to undirected form.
· For a boundary interval A, count edges crossing between A and its complement.
· Each crossing edge contributes \log \chi to the entropy.
· For small N, exact min‑cut via Edmonds–Karp (NetworkX) is used for verification.

---

🧪 Extending the Model

This repository is designed to be easily extended. Common modifications:

· Perfect tensors: Replace the conceptual random isometries with perfect tensors (HaPPY‑style).
· Higher bond dimensions: Increase \chi (tested up to 6).
· Alternative boundary dynamics: Adjust p_b or implement deterministic boundary chains.
· 2D boundaries: Modify the boundary‑ordering logic in src/utils.py.

Pull requests and issues are welcome!

---

🏷️ Citation

If you use this code or ideas from the paper in your own research, please cite:

```bibtex
@article{Anderson2026ensemble,
  title     = {Ensemble Statistics of Linear Entanglement Scaling in a Causal Graph Model for de Sitter Holography},
  author    = {Anderson, Heidi},
  journal   = {arXiv preprint arXiv:YYMM.NNNNN},
  year      = {2026},
  note      = {Submitted to Physical Review E},
  url       = {https://arxiv.org/abs/YYMM.NNNNN},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

A.

---

📄 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

🙏 Acknowledgements

· The open‑source scientific Python community (NumPy, NetworkX, Matplotlib, SciPy, Pandas).
· The anonymous reviewers for their constructive feedback.
· This research did not receive any specific grant from funding agencies in the public, commercial, or not‑for‑profit sectors.

---

Maintained by Heidi Anderson
📧 heidilanderson0@gmail.com
🐙 https://github.com/ComradeHB
🗓️ Last updated: February 12, 2026

```
