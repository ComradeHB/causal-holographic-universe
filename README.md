# Causal Quantum Networks: A Discrete Foundation for Spacetime

[![arXiv](https://img.shields.io/badge/arXiv-2502.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2502.XXXXX)
[![GitHub license](https://img.shields.io/github/license/ComradeHB/causal-holographic-universe)](LICENSE)

This repository contains the complete simulation code for:

**Anderson, H. (2026). *Causal Quantum Networks: A Discrete Foundation for Spacetime with a 16 Myr Cosmological Signature.** *

The code implements causal quantum networks (CQNs), a discrete, finitary model of spacetime from which general relativity, quantum statistics, and a testable 16 Myr cosmological timescale emerge. All results in the paper are fully reproducible.

---

## 📁 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| [`causal_quantum_network.py`](./causal_quantum_network.py) | Core `CausalQuantumNetwork` class and methods (CQN definition, evolution, entanglement entropy, boundary CFT diagnostics, emergent metric extraction). |
| [`holographic_code_construction.py`](./holographic_code_construction.py) | Explicit hypergraph stabilizer code (HaPPY‑like) and Ryu–Takayanagi formula test – Appendix C. |
| [`newtons_law_emergence.py`](./newtons_law_emergence.py) | Numerical verification of inverse‑square law in emergent 3D spacetime – Section 2.4 / Theorem 2.4. |
| [`Requirements.txt`](./Requirements.txt) | Python dependencies (NumPy, NetworkX, SciPy, scikit‑learn). |
| [`notebooks/`](./notebooks) | Jupyter notebooks with interactive demonstrations and figure generation. |
| [`LICENSE`](./LICENSE) | MIT License. |
| [`.gitignore`](./.gitignore) | Python‑specific ignore rules. |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ComradeHB/causal-holographic-universe
cd causal-holographic-universe
2. Install dependencies
pip install -r Requirements.txt
3. Run the core CQN simulation
python causal_quantum_network.py
4. Reproduce the gravity law test
python newtons_law_emergence.py
5. Explore the holographic code construction
python holographic_code_construction.py
All scripts are self‑contained and produce output matching the tables and figures in the paper.

---

📓 Jupyter Notebooks

The notebooks/ folder contains a fully documented notebook that reproduces every simulation and plot from the paper.

Run it interactively in Google Colab:
https://colab.research.google.com/assets/colab-badge.svg

---

📊 Reproducing Paper Results

· Table 4.1 (central charge): Run the central charge extraction routine in causal_quantum_network.py (see notebook).
· Table 6.2 (fermion masses): Analytic calculation; derivation provided in paper.
· Table 7.1 (16 Myr timescale): Run cosmology.py (to be added) or see notebook.
· Table 8.1 (LHC resonances): Phenomenological estimates; code for cross‑section calculations is available on request.

All numerical experiments use fixed random seeds (included in the notebook) for full reproducibility.

---

📜 Citation

If you use this code or ideas from the paper, please cite:

```bibtex
@article{Anderson2026,
  author    = {Anderson, Heidi},
  title     = {Causal Quantum Networks: A Discrete Foundation for Spacetime with a 16 Myr Cosmological Signature},
  journal   = {arXiv preprint},
  year      = {2026},
  volume    = {arXiv:2502.XXXXX}
}
```

---

🧠 Contact / Feedback

· Author: Heidi Anderson
· Email: hla254@gmail.com
· GitHub: @ComradeHB

Questions, suggestions, and collaborations are very welcome.

---

📜 License

Distributed under the MIT License. See LICENSE for more information.

```
