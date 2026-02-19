

Causal Graph Models for de Sitter Holography

https://zenodo.org/badge/DOI/10.5281/zenodo.18626027.svg
https://zenodo.org/badge/DOI/10.5281/zenodo.18647441.svg
https://img.shields.io/badge/License-MIT-yellow.svg

This repository contains the complete simulation code and analysis pipeline for two companion papers investigating holographic entanglement in stochastic causal graphs.

---

📄 Papers

Paper I: Classical Geometry

"Ensemble Statistics of Linear Entanglement Scaling in a Causal Graph Model for de Sitter Holography"

· Grows causal graphs up to 60,000 nodes
· Demonstrates linear scaling of minimal cuts with boundary interval length
· Establishes classical geometry of the model family
· Key parameters: p_a (ancestor bias), p_g (growth probability), p_b (boundary-linking probability)

Paper II: Quantum Structure

"From Hubs to Holography: How Graph Regularity Unlocks Entanglement"

· Embeds random and perfect tensors (χ = 2–6) into causal graphs
· Shows that capping high-degree nodes ("hubs") increases entanglement by a factor of 2.6
· Demonstrates that graph regularity, not perfect tensors, is the dominant factor
· Includes full parameter sweeps over p_a, χ, and p_g

---

🚀 Getting Started

Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:

· numpy, networkx — graph generation and analysis
· quimb — tensor network construction and perfect tensor support
· matplotlib — visualization
· tqdm — progress tracking

Repository Structure

```
├── src/               # Core CTN class and utilities
├── scripts/           # Parameter sweeps and ensemble runs
├── paper2/            # Perfect tensor simulations (χ scaling)
├── data/              # Generated results (CSV files)
├── figures/           # Publication-ready plots
├── docs/              # Supplementary documentation
├── requirements.txt   # Python dependencies
└── LICENSE            # MIT License
```

---

🔬 Reproducibility

Each paper's results can be reproduced using the scripts in scripts/:

· Paper I: finite_size_scaling.py
· Paper II: sweep_pa_chi_pg_ensemble.py

All raw data is saved to data/ in CSV format. Figures are generated automatically and stored in figures/.

DOIs for each paper are provided above. Please cite the relevant paper(s) if you use this code in your own work.

---

📊 Key Results Preview

Paper Finding
I Entanglement entropy scales linearly with boundary interval length
II Hub capping increases entanglement ratio by 2.6× (p < 10⁻¹⁵)

---

🤝 Contributing

This project is developed by an independent researcher. Feedback, bug reports, and collaborations are welcome via GitHub Issues.

---

📜 License

MIT License — free to use, modify, and distribute with attribution.

---

🧠 About

This work grew from a simple question: What makes holographic entanglement work? The answer, iturns out, is not perfect tensors — it's graph regularity. Hubs concentrate entanglement, and regularizing the graph unlocks it. These ideas connect not only to quantum gravity but to broader questions about information, networks, and emergence.

For questions or collaborations, reach out via GitHub.
