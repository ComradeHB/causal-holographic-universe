# Causal Holographic Universe

[![Paper](https://img.shields.io/badge/Paper-arXiv.YYMM.NNNNN-B3181B?logo=arXiv)](https://arxiv.org/abs/YYMM.NNNNN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

This repository contains the complete simulation code and analysis pipeline for the paper:

**"Linear Entanglement Scaling in a Causal Graph Model for de Sitter Holography"**  
Heidi Anderson (Independent Researcher), February 2026

📄 **Paper:** [arXiv:YYMM.NNNNN [gr-qc, hep-th, cond-mat.stat-mech]](https://arxiv.org/abs/YYMM.NNNNN) *(update when available)*  
🐙 **Code:** https://github.com/ComradeHB/causal-holographic-universe  
📦 **DOI:** *[Add Zenodo DOI after archiving – see Step 4]*

---

## 📋 Description

This repository implements a **causal random graph model** for de Sitter holography. The model grows directed acyclic graphs via a stochastic process with tunable **ancestor probability** `p_a`. Entanglement entropy is defined via the Ryu–Takayanagi prescription as the **minimal cut** through the undirected graph, with each edge weighted by `log χ` (bond dimension).

**Key contributions:**
- First tunable, numerically tractable causal graph model for dS/CFT
- Perfect linear scaling of boundary minimal cuts across all parameters
- Ensemble simulations with error bars and finite-size scaling
- Analytical mean-field estimate `⟨k_bulk⟩ = 1/(1-p_a)`

**Main results:**
- Slope `dS/dL` increases monotonically with `p_a`
- Slope is linear in `log χ` with coefficient `1 + p_a`
- Slope is independent of growth probability `p_g`
- Pure tree limit (`p_a = 0`) gives slope exactly `log χ` (zero variance)

---

## 📁 Repository Structure
