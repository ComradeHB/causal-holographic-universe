causal-holographic-universe/
│
├── README.md                 # Main project documentation (full text below)
├── LICENSE                  # MIT or Apache 2.0 (choose one)
├── CITATION.cff            # Citation metadata
├── .gitignore              # Tells Git what to ignore
├── environment.yml         # Conda environment (recommended) OR
├── requirements.txt        # Pip requirements
│
├── src/                    # Core simulation code
│   ├── causal_graph.py     # Main graph growth algorithm
│   ├── mincut.py           # Entanglement entropy calculation
│   └── utils.py           # Ancestor cache, helpers
│
├── scripts/                # Run scripts to reproduce paper results
│   ├── run_pa_sweep.py    # Figure 1 (ancestor probability)
│   ├── run_chi_sweep.py   # Figure 2 (bond dimension)
│   ├── run_pg_sweep.py    # Figure 3 (growth probability)
│   └── run_finite_size.py # Figure 4 (finite-size scaling)
│
├── figures/                # All generated figures (PDF/PNG)
│   ├── fig1_pa_slope.pdf
│   ├── fig2_logchi_slope.pdf
│   ├── fig3_pg_flat.pdf
│   └── fig4_finite_size.pdf
│
├── data/                   # Sample data (not raw simulation outputs)
│   └── README.md          # Explanation of where full data is stored
│
└── docs/                   # Supplementary documentation
    └── methods.md         # Extended algorithm description (optional)
