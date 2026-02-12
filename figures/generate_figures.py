import matplotlib.pyplot as plt
import numpy as np
import zipfile
import os
from google.colab import files

# ------------------------------------------------------------
# FIGURE 1: Slope vs p_a (χ=4, p_g=0.3, 10 seeds)
# ------------------------------------------------------------
p_vals = [0.0, 0.3, 0.9]
means = [1.386, 1.730, 2.545]
stds  = [0.000, 0.042, 0.051]

plt.figure(figsize=(5,4))
plt.errorbar(p_vals, means, yerr=stds, fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6, color='blue')
plt.xlabel(r'Ancestor probability $p_a$', fontsize=12)
plt.ylabel(r'Linear slope $dS/dL$', fontsize=12)
plt.title(r'Slope vs $p_a$ ($\chi=4$, $p_g=0.3$, 10 seeds)', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('slope_vs_pa_ensemble.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# FIGURE 2: Slope vs log χ (p_a=0.3, p_g=0.3, 10 seeds)
# ------------------------------------------------------------
chi_vals = [2,3,4,6]
logchi = np.log(chi_vals)
means = [0.902, 1.429, 1.730, 2.348]
stds  = [0.047, 0.062, 0.042, 0.078]

plt.figure(figsize=(5,4))
plt.errorbar(logchi, means, yerr=stds, fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6, color='red')
plt.xlabel(r'$\log \chi$', fontsize=12)
plt.ylabel(r'Linear slope $dS/dL$', fontsize=12)
plt.title(r'Slope vs $\log\chi$ ($p_a=0.3$, $p_g=0.3$, 10 seeds)', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('slope_vs_logchi_ensemble.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# FIGURE 3: Slope vs p_g (p_a=0.3, χ=4, 10 seeds)
# ------------------------------------------------------------
pg_vals = [0.2, 0.3, 0.4]
means = [1.727, 1.730, 1.728]
stds  = [0.044, 0.042, 0.048]

plt.figure(figsize=(5,4))
plt.errorbar(pg_vals, means, yerr=stds, fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=6, color='green')
plt.xlabel(r'Growth probability $p_g$', fontsize=12)
plt.ylabel(r'Linear slope $dS/dL$', fontsize=12)
plt.title(r'Slope vs $p_g$ ($p_a=0.3$, $\chi=4$, 10 seeds)', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('slope_vs_pg_ensemble.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# FIGURE 4: Finite-size scaling (from your table)
# ------------------------------------------------------------
N = [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]
slope = [1.760, 1.780, 1.795, 1.810, 1.830, 1.820, 1.800, 1.790, 1.785]

plt.figure(figsize=(5,4))
plt.plot(N, slope, 'o-', linewidth=2, markersize=6, color='purple')
plt.xlabel(r'Network size $N$', fontsize=12)
plt.ylabel(r'Entanglement slope $dS/dL$', fontsize=12)
plt.title(r'Finite-size scaling ($p_a=0.3$, $\chi=4$, $p_g=0.3$)', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Figure10_finite_size_scaling.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------
# BUNDLE EVERYTHING INTO A ZIP AND FORCE DOWNLOAD
# ------------------------------------------------------------
with zipfile.ZipFile('figures.zip', 'w') as z:
    for f in ['slope_vs_pa_ensemble.pdf', 'slope_vs_logchi_ensemble.pdf',
                  'slope_vs_pg_ensemble.pdf', 'Figure10_finite_size_scaling.pdf']:
                          z.write(f)

                          print("✅ All four figures are now in 'figures.zip'.")
                          print("📥 Downloading now...")
                          files.download('figures.zip')
                          print("🎯 Done. Unzip the file on your phone – the PDFs inside are the real ones.")
