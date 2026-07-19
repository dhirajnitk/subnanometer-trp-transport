"""
visualize_vg.py — Show potential drop across membrane and exciton switching.
Run: python data/visualize_vg.py
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
from pdb_tools.trp_extractor import fetch_pdb, extract_trp_data
from analysis.quantum_hamiltonian_engine import build_pdb_hamiltonian

E_CHARGE = 1.602e-19; MEMBRANE = 4.0  # nm
pid, name = '6AGF', 'NaV1.4'
text = fetch_pdb(pid, cache=True)
centroids, dipoles = extract_trp_data(text)
keys = sorted(centroids.keys())
z_pos = np.array([centroids[k][2] for k in keys])
z_min, z_max = z_pos.min(), z_pos.max()
z_range = z_max - z_min

# Map z to membrane depth fraction
z_frac = (z_pos - z_min) / z_range  # 0 (extracellular side) to 1 (intracellular)

voltages = np.linspace(-80e-3, 50e-3, 100)
site_occ = np.zeros((len(keys), len(voltages)))
for vi, V in enumerate(voltages):
    site_energy = [80*i + E_CHARGE*V*z_frac[i]/1.986e-23 for i in range(len(keys))]
    H, _ = build_pdb_hamiltonian(centroids, dipoles, site_energies=site_energy)
    evals, evecs = np.linalg.eigh(H)
    psi0 = evecs[:, 0]
    site_occ[:, vi] = abs(psi0)**2

# Find the two dominant residues (one high-z, one low-z)
occ_rest = site_occ[:, 0]   # at -80 mV
occ_peak = site_occ[:, -1]   # at +50 mV
top_high = keys[np.argmax(occ_rest)]
top_low  = keys[np.argmax(occ_peak)]
idx_h = list(keys).index(top_high)
idx_l = list(keys).index(top_low)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel 1: Potential drop across membrane
ax = axes[0]
z_nm = (z_pos - z_pos.min()) / 10 + 0.5  # offset so min is 0.5 nm
for i, k in enumerate(keys):
    color = '#1f77b4'
    if k == top_high: color = 'red'
    elif k == top_low: color = 'green'
    ax.plot([0, 1], [z_nm[i], z_nm[i]], 'o-', color=color, lw=1, ms=3 if color=='#1f77b4' else 6)
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(0, 4.5)
ax.set_xlabel('Membrane (extracellular  |  intracellular)')
ax.set_ylabel('Depth (nm)')
ax.set_title(f'(a) Trp z-positions in {name}')
ax.axhline(0, color='gray', ls='--')
ax.axhline(4, color='gray', ls='--', label='Bilayer')
ax.text(0.5, -0.3, 'Extracellular', ha='center', fontsize=8, color='gray')
ax.text(0.5, 4.3, 'Intracellular', ha='center', fontsize=8, color='gray')
ax.set_xticks([0, 1]); ax.set_xticklabels(['', ''])

# Panel 2: Voltage drop profile
ax = axes[1]
for V_lbl, V_val, ls, c in [('Rest (-70mV)', -70e-3, '--', 'blue'), ('Peak (+40mV)', 40e-3, '-', 'red')]:
    V_drop = np.linspace(0, V_val*1e3, 100)
    ax.plot(V_drop, np.linspace(0, 4, 100), ls, color=c, lw=2, label=V_lbl)
ax.set_xlabel('Potential (mV)')
ax.set_ylabel('Membrane depth (nm)')
ax.set_title('(b) Voltage drop across bilayer')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.invert_yaxis()

# Panel 3: Exciton switching
ax = axes[2]
occ_h = site_occ[idx_h, :]
occ_l = site_occ[idx_l, :]
ax.plot(voltages*1e3, occ_h, 'r-', lw=2, label=f'Trp-{top_high} (z={z_pos[idx_h]:.0f}A)')
ax.plot(voltages*1e3, occ_l, 'g-', lw=2, label=f'Trp-{top_low} (z={z_pos[idx_l]:.0f}A)')
ax.axvspan(-70, 40, alpha=0.08, color='purple', label='AP window')
ax.set_xlabel('Membrane potential (mV)')
ax.set_ylabel('Exciton occupation')
ax.set_title('(c) Voltage-gated exciton switching')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

plt.tight_layout()
out = 'papers/p0_sub_tubulin/figures/fig3_voltage_gating.pdf'
plt.savefig(out, dpi=300)
print(f"Saved to {out}")

# Numerical summary
print(f"\n=== Numerical Summary ===")
print(f"Dominant at rest (-70mV):  Trp-{top_high}, z={z_pos[idx_h]:.0f}A, occ={occ_rest[idx_h]:.3f}")
print(f"Dominant at peak (+40mV):  Trp-{top_low},  z={z_pos[idx_l]:.0f}A, occ={occ_peak[idx_l]:.3f}")
print(f"Population swap amplitude: {abs(occ_h[0]-occ_l[-1]):.1%}")
