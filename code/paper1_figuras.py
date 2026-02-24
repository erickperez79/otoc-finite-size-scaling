#!/usr/bin/env python3
"""
============================================================
PAPER 1 — GENERACIÓN DE FIGURAS
============================================================
Proyecto Kaelion (KB)
Ejecutar en Google Colab. No requiere instalar nada extra.

Genera 4 figuras publicables:
  Fig 1: C(d) vs d — IBM vs Exacta, todos los modelos N=4
  Fig 2: Escalamiento con N — KI N=4,8,12,20
  Fig 3: Tabla visual de Ω y clasificación
  Fig 4: Noise floor vs 1/2^N
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Configuración global
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14]
C0 = 0.5

# ============================================================
# DATOS (verificados, 3 ejecuciones independientes concordantes)
# ============================================================

# --- Simulación exacta ---
exact = {
    "KI N=4":  [0.014444, 0.013043, 0.048258, 0.242499, 0.076539, 0.019037, 0.088046, 0.000181, 0.024876, 0.213425, 0.006876],
    "KI N=8":  [0.014444, 0.013043, 0.000015, 0.000001, 0.000024, 0.001058, 0.001217, 0.006184, 0.009900, 0.000143, 0.000836],
    "KI N=12": [0.014444, 0.013043, 0.000002, 0.000000, 0.000000, 0.000001, 0.000003, 0.000001, 0.000283, 0.000004, 0.003369],
    "KI N=20": [0.014444, 0.013043, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    "Integrable": [0.000000, 0.500000, 0.500000, 0.500000, 0.000000, 0.500000, 0.000000, 0.500000, 0.500000, 0.500000, 0.500000],
    "Floquet": [0.000232, 0.341751, 0.260664, 0.250834, 0.199813, 0.376314, 0.115672, 0.039800, 0.062062, 0.151331, 0.163113],
}

# --- IBM Quantum (ibm_marrakesh, 14 feb 2026, 5 runs cada uno) ---
ibm = {
    "KI N=4": {
        "run1": [0.016357, 0.039307, 0.052002, 0.199219, 0.065430, 0.020996, 0.102051, 0.024902, 0.043701, 0.115723, 0.047119],
        "run2": [0.018066, 0.047607, 0.048096, 0.189941, 0.075684, 0.043945, 0.086182, 0.028320, 0.039307, 0.079834, 0.047119],
        "run3": [0.022705, 0.042480, 0.056152, 0.177734, 0.063477, 0.049072, 0.072266, 0.030273, 0.076416, 0.101562, 0.056396],
        "run4": [0.024170, 0.041260, 0.054932, 0.200439, 0.078369, 0.047852, 0.092285, 0.033447, 0.047607, 0.099854, 0.050049],
        "run5": [0.017334, 0.045166, 0.049072, 0.189697, 0.067139, 0.036133, 0.083008, 0.029297, 0.050781, 0.094238, 0.049805],
    },
    "KI N=8": {
        "run1": [0.031494, 0.024902, 0.006592, 0.004395, 0.004150, 0.004883, 0.006104, 0.003662, 0.006348, 0.004395, 0.004395],
        "run2": [0.015381, 0.019287, 0.006348, 0.004639, 0.003418, 0.004639, 0.008545, 0.009277, 0.003662, 0.004639, 0.002686],
        "run3": [0.024902, 0.016602, 0.002686, 0.003906, 0.004150, 0.005615, 0.006592, 0.004639, 0.005615, 0.003174, 0.005371],
        "run4": [0.017822, 0.026855, 0.006104, 0.002441, 0.004639, 0.003906, 0.005371, 0.006592, 0.005615, 0.003418, 0.007324],
        "run5": [0.021240, 0.010010, 0.001465, 0.003418, 0.003906, 0.004883, 0.004639, 0.003906, 0.006836, 0.004639, 0.004883],
    },
    "KI N=12": {
        "run1": [0.020996, 0.014893, 0.001221, 0.000244, 0.000244, 0.000488, 0.000000, 0.001221, 0.000488, 0.000732, 0.000244],
        "run2": [0.020752, 0.014893, 0.002686, 0.000488, 0.000244, 0.000732, 0.000488, 0.000000, 0.000000, 0.000244, 0.000732],
        "run3": [0.024170, 0.011963, 0.002686, 0.000732, 0.000488, 0.000488, 0.000732, 0.001221, 0.000488, 0.000244, 0.001221],
        "run4": [0.019287, 0.010742, 0.002930, 0.000488, 0.000244, 0.000244, 0.000244, 0.000000, 0.000488, 0.000732, 0.000000],
        "run5": [0.025146, 0.011230, 0.002930, 0.000977, 0.000244, 0.000732, 0.000732, 0.000488, 0.000488, 0.000000, 0.000000],
    },
    "KI N=20": {
        "run1": [0.020264, 0.002197, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        "run2": [0.019531, 0.000977, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        "run3": [0.021484, 0.002930, 0.000244, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        "run4": [0.018311, 0.001709, 0.000732, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
        "run5": [0.019043, 0.003174, 0.000244, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    },
    "Integrable": {
        "run1": [0.000000, 0.496582, 0.500732, 0.499268, 0.003662, 0.501953, 0.004150, 0.506348, 0.493408, 0.495117, 0.490234],
        "run2": [0.000000, 0.509521, 0.491455, 0.507568, 0.002686, 0.507568, 0.003174, 0.488525, 0.496582, 0.503174, 0.496826],
        "run3": [0.000000, 0.503418, 0.495117, 0.500244, 0.004639, 0.493896, 0.003418, 0.510986, 0.493164, 0.497070, 0.491943],
        "run4": [0.000000, 0.488770, 0.498291, 0.503174, 0.004395, 0.492432, 0.002930, 0.485596, 0.500977, 0.501465, 0.491699],
        "run5": [0.000000, 0.496338, 0.506836, 0.497803, 0.003906, 0.490723, 0.004639, 0.493896, 0.478516, 0.507080, 0.517822],
    },
    "Floquet": {
        "run1": [0.011230, 0.299561, 0.222412, 0.200928, 0.138916, 0.265381, 0.083252, 0.051758, 0.063721, 0.080811, 0.104980],
        "run2": [0.006592, 0.286133, 0.217529, 0.174072, 0.144775, 0.233154, 0.104004, 0.054932, 0.046631, 0.099121, 0.080811],
        "run3": [0.008545, 0.274902, 0.200684, 0.167236, 0.162354, 0.221680, 0.106445, 0.064697, 0.078613, 0.096436, 0.085938],
        "run4": [0.010010, 0.286865, 0.211914, 0.195068, 0.145752, 0.266357, 0.091309, 0.057861, 0.066895, 0.082275, 0.075439],
        "run5": [0.013672, 0.285889, 0.187500, 0.198975, 0.162354, 0.205566, 0.091797, 0.062500, 0.060547, 0.095215, 0.077148],
    },
}

# SYK IBM (9 seeds)
syk_ibm = {
    1000: [0.397217, 0.256104, 0.320801, 0.294434, 0.053711, 0.059326, 0.044678, 0.049316, 0.049316, 0.057617, 0.046143],
    1137: [0.335693, 0.018555, 0.040039, 0.037109, 0.036133, 0.037842, 0.064209, 0.044434, 0.041016, 0.098633, 0.059814],
    1274: [0.175537, 0.175537, 0.047119, 0.179688, 0.071045, 0.097168, 0.123047, 0.064209, 0.049316, 0.044434, 0.052734],
    1411: [0.140869, 0.108154, 0.188721, 0.084229, 0.095215, 0.148926, 0.054932, 0.066650, 0.062256, 0.040771, 0.067383],
    1548: [0.293213, 0.030029, 0.021484, 0.054199, 0.061035, 0.085205, 0.034180, 0.052002, 0.075684, 0.052246, 0.045410],
    1685: [0.324951, 0.280029, 0.029297, 0.064209, 0.068848, 0.075439, 0.095947, 0.051025, 0.071289, 0.102783, 0.049072],
    1822: [0.271973, 0.037842, 0.015869, 0.069580, 0.039062, 0.041260, 0.077637, 0.041016, 0.063477, 0.072510, 0.048584],
    1959: [0.086182, 0.179688, 0.163330, 0.033203, 0.105469, 0.058838, 0.043457, 0.155518, 0.062988, 0.038818, 0.048584],
    2096: [0.033691, 0.051758, 0.053711, 0.052002, 0.052979, 0.054199, 0.033936, 0.040283, 0.052490, 0.039795, 0.085693],
}


def get_ibm_stats(runs_dict):
    """Return mean and std over runs."""
    arrays = np.stack([np.array(v) for v in runs_dict.values()])
    return np.mean(arrays, axis=0), np.std(arrays, axis=0, ddof=1)


# Colors
C_EXACT = '#1b1b1b'
C_KI = '#d62728'
C_INT = '#2ca02c'
C_FLOQ = '#ff7f0e'
C_SYK = '#9467bd'
C_IBM = '#1f77b4'

# ============================================================
# FIGURA 1: C(d) vs d — Todos los modelos N=4
# ============================================================
print("Generando Figura 1...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Figure 1: OTOC C(d) — Exact simulation vs IBM Quantum (ibm_marrakesh)',
             fontsize=14, fontweight='bold', y=0.98)

models_fig1 = [
    ("(a) Kicked Ising N=4", "KI N=4", C_KI),
    ("(b) Integrable N=4", "Integrable", C_INT),
    ("(c) Floquet N=4", "Floquet", C_FLOQ),
    ("(d) SYK N=4 (9 disorder realizations)", None, C_SYK),
]

for idx, (title, key, color) in enumerate(models_fig1):
    ax = axes[idx // 2][idx % 2]

    if key is not None:
        # Exact
        ax.plot(DEPTHS, exact[key], 'o-', color=C_EXACT, markersize=5,
                linewidth=1.5, label='Exact (statevector)', zorder=3)
        # IBM
        ibm_mean, ibm_std = get_ibm_stats(ibm[key])
        ax.errorbar(DEPTHS, ibm_mean, yerr=ibm_std, fmt='s', color=color,
                    markersize=5, capsize=3, linewidth=1.2,
                    label=f'IBM (5 runs)', zorder=2)
    else:
        # SYK: individual seeds + mean
        syk_arrays = np.stack([np.array(v) for v in syk_ibm.values()])
        syk_mean = np.mean(syk_arrays, axis=0)
        syk_std = np.std(syk_arrays, axis=0, ddof=1)

        for i, (seed, vals) in enumerate(syk_ibm.items()):
            ax.plot(DEPTHS, vals, 'o-', color=color, alpha=0.15, markersize=2,
                    linewidth=0.5, label='Individual seeds' if i == 0 else None)

        ax.errorbar(DEPTHS, syk_mean, yerr=syk_std, fmt='s-', color='black',
                    markersize=5, capsize=3, linewidth=1.5,
                    label='Disorder average (9 seeds)', zorder=3)

    ax.set_xlabel('Circuit depth d')
    ax.set_ylabel('C(d) = |⟨0|ψ⟩|²')
    ax.set_title(title)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.02, max(0.55, ax.get_ylim()[1]))
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('fig1_otoc_all_models.png', dpi=300, bbox_inches='tight')
plt.savefig('fig1_otoc_all_models.pdf', bbox_inches='tight')
print("  → fig1_otoc_all_models.png/pdf ✓")


# ============================================================
# FIGURA 2: Escalamiento con N — KI solamente
# ============================================================
print("Generando Figura 2...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle('Figure 2: Kicked Ising scaling with system size N',
             fontsize=14, fontweight='bold')

# Panel (a): C(d) para cada N
ax = axes[0]
colors_n = {4: '#d62728', 8: '#ff7f0e', 12: '#2ca02c', 20: '#1f77b4'}
markers_n = {4: 'o', 8: 's', 12: '^', 20: 'D'}

for N in [4, 8, 12, 20]:
    key = f"KI N={N}"

    # Exact (línea)
    ax.plot(DEPTHS, exact[key], '-', color=colors_n[N], linewidth=1,
            alpha=0.4)

    # IBM (puntos con error)
    ibm_mean, ibm_std = get_ibm_stats(ibm[key])
    ax.errorbar(DEPTHS, ibm_mean, yerr=ibm_std, fmt=markers_n[N],
                color=colors_n[N], markersize=5, capsize=2, linewidth=1,
                label=f'N={N}')

ax.set_xlabel('Circuit depth d')
ax.set_ylabel('C(d)')
ax.set_title('(a) C(d) vs depth — IBM data')
ax.legend(title='System size', framealpha=0.9)
ax.set_xlim(0, 15)
ax.set_yscale('log')
ax.set_ylim(1e-5, 1)
ax.grid(True, alpha=0.3, which='both')

# Panel (b): Ω vs N
ax = axes[1]
Ns = [4, 8, 12, 20]
omega_exact_vals = []
omega_ibm_vals = []
omega_ibm_err = []

for N in Ns:
    key = f"KI N={N}"
    oe = np.mean(exact[key]) / C0
    omega_exact_vals.append(oe)

    ibm_mean, ibm_std = get_ibm_stats(ibm[key])
    oi = np.mean(ibm_mean) / C0
    # Error propagation: σ_Ω = σ_mean / C0 / sqrt(n_depths)
    oi_err = np.mean(ibm_std) / C0 / np.sqrt(len(DEPTHS))
    omega_ibm_vals.append(oi)
    omega_ibm_err.append(oi_err)

ax.plot(Ns, omega_exact_vals, 'o-', color=C_EXACT, markersize=8,
        linewidth=2, label='Exact simulation', zorder=3)
ax.errorbar(Ns, omega_ibm_vals, yerr=omega_ibm_err, fmt='s-', color=C_IBM,
            markersize=8, capsize=4, linewidth=2, label='IBM Quantum', zorder=2)

# Reference line: 1/2^N behavior
N_ref = np.linspace(4, 20, 100)
ax.plot(N_ref, 2 / 2**N_ref, '--', color='gray', alpha=0.5,
        label='~ 1/2ᴺ (guide)')

ax.set_xlabel('System size N')
ax.set_ylabel('Ω = ⟨C⟩/C₀')
ax.set_title('(b) Scrambling parameter Ω vs N')
ax.legend(framealpha=0.9)
ax.set_yscale('log')
ax.set_ylim(1e-3, 0.5)
ax.set_xticks(Ns)
ax.grid(True, alpha=0.3, which='both')

# Annotate regimes
ax.axhspan(0, 0.05, alpha=0.08, color='red')
ax.axhspan(0.05, 0.15, alpha=0.06, color='orange')
ax.text(17, 0.003, 'Complete\nscrambling', fontsize=8, color='red',
        ha='center', style='italic')
ax.text(17, 0.08, 'Strong', fontsize=8, color='orange',
        ha='center', style='italic')

plt.tight_layout()
plt.savefig('fig2_scaling_N.png', dpi=300, bbox_inches='tight')
plt.savefig('fig2_scaling_N.pdf', bbox_inches='tight')
print("  → fig2_scaling_N.png/pdf ✓")


# ============================================================
# FIGURA 3: Clasificación de regímenes (bar chart)
# ============================================================
print("Generando Figura 3...")

fig, ax = plt.subplots(figsize=(10, 5))

# Compute all Ω values
models_bar = []
for key in ["KI N=20", "KI N=12", "KI N=8", "KI N=4"]:
    oe = np.mean(exact[key]) / C0
    ibm_mean, _ = get_ibm_stats(ibm[key])
    oi = np.mean(ibm_mean) / C0
    models_bar.append((key, oe, oi))

# SYK
syk_arrays = np.stack([np.array(v) for v in syk_ibm.values()])
syk_mean_all = np.mean(syk_arrays, axis=0)
oi_syk = np.mean(syk_mean_all) / C0
models_bar.append(("SYK N=4\n(9 seeds)", None, oi_syk))

# Floquet
oe_f = np.mean(exact["Floquet"]) / C0
ibm_mean_f, _ = get_ibm_stats(ibm["Floquet"])
oi_f = np.mean(ibm_mean_f) / C0
models_bar.append(("Floquet\nN=4", oe_f, oi_f))

# Integrable
oe_i = np.mean(exact["Integrable"]) / C0
ibm_mean_i, _ = get_ibm_stats(ibm["Integrable"])
oi_i = np.mean(ibm_mean_i) / C0
models_bar.append(("Integrable\nN=4", oe_i, oi_i))

labels = [m[0] for m in models_bar]
exact_vals = [m[1] for m in models_bar]
ibm_vals = [m[2] for m in models_bar]

x = np.arange(len(labels))
width = 0.35

bars_exact = []
bars_ibm = []
for i in range(len(labels)):
    if exact_vals[i] is not None:
        b = ax.bar(x[i] - width/2, exact_vals[i], width, color=C_EXACT,
                   alpha=0.7, edgecolor='black', linewidth=0.5)
        bars_exact.append(b)
    b2 = ax.bar(x[i] + width/2, ibm_vals[i], width, color=C_IBM,
                alpha=0.7, edgecolor='black', linewidth=0.5)
    bars_ibm.append(b2)

# Regime backgrounds
ax.axhspan(0, 0.05, alpha=0.08, color='red', zorder=0)
ax.axhspan(0.05, 0.15, alpha=0.06, color='orange', zorder=0)
ax.axhspan(0.15, 0.35, alpha=0.05, color='yellow', zorder=0)
ax.axhspan(0.35, 0.60, alpha=0.04, color='green', zorder=0)
ax.axhspan(0.60, 1.0, alpha=0.04, color='blue', zorder=0)

# Regime labels
ax.text(6.8, 0.025, 'Complete scrambling', fontsize=8, style='italic', color='darkred')
ax.text(6.8, 0.10, 'Strong scrambling', fontsize=8, style='italic', color='darkorange')
ax.text(6.8, 0.25, 'Intermediate', fontsize=8, style='italic', color='olive')
ax.text(6.8, 0.47, 'Weak scrambling', fontsize=8, style='italic', color='darkgreen')
ax.text(6.8, 0.80, 'No scrambling', fontsize=8, style='italic', color='darkblue')

ax.set_ylabel('Ω = ⟨C(d)⟩ / C₀')
ax.set_title('Figure 3: Scrambling classification — Exact vs IBM Quantum',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 0.95)
ax.legend([mpatches.Patch(color=C_EXACT, alpha=0.7),
           mpatches.Patch(color=C_IBM, alpha=0.7)],
          ['Exact simulation', 'IBM Quantum (ibm_marrakesh)'],
          loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('fig3_classification.png', dpi=300, bbox_inches='tight')
plt.savefig('fig3_classification.pdf', bbox_inches='tight')
print("  → fig3_classification.png/pdf ✓")


# ============================================================
# FIGURA 4: Noise floor analysis
# ============================================================
print("Generando Figura 4...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 4: Hardware noise floor characterization',
             fontsize=14, fontweight='bold')

# Panel (a): C(d≥6) promedio vs N
ax = axes[0]
Ns_floor = [4, 8, 12, 20]
floor_ibm = []
floor_exact = []
floor_theory = []  # 1/2^N

for N in Ns_floor:
    key = f"KI N={N}"
    ibm_mean, _ = get_ibm_stats(ibm[key])
    f_ibm = np.mean(ibm_mean[5:])  # d >= 6 (index 5 onwards)
    f_exact = np.mean(np.array(exact[key])[5:])
    floor_ibm.append(f_ibm)
    floor_exact.append(f_exact)
    floor_theory.append(1.0 / 2**N)

ax.semilogy(Ns_floor, floor_ibm, 'o-', color=C_IBM, markersize=8,
            linewidth=2, label='IBM ⟨C(d≥6)⟩')
ax.semilogy(Ns_floor, [max(f, 1e-8) for f in floor_exact], 's--',
            color=C_EXACT, markersize=8, linewidth=1.5,
            label='Exact ⟨C(d≥6)⟩')
ax.semilogy(Ns_floor, floor_theory, '^:', color='gray', markersize=8,
            linewidth=1.5, label='1/2ᴺ (uniform noise)')

ax.set_xlabel('System size N')
ax.set_ylabel('⟨C(d ≥ 6)⟩')
ax.set_title('(a) Deep-circuit noise floor')
ax.legend(framealpha=0.9)
ax.set_xticks(Ns_floor)
ax.set_ylim(1e-7, 0.2)
ax.grid(True, alpha=0.3, which='both')

# Panel (b): Signal-to-noise for KI N=4
ax = axes[1]
ibm_mean_4, ibm_std_4 = get_ibm_stats(ibm["KI N=4"])
exact_4 = np.array(exact["KI N=4"])

# Noise floor estimate for N=4
noise_floor = 1.0/16  # uniform noise

signal = exact_4
noise = ibm_mean_4 - exact_4

ax.bar(np.arange(len(DEPTHS)) - 0.2, exact_4, 0.35, color=C_EXACT,
       alpha=0.7, label='Signal (exact C(d))')
ax.bar(np.arange(len(DEPTHS)) + 0.2, np.abs(noise), 0.35, color=C_KI,
       alpha=0.5, label='|Noise| (IBM − exact)')
ax.axhline(y=noise_floor, color='gray', linewidth=1, linestyle='--',
           label=f'1/2⁴ = {noise_floor:.4f}')

ax.set_xlabel('Depth index')
ax.set_ylabel('C(d)')
ax.set_title('(b) KI N=4: Signal vs noise decomposition')
ax.set_xticks(range(len(DEPTHS)))
ax.set_xticklabels([str(d) for d in DEPTHS], fontsize=9)
ax.legend(framealpha=0.9, fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('fig4_noise_floor.png', dpi=300, bbox_inches='tight')
plt.savefig('fig4_noise_floor.pdf', bbox_inches='tight')
print("  → fig4_noise_floor.png/pdf ✓")


# ============================================================
# RESUMEN
# ============================================================
print("\n" + "=" * 50)
print("FIGURAS GENERADAS:")
print("=" * 50)
print("  fig1_otoc_all_models.png/pdf  — C(d) todos los modelos")
print("  fig2_scaling_N.png/pdf        — Escalamiento KI con N")
print("  fig3_classification.png/pdf   — Clasificación de regímenes")
print("  fig4_noise_floor.png/pdf      — Análisis de noise floor")
print("\nTodas en 300 DPI, formato publicación.")
print("Copiar a Google Drive o descargar directamente.")
