# Probing Quantum Scrambling via OTOCs in Digital Circuits

**Finite-Size Recurrences, System-Size Scaling, and Validation on IBM Quantum Hardware**

**Author:** Erick Francisco Perez Eugenio  
**ORCID:** [0009-0006-3228-4847](https://orcid.org/0009-0006-3228-4847)  
**Contact:** erick.fpe79@gmail.com

---

## Overview

We study quantum scrambling via out-of-time-ordered correlators (OTOCs) in the Kicked Ising model across system sizes N = 4, 8, 12, 20, combining exact statevector simulation with experiments on IBM Quantum hardware (ibm_marrakesh, 156 qubits). We introduce the scrambling residual Ω as a fit-free diagnostic that correctly classifies dynamical regimes on both simulated and hardware data.

## Key Results

- Quantum Poincaré recurrences are the limiting factor for OTOC-based scrambling at small N
- Recurrences are directly observable on IBM hardware (Pearson r = 0.91 at N = 4)
- Clean exponential decay emerges at N = 20 (R² = 0.97, λ_L = 3.12 ± 0.1)
- The scrambling residual Ω provides regime classification consistent between simulation and hardware

## Reproducing Results

### Requirements
- Python 3.8+
- numpy, scipy, matplotlib
- qiskit 2.3.0 (for IBM hardware execution only)

### Data
All raw data is in the `data/` directory:
- `paper1_raw_data.json` — 616 exact simulation data points (all models, all N)
- `paper1_recovered_ibm_data.json` — 429 IBM experimental data points (40 jobs)

### Figures
Pre-generated figures are in `figures/`. To regenerate:
```bash
python code/paper1_figuras.py
```

### Exact Simulation
The simulation code in `code/paper1_analisis_ibm_v1.py` contains:
- Statevector OTOC computation for Kicked Ising, Integrable, Floquet, and SYK models
- IBM data recovery and comparison
- Statistical analysis (Ω, R², Pearson/Spearman correlations)

## Repository Structure

```
otoc-finite-size-scaling/
├── README.md
├── LICENSE
├── .zenodo.json
├── .gitignore
├── code/
│   ├── paper1_analisis_ibm_v1.py    — IBM data analysis and exact simulation
│   └── paper1_figuras.py            — Figure generation (4 publication figures)
├── data/
│   ├── paper1_raw_data.json         — 616 exact simulation points
│   └── paper1_recovered_ibm_data.json — 429 IBM experimental points
├── figures/
│   ├── fig1_otoc_all_models.png     — OTOC C(d) all models N=4
│   ├── fig2_scaling_N.png           — KI scaling with system size
│   ├── fig3_classification.png      — Ω classification comparison
│   └── fig4_noise_floor.png         — Hardware noise floor
└── tex/
    ├── paper1_main.tex              — Main manuscript
    ├── paper1_supplemental.tex      — Supplemental material
    ├── kaelion.bib                  — Bibliography (50 entries)
    ├── Kaelion_Paper1_main.pdf      — Compiled main paper
    └── Kaelion_Paper1_supplemental.pdf — Compiled supplemental
```

## Models

| Model | Parameters | Boundary | Regime |
|-------|-----------|----------|--------|
| Kicked Ising | J=0.9, h=0.7 | PBC | Chaotic |
| Integrable | H + CNOT | OBC | Clifford |
| Floquet | θ=0.8, φ=1.2, J=0.9 | PBC | Prethermal |
| SYK | J_ij ~ U[0.5,1.5] | All-to-all | Disorder-averaged |

## Hardware

- **Backend:** ibm_marrakesh (156 superconducting qubits)
- **Processor:** Heron (tunable-coupler architecture)
- **Native gate set:** CZ, RZ, SX, X
- **Qiskit:** 2.3.0, Sampler V2
- **Shots:** 4096 per circuit, 5 runs per configuration
- **Total experimental points:** 429
- **Date:** February 12–14, 2026

## Citation

If you use this code or data, please cite:

```bibtex
@article{PerezEugenio2026otoc,
  author = {Perez Eugenio, Erick Francisco},
  title = {Probing Quantum Scrambling via OTOCs in Digital Circuits:
           Finite-Size Recurrences, System-Size Scaling,
           and Validation on IBM Quantum Hardware},
  year = {2026},
  note = {DOI: 10.5281/zenodo.XXXXXXX}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

We acknowledge the use of IBM Quantum services. Simulations were performed on Google Colab.
