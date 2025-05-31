#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from qiskit.quantum_info import Pauli
from pauli_propagation import PauliTerm, MonteCarlo
from pauli_propagation.utils import encode_pauli
from pauli_propagation.gates import QuantumGate
from pauli_propagation.decomposition import su4_kak_reconstruct
from pauli_propagation import staircasetopology2d_qc

# =============================================================================
# Parameters (you may override via command-line args or environment if desired)
# =============================================================================
# L_values: list of circuit depths to run Monte Carlo over
L_values = [4]              # for example, only L=4; adjust as needed
nx, ny = 5, 5               # lattice size (5¡Á5 heavy-hex becomes 25 qubits)
M = 10000 * 200             # total number of Monte Carlo samples
max_kk = 6                  # maximum truncation weight k
out_dir = "results"         # directory to store JSON output

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

for L in L_values:
    # 0) Build 2D circuit & initial Pauli term
    qc_2d = staircasetopology2d_qc(nx, ny, L)
    qc_2d = su4_kak_reconstruct(qc_2d)
    n = qc_2d.num_qubits

    # Prepare initial Pauli Z on qubit 0 (Z ? I ? I ¡­)
    pauli_label = 'Z' + "I" * (n - 1)
    key = encode_pauli(Pauli(pauli_label))
    init_term = PauliTerm(1.0, key, n)
    product_label = "0" * n  # computational©\basis state |0¡­0?
    prop_2d = None  # not needed for raw MC sampling

    # 1) Monte Carlo sampling (all k share the same random seeds/weights)
    mc = MonteCarlo(qc_2d)
    _, _, last_weights, _ = mc.monte_carlo_samples(
        init_term=init_term,
        M=M
    )

    # 2) Compute weight histogram (counts & normalized probabilities)
    bins = np.arange(0.5, n + 1.5)  # integer bins for weights 1¡­n
    hist_counts, bin_edges = np.histogram(last_weights, bins=bins, density=False)
    hist_vals = hist_counts / float(len(last_weights))

    # 3) Estimate MC MSE for all k up to max_kk
    prop_2d = MonteCarlo(qc_2d)._propagator  # reuse internal PauliPropagator
    mse_mc_results_dict = mc.estimate_mse_for_truncation(
        propagator=prop_2d,
        product_label=product_label
    )
    cum = mse_mc_results_dict['cumulative']
    layer = mse_mc_results_dict['layer']

    # 4) Package results for this L
    result_for_L = {
        "L": L,
        "weight_hist": {
            "hist_vals": hist_vals.tolist(),
            "bin_edges": bin_edges.tolist(),
            "hist_counts": hist_counts.tolist()
        },
        "cum_mse": { str(k): float(cum.get(k, 0.0)) for k in range(max_kk + 1) },
        "weight_mse": { str(k): float(layer.get(k, 0.0)) for k in range(max_kk + 1) }
    }

    # 5) Write to JSON
    filename = os.path.join(out_dir, f"kak_mc_results_L{L}.json")
    with open(filename, 'w') as f:
        json.dump(result_for_L, f, indent=2, ensure_ascii=False)
    print(f"Completed MC sampling for L={L}; saved to: {filename}")
