# %%
# Standard library imports
import os
import json

# Third-party imports
import numpy as np

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

# Custom package imports
from pauli_propagation import PauliTerm, PauliPropagator
from pauli_propagation.utils import encode_pauli
from pauli_propagation.monte_carlo import MonteCarlo
from pauli_propagation import staircasetopology2d_qc

# %%
# Monte Carlo parameters
nx = ny = 5
L_vals = [1,2,3,4]
M = 10000 * 200
max_kk = 6

# Prepare output directory
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

# %%
for L in L_vals:

    # === 0) Build circuit and initial PauliTerm ===
    qc_2d = staircasetopology2d_qc(nx, ny, L)
    n = qc_2d.num_qubits

    pauli_label = 'Z' + "I" * (n - 1)
    key = encode_pauli(Pauli(pauli_label))
    init_term = PauliTerm(1.0, key, n)
    product_label = "0" * n
    prop_2d = PauliPropagator(qc_2d)

    # === 1) Monte Carlo sampling step ===
    mc = MonteCarlo(qc_2d)
    _, _, last_weights, _ = mc.monte_carlo_samples(init_term=init_term, M=M)

    # === 2) Compute weight histogram ===
    bins = np.arange(0.5, n + 1.5)
    hist_counts, bin_edges = np.histogram(last_weights, bins=bins, density=False)
    hist_vals = hist_counts / float(len(last_weights))

    # === 3) Estimate MSE for all thresholds k ===
    mse_mc_results_dict = mc.estimate_mse_for_truncation(propagator=prop_2d,
                                                         product_label=product_label)

    cum = mse_mc_results_dict['cumulative']
    layer = mse_mc_results_dict['layer']

    # === 4) Package results into a dictionary ===
    result_for_L = {
        "L": L,
        "weight_hist": {
            "hist_vals": hist_vals.tolist(),
            "bin_edges": bin_edges.tolist(),
            "hist_counts": hist_counts.tolist()
        },
        "cum_mse": {str(k): float(cum.get(k, 0.0)) for k in range(max_kk + 1)},
        "weight_mse": {str(k): float(layer.get(k, 0.0)) for k in range(max_kk + 1)}
    }

    # === 5) Write results to JSON file ===
    filename = os.path.join(out_dir, f"su4_mc_results_L{L}.json")
    with open(filename, 'w') as f:
        json.dump(result_for_L, f, indent=2)

    print(f"Completed MC sampling for L={L}, results saved to: {filename}")
