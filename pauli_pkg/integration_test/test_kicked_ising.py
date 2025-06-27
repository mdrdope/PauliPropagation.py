# -*- coding: utf-8 -*-

"""
Test IBM Eagle Topology TFI Circuit with Randomized Observables and Angles

This test module validates consistency between Pauli propagation and statevector
simulation for TFI (Transverse Field Ising) circuits on random single-Pauli observables,
random RX/RZZ angles, for qubit counts in [2..7] and layers in [5,10].
"""

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, Statevector

from pauli_propagation import PauliTerm, PauliPropagator
from pauli_propagation.utils import encode_pauli, random_pauli_label, random_state_label
from pauli_propagation.circuit_topologies import tfi_trotter_circuit, ibm_eagle_topology


@pytest.mark.parametrize("n_qubits", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("n_layers", [3, 5, 10])
def test_tfi_circuit_random_pauli_and_angles(n_qubits, n_layers):
    """
    Test consistency between Pauli propagation and statevector simulation
    for TFI circuits with IBM Eagle topology, using:
      - a random single-Pauli observable (exactly one non-I),
      - random rotation angles for RX and RZZ,
      - qubit counts in [2..7],
      - layer counts in [5,10].
    """
    # Extract subgraph edges for qubits 1..n_qubits (1-based topology)
    subgraph_edges = [
        (u, v) for (u, v) in ibm_eagle_topology
        if u <= n_qubits and v <= n_qubits
    ]

    # Generate random RX and RZZ angles in (0, π)

    theta_h = np.random.uniform(0.0 , np.pi  )
    rzz_theta = np.random.uniform(0.0 , np.pi  )

    # Construct the TFI Trotter circuit with random angles
    qc = tfi_trotter_circuit(
        n_qubits,
        n_layers,
        subgraph_edges,
        start_with_ZZ=True,
        rx_theta=theta_h,
        rzz_theta=rzz_theta
    )

    #  Build a random single-Pauli observable (one non-I at a random position)
    pauli_lbl = random_pauli_label(n_qubits)  # random non-identity Pauli label
    key = encode_pauli(Pauli(pauli_lbl))
    init_term = PauliTerm(1.0, key, n_qubits)

    # Pauli Propagation Method 
    prop = PauliPropagator(qc)
    layers = prop.propagate(init_term, max_weight=None, use_parallel=False)
    state_lbl = random_state_label(n_qubits)  # random product state label
    exp_pauli = prop.expectation_pauli_sum(layers[-1], state_lbl)

    # ==== Statevector Simulation Method ====
    sv = Statevector.from_label(state_lbl)
    sv_evolved = sv.evolve(qc)
    pauli_op = Pauli(pauli_lbl)
    exp_statevector = sv_evolved.expectation_value(pauli_op).real

    # Assert consistency within tolerance
    tolerance = 5e-10
    difference = abs(exp_pauli - exp_statevector)
    assert difference < tolerance, (
        f"Inconsistency detected for {n_qubits} qubits, {n_layers} layers, "
        f"Pauli observable {pauli_lbl}, angles RX={theta_h:.6f}, RZZ={rzz_theta:.6f}:\n"
        f"Pauli Propagation: {exp_pauli:.12f}\n"
        f"Statevector:       {exp_statevector:.12f}\n"
        f"Difference:        {difference:.2e}"
    )
