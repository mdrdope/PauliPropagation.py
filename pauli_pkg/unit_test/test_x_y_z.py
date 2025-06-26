# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from qiskit.quantum_info import Pauli, Statevector, Operator
from qiskit import QuantumCircuit

from pauli_propagation.utils import (
    encode_pauli,
    decode_pauli,
    random_pauli_label,
    random_state_label,
    pauli_terms_to_matrix,
)
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates import QuantumGate
from pauli_propagation.propagator import PauliPropagator

# -----------------------------------------------------------------------------
# Helper data & functions that the remaining tests rely on
# -----------------------------------------------------------------------------

X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
GATE_MATRICES = {"x": X_GATE, "y": Y_GATE, "z": Z_GATE}

@pytest.mark.parametrize("gate_name", ["x", "y", "z"])
@pytest.mark.parametrize("label", ["I", "X", "Y", "Z"])
def test_x_y_z_rules(gate_name, label):
    """Verify U^dagger P U for a single–qubit Pauli gate."""
    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 1)

    output_terms = QuantumGate.get(gate_name)(input_term, 0)
    result = pauli_terms_to_matrix(output_terms, 1)

    gate_matrix = GATE_MATRICES[gate_name]
    expected = gate_matrix.conj().T @ Pauli(label).to_matrix() @ gate_matrix

    assert np.allclose(result, expected), f"{gate_name.upper()} on P={label} failed"

TRIALS = 20

@pytest.mark.parametrize("gate_name", ["x", "y", "z"])
@pytest.mark.parametrize("trial", range(TRIALS))
def test_x_y_z_random_embedded(gate_name, trial):
    """Embed X/Y/Z on a random qubit of an 8-qubit Pauli and compare matrices."""
    num_qubits = 8
    label = "".join(random.choice("IXYZ") for _ in range(num_qubits))
    target_q = random.randrange(num_qubits)

    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, num_qubits)

    # Propagation via bit-kernel
    output_terms = QuantumGate.get(gate_name)(input_term, target_q)
    matsum = pauli_terms_to_matrix(output_terms, num_qubits) 

    # Reference matrix via Qiskit's Operator utility (simpler & less error-prone)
    qc_ref = QuantumCircuit(num_qubits)
    getattr(qc_ref, gate_name)(target_q)  # apply the single-qubit gate on the target qubit
    G_full = Operator(qc_ref).data  # full 2^n × 2^n unitary for the embedded gate

    P_mat = Pauli(label).to_matrix()  # dense matrix for the Pauli operator
    ref = G_full.conj().T @ P_mat @ G_full

    assert np.allclose(matsum, ref), (
        f"Embedded {gate_name.upper()} mismatch on qubit {target_q}, label {label}")

# -----------------------------------------------------------------------------
# 3. test_xyz_random_circuits
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("trial", range(10))
def test_x_y_z_random_circuits(trial):
    """Compare PauliPropagator expectation with full state-vector simulation."""
    np.random.seed(trial + 7000)

    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits
    n_gates = np.random.randint(3, 8)

    # Random initial state |s> and observable P
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable = PauliTerm(1.0, encode_pauli(Pauli(pauli_label)), n)

    # Build random circuit consisting only of X/Y/Z gates
    qc = QuantumCircuit(n, name=f"xyz_rand_{n}q_{n_gates}g")
    gate_choices = ["x", "y", "z"]
    for _ in range(n_gates):
        gate = np.random.choice(gate_choices)
        q = np.random.randint(0, n)
        getattr(qc, gate)(q)  # call qc.x / qc.y / qc.z

    # Expectation via Pauli propagation (no truncation)
    prop = PauliPropagator(qc)
    final_layer = prop.propagate(observable, max_weight=None)[-1]
    pauli_exp = prop.expectation_pauli_sum(final_layer, state_label)

    # Exact expectation via state-vector evolution
    psi0 = Statevector.from_label(state_label)
    psi1 = psi0.evolve(qc)
    exact_exp = psi1.expectation_value(Pauli(pauli_label)).real

    assert abs(pauli_exp - exact_exp) < 1e-10, (
        f"Trial {trial}: {pauli_exp} vs {exact_exp} (state={state_label}, P={pauli_label})"
    ) 