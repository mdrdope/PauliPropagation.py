# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from math import pi
from qiskit.quantum_info import Pauli, Operator, Statevector
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

T_GATE = np.diag([1.0, np.exp(1j * pi / 4)])

@pytest.mark.parametrize("label", ["I", "X", "Y", "Z"])
def test_t_rule(label):
    """
    Direct test of single-qubit T gate conjugation:
    T^dagger P T == sum of output terms
    """
    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 1)
    output_terms = QuantumGate.get("t")(input_term, 0)
    
    result = pauli_terms_to_matrix(output_terms, 1)
    expected = T_GATE.conj().T @ Pauli(label).to_matrix() @ T_GATE
    assert np.allclose(result, expected), f"Failed for P={label}"

TRIALS = 20

@pytest.mark.parametrize("trial", range(TRIALS))
def test_t_random_embedded(trial):
    """
    Embed T gate on a random qubit in 4-qubit Pauli, compare with full tensor product construction.
    """
    num_qubits = 6
    # Generate random 4-qubit Pauli label
    label = "".join(random.choice("IXYZ") for _ in range(num_qubits))
    q      = random.randrange(num_qubits)
    key    = encode_pauli(Pauli(label))

    # Apply T gate using bit-kernel
    input_term = PauliTerm(1.0, key, num_qubits)
    output_terms = QuantumGate.get("t")(input_term, q)
    matsum = pauli_terms_to_matrix(output_terms, num_qubits)

    # Reference calculation via Qiskit's Operator utility (avoids manual kron loop)
    qc_ref = QuantumCircuit(num_qubits)
    qc_ref.t(q)
    G_full = Operator(qc_ref).data

    P_mat = Pauli(label).to_matrix()
    ref = G_full.conj().T @ P_mat @ G_full

    assert np.allclose(matsum, ref), f"Embedded T mismatch on qubit {q}, label {label}"



@pytest.mark.parametrize("trial", range(10))
def test_t_random_circuits(trial):
    """Test T gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 8000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Create quantum circuit with random T gates
    qc = QuantumCircuit(n, name=f"t_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random qubit for T gate
        q = np.random.randint(0, n)
        qc.t(q)
    
    # Method 1: PauliPropagator expectation
    prop = PauliPropagator(qc)
    layers = prop.propagate(observable, max_weight=None)
    pauli_expectation = prop.expectation_pauli_sum(layers[-1], state_label)
    
    # Method 2: Qiskit statevector expectation
    initial_state = Statevector.from_label(state_label)
    final_state = initial_state.evolve(qc)
    qiskit_expectation = final_state.expectation_value(Pauli(pauli_label)).real
    
    # Compare results
    assert abs(pauli_expectation - qiskit_expectation) < 1e-10, (
        f"Trial {trial}: expectation mismatch {pauli_expectation} vs {qiskit_expectation} "
        f"on state {state_label}, observable {pauli_label}, circuit {n}q {n_gates}g"
    )
