# -*- coding: utf-8 -*-

import itertools
import random
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli, Operator, Statevector
from pauli_propagation.utils import (
    encode_pauli,
    decode_pauli,
    random_su4,
    random_pauli_label,
    random_state_label,
    pauli_terms_to_matrix,
)
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate
from pauli_propagation.propagator import PauliPropagator

# All 2-qubit Pauli labels for exhaustive testing
LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

def random_unitary(n: int) -> np.ndarray:
    """Generate a random n��n unitary matrix."""
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D

# Test with random unitaries - reduced test set for speed
@pytest.mark.parametrize("trial", range(5))  # 5 random unitary matrices
@pytest.mark.parametrize("label", ["II", "XX", "YY", "ZZ", "XY", "ZI"])  # Key cases
def test_su4_rule(trial, label):
    """Test SU4 gate with random unitary matrices on key Pauli cases."""
    np.random.seed(trial + 100)
    
    # Generate random 4x4 unitary matrix
    U = random_unitary(4)
    
    key = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = QuantumGate.get("su4")(input_term, 0, 1, U)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    
    assert np.allclose(matsum, expected), f"SU4 random unitary failed for trial {trial}, P={label}"

TRIALS_EMB = 20

@pytest.mark.parametrize("trial", range(TRIALS_EMB))
def test_su4_random_embedded(trial):
    """Embed a random SU4 gate on random qubit pair of a 6-qubit Pauli and compare matrices."""

    np.random.seed(trial + 12000)
    num_qubits = 6

    # Random Pauli label and target qubits
    label = "".join(random.choice("IXYZ") for _ in range(num_qubits))
    q1, q2 = np.random.choice(num_qubits, 2, replace=False)

    # Random SU4 unitary
    U = random_su4()

    # Prepare input Pauli term and propagate via bit-kernel
    key        = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, num_qubits)
    output_terms = QuantumGate.get("su4")(input_term, q1, q2, U)
    matsum = pauli_terms_to_matrix(output_terms, num_qubits)

    # Reference via Qiskit Operator (no manual kron loop)
    qc_ref = QuantumCircuit(num_qubits)
    gate_unitary = UnitaryGate(U, label="su4")
    gate_unitary._name = "su4"
    qc_ref.append(gate_unitary, [q1, q2])

    G_full = Operator(qc_ref).data
    P_mat  = Pauli(label).to_matrix()
    ref    = G_full.conj().T @ P_mat @ G_full

    assert np.allclose(matsum, ref), (
        f"Embedded SU4 mismatch on qubits ({q1},{q2}), label {label}")

@pytest.mark.parametrize("trial", range(10))
def test_su4_random_circuits(trial):
    """Test SU4 gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 11000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits (minimum 2 for two-qubit gates)
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Create quantum circuit with random SU4 gates
    qc = QuantumCircuit(n, name=f"su4_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random two qubits and generate random unitary matrix
        q1, q2 = np.random.choice(n, 2, replace=False)
        U = random_su4()
        
        # Add SU4 gate to circuit using UnitaryGate with correct naming
        gate = UnitaryGate(U, label="su4")
        gate._name = "su4"
        qc.append(gate, [q1, q2])
    
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
