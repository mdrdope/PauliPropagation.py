# -*- coding: utf-8 -*-

import itertools
import random
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli, Operator

from pauli_propagation.utils      import encode_pauli, decode_pauli, random_su4, random_pauli_label, random_state_label
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

def pauli_matrix(label: str) -> np.ndarray:
    """Convert Pauli label to matrix representation."""
    return Pauli(label).to_matrix()

def pauli_terms_to_matrix(terms: list, n: int) -> np.ndarray:
    """Convert list of PauliTerm objects to their matrix sum representation."""
    total_matrix = np.zeros((2**n, 2**n), dtype=complex)
    for term in terms:
        pauli = decode_pauli(term.key, term.n)
        total_matrix += term.coeff * pauli.to_matrix()
    return total_matrix

# Test with random unitaries - reduced test set for speed
@pytest.mark.parametrize("trial", range(5))  # 5 random unitary matrices
@pytest.mark.parametrize("label", ["II", "XX", "YY", "ZZ", "XY", "ZI"])  # Key cases
def test_su4_random_unitary(trial, label):
    """Test SU4 gate with random unitary matrices on key Pauli cases."""
    np.random.seed(trial + 100)
    
    # Generate random 4x4 unitary matrix
    U = random_unitary(4)
    
    key = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = QuantumGate.get("su4")(input_term, 0, 1, U)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"SU4 random unitary failed for trial {trial}, P={label}"

# Test coefficient preservation
def test_su4_coefficient_preservation():
    """Test that SU4 preserves the magnitude of coefficients."""
    U = random_unitary(4)
    label = "XY"
    coeff = 2.5 + 1.5j
    key = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(coeff, key, 2)
    output_terms = QuantumGate.get("su4")(input_term, 0, 1, U)
    
    # Sum of squared magnitudes should be preserved
    input_mag_sq = abs(coeff)**2
    output_mag_sq = sum(abs(term.coeff)**2 for term in output_terms)
    
    assert abs(input_mag_sq - output_mag_sq) < 1e-12, "SU4 coefficient magnitude not preserved"

def apply_gate_via_propagator(qc: QuantumCircuit, pauli_term: PauliTerm) -> list:
    """Apply quantum circuit via propagator."""
    prop = PauliPropagator(qc)
    history = prop.propagate(pauli_term)
    return history[-1]

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
    from qiskit.quantum_info import Statevector
    initial_state = Statevector.from_label(state_label)
    final_state = initial_state.evolve(qc)
    qiskit_expectation = final_state.expectation_value(Pauli(pauli_label)).real
    
    # Compare results
    assert abs(pauli_expectation - qiskit_expectation) < 1e-10, (
        f"Trial {trial}: expectation mismatch {pauli_expectation} vs {qiskit_expectation} "
        f"on state {state_label}, observable {pauli_label}, circuit {n}q {n_gates}g"
    )
