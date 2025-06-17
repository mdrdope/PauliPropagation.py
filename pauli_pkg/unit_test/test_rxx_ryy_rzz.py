#!/usr/bin/env python3

import itertools
import numpy as np
import pytest
from math import pi
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator, Statevector

from pauli_propagation.utils      import encode_pauli, decode_pauli, random_pauli_label, random_state_label
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate
from pauli_propagation.propagator import PauliPropagator

LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

def pauli_matrix(label: str) -> np.ndarray:
    """Convert Pauli label to matrix representation."""
    return Pauli(label).to_matrix()

def rxx_matrix(q1: int, q2: int, theta: float) -> np.ndarray:
    """Generate RXX gate matrix for given qubits and angle."""
    qc = QuantumCircuit(2)
    qc.rxx(theta, q1, q2)
    return Operator(qc).data

def ryy_matrix(q1: int, q2: int, theta: float) -> np.ndarray:
    """Generate RYY gate matrix for given qubits and angle."""
    qc = QuantumCircuit(2)
    qc.ryy(theta, q1, q2)
    return Operator(qc).data

def rzz_matrix(q1: int, q2: int, theta: float) -> np.ndarray:
    """Generate RZZ gate matrix for given qubits and angle."""
    qc = QuantumCircuit(2)
    qc.rzz(theta, q1, q2)
    return Operator(qc).data

def pauli_terms_to_matrix(terms: list, n: int) -> np.ndarray:
    """Convert list of PauliTerm objects to their matrix sum representation."""
    total_matrix = np.zeros((2**n, 2**n), dtype=complex)
    for term in terms:
        pauli = decode_pauli(term.key, term.n)
        total_matrix += term.coeff * pauli.to_matrix()
    return total_matrix

@pytest.mark.parametrize("gate_name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("q1,q2", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
@pytest.mark.parametrize("theta", [0.0, pi/4, pi/2, pi, 3*pi/2, 2*pi])
def test_rxx_ryy_rzz_matrix_equivalence(gate_name, q1, q2, label, theta):
    """Test that pauli propagation matches direct matrix propagation."""
    # Get gate matrix
    if gate_name == "rxx":
        U = rxx_matrix(q1, q2, theta)
    elif gate_name == "ryy":
        U = ryy_matrix(q1, q2, theta)
    else:  # rzz
        U = rzz_matrix(q1, q2, theta)
    
    # Get gate kernel
    kernel = QuantumGate.get(gate_name)
    key = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate via pauli propagation
    output_terms = kernel(input_term, q1, q2, theta)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    # Calculate expected result via direct matrix propagation
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"Matrix mismatch for {gate_name}(θ={theta}) on {label}, qubits ({q1},{q2})"

@pytest.mark.parametrize("trial", range(30))
def test_rxx_ryy_rzz_expectation_values(trial):
    """Test RXX, RYY, RZZ gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 1000)  # Set seed for reproducibility
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits (minimum 2 for two-qubit gates)
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Available gate types
    gate_types = ['rxx', 'ryy', 'rzz']
    
    # Create quantum circuit with random two-qubit rotation gates
    qc = QuantumCircuit(n, name=f"rxx_ryy_rzz_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random gate type and qubits
        gate_type = np.random.choice(gate_types)
        q1, q2 = np.random.choice(n, 2, replace=False)
        theta = np.random.uniform(0, 2*np.pi)
        
        # Add gate to circuit
        if gate_type == 'rxx':
            qc.rxx(theta, q1, q2)
        elif gate_type == 'ryy':
            qc.ryy(theta, q1, q2)
        elif gate_type == 'rzz':
            qc.rzz(theta, q1, q2)
    
    # Method 1: PauliPropagator expectation
    prop = PauliPropagator(qc)
    layers = prop.propagate(observable, max_weight=None)
    pauli_expectation = prop.expectation_pauli_sum(layers[-1], state_label)
    
    # Method 2: Qiskit statevector expectation
    sv_expectation = Statevector.from_label(state_label).evolve(qc).expectation_value(Pauli(pauli_label)).real
    
    # Compare results
    assert abs(pauli_expectation - sv_expectation) < 1e-10, (
        f"Trial {trial}: expectation mismatch {pauli_expectation} vs {sv_expectation} "
        f"on state {state_label}, observable {pauli_label}, circuit {n}q {n_gates}g"
    ) 