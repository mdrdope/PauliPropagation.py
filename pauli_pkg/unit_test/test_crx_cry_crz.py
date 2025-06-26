# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator, Statevector

from pauli_propagation.utils import (
    encode_pauli,
    decode_pauli,
    random_pauli_label,
    random_state_label,
    pauli_terms_to_matrix,
)
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate
from pauli_propagation.propagator import PauliPropagator

# Reduced test set - only key Pauli operators
LABELS_2Q_REDUCED = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", 
                     "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]

def crx_matrix(ctrl: int, tgt: int, theta: float) -> np.ndarray:
    """Generate CRX gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.crx(theta, ctrl, tgt)
    return Operator(qc).data

def cry_matrix(ctrl: int, tgt: int, theta: float) -> np.ndarray:
    """Generate CRY gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.cry(theta, ctrl, tgt)
    return Operator(qc).data

def crz_matrix(ctrl: int, tgt: int, theta: float) -> np.ndarray:
    """Generate CRZ gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.crz(theta, ctrl, tgt)
    return Operator(qc).data

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q_REDUCED)
@pytest.mark.parametrize("theta", [np.pi/4, np.pi/2])  # Reduced from 6 to 2 angles
def test_crx_rule(ctrl, tgt, label, theta):
    """Exhaustively test CRX gate conjugation for key 2-qubit Paulis."""
    U      = crx_matrix(ctrl, tgt, theta)
    kernel = QuantumGate.get("crx")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - now returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt, theta)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}, theta={theta}"

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q_REDUCED)
@pytest.mark.parametrize("theta", [np.pi/4, np.pi/2])  # Reduced from 6 to 2 angles
def test_cry_rule(ctrl, tgt, label, theta):
    """Exhaustively test CRY gate conjugation for key 2-qubit Paulis."""
    U      = cry_matrix(ctrl, tgt, theta)
    kernel = QuantumGate.get("cry")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt, theta)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}, theta={theta}"

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q_REDUCED)
@pytest.mark.parametrize("theta", [np.pi/4, np.pi/2])  # Reduced from 6 to 2 angles
def test_crz_rule(ctrl, tgt, label, theta):
    """Exhaustively test CRZ gate conjugation for key 2-qubit Paulis."""
    U      = crz_matrix(ctrl, tgt, theta)
    kernel = QuantumGate.get("crz")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt, theta)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}, theta={theta}"

@pytest.mark.parametrize("trial", range(10))
def test_crx_cry_crz_random_circuits(trial):
    """Test CRX, CRY, CRZ gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 2000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits (minimum 2 for controlled gates)
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Available gate types
    gate_types = ['crx', 'cry', 'crz']
    
    # Create quantum circuit with random controlled rotation gates
    qc = QuantumCircuit(n, name=f"crxyz_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random gate type and qubits
        gate_type = np.random.choice(gate_types)
        ctrl, tgt = np.random.choice(n, 2, replace=False)
        theta = np.random.uniform(0, 2*np.pi)
        
        # Add gate to circuit
        if gate_type == 'crx':
            qc.crx(theta, ctrl, tgt)
        elif gate_type == 'cry':
            qc.cry(theta, ctrl, tgt)
        elif gate_type == 'crz':
            qc.crz(theta, ctrl, tgt)
    
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
    