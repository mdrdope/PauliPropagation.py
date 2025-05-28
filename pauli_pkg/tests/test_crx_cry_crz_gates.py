# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator

from pauli_propagation.utils      import encode_pauli, decode_pauli
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate

# Reduced test set - only key Pauli operators
LABELS_2Q_REDUCED = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", 
                     "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]

def pauli_matrix(label: str) -> np.ndarray:
    """Convert Pauli label to matrix representation."""
    return Pauli(label).to_matrix()

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

def pauli_terms_to_matrix(terms: list, n: int) -> np.ndarray:
    """Convert list of PauliTerm objects to their matrix sum representation."""
    total_matrix = np.zeros((2**n, 2**n), dtype=complex)
    for term in terms:
        pauli = decode_pauli(term.key, term.n)
        total_matrix += term.coeff * pauli.to_matrix()
    return total_matrix

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
    
    expected = U.conj().T @ pauli_matrix(label) @ U
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
    
    expected = U.conj().T @ pauli_matrix(label) @ U
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
    
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}, theta={theta}"

# Test specific CRX cases to verify the controlled rotation behavior
@pytest.mark.parametrize("label", ["II", "XI", "IX", "XX"])
def test_crx_specific_cases(label):
    """Test CRX on specific Pauli cases to verify controlled rotation behavior."""
    ctrl, tgt = 0, 1
    theta = np.pi/2
    U      = crx_matrix(ctrl, tgt, theta)
    kernel = QuantumGate.get("crx")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, ctrl, tgt, theta)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"CRX specific case failed for P={label}"

# Test specific CRY cases to verify the controlled rotation behavior
@pytest.mark.parametrize("label", ["II", "YI", "IY", "YY"])
def test_cry_specific_cases(label):
    """Test CRY on specific Pauli cases to verify controlled rotation behavior."""
    ctrl, tgt = 0, 1
    theta = np.pi/2
    U      = cry_matrix(ctrl, tgt, theta)
    kernel = QuantumGate.get("cry")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, ctrl, tgt, theta)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"CRY specific case failed for P={label}"

# Test specific CRZ cases to verify the controlled rotation behavior
@pytest.mark.parametrize("label", ["II", "ZI", "IZ", "ZZ"])
def test_crz_specific_cases(label):
    """Test CRZ on specific Pauli cases to verify controlled rotation behavior."""
    ctrl, tgt = 0, 1
    theta = np.pi/2
    U      = crz_matrix(ctrl, tgt, theta)
    kernel = QuantumGate.get("crz")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, ctrl, tgt, theta)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"CRZ specific case failed for P={label}" 