# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator

from pauli_propagation.utils      import encode_pauli, decode_pauli
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate

LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

def pauli_matrix(label: str) -> np.ndarray:
    """Convert Pauli label to matrix representation."""
    return Pauli(label).to_matrix()

def swap_matrix(q1: int, q2: int) -> np.ndarray:
    """Generate SWAP gate matrix for given qubits."""
    qc = QuantumCircuit(2)
    qc.swap(q1, q2)
    return Operator(qc).data

def iswap_matrix(q1: int, q2: int) -> np.ndarray:
    """Generate iSWAP gate matrix for given qubits."""
    qc = QuantumCircuit(2)
    qc.iswap(q1, q2)
    return Operator(qc).data

def pauli_terms_to_matrix(terms: list, n: int) -> np.ndarray:
    """Convert list of PauliTerm objects to their matrix sum representation."""
    total_matrix = np.zeros((2**n, 2**n), dtype=complex)
    for term in terms:
        pauli = decode_pauli(term.key, term.n)
        total_matrix += term.coeff * pauli.to_matrix()
    return total_matrix

@pytest.mark.parametrize("q1,q2", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_swap_rule(q1, q2, label):
    """Exhaustively test SWAP gate conjugation for all 2-qubit Paulis."""
    U      = swap_matrix(q1, q2)
    kernel = QuantumGate.get("swap")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - now returns List[PauliTerm]
    output_terms = kernel(input_term, q1, q2)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for q1={q1}, q2={q2}, P={label}"

@pytest.mark.parametrize("q1,q2", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_iswap_rule(q1, q2, label):
    """Exhaustively test iSWAP gate conjugation for all 2-qubit Paulis."""
    U      = iswap_matrix(q1, q2)
    kernel = QuantumGate.get("iswap")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, q1, q2)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for q1={q1}, q2={q2}, P={label}"

# Test specific SWAP cases to verify the bit manipulation rules
@pytest.mark.parametrize("label", ["II", "XI", "IX", "XX", "YI", "IY", "YY", "ZI", "IZ", "ZZ"])
def test_swap_specific_cases(label):
    """Test SWAP on specific Pauli cases to verify bit manipulation rules."""
    q1, q2 = 0, 1
    U      = swap_matrix(q1, q2)
    kernel = QuantumGate.get("swap")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, q1, q2)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"SWAP specific case failed for P={label}"

# Test specific iSWAP cases to verify the behavior
@pytest.mark.parametrize("label", ["II", "XI", "IX", "XX", "YI", "IY", "YY", "ZI", "IZ", "ZZ"])
def test_iswap_specific_cases(label):
    """Test iSWAP on specific Pauli cases to verify behavior."""
    q1, q2 = 0, 1
    U      = iswap_matrix(q1, q2)
    kernel = QuantumGate.get("iswap")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, q1, q2)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"iSWAP specific case failed for P={label}" 