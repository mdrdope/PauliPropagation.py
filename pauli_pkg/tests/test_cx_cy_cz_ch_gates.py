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

def cx_matrix(ctrl: int, tgt: int) -> np.ndarray:
    """Generate CX gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.cx(ctrl, tgt)
    return Operator(qc).data

def cz_matrix(ctrl: int, tgt: int) -> np.ndarray:
    """Generate CZ gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.cz(ctrl, tgt)
    return Operator(qc).data

def cy_matrix(ctrl: int, tgt: int) -> np.ndarray:
    """Generate CY gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.cy(ctrl, tgt)
    return Operator(qc).data

def ch_matrix(ctrl: int, tgt: int) -> np.ndarray:
    """Generate CH gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.ch(ctrl, tgt)
    return Operator(qc).data

def pauli_terms_to_matrix(terms: list, n: int) -> np.ndarray:
    """Convert list of PauliTerm objects to their matrix sum representation."""
    total_matrix = np.zeros((2**n, 2**n), dtype=complex)
    for term in terms:
        pauli = decode_pauli(term.key, term.n)
        total_matrix += term.coeff * pauli.to_matrix()
    return total_matrix

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_cx_rule(ctrl, tgt, label):
    """Exhaustively test CX gate conjugation for all 2-qubit Paulis."""
    U      = cx_matrix(ctrl, tgt)
    kernel = QuantumGate.get("cx")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - now returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}"

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_cz_rule(ctrl, tgt, label):
    """Exhaustively test CZ gate conjugation for all 2-qubit Paulis."""
    U      = cz_matrix(ctrl, tgt)
    kernel = QuantumGate.get("cz")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}"

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_cy_rule(ctrl, tgt, label):
    """Exhaustively test CY gate conjugation for all 2-qubit Paulis."""
    U      = cy_matrix(ctrl, tgt)
    kernel = QuantumGate.get("cy")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}"

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_ch_rule(ctrl, tgt, label):
    """Exhaustively test CH gate conjugation for all 2-qubit Paulis."""
    U      = ch_matrix(ctrl, tgt)
    kernel = QuantumGate.get("ch")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}"

# Test specific CZ cases to verify the bit manipulation rules
@pytest.mark.parametrize("label", ["XI", "IX", "XX", "YY", "ZZ"])
def test_cz_specific_cases(label):
    """Test CZ on specific Pauli cases to verify bit manipulation rules."""
    ctrl, tgt = 0, 1
    U      = cz_matrix(ctrl, tgt)
    kernel = QuantumGate.get("cz")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, ctrl, tgt)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"CZ specific case failed for P={label}"

# Test specific CY cases to verify the bit manipulation rules
@pytest.mark.parametrize("label", ["XI", "IX", "XX", "YI", "IY", "YY"])
def test_cy_specific_cases(label):
    """Test CY on specific Pauli cases to verify bit manipulation rules."""
    ctrl, tgt = 0, 1
    U      = cy_matrix(ctrl, tgt)
    kernel = QuantumGate.get("cy")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, ctrl, tgt)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"CY specific case failed for P={label}"

# Test specific CH cases to verify the controlled-H behavior
@pytest.mark.parametrize("label", ["II", "XI", "IX", "XX", "ZI", "IZ", "ZZ"])
def test_ch_specific_cases(label):
    """Test CH on specific Pauli cases to verify controlled-H behavior."""
    ctrl, tgt = 0, 1
    U      = ch_matrix(ctrl, tgt)
    kernel = QuantumGate.get("ch")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, ctrl, tgt)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"CH specific case failed for P={label}" 