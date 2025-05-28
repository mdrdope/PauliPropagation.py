# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator

from pauli_propagation.utils      import encode_pauli, decode_pauli
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate

# All 3-qubit Pauli labels (64 total)
LABELS_3Q = ["".join(p) for p in itertools.product("IXYZ", repeat=3)]

def pauli_matrix(label: str) -> np.ndarray:
    """Convert Pauli label to matrix representation."""
    return Pauli(label).to_matrix()

def ccx_matrix(ctrl1: int, ctrl2: int, tgt: int) -> np.ndarray:
    """Generate CCX (Toffoli) gate matrix for given control and target qubits."""
    qc = QuantumCircuit(3)
    qc.ccx(ctrl1, ctrl2, tgt)
    return Operator(qc).data

def pauli_terms_to_matrix(terms: list, n: int) -> np.ndarray:
    """Convert list of PauliTerm objects to their matrix sum representation."""
    total_matrix = np.zeros((2**n, 2**n), dtype=complex)
    for term in terms:
        pauli = decode_pauli(term.key, term.n)
        total_matrix += term.coeff * pauli.to_matrix()
    return total_matrix

@pytest.mark.parametrize("ctrl1,ctrl2,tgt", [
    (0, 1, 2), (0, 2, 1), (1, 0, 2), 
    (1, 2, 0), (2, 0, 1), (2, 1, 0)
])
@pytest.mark.parametrize("label", LABELS_3Q)
def test_ccx_rule(ctrl1, ctrl2, tgt, label):
    """Exhaustively test CCX gate conjugation for all 3-qubit Paulis."""
    U      = ccx_matrix(ctrl1, ctrl2, tgt)
    kernel = QuantumGate.get("ccx")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 3)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl1, ctrl2, tgt)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 3)
    
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl1={ctrl1}, ctrl2={ctrl2}, tgt={tgt}, P={label}"

# Test specific cases to verify the decomposition works correctly
@pytest.mark.parametrize("label", ["XII", "IXI", "IIX", "XYZ", "YYY"])
def test_ccx_specific_cases(label):
    """Test CCX on specific Pauli cases to verify correctness."""
    ctrl1, ctrl2, tgt = 0, 1, 2
    U      = ccx_matrix(ctrl1, ctrl2, tgt)
    kernel = QuantumGate.get("ccx")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 3)
    output_terms = kernel(input_term, ctrl1, ctrl2, tgt)
    
    matsum = pauli_terms_to_matrix(output_terms, 3)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"Specific case failed for P={label}"

# Test that CCX preserves coefficient magnitude
def test_ccx_coefficient_preservation():
    """Test that CCX preserves the magnitude of coefficients."""
    label = "XYZ"
    coeff = 2.5 + 1.5j
    key = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(coeff, key, 3)
    output_terms = QuantumGate.get("ccx")(input_term, 0, 1, 2)
    
    # Sum of squared magnitudes should be preserved
    input_mag_sq = abs(coeff)**2
    output_mag_sq = sum(abs(term.coeff)**2 for term in output_terms)
    
    assert abs(input_mag_sq - output_mag_sq) < 1e-12, "Coefficient magnitude not preserved"

# Test edge cases
def test_ccx_identity():
    """Test CCX on identity Pauli."""
    key = encode_pauli(Pauli("III"))
    input_term = PauliTerm(1.0, key, 3)
    output_terms = QuantumGate.get("ccx")(input_term, 0, 1, 2)
    
    assert len(output_terms) == 1
    assert output_terms[0].key == key
    assert abs(output_terms[0].coeff - 1.0) < 1e-12 