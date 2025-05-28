# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from qiskit.quantum_info import Pauli

from pauli_propagation.utils import encode_pauli, decode_pauli
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates import QuantumGate

# Pauli gate matrices
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)

GATE_MATRICES = {"x": X_GATE, "y": Y_GATE, "z": Z_GATE}

def pauli_terms_to_matrix(terms, n):
    """
    Reconstruct sum(alpha_j * P_j) from a list of PauliTerm objects.
    """
    if isinstance(terms, PauliTerm):
        terms = [terms]
    
    total_matrix = np.zeros((2**n, 2**n), dtype=complex)
    for term in terms:
        pauli = decode_pauli(term.key, term.n)
        total_matrix += term.coeff * pauli.to_matrix()
    return total_matrix

@pytest.mark.parametrize("gate_name", ["x", "y", "z"])
@pytest.mark.parametrize("label", ["I", "X", "Y", "Z"])
def test_xyz_single(gate_name, label):
    """
    Direct test of single-qubit X/Y/Z gate conjugation:
    G^dagger P G == sum of output terms
    """
    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 1)
    output_terms = QuantumGate.get(gate_name)(input_term, 0)
    
    result = pauli_terms_to_matrix(output_terms, 1)
    gate_matrix = GATE_MATRICES[gate_name]
    expected = gate_matrix.conj().T @ Pauli(label).to_matrix() @ gate_matrix
    assert np.allclose(result, expected), f"Failed for gate {gate_name.upper()} on P={label}"

TRIALS = 20

@pytest.mark.parametrize("gate_name", ["x", "y", "z"])
@pytest.mark.parametrize("trial", range(TRIALS))
def test_xyz_random_embedded(gate_name, trial):
    """
    Embed X/Y/Z gate on a random qubit in 4-qubit Pauli, compare with full tensor product construction.
    """
    # Generate random 4-qubit Pauli label
    num_qubits = 8
    label = "".join(random.choice("IXYZ") for _ in range(num_qubits))
    q      = random.randrange(num_qubits)
    key    = encode_pauli(Pauli(label))

    # Apply gate using bit-kernel
    input_term = PauliTerm(1.0, key, num_qubits)
    output_terms = QuantumGate.get(gate_name)(input_term, q)
    matsum = pauli_terms_to_matrix(output_terms, num_qubits)

    # Build reference via tensor products (Qiskit little-endian)
    mats = []
    gate_matrix = GATE_MATRICES[gate_name]
    for idx, ch in enumerate(reversed(label)):  # qubit-0 = rightmost
        base = Pauli(ch).to_matrix()
        if idx == q:
            base = gate_matrix.conj().T @ base @ gate_matrix
        mats.append(base)

    ref = mats[0]
    for m in mats[1:]:
        ref = np.kron(m, ref)

    assert np.allclose(matsum, ref), f"Embedded {gate_name.upper()} mismatch on qubit {q}, label {label}" 