# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from qiskit.quantum_info import Pauli

from pauli_propagation.utils import encode_pauli, decode_pauli
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates import QuantumGate

# Hadamard gate matrix
H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

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

@pytest.mark.parametrize("label", ["I", "X", "Y", "Z"])
def test_h_single(label):
    """
    Direct test of single-qubit H gate conjugation:
    H^dagger P H == sum of output terms
    """
    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 1)
    output_terms = QuantumGate.get("h")(input_term, 0)
    
    result = pauli_terms_to_matrix(output_terms, 1)
    expected = H_GATE.conj().T @ Pauli(label).to_matrix() @ H_GATE
    assert np.allclose(result, expected), f"Failed for P={label}"

TRIALS = 20

@pytest.mark.parametrize("trial", range(TRIALS))
def test_h_random_embedded(trial):
    """
    Embed H gate on a random qubit in 4-qubit Pauli, compare with full tensor product construction.
    """
    num_qubits = 5
    # Generate random 4-qubit Pauli label
    label = "".join(random.choice("IXYZ") for _ in range(num_qubits))
    q      = random.randrange(num_qubits)
    key    = encode_pauli(Pauli(label))

    # Apply H gate using bit-kernel
    input_term = PauliTerm(1.0, key, num_qubits)
    output_terms = QuantumGate.get("h")(input_term, q)
    matsum = pauli_terms_to_matrix(output_terms, num_qubits)

    # Build reference via tensor products (Qiskit little-endian)
    mats = []
    for idx, ch in enumerate(reversed(label)):  # qubit-0 = rightmost
        base = Pauli(ch).to_matrix()
        if idx == q:
            base = H_GATE.conj().T @ base @ H_GATE
        mats.append(base)

    ref = mats[0]
    for m in mats[1:]:
        ref = np.kron(m, ref)

    assert np.allclose(matsum, ref), f"Embedded H mismatch on qubit {q}, label {label}" 