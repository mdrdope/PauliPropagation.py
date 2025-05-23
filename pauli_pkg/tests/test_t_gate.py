# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from math import pi
from qiskit.quantum_info import Pauli

from pauli_propagation.utils import encode_pauli, decode_pauli
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates import QuantumGate

T_GATE = np.diag([1.0, np.exp(1j * pi / 4)])

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
def test_t_single(label):
    """
    Direct test of single-qubit T gate conjugation:
    T^dagger P T == sum of output terms
    """
    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 1)
    output_terms = QuantumGate.get("t")(input_term, 0)
    
    result = pauli_terms_to_matrix(output_terms, 1)
    expected = T_GATE.conj().T @ Pauli(label).to_matrix() @ T_GATE
    assert np.allclose(result, expected), f"Failed for P={label}"

TRIALS = 50

@pytest.mark.parametrize("trial", range(TRIALS))
def test_t_random_embedded(trial):
    """
    Embed T gate on a random qubit in 4-qubit Pauli, compare with full tensor product construction.
    """
    # Generate random 4-qubit Pauli label
    label4 = "".join(random.choice("IXYZ") for _ in range(4))
    q      = random.randrange(4)
    key    = encode_pauli(Pauli(label4))

    # Apply T gate using bit-kernel
    input_term = PauliTerm(1.0, key, 4)
    output_terms = QuantumGate.get("t")(input_term, q)
    matsum = pauli_terms_to_matrix(output_terms, 4)

    # Build reference via tensor products (Qiskit little-endian)
    mats = []
    for idx, ch in enumerate(reversed(label4)):  # qubit-0 = rightmost
        base = Pauli(ch).to_matrix()
        if idx == q:
            base = T_GATE.conj().T @ base @ T_GATE
        mats.append(base)

    ref = mats[0]
    for m in mats[1:]:
        ref = np.kron(m, ref)

    assert np.allclose(matsum, ref), f"Embedded T mismatch on qubit {q}, label {label4}"
