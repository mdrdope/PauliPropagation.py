# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from math import pi
from qiskit.quantum_info import Pauli

from pauli_propagation.utils import encode_pauli, decode_pauli
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates import QuantumGate

# Rotation gate matrices
def rx_matrix(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

def ry_matrix(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def rz_matrix(theta):
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]], dtype=complex)

GATE_MATRICES = {"rx": rx_matrix, "ry": ry_matrix, "rz": rz_matrix}

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

@pytest.mark.parametrize("gate_name", ["rx", "ry", "rz"])
@pytest.mark.parametrize("label", ["I", "X", "Y", "Z"])
@pytest.mark.parametrize("theta", [0.0, pi/4, pi/2, pi, 3*pi/2])
def test_rxyz_single(gate_name, label, theta):
    """
    Direct test of single-qubit rotation gate conjugation:
    R(θ)^dagger P R(θ) == sum of output terms
    """
    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 1)
    output_terms = QuantumGate.get(gate_name)(input_term, 0, theta)
    
    result = pauli_terms_to_matrix(output_terms, 1)
    gate_matrix = GATE_MATRICES[gate_name](theta)
    expected = gate_matrix.conj().T @ Pauli(label).to_matrix() @ gate_matrix
    assert np.allclose(result, expected), f"Failed for gate {gate_name.upper()}({theta}) on P={label}"

TRIALS = 20

@pytest.mark.parametrize("gate_name", ["rx", "ry", "rz"])
@pytest.mark.parametrize("trial", range(TRIALS))
def test_rxyz_random_embedded(gate_name, trial):
    """
    Embed rotation gate on a random qubit in 4-qubit Pauli, compare with full tensor product construction.
    """
    # Generate random 4-qubit Pauli label and random angle
    label4 = "".join(random.choice("IXYZ") for _ in range(4))
    q      = random.randrange(4)
    theta  = random.uniform(0, 2*pi)
    key    = encode_pauli(Pauli(label4))

    # Apply rotation gate using bit-kernel
    input_term = PauliTerm(1.0, key, 4)
    output_terms = QuantumGate.get(gate_name)(input_term, q, theta)
    matsum = pauli_terms_to_matrix(output_terms, 4)

    # Build reference via tensor products (Qiskit little-endian)
    mats = []
    gate_matrix = GATE_MATRICES[gate_name](theta)
    for idx, ch in enumerate(reversed(label4)):  # qubit-0 = rightmost
        base = Pauli(ch).to_matrix()
        if idx == q:
            base = gate_matrix.conj().T @ base @ gate_matrix
        mats.append(base)

    ref = mats[0]
    for m in mats[1:]:
        ref = np.kron(m, ref)

    assert np.allclose(matsum, ref), f"Embedded {gate_name.upper()}({theta}) mismatch on qubit {q}, label {label4}" 