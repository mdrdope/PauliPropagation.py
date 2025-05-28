#!/usr/bin/env python3

import pytest
import numpy as np
from math import pi
from qiskit import QuantumCircuit

from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.propagator import PauliPropagator
from pauli_propagation.gates import QuantumGate

# Test parameters
LABELS_2Q = ["II", "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", 
             "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]

def pauli_from_label(label: str, n: int) -> PauliTerm:
    """Convert Pauli label to PauliTerm."""
    key = 0
    for i, p in enumerate(reversed(label)):
        if p == 'X':
            key |= 1 << i
        elif p == 'Y':
            key |= (1 << i) | (1 << (n + i))
        elif p == 'Z':
            key |= 1 << (n + i)
    return PauliTerm(1.0, key, n)

def apply_gate_direct(gate_name: str, pauli_term: PauliTerm, q1: int, q2: int, theta: float) -> list:
    """Apply gate directly using gate function."""
    gate_func = QuantumGate.get(gate_name)
    return gate_func(pauli_term, q1, q2, theta)

def apply_gate_via_propagator(gate_name: str, pauli_term: PauliTerm, q1: int, q2: int, theta: float) -> list:
    """Apply gate via propagator."""
    qc = QuantumCircuit(pauli_term.n)
    if gate_name == "rxx":
        qc.rxx(theta, q1, q2)
    elif gate_name == "ryy":
        qc.ryy(theta, q1, q2)
    elif gate_name == "rzz":
        qc.rzz(theta, q1, q2)
    else:
        raise ValueError(f"Unknown gate: {gate_name}")
    
    prop = PauliPropagator(qc)
    history = prop.propagate(pauli_term)
    return history[-1]

@pytest.mark.parametrize("gate_name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("label", LABELS_2Q)
@pytest.mark.parametrize("theta", [0.0, pi/4, pi/2, pi, 3*pi/2])
def test_rxxyyzz_single(gate_name, label, theta):
    """Test single RXX/RYY/RZZ gate application on 2-qubit Pauli terms."""
    n = 2
    pauli_term = pauli_from_label(label, n)
    
    # Apply gate directly
    direct_result = apply_gate_direct(gate_name, pauli_term, 0, 1, theta)
    
    # Apply gate via propagator
    prop_result = apply_gate_via_propagator(gate_name, pauli_term, 0, 1, theta)
    
    # Compare results
    assert len(direct_result) == len(prop_result), f"Length mismatch for {gate_name}({theta}) on {label}"
    
    # Sort both results by key for comparison
    direct_sorted = sorted(direct_result, key=lambda t: t.key)
    prop_sorted = sorted(prop_result, key=lambda t: t.key)
    
    for d_term, p_term in zip(direct_sorted, prop_sorted):
        assert d_term.key == p_term.key, f"Key mismatch for {gate_name}({theta}) on {label}"
        assert abs(d_term.coeff - p_term.coeff) < 1e-12, f"Coefficient mismatch for {gate_name}({theta}) on {label}"

TRIALS = 20

@pytest.mark.parametrize("gate_name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("trial", range(TRIALS))
def test_rxxyyzz_random_embedded(gate_name, trial):
    """Test RXX/RYY/RZZ gates on random embedded Pauli terms."""
    np.random.seed(trial)
    
    # Random circuit parameters
    n = np.random.randint(3, 6)  # 3-5 qubits
    q1, q2 = np.random.choice(n, 2, replace=False)
    theta = np.random.uniform(0, 2*pi)
    
    # Random Pauli term
    label = "".join(np.random.choice(["I", "X", "Y", "Z"], n))
    pauli_term = pauli_from_label(label, n)
    
    # Apply gate directly
    direct_result = apply_gate_direct(gate_name, pauli_term, q1, q2, theta)
    
    # Apply gate via propagator
    qc = QuantumCircuit(n)
    if gate_name == "rxx":
        qc.rxx(theta, q1, q2)
    elif gate_name == "ryy":
        qc.ryy(theta, q1, q2)
    elif gate_name == "rzz":
        qc.rzz(theta, q1, q2)
    
    prop = PauliPropagator(qc)
    history = prop.propagate(pauli_term)
    prop_result = history[-1]
    
    # Compare results
    assert len(direct_result) == len(prop_result), f"Length mismatch for {gate_name}({theta}) on {label}, qubits ({q1},{q2})"
    
    # Sort both results by key for comparison
    direct_sorted = sorted(direct_result, key=lambda t: t.key)
    prop_sorted = sorted(prop_result, key=lambda t: t.key)
    
    for d_term, p_term in zip(direct_sorted, prop_sorted):
        assert d_term.key == p_term.key, f"Key mismatch for {gate_name}({theta}) on {label}, qubits ({q1},{q2})"
        assert abs(d_term.coeff - p_term.coeff) < 1e-12, f"Coefficient mismatch for {gate_name}({theta}) on {label}, qubits ({q1},{q2})" 