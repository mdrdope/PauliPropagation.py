# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from math import pi
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit

from pauli_propagation.utils import encode_pauli, decode_pauli, random_pauli_label, random_state_label
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates import QuantumGate
from pauli_propagation.propagator import PauliPropagator

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

TRIALS = 20

@pytest.mark.parametrize("trial", range(TRIALS))
def test_t_random_embedded(trial):
    """
    Embed T gate on a random qubit in 4-qubit Pauli, compare with full tensor product construction.
    """
    num_qubits = 5
    # Generate random 4-qubit Pauli label
    label = "".join(random.choice("IXYZ") for _ in range(num_qubits))
    q      = random.randrange(num_qubits)
    key    = encode_pauli(Pauli(label))

    # Apply T gate using bit-kernel
    input_term = PauliTerm(1.0, key, num_qubits)
    output_terms = QuantumGate.get("t")(input_term, q)
    matsum = pauli_terms_to_matrix(output_terms, num_qubits)

    # Build reference via tensor products (Qiskit little-endian)
    mats = []
    for idx, ch in enumerate(reversed(label)):  # qubit-0 = rightmost
        base = Pauli(ch).to_matrix()
        if idx == q:
            base = T_GATE.conj().T @ base @ T_GATE
        mats.append(base)

    ref = mats[0]
    for m in mats[1:]:
        ref = np.kron(m, ref)

    assert np.allclose(matsum, ref), f"Embedded T mismatch on qubit {q}, label {label}"

def generate_random_pauli_label(n: int) -> str:
    """Generate a random Pauli label of length n."""
    return "".join(np.random.choice(["I", "X", "Y", "Z"], n))

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

def apply_gate_via_propagator(qc: QuantumCircuit, pauli_term: PauliTerm) -> list:
    """Apply quantum circuit via propagator."""
    prop = PauliPropagator(qc)
    history = prop.propagate(pauli_term)
    return history[-1]

@pytest.mark.parametrize("trial", range(10))
def test_t_random_circuits(trial):
    """Test T gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 8000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Create quantum circuit with random T gates
    qc = QuantumCircuit(n, name=f"t_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random qubit for T gate
        q = np.random.randint(0, n)
        qc.t(q)
    
    # Method 1: PauliPropagator expectation
    prop = PauliPropagator(qc)
    layers = prop.propagate(observable, max_weight=None)
    pauli_expectation = prop.expectation_pauli_sum(layers[-1], state_label)
    
    # Method 2: Qiskit statevector expectation
    from qiskit.quantum_info import Statevector
    initial_state = Statevector.from_label(state_label)
    final_state = initial_state.evolve(qc)
    qiskit_expectation = final_state.expectation_value(Pauli(pauli_label)).real
    
    # Compare results
    assert abs(pauli_expectation - qiskit_expectation) < 1e-10, (
        f"Trial {trial}: expectation mismatch {pauli_expectation} vs {qiskit_expectation} "
        f"on state {state_label}, observable {pauli_label}, circuit {n}q {n_gates}g"
    )
