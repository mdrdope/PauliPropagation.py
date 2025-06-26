# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator, Statevector
from pauli_propagation.utils import pauli_terms_to_matrix, encode_pauli, decode_pauli, random_pauli_label, random_state_label
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate
from pauli_propagation.propagator import PauliPropagator
import random

# All 2-qubit Pauli labels (16 total)
LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

def swap_matrix(q1: int, q2: int, n: int) -> np.ndarray:
    """Generate SWAP gate matrix for given qubits in n-qubit system."""
    qc = QuantumCircuit(n)
    qc.swap(q1, q2)
    return Operator(qc).data

def iswap_matrix(q1: int, q2: int, n: int) -> np.ndarray:
    """Generate iSWAP gate matrix for given qubits in n-qubit system."""
    qc = QuantumCircuit(n)
    qc.iswap(q1, q2)
    return Operator(qc).data

# -----------------------------------------------------------------------------
# 1. Unified SWAP / iSWAP conjugation rules
# -----------------------------------------------------------------------------

# Parameterize over gate type, qubit pair, and Pauli label (16 possibilities)
@pytest.mark.parametrize("gate_name", ["swap", "iswap"])
@pytest.mark.parametrize("q_pair", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_swap_iswap_rules(gate_name, q_pair, label):
    """Verify U^† P U for SWAP / iSWAP on all 2-qubit Paulis."""
    q1, q2 = q_pair

    # Select correct unitary generator and kernel
    if gate_name == "swap":
        U = swap_matrix(q1, q2, 2)
    else:  # iswap
        U = iswap_matrix(q1, q2, 2)

    kernel = QuantumGate.get(gate_name)
    key    = encode_pauli(Pauli(label))

    input_term   = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, q1, q2)

    matsum   = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ Pauli(label).to_matrix() @ U

    assert np.allclose(matsum, expected), (
        f"Mismatch for {gate_name.upper()}({q1},{q2}) on P={label}")

# -----------------------------------------------------------------------------
# 2. Random embedded test (single gate in larger system)
# -----------------------------------------------------------------------------

TRIALS = 20

@pytest.mark.parametrize("gate_name", ["swap", "iswap"])
@pytest.mark.parametrize("trial", range(TRIALS))
def test_swap_iswap_random_embedded(gate_name, trial):
    """Embed SWAP / iSWAP on random qubit pair of an 6-qubit Pauli and compare matrices."""
    num_qubits = 6

    # Random Pauli label and random distinct qubits
    label = "".join(random.choice("IXYZ") for _ in range(num_qubits))
    q1, q2 = random.sample(range(num_qubits), 2)

    # Prepare input Pauli term
    key        = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, num_qubits)

    # Propagation via bit-kernel
    output_terms = QuantumGate.get(gate_name)(input_term, q1, q2)
    matsum       = pauli_terms_to_matrix(output_terms, num_qubits)

    # Reference matrix using Qiskit Operator (no manual kronecker products)
    qc_ref = QuantumCircuit(num_qubits)
    if gate_name == "swap":
        qc_ref.swap(q1, q2)
    else:  # iswap
        qc_ref.iswap(q1, q2)

    G_full = Operator(qc_ref).data  # full 2^n × 2^n unitary
    P_mat  = Pauli(label).to_matrix()
    ref    = G_full.conj().T @ P_mat @ G_full

    assert np.allclose(matsum, ref), (
        f"Embedded {gate_name.upper()} mismatch on qubits ({q1},{q2}), label {label}")


@pytest.mark.parametrize("trial", range(10))
def test_swap_iswap_random_circuits(trial):
    """Test SWAP, iSWAP gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 9000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits (minimum 2 for swap gates)
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Available gate types
    gate_types = ['swap', 'iswap']
    
    # Create quantum circuit with random swap gates
    qc = QuantumCircuit(n, name=f"swap_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random gate type and qubits
        gate_type = np.random.choice(gate_types)
        q1, q2 = np.random.choice(n, 2, replace=False)
        
        # Add gate to circuit
        if gate_type == 'swap':
            qc.swap(q1, q2)
        elif gate_type == 'iswap':
            qc.iswap(q1, q2)
    
    # Method 1: PauliPropagator expectation
    prop = PauliPropagator(qc)
    layers = prop.propagate(observable, max_weight=None)
    pauli_expectation = prop.expectation_pauli_sum(layers[-1], state_label)
    
    # Method 2: Qiskit statevector expectation
    initial_state = Statevector.from_label(state_label)
    final_state = initial_state.evolve(qc)
    qiskit_expectation = final_state.expectation_value(Pauli(pauli_label)).real
    
    # Compare results
    assert abs(pauli_expectation - qiskit_expectation) < 1e-10, (
        f"Trial {trial}: expectation mismatch {pauli_expectation} vs {qiskit_expectation} "
        f"on state {state_label}, observable {pauli_label}, circuit {n}q {n_gates}g"
    ) 
