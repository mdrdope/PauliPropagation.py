# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit

from pauli_propagation.utils import encode_pauli, decode_pauli, random_pauli_label, random_state_label
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates import QuantumGate
from pauli_propagation.propagator import PauliPropagator

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

# All single-qubit Pauli labels
LABELS = ["I", "X", "Y", "Z"]

def decode_key(key, n):
    """Helper to decode key back to Pauli"""
    pauli = decode_pauli(key, n)
    return str(pauli)

@pytest.mark.parametrize("gate", ["x", "y", "z"])
@pytest.mark.parametrize("label", LABELS)
def test_pauli_gates_single(gate, label):
    """Test X, Y, Z gates on single-qubit Paulis."""
    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 1)
    
    output_terms = QuantumGate.get(gate)(input_term, 0)
    
    # Should always return exactly one term for Pauli gates
    assert len(output_terms) == 1, f"Expected 1 term, got {len(output_terms)} for {gate} on {label}"
    
    result = output_terms[0]
    
    # Check known transformations
    if gate == "x":
        if label == "I": expected = ("I", 1.0)
        elif label == "X": expected = ("X", 1.0)
        elif label == "Y": expected = ("Y", -1.0)
        elif label == "Z": expected = ("Z", -1.0)
    elif gate == "y":
        if label == "I": expected = ("I", 1.0)
        elif label == "X": expected = ("X", -1.0)
        elif label == "Y": expected = ("Y", 1.0)
        elif label == "Z": expected = ("Z", -1.0)
    elif gate == "z":
        if label == "I": expected = ("I", 1.0)
        elif label == "X": expected = ("X", -1.0)
        elif label == "Y": expected = ("Y", -1.0)
        elif label == "Z": expected = ("Z", 1.0)
    
    expected_label, expected_coeff = expected
    result_label = decode_key(result.key, 1)
    
    assert result_label == expected_label, f"{gate} on {label}: expected {expected_label}, got {result_label}"
    assert abs(result.coeff - expected_coeff) < 1e-12, f"{gate} on {label}: expected coeff {expected_coeff}, got {result.coeff}"

@pytest.mark.parametrize("trial", range(10))
def test_pauli_gates_embedded(trial):
    """Test X, Y, Z gates embedded in random multi-qubit Paulis."""
    np.random.seed(trial)
    n_qubits = random.randint(2, 5)
    
    # Generate random Pauli string
    label = "".join(random.choice(LABELS) for _ in range(n_qubits))
    target_qubit = random.randint(0, n_qubits - 1)
    gate = random.choice(["x", "y", "z"])
    
    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, n_qubits)
    
    output_terms = QuantumGate.get(gate)(input_term, target_qubit)
    
    assert len(output_terms) == 1, f"Expected 1 term for {gate} gate"
    
    result = output_terms[0]
    result_label = decode_key(result.key, n_qubits)
    
    # Verify only the target qubit is affected according to Pauli gate rules
    original_char = label[n_qubits - 1 - target_qubit]  # Qiskit little-endian
    
    # Build expected result
    expected_chars = list(label)
    if gate == "x":
        if original_char == "Y": expected_coeff = -1.0
        elif original_char == "Z": expected_coeff = -1.0
        else: expected_coeff = 1.0
    elif gate == "y":
        if original_char == "X": expected_coeff = -1.0
        elif original_char == "Z": expected_coeff = -1.0
        else: expected_coeff = 1.0
    elif gate == "z":
        if original_char == "X": expected_coeff = -1.0
        elif original_char == "Y": expected_coeff = -1.0
        else: expected_coeff = 1.0
    
    expected_label = "".join(expected_chars)
    
    assert result_label == expected_label, f"Embedded {gate} gate failed: expected {expected_label}, got {result_label}"
    assert abs(result.coeff - expected_coeff) < 1e-12, f"Embedded {gate} gate coeff mismatch"

def test_pauli_gates_coefficient_preservation():
    """Test that Pauli gates preserve coefficient magnitude."""
    coeff = 2.5 + 1.5j
    label = "X"
    key = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(coeff, key, 1)
    
    for gate in ["x", "y", "z"]:
        output_terms = QuantumGate.get(gate)(input_term, 0)
        result = output_terms[0]
        
        # Magnitude should be preserved (only phase can change)
        assert abs(abs(result.coeff) - abs(coeff)) < 1e-12, f"{gate} gate magnitude not preserved"

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
def test_xyz_random_circuits(trial):
    """Test X, Y, Z gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 7000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Available gate types
    gate_types = ['x', 'y', 'z']
    
    # Create quantum circuit with random Pauli gates
    qc = QuantumCircuit(n, name=f"xyz_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random gate type and qubit
        gate_type = np.random.choice(gate_types)
        q = np.random.randint(0, n)
        
        # Add gate to circuit
        if gate_type == 'x':
            qc.x(q)
        elif gate_type == 'y':
            qc.y(q)
        elif gate_type == 'z':
            qc.z(q)
    
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