# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from math import pi
from qiskit.quantum_info import Pauli

from pauli_propagation.utils import encode_pauli, decode_pauli, random_pauli_label, random_state_label
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

import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator

from pauli_propagation.utils      import encode_pauli, decode_pauli, random_pauli_label
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate
from pauli_propagation.propagator import PauliPropagator

# All single-qubit Pauli labels
LABELS_1Q = ["I", "X", "Y", "Z"]

# Test angles - 8 evenly spaced angles including edge cases
ANGLES = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, np.pi]

def pauli_matrix(label: str) -> np.ndarray:
    """Convert Pauli label to matrix representation."""
    return Pauli(label).to_matrix()

def gate_matrix(gate_name: str, q: int, n: int, theta: float = None) -> np.ndarray:
    """Generate gate matrix for given gate and qubit position in n-qubit system."""
    qc = QuantumCircuit(n)
    if gate_name == "rx":
        qc.rx(theta, q)
    elif gate_name == "ry":
        qc.ry(theta, q)
    elif gate_name == "rz":
        qc.rz(theta, q)
    else:
        raise ValueError(f"Unknown gate: {gate_name}")
    return Operator(qc).data

def pauli_terms_to_matrix(terms: list, n: int) -> np.ndarray:
    """Convert list of PauliTerm objects to their matrix sum representation."""
    total_matrix = np.zeros((2**n, 2**n), dtype=complex)
    for term in terms:
        pauli = decode_pauli(term.key, term.n)
        total_matrix += term.coeff * pauli.to_matrix()
    return total_matrix

@pytest.mark.parametrize("gate_name", ["rx", "ry", "rz"])
@pytest.mark.parametrize("q", [0, 1])
@pytest.mark.parametrize("label", LABELS_1Q)
@pytest.mark.parametrize("theta", ANGLES)
def test_rotation_gates(gate_name, q, label, theta):
    """Test rotation gates on single-qubit Paulis embedded in 2-qubit system."""
    # Create 2-qubit label with rotation applied to qubit q
    if q == 0:
        full_label = label + "I"
    else:
        full_label = "I" + label
    
    U      = gate_matrix(gate_name, q, 2, theta)
    kernel = QuantumGate.get(gate_name)
    key    = encode_pauli(Pauli(full_label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, q, theta)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ pauli_matrix(full_label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for {gate_name}({theta}) on q={q}, P={full_label}"

# Test specific cases for each rotation gate
@pytest.mark.parametrize("gate_name", ["rx", "ry", "rz"])
@pytest.mark.parametrize("label", ["X", "Y", "Z", "I"])
def test_rotation_specific_cases(gate_name, label):
    """Test rotation gates on specific single-qubit Pauli cases."""
    q = 0
    theta = np.pi/2
    full_label = label + "I"  # Apply rotation to first qubit
    U      = gate_matrix(gate_name, q, 2, theta)
    kernel = QuantumGate.get(gate_name)
    key    = encode_pauli(Pauli(full_label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, q, theta)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(full_label) @ U
    
    assert np.allclose(matsum, expected), f"{gate_name} specific case failed for P={full_label}"

def apply_gate_via_propagator(qc: QuantumCircuit, pauli_term: PauliTerm) -> list:
    """Apply quantum circuit via propagator."""
    prop = PauliPropagator(qc)
    history = prop.propagate(pauli_term)
    return history[-1]

@pytest.mark.parametrize("trial", range(10))
def test_rx_ry_rz_random_circuits(trial):
    """Test RX, RY, RZ gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 5000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Available gate types
    gate_types = ['rx', 'ry', 'rz']
    
    # Create quantum circuit with random rotation gates
    qc = QuantumCircuit(n, name=f"rxyz_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random gate type and qubit
        gate_type = np.random.choice(gate_types)
        q = np.random.randint(0, n)
        theta = np.random.uniform(0, 2*np.pi)
        
        # Add gate to circuit
        if gate_type == 'rx':
            qc.rx(theta, q)
        elif gate_type == 'ry':
            qc.ry(theta, q)
        elif gate_type == 'rz':
            qc.rz(theta, q)
    
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