# -*- coding: utf-8 -*-

import itertools
import random
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli, Operator

from pauli_propagation.utils      import encode_pauli, decode_pauli, random_su2, random_pauli_label, random_state_label
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate
from pauli_propagation.propagator import PauliPropagator

# All single-qubit Pauli labels
LABELS_1Q = ["I", "X", "Y", "Z"]

def pauli_to_matrix(label: str) -> np.ndarray:
    """Convert single-qubit Pauli label to 2x2 matrix."""
    SINGLE = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    return SINGLE[label]

def series_to_matrix(series):
    """Sum alpha*P over a list of PauliTerm, using term.to_label()."""
    acc = np.zeros((2, 2), dtype=complex)
    for term in series:
        mat = pauli_to_matrix(term.to_label())
        acc += term.coeff * mat
    return acc

TRIALS = 30
TOL    = 1e-12

@pytest.mark.parametrize("trial", range(TRIALS))
def test_random_su2_conjugation(trial):
    """Test random SU2 gate conjugation against matrix calculation."""
    # 1) Generate Haar-random SU(2) matrix for qubit 0
    U    = random_su2()
    gate = UnitaryGate(U, label="randSU2")
    gate._name = "su2"

    # 2) Build a 1-qubit circuit with that SU(2) gate
    qc = QuantumCircuit(1)
    qc.append(gate, [0])

    # 3) Pick a random input Pauli operator
    label = np.random.choice(LABELS_1Q)
    key   = encode_pauli(Pauli(label))
    pt    = PauliTerm(1.0, key, 1)

    # 4) Back-propagate through circuit and get the output series
    series = PauliPropagator(qc).propagate(pt, max_weight=None)[-1]

    # 5) Compare matrices: U^dagger P U vs sum alpha_i P_i
    lhs = U.conj().T @ pauli_to_matrix(label) @ U
    rhs = series_to_matrix(series)

    assert np.allclose(lhs, rhs, atol=TOL), (
        f"Trial {trial}: mismatch for Pauli {label}"
    )

@pytest.mark.parametrize("label", LABELS_1Q)
def test_su2_identity_gate(label):
    """Test that identity SU(2) gate leaves Pauli operators unchanged."""
    # Identity matrix
    U = np.eye(2, dtype=complex)
    gate = UnitaryGate(U, label="Identity")
    gate._name = "su2"

    qc = QuantumCircuit(1)
    qc.append(gate, [0])

    key = encode_pauli(Pauli(label))
    pt = PauliTerm(1.0, key, 1)

    series = PauliPropagator(qc).propagate(pt, max_weight=None)[-1]

    # Should get back exactly the same Pauli operator
    assert len(series) == 1, f"Identity should return single term for {label}"
    assert series[0].to_label() == label, f"Identity should preserve Pauli {label}"
    assert abs(series[0].coeff - 1.0) < TOL, f"Identity should preserve coefficient for {label}"

@pytest.mark.parametrize("label", LABELS_1Q)
def test_su2_pauli_gates(label):
    """Test that Pauli gates implemented as SU(2) gates work correctly."""
    pauli_matrices = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex)
    }
    
    for gate_name, matrix in pauli_matrices.items():
        gate = UnitaryGate(matrix, label=gate_name)
        gate._name = "su2"

        qc = QuantumCircuit(1)
        qc.append(gate, [0])

        key = encode_pauli(Pauli(label))
        pt = PauliTerm(1.0, key, 1)

        series = PauliPropagator(qc).propagate(pt, max_weight=None)[-1]

        # Calculate expected result: gate^dagger P gate = gate P gate (since Pauli gates are self-inverse)
        expected = matrix.conj().T @ pauli_to_matrix(label) @ matrix
        actual = series_to_matrix(series)

        assert np.allclose(actual, expected, atol=TOL), (
            f"SU2 {gate_name} gate mismatch for Pauli {label}"
        )

@pytest.mark.parametrize("trial", range(10))
def test_su2_embedded_in_multiqubit(trial):
    """Test SU(2) gate on a random qubit in multi-qubit system."""
    n_qubits = np.random.randint(2, 5)  # 2-4 qubits
    target_qubit = np.random.randint(n_qubits)
    
    # Generate random SU(2) matrix
    U = random_su2()
    gate = UnitaryGate(U, label="randSU2")
    gate._name = "su2"

    # Build circuit
    qc = QuantumCircuit(n_qubits)
    qc.append(gate, [target_qubit])

    # Generate random multi-qubit Pauli operator
    pauli_label = "".join(np.random.choice(LABELS_1Q) for _ in range(n_qubits))
    key = encode_pauli(Pauli(pauli_label))
    pt = PauliTerm(1.0, key, n_qubits)

    # Propagate
    series = PauliPropagator(qc).propagate(pt, max_weight=None)[-1]

    # Build expected result using tensor products
    # For Qiskit little-endian: qubit 0 is rightmost
    single_matrices = []
    for i, pauli_char in enumerate(reversed(pauli_label)):
        if i == target_qubit:
            # Apply U? P U to this qubit
            base = pauli_to_matrix(pauli_char)
            single_matrices.append(U.conj().T @ base @ U)
        else:
            # Keep original Pauli operator
            single_matrices.append(pauli_to_matrix(pauli_char))

    # Build full matrix via tensor products
    expected = single_matrices[0]
    for mat in single_matrices[1:]:
        expected = np.kron(mat, expected)

    # Build actual result from series
    actual = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for term in series:
        pauli_mat = Pauli(term.to_label()).to_matrix()
        actual += term.coeff * pauli_mat

    assert np.allclose(actual, expected, atol=TOL), (
        f"Trial {trial}: SU2 embedded gate mismatch for {pauli_label} on qubit {target_qubit}"
    )

# Generate test matrices
def random_unitary(n: int) -> np.ndarray:
    """Generate a random n×n unitary matrix."""
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D

def pauli_matrix(label: str) -> np.ndarray:
    """Convert Pauli label to matrix representation."""
    return Pauli(label).to_matrix()

def pauli_terms_to_matrix(terms: list, n: int) -> np.ndarray:
    """Convert list of PauliTerm objects to their matrix sum representation."""
    total_matrix = np.zeros((2**n, 2**n), dtype=complex)
    for term in terms:
        pauli = decode_pauli(term.key, term.n)
        total_matrix += term.coeff * pauli.to_matrix()
    return total_matrix

# Test known unitary matrices first
@pytest.mark.parametrize("gate_name", ["h", "x", "y", "z", "s", "t"])
@pytest.mark.parametrize("label", LABELS_1Q)
def test_su2_known_gates(gate_name, label):
    """Test SU2 gate against known single-qubit gates."""
    # Create 2-qubit circuit and get the matrix for the gate
    qc = QuantumCircuit(2)
    if gate_name == "h":
        qc.h(0)
    elif gate_name == "x":
        qc.x(0)
    elif gate_name == "y":
        qc.y(0)
    elif gate_name == "z":
        qc.z(0)
    elif gate_name == "s":
        qc.s(0)
    elif gate_name == "t":
        qc.t(0)
    
    U_full = Operator(qc).data
    # Extract single-qubit part (first qubit)
    U = U_full[:2, :2]  # Top-left 2x2 block
    
    # Test our su2 implementation
    full_label = label + "I"  # Apply to first qubit, second qubit is identity
    key = encode_pauli(Pauli(full_label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = QuantumGate.get("su2")(input_term, 0, U)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U_full.conj().T @ pauli_matrix(full_label) @ U_full
    
    assert np.allclose(matsum, expected), f"SU2 mismatch for {gate_name} gate on P={full_label}"

# Test with random unitaries
@pytest.mark.parametrize("trial", range(10))  # 10 random unitary matrices
@pytest.mark.parametrize("label", LABELS_1Q)
def test_su2_random_unitary(trial, label):
    """Test SU2 gate with random unitary matrices."""
    np.random.seed(trial + 1000)
    
    # Generate random 2x2 unitary matrix
    U = random_unitary(2)
    
    # Test on 2-qubit system (apply to first qubit)
    full_label = label + "I"
    key = encode_pauli(Pauli(full_label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = QuantumGate.get("su2")(input_term, 0, U)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    # Build expected result using tensor product
    U_full = np.kron(np.eye(2), U)  # Apply to first qubit (qiskit ordering)
    expected = U_full.conj().T @ pauli_matrix(full_label) @ U_full
    
    assert np.allclose(matsum, expected), f"SU2 random unitary failed for trial {trial}, P={full_label}"

# Test coefficient preservation
def test_su2_coefficient_preservation():
    """Test that SU2 preserves the magnitude of coefficients."""
    U = random_unitary(2)
    label = "X"
    coeff = 2.5 + 1.5j
    full_label = label + "I"
    key = encode_pauli(Pauli(full_label))
    
    input_term = PauliTerm(coeff, key, 2)
    output_terms = QuantumGate.get("su2")(input_term, 0, U)
    
    # Sum of squared magnitudes should be preserved
    input_mag_sq = abs(coeff)**2
    output_mag_sq = sum(abs(term.coeff)**2 for term in output_terms)
    
    assert abs(input_mag_sq - output_mag_sq) < 1e-12, "SU2 coefficient magnitude not preserved"

# Test embedding in larger systems
@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("target_qubit", [0, 1, 2])
def test_su2_embedded(n_qubits, target_qubit):
    """Test SU2 gate embedded in larger systems."""
    if target_qubit >= n_qubits:
        pytest.skip("Target qubit index exceeds number of qubits")
    
    np.random.seed(42 + target_qubit)
    U = random_unitary(2)
    
    # Generate random Pauli label
    label = "".join(random.choice("IXYZ") for _ in range(n_qubits))
    key = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, n_qubits)
    output_terms = QuantumGate.get("su2")(input_term, target_qubit, U)
    
    matsum = pauli_terms_to_matrix(output_terms, n_qubits)
    
    # Build expected using qiskit circuit
    qc = QuantumCircuit(n_qubits)
    qc.unitary(U, target_qubit)
    U_full = Operator(qc).data
    
    expected = U_full.conj().T @ pauli_matrix(label) @ U_full
    
    assert np.allclose(matsum, expected), f"SU2 embedded failed for n={n_qubits}, q={target_qubit}, P={label}"

def apply_gate_via_propagator(qc: QuantumCircuit, pauli_term: PauliTerm) -> list:
    """Apply quantum circuit via propagator."""
    prop = PauliPropagator(qc)
    history = prop.propagate(pauli_term)
    return history[-1]

@pytest.mark.parametrize("trial", range(10))
def test_su2_random_circuits(trial):
    """Test SU2 gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 6000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Create quantum circuit with random SU2 gates
    qc = QuantumCircuit(n, name=f"su2_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random qubit and generate random unitary matrix
        q = np.random.randint(0, n)
        U = random_su2()
        
        # Add SU2 gate to circuit using UnitaryGate with correct naming
        gate = UnitaryGate(U, label="su2")
        gate._name = "su2"
        qc.append(gate, [q])
    
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