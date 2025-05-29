# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli

from pauli_propagation.utils      import encode_pauli, random_su2
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation             import PauliPropagator

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