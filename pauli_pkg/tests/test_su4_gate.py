# pauli_pkg/tests/test_su4_gate.py

# Random SU(4) conjugation rule test for QuantumGate.get("su4").

# A Haar-random SU(4) is appended to qubits [0,1] (big-endian order).
# For each trial we check that 危 c岬� P岬� from PauliPropagator equals
# U鈥� P U in matrix form, within a tight tolerance.

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate  
from qiskit.quantum_info import Pauli
from pauli_propagation import PauliPropagator, PauliTerm
from pauli_propagation.utils import random_su4
import itertools

# ----------------- helpers -----------------
LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]


def pauli_to_matrix(label: str) -> np.ndarray:
    """2-qubit label (big-endian) 鈫� 4脳4 matrix."""

    _SINGLE = {"I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
    return np.kron(_SINGLE[label[0]], _SINGLE[label[1]])


def series_to_matrix(series):
    """list[PauliTerm] 鈫� summed 4脳4 matrix."""
    acc = np.zeros((4, 4), dtype=complex)
    for term in series:
        acc += term.coeff * pauli_to_matrix(term.pauli.to_label())
    return acc

# ----------------- test -----------------
TRIALS = 50
TOL = 1e-12

def test_random_su4_trials():
    failures = 0
    for _ in range(TRIALS):
        # 1) random SU(4) gate
        U = random_su4()
        gate = UnitaryGate(U, label="randSU4")
        gate._name = "su4"

        # 2) 2-qubit circuit
        qc = QuantumCircuit(2)
        qc.append(gate, [0, 1])  # big-endian: qubit-1 left, qubit-0 right

        # 3) random input Pauli
        label = np.random.choice(LABELS_2Q)
        propagator = PauliPropagator(qc)
        series = propagator.propagate(PauliTerm(1.0, Pauli(label)))[-1]

        lhs = U.conj().T @ pauli_to_matrix(label) @ U
        rhs = series_to_matrix(series)

        if not np.allclose(lhs, rhs, atol=TOL):
            failures += 1

    assert failures == 0, f"{failures} / {TRIALS} random SU(4) trials failed"
