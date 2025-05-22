import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli

from pauli_propagation.utils      import random_su4, encode_pauli
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation             import PauliPropagator

# All 2-qubit Pauli labels in big-endian order
LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

def pauli_to_matrix(label: str) -> np.ndarray:
    """2-qubit Pauli label -> 4x4 matrix (big-endian)."""
    SINGLE = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    return np.kron(SINGLE[label[0]], SINGLE[label[1]])

def series_to_matrix(series):
    """Sum alpha*P over a list of PauliTerm, using term.to_label()."""
    acc = np.zeros((4, 4), dtype=complex)
    for term in series:
        mat = pauli_to_matrix(term.to_label())
        acc += term.coeff * mat
    return acc

TRIALS = 30
TOL    = 1e-12

@pytest.mark.parametrize("trial", range(TRIALS))
def test_random_su4_conjugation(trial):
    # 1) Haar-random SU(4) on qubits [0,1]
    U    = random_su4()
    gate = UnitaryGate(U, label="randSU4")
    gate._name = "su4"

    # 2) Build a 2-qubit circuit with that SU(4)
    qc = QuantumCircuit(2)
    qc.append(gate, [0, 1])

    # 3) Pick a random input Pauli
    label = np.random.choice(LABELS_2Q)
    key   = encode_pauli(Pauli(label))
    pt    = PauliTerm(1.0, key, 2)

    # 4) Back-propagate and get the output series
    series = PauliPropagator(qc).propagate(pt, max_weight=None)[-1]

    # 5) Compare matrices: U^dagger P U vs sum alpha_i P_i
    lhs = U.conj().T @ pauli_to_matrix(label) @ U
    rhs = series_to_matrix(series)

    assert np.allclose(lhs, rhs, atol=TOL), (
        f"Trial {trial}: mismatch for Pauli {label}"
    )
