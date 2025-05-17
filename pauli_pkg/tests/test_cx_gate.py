# pauli_pkg/tests/test_cx_gate.py

# Analytical cross-check of the CX conjugation rule.
# We compare 危 c? P? produced by QuantumGate.get("cx") with
# U? P U obtained from Qiskit鈥檚 Operator, for both control-target
# orders (0,1) and (1,0).


import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator
from pauli_propagation import PauliTerm, QuantumGate

LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]


def pauli_matrix(label: str) -> np.ndarray:
    return Pauli(label).to_matrix()


def cx_matrix(ctrl: int, tgt: int) -> np.ndarray:
    qc = QuantumCircuit(2)
    qc.cx(ctrl, tgt)
    return Operator(qc).data


def sum_paths(paths):
    """危 coeff 路 Pauli_matrix for a list[PauliTerm]."""
    return sum(p.coeff * p.pauli.to_matrix() for p in paths)


@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_cx_rule(ctrl, tgt, label):
    """Exhaustive test over all 2-qubit Pauli labels."""
    U = cx_matrix(ctrl, tgt)
    inp = PauliTerm(1.0, Pauli(label))
    out = QuantumGate.get("cx")(inp, ctrl, tgt)
    assert np.allclose(sum_paths(out), U.conj().T @ pauli_matrix(label) @ U)
