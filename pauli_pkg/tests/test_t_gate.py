# pauli_pkg/tests/test_t_gate.py

# Analytical cross-check of the single-qubit T conjugation rule.

import random
import numpy as np
import pytest
from math import pi
from qiskit.quantum_info import Pauli
from pauli_propagation import PauliTerm, QuantumGate

T_GATE = np.diag([1.0, np.exp(1j * pi / 4)])


def sum_paths(paths):
    return sum(p.coeff * p.pauli.to_matrix() for p in paths)

@pytest.mark.parametrize("label", list("IXYZ"))
def test_t_single(label):
    """Direct comparison for a single-qubit Pauli."""
    inp = PauliTerm(1.0, Pauli(label))
    out = QuantumGate.get("t")(inp, 0)
    assert np.allclose(sum_paths(out), T_GATE.conj().T @ Pauli(label).to_matrix() @ T_GATE)

def test_t_random_embedded():
    """
    Embed a T gate on a random qubit within 4 qubits and compare
    against explicit tensor-product construction.
    """
    for _ in range(100):
        label4 = "".join(random.choice("IXYZ") for _ in range(4))
        q = random.randrange(4)
        inp = PauliTerm(1.0, Pauli(label4))
        out = QuantumGate.get("t")(inp, q)

        # build reference matrix little-endian (Qiskit ordering)
        mats = []
        for idx, ch in enumerate(reversed(label4)):  # qubit-0 is rightmost
            base = Pauli(ch).to_matrix()
            mats.append(T_GATE.conj().T @ base @ T_GATE if idx == q else base)

        ref = mats[0]
        for m in mats[1:]:
            ref = np.kron(m, ref)

        assert np.allclose(sum_paths(out), ref)
