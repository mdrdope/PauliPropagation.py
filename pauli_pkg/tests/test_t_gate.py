import random
import numpy as np
import pytest
from math import pi
from qiskit.quantum_info import Pauli

from pauli_propagation.utils import encode_pauli, decode_pauli
from pauli_propagation.gates import QuantumGate

T_GATE = np.diag([1.0, np.exp(1j * pi / 4)])

def sum_paths_from_kernel(out, n):
    """
    Reconstruct sum(alpha_j * P_j) from the bit-kernel 5-tuple.
    out = (L, c1, k1, c2, k2)
    """
    L, c1, k1, c2, k2 = out
    mat = c1 * Pauli(decode_pauli(k1, n).to_label()).to_matrix()
    if int(L) == 2:
        mat += c2 * Pauli(decode_pauli(k2, n).to_label()).to_matrix()
    return mat

@pytest.mark.parametrize("label", ["I", "X", "Y", "Z"])
def test_t_single(label):
    """
    Direct test of single-qubit T conjugation:
    T^dagger P T == sum kernel paths
    """
    key = encode_pauli(Pauli(label))
    out = QuantumGate.get("t")(1.0, key, 1, 0)
    result   = sum_paths_from_kernel(out, 1)
    expected = T_GATE.conj().T @ Pauli(label).to_matrix() @ T_GATE
    assert np.allclose(result, expected), f"Failed for P={label}"

TRIALS = 50

@pytest.mark.parametrize("trial", range(TRIALS))
def test_t_random_embedded(trial):
    """
    Embed T on a random qubit in 4-qubit Pauli, compare with full tensor product construction.
    """
    # random 4-qubit Pauli label
    label4 = "".join(random.choice("IXYZ") for _ in range(4))
    q      = random.randrange(4)
    key    = encode_pauli(Pauli(label4))

    # apply bit-kernel
    out = QuantumGate.get("t")(1.0, key, 4, q)
    matsum = sum_paths_from_kernel(out, 4)

    # build reference via tensor products (Qiskit little-endian)
    mats = []
    for idx, ch in enumerate(reversed(label4)):  # qubit-0 = rightmost
        base = Pauli(ch).to_matrix()
        if idx == q:
            base = T_GATE.conj().T @ base @ T_GATE
        mats.append(base)

    ref = mats[0]
    for m in mats[1:]:
        ref = np.kron(m, ref)

    assert np.allclose(matsum, ref), f"Embedded T mismatch on qubit {q}, label {label4}"
