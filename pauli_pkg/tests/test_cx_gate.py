import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator

from pauli_propagation.utils      import encode_pauli, decode_pauli
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate

LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

def pauli_matrix(label: str) -> np.ndarray:
    return Pauli(label).to_matrix()

def cx_matrix(ctrl: int, tgt: int) -> np.ndarray:
    qc = QuantumCircuit(2)
    qc.cx(ctrl, tgt)
    return Operator(qc).data

def sum_paths_from_kernel(out, n: int) -> np.ndarray:
    """Reconstruct ∑ αᵢ Pᵢ from the bit-kernel output 5-tuple."""
    L, c1, k1, c2, k2 = out
    # first term
    mat = c1 * Pauli(decode_pauli(k1, n).to_label()).to_matrix()
    # possible second term
    if int(L) == 2:
        mat += c2 * Pauli(decode_pauli(k2, n).to_label()).to_matrix()
    return mat

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_cx_rule(ctrl, tgt, label):
    """Exhaustively check CX conjugation for all 2-qubit Paulis."""
    U      = cx_matrix(ctrl, tgt)
    kernel = QuantumGate.get("cx")
    key    = encode_pauli(Pauli(label))
    out    = kernel(1.0, key, 2, ctrl, tgt)
    matsum = sum_paths_from_kernel(out, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}"
