
# # pauli_pkg/tests/test_gates.py

# # Exhaustive and analytical checks for CX / T conjugation rules.

# # No state-vector sims here: we compare the matrices from the rule against
# # U閳ワ拷 P U directly.


# import itertools, numpy as np, pytest
# from qiskit import QuantumCircuit
# from qiskit.quantum_info import Pauli, Operator
# from pauli_propagation import PauliTerm, QuantumGate

# LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

# def pauli_matrix(lbl):           # helper
#     return Pauli(lbl).to_matrix()

# def cx_matrix(ctrl, tgt):
#     qc = QuantumCircuit(2)
#     qc.cx(ctrl, tgt)
#     return Operator(qc).data

# def sum_paths(paths):
#     return sum(p.coeff * p.pauli.to_matrix() for p in paths)

# @pytest.mark.parametrize("ctrl,tgt", [(0,1), (1,0)])
# @pytest.mark.parametrize("label", LABELS_2Q)
# def test_cx_rule(ctrl, tgt, label):
#     U  = cx_matrix(ctrl, tgt)
#     inp = PauliTerm(1.0, Pauli(label))
#     out = QuantumGate.get("cx")(inp, ctrl, tgt)
#     assert np.allclose(sum_paths(out), U.conj().T @ pauli_matrix(label) @ U)

# @pytest.mark.parametrize("label", list("IXYZ"))
# def test_t_single(label):
#     from math import pi
#     T = np.diag([1.0, np.exp(1j*pi/4)])
#     inp = PauliTerm(1.0, Pauli(label))
#     out = QuantumGate.get("t")(inp, 0)
#     assert np.allclose(sum_paths(out), T.conj().T @ Pauli(label).to_matrix() @ T)

# def test_t_random_embedded():
#     from math import pi
#     import random
#     T = np.diag([1.0, np.exp(1j*pi/4)])

#     for _ in range(100):
#         label4 = "".join(random.choice("IXYZ") for _ in range(4))
#         q      = random.randrange(4)
#         inp    = PauliTerm(1.0, Pauli(label4))
#         out    = QuantumGate.get("t")(inp, q)

#         # build reference matrix 閳?锟?-wise (little-endian)
#         mats = []
#         for idx, ch in enumerate(reversed(label4)):
#             base = Pauli(ch).to_matrix()
#             mats.append(T.conj().T @ base @ T if idx == q else base)

#         ref = mats[0]
#         for m in mats[1:]:
#             ref = np.kron(m, ref)

#         assert np.allclose(sum_paths(out), ref)