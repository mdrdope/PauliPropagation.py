# -*- coding: utf-8 -*-

"""Unit tests for two-qubit controlled gates: CX, CY, CZ, CH.

Three test groups are provided – rules, random embedded gate, and random
circuits – mirroring the style of other rewritten test files.
"""

import random
import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator, Statevector

from pauli_propagation.utils import (
    encode_pauli,
    decode_pauli,
    random_pauli_label,
    random_state_label,
    pauli_terms_to_matrix,
)
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate
from pauli_propagation.propagator import PauliPropagator

LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

def cx_matrix(ctrl: int, tgt: int, n: int) -> np.ndarray:
    qc = QuantumCircuit(n)
    qc.cx(ctrl, tgt)
    return Operator(qc).data

def cy_matrix(ctrl: int, tgt: int, n: int) -> np.ndarray:
    qc = QuantumCircuit(n)
    qc.cy(ctrl, tgt)
    return Operator(qc).data

def cz_matrix(ctrl: int, tgt: int, n: int) -> np.ndarray:
    qc = QuantumCircuit(n)
    qc.cz(ctrl, tgt)
    return Operator(qc).data

def ch_matrix(ctrl: int, tgt: int, n: int) -> np.ndarray:
    qc = QuantumCircuit(n)
    qc.ch(ctrl, tgt)
    return Operator(qc).data

MAT_FUNCS = {
    "cx": cx_matrix,
    "cy": cy_matrix,
    "cz": cz_matrix,
    "ch": ch_matrix,
}

@pytest.mark.parametrize("gate_name", ["cx", "cy", "cz", "ch"])
@pytest.mark.parametrize("q_pair", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_cx_cy_cz_ch_rules(gate_name, q_pair, label):
    ctrl, tgt = q_pair

    U = MAT_FUNCS[gate_name](ctrl, tgt, 2)

    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 2)

    output_terms = QuantumGate.get(gate_name)(input_term, ctrl, tgt)
    matsum = pauli_terms_to_matrix(output_terms, 2)

    expected = U.conj().T @ Pauli(label).to_matrix() @ U

    assert np.allclose(matsum, expected), (
        f"Mismatch {gate_name.upper()} on qubits ({ctrl},{tgt}) P={label}")

TRIALS_EMB = 20

@pytest.mark.parametrize("gate_name", ["cx", "cy", "cz", "ch"])
@pytest.mark.parametrize("trial", range(TRIALS_EMB))
def test_cx_cy_cz_ch_random_embedded(gate_name, trial):
    np_random = np.random.RandomState(trial + 17000)
    n = 6

    label = "".join(np_random.choice(list("IXYZ"), n))
    ctrl, tgt = np_random.choice(n, 2, replace=False)

    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, n)

    output_terms = QuantumGate.get(gate_name)(input_term, ctrl, tgt)
    matsum = pauli_terms_to_matrix(output_terms, n)

    qc_ref = QuantumCircuit(n)
    getattr(qc_ref, gate_name)(ctrl, tgt)
    G_full = Operator(qc_ref).data
    ref = G_full.conj().T @ Pauli(label).to_matrix() @ G_full

    assert np.allclose(matsum, ref), (
        f"Embedded {gate_name.upper()} mismatch qubits ({ctrl},{tgt}) label {label}")

@pytest.mark.parametrize("trial", range(10))
def test_cx_cy_cz_ch_random_circuits(trial):
    np.random.seed(trial + 17100)

    n = np.random.randint(2, 6)
    n_gates = np.random.randint(3, 8)

    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable = PauliTerm(1.0, encode_pauli(Pauli(pauli_label)), n)

    qc = QuantumCircuit(n, name=f"cctl_rand_{n}q_{n_gates}g")
    gate_choices = ["cx", "cy", "cz", "ch"]
    for _ in range(n_gates):
        gate = random.choice(gate_choices)
        ctrl, tgt = random.sample(range(n), 2)
        getattr(qc, gate)(ctrl, tgt)

    prop = PauliPropagator(qc)
    final_layer = prop.propagate(observable, max_weight=None)[-1]
    pauli_exp = prop.expectation_pauli_sum(final_layer, state_label)

    psi0 = Statevector.from_label(state_label)
    psi1 = psi0.evolve(qc)
    exact_exp = psi1.expectation_value(Pauli(pauli_label)).real

    assert abs(pauli_exp - exact_exp) < 1e-10, (
        f"Trial {trial}: {pauli_exp} vs {exact_exp} (state={state_label}, P={pauli_label})")

