# -*- coding: utf-8 -*-
"""Unit tests for controlled rotations CRX, CRY, CRZ with exactly three test functions."""
import random
import itertools
from math import pi

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator, Statevector

from pauli_propagation.utils import (
    encode_pauli,
    pauli_terms_to_matrix,
    random_pauli_label,
    random_state_label,
)
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates import QuantumGate
from pauli_propagation.propagator import PauliPropagator

# Helper utilities
# -----------------------------------------------------------------------------

LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

def _u_matrix(gate: str, ctrl: int, tgt: int, theta: float, n: int):
    qc = QuantumCircuit(n)
    getattr(qc, gate)(theta, ctrl, tgt)
    return Operator(qc).data

# -----------------------------------------------------------------------------
# 1. Rules
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("gate", ["crx", "cry", "crz"])
@pytest.mark.parametrize("q_pair", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_crx_cry_crz_rules(gate, q_pair, label):
    ctrl, tgt = q_pair
    theta = random.uniform(0, 2 * pi)

    qc_u = QuantumCircuit(2)
    getattr(qc_u, gate)(theta, ctrl, tgt)
    U = Operator(qc_u).data
    out_terms = QuantumGate.get(gate)(PauliTerm(1.0, encode_pauli(Pauli(label)), 2), ctrl, tgt, theta)
    matsum = pauli_terms_to_matrix(out_terms,2)
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    assert np.allclose(matsum, expected)

# -----------------------------------------------------------------------------
# 2. Embedded
# -----------------------------------------------------------------------------

TRIALS_EMB = 20

@pytest.mark.parametrize("gate", ["crx", "cry", "crz"])
@pytest.mark.parametrize("trial", range(TRIALS_EMB))
def test_crx_cry_crz_random_embedded(gate, trial):
    rng = np.random.RandomState(trial + 21000)
    n = 6
    label = "".join(rng.choice(list("IXYZ"), n))
    ctrl, tgt = rng.choice(n, 2, replace=False)
    theta = rng.uniform(0, 2 * pi)

    qc_u = QuantumCircuit(n)
    getattr(qc_u, gate)(theta, ctrl, tgt)
    U = Operator(qc_u).data
    out_terms = QuantumGate.get(gate)(PauliTerm(1.0, encode_pauli(Pauli(label)), n), ctrl, tgt, theta)
    matsum = pauli_terms_to_matrix(out_terms,n)
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    assert np.allclose(matsum, expected)

# -----------------------------------------------------------------------------
# 3. Circuits
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("trial", range(10))
def test_crx_cry_crz_random_circuits(trial):
    rng = np.random.RandomState(trial + 21100)
    n = rng.randint(2, 6)
    n_gates = rng.randint(3, 8)
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable = PauliTerm(1.0, encode_pauli(Pauli(pauli_label)), n)

    qc = QuantumCircuit(n)
    for _ in range(n_gates):
        gate = rng.choice(["crx", "cry", "crz"])
        ctrl, tgt = rng.choice(n, 2, replace=False)
        theta = rng.uniform(0, 2 * pi)
        getattr(qc, gate)(theta, ctrl, tgt)

    prop = PauliPropagator(qc)
    pauli_exp = prop.expectation_pauli_sum(prop.propagate(observable)[-1], state_label)
    exact_exp = Statevector.from_label(state_label).evolve(qc).expectation_value(Pauli(pauli_label)).real
    assert abs(pauli_exp - exact_exp) < 1e-10
    