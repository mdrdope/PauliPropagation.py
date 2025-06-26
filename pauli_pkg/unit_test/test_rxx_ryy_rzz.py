#!/usr/bin/env python3

# -*- coding: utf-8 -*-


import itertools
import random
import numpy as np
import pytest
from math import pi
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
from pauli_propagation.gates import QuantumGate
from pauli_propagation.propagator import PauliPropagator


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]


def rxx_matrix(q1: int, q2: int, theta: float, n: int) -> np.ndarray:
    qc = QuantumCircuit(n)
    qc.rxx(theta, q1, q2)
    return Operator(qc).data


def ryy_matrix(q1: int, q2: int, theta: float, n: int) -> np.ndarray:
    qc = QuantumCircuit(n)
    qc.ryy(theta, q1, q2)
    return Operator(qc).data


def rzz_matrix(q1: int, q2: int, theta: float, n: int) -> np.ndarray:
    qc = QuantumCircuit(n)
    qc.rzz(theta, q1, q2)
    return Operator(qc).data


GATE_MAT_FUNCS = {"rxx": rxx_matrix, "ryy": ryy_matrix, "rzz": rzz_matrix}



ANGLES_SAMPLE = [0.0, pi / 4, pi / 2, pi]  # representative angles incl. edge cases


# Use random angle per case to reduce total combinations
@pytest.mark.parametrize("gate_name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("q_pair", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_rxx_ryy_rzz_rules(gate_name, q_pair, label):
    """Validate conjugation U† P U for RXX/RYY/RZZ on all 2-qubit Paulis."""

    q1, q2 = q_pair
    theta = random.uniform(0, 2 * pi)
    U = GATE_MAT_FUNCS[gate_name](q1, q2, theta, 2)

    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 2)

    output_terms = QuantumGate.get(gate_name)(input_term, q1, q2, theta)
    matsum = pauli_terms_to_matrix(output_terms, 2)

    expected = U.conj().T @ Pauli(label).to_matrix() @ U

    assert np.allclose(matsum, expected), (
        f"Mismatch {gate_name.upper()}(θ={theta}) on qubits ({q1},{q2}) P={label}")


TRIALS_EMB = 20


@pytest.mark.parametrize("gate_name", ["rxx", "ryy", "rzz"])
@pytest.mark.parametrize("trial", range(TRIALS_EMB))
def test_rxx_ryy_rzz_random_embedded(gate_name, trial):
    np.random.seed(trial + 15000)
    num_qubits = 6

    label = "".join(random.choice("IXYZ") for _ in range(num_qubits))
    q1, q2 = random.sample(range(num_qubits), 2)
    theta = random.uniform(0, 2 * pi)

    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, num_qubits)

    output_terms = QuantumGate.get(gate_name)(input_term, q1, q2, theta)
    matsum = pauli_terms_to_matrix(output_terms, num_qubits)

    # Dense reference
    qc_ref = QuantumCircuit(num_qubits)
    getattr(qc_ref, gate_name)(theta, q1, q2)
    G_full = Operator(qc_ref).data
    ref = G_full.conj().T @ Pauli(label).to_matrix() @ G_full

    assert np.allclose(matsum, ref), (
        f"Embedded {gate_name.upper()} mismatch θ={theta:.3f} qubits ({q1},{q2}) label {label}")


@pytest.mark.parametrize("trial", range(10))
def test_rxx_ryy_rzz_random_circuits(trial):
    np.random.seed(trial + 15100)

    n = np.random.randint(2, 6)  # 2–5 qubits
    n_gates = np.random.randint(3, 8)

    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable = PauliTerm(1.0, encode_pauli(Pauli(pauli_label)), n)

    qc = QuantumCircuit(n, name=f"rpair_rand_{n}q_{n_gates}g")
    gate_choices = ["rxx", "ryy", "rzz"]
    for _ in range(n_gates):
        gate = random.choice(gate_choices)
        q1, q2 = random.sample(range(n), 2)
        theta = random.uniform(0, 2 * pi)
        getattr(qc, gate)(theta, q1, q2)

    # Pauli propagation
    prop = PauliPropagator(qc)
    final_layer = prop.propagate(observable, max_weight=None)[-1]
    pauli_exp = prop.expectation_pauli_sum(final_layer, state_label)

    # Exact state-vector simulation
    psi0 = Statevector.from_label(state_label)
    psi1 = psi0.evolve(qc)
    exact_exp = psi1.expectation_value(Pauli(pauli_label)).real

    assert abs(pauli_exp - exact_exp) < 1e-10, (
        f"Trial {trial}: {pauli_exp} vs {exact_exp} (state={state_label}, P={pauli_label})") 