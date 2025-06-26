# -*- coding: utf-8 -*-


import random
from math import pi

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator, Statevector

from pauli_propagation.utils import (
    encode_pauli,
    random_pauli_label,
    random_state_label,
    pauli_terms_to_matrix,
)
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates import QuantumGate
from pauli_propagation.propagator import PauliPropagator


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

LABELS_1Q = ["I", "X", "Y", "Z"]

def rx_matrix(theta: float) -> np.ndarray:
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    return Operator(qc).data


def ry_matrix(theta: float) -> np.ndarray:
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    return Operator(qc).data


def rz_matrix(theta: float) -> np.ndarray:
    qc = QuantumCircuit(1)
    qc.rz(theta, 0)
    return Operator(qc).data


MAT_FUNCS = {"rx": rx_matrix, "ry": ry_matrix, "rz": rz_matrix}


# -----------------------------------------------------------------------------
# 1. Conjugation rules on single-qubit system
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("gate_name", ["rx", "ry", "rz"])
@pytest.mark.parametrize("label", LABELS_1Q)
def test_rx_ry_rz_rules(gate_name, label):
    theta = random.uniform(0, 2 * pi)

    U = MAT_FUNCS[gate_name](theta)

    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 1)

    output_terms = QuantumGate.get(gate_name)(input_term, 0, theta)
    matsum = pauli_terms_to_matrix(output_terms, 1)

    expected = U.conj().T @ Pauli(label).to_matrix() @ U

    assert np.allclose(matsum, expected), (
        f"{gate_name.upper()} rule mismatch θ={theta} on P={label}")


# -----------------------------------------------------------------------------
# 2. Random embedded single rotation in larger system
# -----------------------------------------------------------------------------

TRIALS_EMB = 20

@pytest.mark.parametrize("gate_name", ["rx", "ry", "rz"])
@pytest.mark.parametrize("trial", range(TRIALS_EMB))
def test_rx_ry_rz_random_embedded(gate_name, trial):
    np.random.seed(trial + 16000)
    num_qubits = 6

    label = "".join(random.choice("IXYZ") for _ in range(num_qubits))
    target_q = np.random.randint(num_qubits)
    theta = random.uniform(0, 2 * pi)

    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, num_qubits)

    output_terms = QuantumGate.get(gate_name)(input_term, target_q, theta)
    matsum = pauli_terms_to_matrix(output_terms, num_qubits)

    qc_ref = QuantumCircuit(num_qubits)
    getattr(qc_ref, gate_name)(theta, target_q)
    G_full = Operator(qc_ref).data
    ref = G_full.conj().T @ Pauli(label).to_matrix() @ G_full

    assert np.allclose(matsum, ref), (
        f"Embedded {gate_name.upper()} mismatch θ={theta:.3f} q={target_q} label {label}")


# -----------------------------------------------------------------------------
# 3. Random circuits with many rotations
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(10))
def test_rx_ry_rz_random_circuits(trial):
    np.random.seed(trial + 16100)

    n = np.random.randint(2, 6)
    n_gates = np.random.randint(3, 8)

    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable = PauliTerm(1.0, encode_pauli(Pauli(pauli_label)), n)

    qc = QuantumCircuit(n, name=f"r1_rand_{n}q_{n_gates}g")
    gate_choices = ["rx", "ry", "rz"]
    for _ in range(n_gates):
        gate = random.choice(gate_choices)
        q = np.random.randint(n)
        theta = random.uniform(0, 2 * pi)
        getattr(qc, gate)(theta, q)

    prop = PauliPropagator(qc)
    final_layer = prop.propagate(observable, max_weight=None)[-1]
    pauli_exp = prop.expectation_pauli_sum(final_layer, state_label)

    psi0 = Statevector.from_label(state_label)
    psi1 = psi0.evolve(qc)
    exact_exp = psi1.expectation_value(Pauli(pauli_label)).real

    assert abs(pauli_exp - exact_exp) < 1e-10, (
        f"Trial {trial}: {pauli_exp} vs {exact_exp} (state={state_label}, P={pauli_label})") 