# -*- coding: utf-8 -*-


import random
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli, Operator, Statevector

from pauli_propagation.utils import (
    encode_pauli,
    decode_pauli,
    random_pauli_label,
    random_state_label,
    random_su2,
    pauli_terms_to_matrix,
)
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates import QuantumGate
from pauli_propagation.propagator import PauliPropagator




# All single-qubit Pauli labels
LABELS_1Q = ["I", "X", "Y", "Z"]



TRIALS_RULE = 10  # number of random SU(2) matrices to test

@pytest.mark.parametrize("trial", range(TRIALS_RULE))
@pytest.mark.parametrize("label", LABELS_1Q)
def test_su2_rule(trial: int, label: str):
    """Verify U† P U = Σᵢ αᵢ Pᵢ for a single-qubit SU2 gate."""

    np.random.seed(trial + 13000)

    # Haar-random 2×2 special-unitary matrix
    U = random_su2()

    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, 1)

    # Propagation via bit-kernel
    output_terms = QuantumGate.get("su2")(input_term, 0, U)
    matsum = pauli_terms_to_matrix(output_terms, 1)

    # Dense-matrix reference
    expected = U.conj().T @ Pauli(label).to_matrix() @ U

    assert np.allclose(matsum, expected), (
        f"SU2 single-qubit conjugation failed (trial={trial}, P={label})")


# -----------------------------------------------------------------------------
# 2. Random embedded SU2 gate inside larger Pauli operator
# -----------------------------------------------------------------------------

TRIALS_EMB = 20


@pytest.mark.parametrize("trial", range(TRIALS_EMB))
def test_su2_random_embedded(trial: int):
    """Embed an SU2 gate on a random qubit of a 6-qubit Pauli operator."""

    np.random.seed(trial + 13100)
    num_qubits = 6

    label = "".join(random.choice("IXYZ") for _ in range(num_qubits))
    target_q = np.random.randint(num_qubits)

    U = random_su2()

    key = encode_pauli(Pauli(label))
    input_term = PauliTerm(1.0, key, num_qubits)

    # Propagate through bit-kernel
    output_terms = QuantumGate.get("su2")(input_term, target_q, U)
    matsum = pauli_terms_to_matrix(output_terms, num_qubits)

    # Dense-matrix reference via Qiskit Operator
    qc_ref = QuantumCircuit(num_qubits)
    gate = UnitaryGate(U, label="su2")
    gate._name = "su2"
    qc_ref.append(gate, [target_q])

    G_full = Operator(qc_ref).data
    P_mat = Pauli(label).to_matrix()
    ref = G_full.conj().T @ P_mat @ G_full

    assert np.allclose(matsum, ref), (
        f"Embedded SU2 mismatch (q={target_q}, label={label})")


# -----------------------------------------------------------------------------
# 3. Random circuits containing many SU2 gates
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("trial", range(10))
def test_su2_random_circuits(trial: int):
    """Compare PauliPropagator expectation to state-vector simulation."""

    np.random.seed(trial + 13200)

    n = np.random.randint(2, 6)  # 2–5 qubits
    n_gates = np.random.randint(3, 8)

    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)

    observable = PauliTerm(1.0, encode_pauli(Pauli(pauli_label)), n)

    # Build random circuit with SU2 gates on random qubits
    qc = QuantumCircuit(n, name=f"su2_rand_{n}q_{n_gates}g")
    for _ in range(n_gates):
        q = np.random.randint(n)
        U = random_su2()
        gate = UnitaryGate(U, label="su2")
        gate._name = "su2"
        qc.append(gate, [q])

    # Method 1: Pauli propagation
    prop = PauliPropagator(qc)
    final_layer = prop.propagate(observable, max_weight=None)[-1]
    pauli_exp = prop.expectation_pauli_sum(final_layer, state_label)

    # Method 2: Exact state-vector evolution
    psi0 = Statevector.from_label(state_label)
    psi1 = psi0.evolve(qc)
    exact_exp = psi1.expectation_value(Pauli(pauli_label)).real

    assert abs(pauli_exp - exact_exp) < 1e-10, (
        f"Trial {trial}: {pauli_exp} vs {exact_exp} (state={state_label}, P={pauli_label})") 