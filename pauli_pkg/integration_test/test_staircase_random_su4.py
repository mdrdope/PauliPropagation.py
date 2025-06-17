# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit.circuit.library import UnitaryGate

from pauli_propagation.utils import encode_pauli, decode_pauli, random_su4
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.propagator import PauliPropagator
from pauli_propagation.circuit_topologies import staircasetopology2d_qc


@pytest.mark.parametrize("nx, ny", [(1,2), (1,3),(2, 2),(1,4),(1,5) ,(2, 3),(1,6)]) # , (3, 4), (2,6), (5,3),(3,6)])
def test_staircase_random_su4(nx, ny):
    """
    Compare expectation value of X on qubit 0 computed by
    PauliPropagator against statevector simulation for a random
    SU(4) staircase circuit. Error must be <= 15%.
    """
    # Fix randomness for reproducibility
    random.seed(42)
    np.random.seed(42)

    qc = staircasetopology2d_qc(nx, ny, seed=42)
    n = qc.num_qubits

    # Build initial PauliTerm for X on qubit 0
    pauli_label = "X" + "I" * (n - 1)
    obs = Pauli(pauli_label)
    key = encode_pauli(obs)
    init_term = PauliTerm(1.0, key, n)

    # Propagate observable through circuit
    prop = PauliPropagator(qc)
    layers = prop.propagate(init_term, max_weight=None)
    prop_exp = prop.expectation_pauli_sum(
        pauli_sum=layers[-1],
        product_label="+" * n)

    # Statevector-based expectation calculation
    psi0 = Statevector.from_label("+" * n)
    psi_final = psi0.evolve(qc)
    sv_exp = psi_final.expectation_value(obs).real

    # Assert relative error within 10%
    rel_err = abs(prop_exp - sv_exp) / abs(sv_exp)
    assert rel_err <= 0.001, (f"Relative error {rel_err:.2%} exceeds 25%: "
                            f"prop={prop_exp:.4f}, statevector={sv_exp:.4f}")
