# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit.circuit.library import UnitaryGate

from pauli_propagation.utils import encode_pauli, decode_pauli, random_su4, random_state_label, random_pauli_label
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.propagator import PauliPropagator
from pauli_propagation.decomposition import su4_kak_reconstruct
from pauli_propagation.circuit_topologies import staircasetopology2d_qc


@pytest.mark.parametrize("nx, ny", [(1, 2), (1, 3), (2, 2), (1, 4), (2, 3)])
# @pytest.mark.parametrize("trial", range(5))
def test_staircase_su4_kak_expectation_consistency(nx, ny):# , trial
    """
    Test expectation consistency between original SU4 staircase circuit
    and its KAK decomposed version for various observables and states.
    """
    # Set seed for reproducibility in each trial
    seed = 42 # + trial
    
    # Build original circuit with random SU4 gates
    qc_original = staircasetopology2d_qc(nx, ny) # , seed=seed
    n = qc_original.num_qubits
    
    # Apply KAK decomposition to get decomposed circuit
    qc_decomposed = su4_kak_reconstruct(qc_original)
    
    # Generate random observable and initial state
    # random.seed(seed + 1000)  # Use different seed for observable/state generation
    # np.random.seed(seed + 1000)
    
    pauli_label = random_pauli_label(n)
    state_label = random_state_label(n)
    
    # Build observable
    obs = Pauli(pauli_label)
    key = encode_pauli(obs)
    init_term = PauliTerm(1.0, key, n)
    
    # Compute expectation using original circuit
    prop_original = PauliPropagator(qc_original)
    layers_original = prop_original.propagate(init_term, max_weight=None, use_parallel=False)
    exp_original = prop_original.expectation_pauli_sum(layers_original[-1], state_label)
    
    # Compute expectation using decomposed circuit
    prop_decomposed = PauliPropagator(qc_decomposed)
    layers_decomposed = prop_decomposed.propagate(init_term, max_weight=None)
    exp_decomposed = prop_decomposed.expectation_pauli_sum(layers_decomposed[-1], state_label)
    
    # Assert consistency
    assert abs(exp_original - exp_decomposed) < 1e-10, (
        f"Expectation mismatch for {nx}x{ny} staircase, trial"#  {trial}\n"
        f"Original: {exp_original}\n"
        f"Decomposed: {exp_decomposed}\n"
        f"Observable: {pauli_label}\n"
        f"State: {state_label}"
    )


@pytest.mark.parametrize("nx, ny", [(1, 2), (1, 3), (2, 2), (1, 4), (2, 3)])
def test_staircase_su4_kak_vs_statevector(nx, ny):
    """
    Test that KAK decomposed staircase circuit gives same expectation
    as exact statevector calculation for X observable on qubit 0.
    """
    # Set seed for reproducibility
    seed = 12345
    
    # Build original circuit
    qc_original = staircasetopology2d_qc(nx, ny, seed=seed)
    n = qc_original.num_qubits
    
    # Apply KAK decomposition
    qc_decomposed = su4_kak_reconstruct(qc_original)
    
    # Use X observable on qubit 0
    pauli_label = "X" + "I" * (n - 1)
    obs = Pauli(pauli_label)
    key = encode_pauli(obs)
    init_term = PauliTerm(1.0, key, n)
    
    # Use |+?^?n state
    state_label = "+" * n
    
    # Compute expectation using PauliPropagator on decomposed circuit
    prop = PauliPropagator(qc_decomposed)
    layers = prop.propagate(init_term, max_weight=None)
    prop_exp = prop.expectation_pauli_sum(layers[-1], state_label)
    
    # Compute exact expectation using statevector
    psi0 = Statevector.from_label(state_label)
    psi_final = psi0.evolve(qc_decomposed)
    sv_exp = psi_final.expectation_value(obs).real
    
    # Assert consistency
    assert abs(prop_exp - sv_exp) < 1e-10, (
        f"Propagator vs statevector mismatch for {nx}x{ny} staircase\n"
        f"Propagator: {prop_exp}\n"
        f"Statevector: {sv_exp}\n"
        f"Observable: {pauli_label}\n"
        f"State: {state_label}"
)

