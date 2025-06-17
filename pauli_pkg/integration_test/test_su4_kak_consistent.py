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
@pytest.mark.parametrize("trial", range(5))
def test_staircase_su4_kak_expectation_consistency(nx, ny, trial):
    """
    Test expectation consistency between original SU4 staircase circuit
    and its KAK decomposed version for various observables and states.
    """
    # Set seed for reproducibility in each trial
    seed = 42 + trial
    
    # Build original circuit with random SU4 gates
    qc_original = staircasetopology2d_qc(nx, ny, seed=seed)
    n = qc_original.num_qubits
    
    # Apply KAK decomposition to get decomposed circuit
    qc_decomposed = su4_kak_reconstruct(qc_original)
    
    # Generate random observable and initial state
    random.seed(seed + 1000)  # Use different seed for observable/state generation
    np.random.seed(seed + 1000)
    
    pauli_label = random_pauli_label(n)
    state_label = random_state_label(n)
    
    # Build observable
    obs = Pauli(pauli_label)
    key = encode_pauli(obs)
    init_term = PauliTerm(1.0, key, n)
    
    # Compute expectation using original circuit
    prop_original = PauliPropagator(qc_original)
    layers_original = prop_original.propagate(init_term, max_weight=None)
    exp_original = prop_original.expectation_pauli_sum(layers_original[-1], state_label)
    
    # Compute expectation using decomposed circuit
    prop_decomposed = PauliPropagator(qc_decomposed)
    layers_decomposed = prop_decomposed.propagate(init_term, max_weight=None)
    exp_decomposed = prop_decomposed.expectation_pauli_sum(layers_decomposed[-1], state_label)
    
    # Assert consistency
    assert abs(exp_original - exp_decomposed) < 1e-10, (
        f"Expectation mismatch for {nx}x{ny} staircase, trial {trial}\n"
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


# @pytest.mark.parametrize("nx, ny", [(1, 3), (2, 2), (1, 4)])
# @pytest.mark.parametrize("qubit_idx", [0, 1])
# def test_staircase_su4_kak_single_qubit_observables(nx, ny, qubit_idx):
#     """
#     Test expectation consistency for single-qubit Pauli observables
#     between original and KAK decomposed staircase circuits.
#     """
#     seed = 777
#     qc_original = staircasetopology2d_qc(nx, ny, seed=seed)
#     n = qc_original.num_qubits
    
#     # Skip if qubit index is out of range
#     if qubit_idx >= n:
#         pytest.skip(f"Qubit index {qubit_idx} out of range for {nx}x{ny} grid")
    
#     qc_decomposed = su4_kak_reconstruct(qc_original)
    
#     # Test X, Y, Z observables on the specified qubit
#     for pauli_op in ["X", "Y", "Z"]:
#         pauli_list = ["I"] * n
#         pauli_list[qubit_idx] = pauli_op
#         pauli_label = "".join(pauli_list)
        
#         obs = Pauli(pauli_label)
#         key = encode_pauli(obs)
#         init_term = PauliTerm(1.0, key, n)
        
#         # Test with different initial states
#         for state_label in ["0" * n, "1" * n, "+" * n, "-" * n]:
#             # Original circuit expectation
#             prop_original = PauliPropagator(qc_original)
#             layers_original = prop_original.propagate(init_term, max_weight=None)
#             exp_original = prop_original.expectation_pauli_sum(layers_original[-1], state_label)
            
#             # Decomposed circuit expectation
#             prop_decomposed = PauliPropagator(qc_decomposed)
#             layers_decomposed = prop_decomposed.propagate(init_term, max_weight=None)
#             exp_decomposed = prop_decomposed.expectation_pauli_sum(layers_decomposed[-1], state_label)
            
#             assert abs(exp_original - exp_decomposed) < 1e-10, (
#                 f"Single-qubit observable mismatch for {nx}x{ny} staircase\n"
#                 f"Observable: {pauli_op} on qubit {qubit_idx}\n"
#                 f"State: {state_label}\n"
#                 f"Original: {exp_original}\n"
#                 f"Decomposed: {exp_decomposed}"
#             )


# @pytest.mark.parametrize("nx, ny", [(2, 2), (1, 4)])
# def test_staircase_su4_kak_two_qubit_observables(nx, ny):
#     """
#     Test expectation consistency for two-qubit Pauli observables
#     between original and KAK decomposed staircase circuits.
#     """
#     seed = 999
#     qc_original = staircasetopology2d_qc(nx, ny, seed=seed)
#     n = qc_original.num_qubits
    
#     if n < 2:
#         pytest.skip(f"Need at least 2 qubits for two-qubit observable test")
    
#     qc_decomposed = su4_kak_reconstruct(qc_original)
    
#     # Test a few two-qubit observables
#     two_qubit_obs = ["XX", "YY", "ZZ", "XZ", "ZX"]
    
#     for obs_type in two_qubit_obs:
#         # Apply observable to first two qubits
#         pauli_list = [obs_type[0], obs_type[1]] + ["I"] * (n - 2)
#         pauli_label = "".join(pauli_list)
        
#         obs = Pauli(pauli_label)
#         key = encode_pauli(obs)
#         init_term = PauliTerm(1.0, key, n)
        
#         # Test with |++...+? state
#         state_label = "+" * n
        
#         # Original circuit expectation
#         prop_original = PauliPropagator(qc_original)
#         layers_original = prop_original.propagate(init_term, max_weight=None)
#         exp_original = prop_original.expectation_pauli_sum(layers_original[-1], state_label)
        
#         # Decomposed circuit expectation
#         prop_decomposed = PauliPropagator(qc_decomposed)
#         layers_decomposed = prop_decomposed.propagate(init_term, max_weight=None)
#         exp_decomposed = prop_decomposed.expectation_pauli_sum(layers_decomposed[-1], state_label)
        
#         assert abs(exp_original - exp_decomposed) < 1e-10, (
#             f"Two-qubit observable mismatch for {nx}x{ny} staircase\n"
#             f"Observable: {pauli_label}\n"
#             f"State: {state_label}\n"
#             f"Original: {exp_original}\n"
#             f"Decomposed: {exp_decomposed}"
#         ) 