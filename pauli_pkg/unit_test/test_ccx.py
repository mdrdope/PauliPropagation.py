# -*- coding: utf-8 -*-

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

# All 3-qubit Pauli labels (64 total)
LABELS_3Q = ["".join(p) for p in itertools.product("IXYZ", repeat=3)]

def ccx_matrix(ctrl1: int, ctrl2: int, tgt: int) -> np.ndarray:
    """Generate CCX (Toffoli) gate matrix for given control and target qubits."""
    qc = QuantumCircuit(3)
    qc.ccx(ctrl1, ctrl2, tgt)
    return Operator(qc).data

@pytest.mark.parametrize("ctrl1,ctrl2,tgt", [
    (0, 1, 2), (0, 2, 1), (1, 0, 2), 
    (1, 2, 0), (2, 0, 1), (2, 1, 0)
])
@pytest.mark.parametrize("label", LABELS_3Q)
def test_ccx_rule(ctrl1, ctrl2, tgt, label):
    """Exhaustively test CCX gate conjugation for all 3-qubit Paulis."""
    U      = ccx_matrix(ctrl1, ctrl2, tgt)
    kernel = QuantumGate.get("ccx")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 3)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl1, ctrl2, tgt)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 3)
    
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl1={ctrl1}, ctrl2={ctrl2}, tgt={tgt}, P={label}"

# Test specific cases to verify the decomposition works correctly
@pytest.mark.parametrize("label", ["XII", "IXI", "IIX", "XYZ", "YYY"])
def test_ccx_specific_cases(label):
    """Test CCX on specific Pauli cases to verify correctness."""
    ctrl1, ctrl2, tgt = 0, 1, 2
    U      = ccx_matrix(ctrl1, ctrl2, tgt)
    kernel = QuantumGate.get("ccx")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 3)
    output_terms = kernel(input_term, ctrl1, ctrl2, tgt)
    
    matsum = pauli_terms_to_matrix(output_terms, 3)
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    
    assert np.allclose(matsum, expected), f"Specific case failed for P={label}"

@pytest.mark.parametrize("trial", range(30))
def test_ccx_random_circuits(trial):
    """Test CCX gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 1000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(3, 6)  # 3-5 qubits (minimum 3 for CCX)
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Create quantum circuit with random CCX gates
    qc = QuantumCircuit(n, name=f"ccx_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose three random qubits for CCX
        ctrl1, ctrl2, tgt = np.random.choice(n, 3, replace=False)
        qc.ccx(ctrl1, ctrl2, tgt)
    
    # Method 1: PauliPropagator expectation
    prop = PauliPropagator(qc)
    layers = prop.propagate(observable, max_weight=None)
    pauli_expectation = prop.expectation_pauli_sum(layers[-1], state_label)
    
    # Method 2: Qiskit statevector expectation
    initial_state = Statevector.from_label(state_label)
    final_state = initial_state.evolve(qc)
    qiskit_expectation = final_state.expectation_value(Pauli(pauli_label)).real
    
    # Compare results
    assert abs(pauli_expectation - qiskit_expectation) < 1e-10, (
        f"Trial {trial}: expectation mismatch {pauli_expectation} vs {qiskit_expectation} "
        f"on state {state_label}, observable {pauli_label}, circuit {n}q {n_gates}g"
    ) 