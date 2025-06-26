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

LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

def cx_matrix(ctrl: int, tgt: int) -> np.ndarray:
    """Generate CX gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.cx(ctrl, tgt)
    return Operator(qc).data

def cz_matrix(ctrl: int, tgt: int) -> np.ndarray:
    """Generate CZ gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.cz(ctrl, tgt)
    return Operator(qc).data

def cy_matrix(ctrl: int, tgt: int) -> np.ndarray:
    """Generate CY gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.cy(ctrl, tgt)
    return Operator(qc).data

def ch_matrix(ctrl: int, tgt: int) -> np.ndarray:
    """Generate CH gate matrix for given control and target qubits."""
    qc = QuantumCircuit(2)
    qc.ch(ctrl, tgt)
    return Operator(qc).data

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_cx_rule(ctrl, tgt, label):
    """Exhaustively test CX gate conjugation for all 2-qubit Paulis."""
    U      = cx_matrix(ctrl, tgt)
    kernel = QuantumGate.get("cx")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - now returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}"

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_cz_rule(ctrl, tgt, label):
    """Exhaustively test CZ gate conjugation for all 2-qubit Paulis."""
    U      = cz_matrix(ctrl, tgt)
    kernel = QuantumGate.get("cz")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}"

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_cy_rule(ctrl, tgt, label):
    """Exhaustively test CY gate conjugation for all 2-qubit Paulis."""
    U      = cy_matrix(ctrl, tgt)
    kernel = QuantumGate.get("cy")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}"

@pytest.mark.parametrize("ctrl,tgt", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_ch_rule(ctrl, tgt, label):
    """Exhaustively test CH gate conjugation for all 2-qubit Paulis."""
    U      = ch_matrix(ctrl, tgt)
    kernel = QuantumGate.get("ch")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, ctrl, tgt)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ Pauli(label).to_matrix() @ U
    assert np.allclose(matsum, expected), f"Mismatch for ctrl={ctrl}, tgt={tgt}, P={label}"


@pytest.mark.parametrize("trial", range(10))
def test_cx_cy_cz_ch_random_circuits(trial):
    """Test CX, CY, CZ, CH gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 3000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits (minimum 2 for controlled gates)
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Available gate types
    gate_types = ['cx', 'cy', 'cz', 'ch']
    
    # Create quantum circuit with random controlled gates
    qc = QuantumCircuit(n, name=f"cxyz_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random gate type and qubits
        gate_type = np.random.choice(gate_types)
        ctrl, tgt = np.random.choice(n, 2, replace=False)
        
        # Add gate to circuit
        if gate_type == 'cx':
            qc.cx(ctrl, tgt)
        elif gate_type == 'cy':
            qc.cy(ctrl, tgt)
        elif gate_type == 'cz':
            qc.cz(ctrl, tgt)
        elif gate_type == 'ch':
            qc.ch(ctrl, tgt)
    
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


# def generate_random_pauli_label(n: int) -> str:
#     """Generate a random Pauli label of length n."""
#     return "".join(np.random.choice(["I", "X", "Y", "Z"], n))

# def pauli_from_label(label: str, n: int) -> PauliTerm:
#     """Convert Pauli label to PauliTerm."""
#     key = 0
#     for i, p in enumerate(reversed(label)):
#         if p == 'X':
#             key |= 1 << i
#         elif p == 'Y':
#             key |= (1 << i) | (1 << (n + i))
#         elif p == 'Z':
#             key |= 1 << (n + i)
#     return PauliTerm(1.0, key, n)