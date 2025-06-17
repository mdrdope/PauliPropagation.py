# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Operator

from pauli_propagation.utils      import encode_pauli, decode_pauli, random_pauli_label, random_state_label
from pauli_propagation.pauli_term import PauliTerm
from pauli_propagation.gates      import QuantumGate
from pauli_propagation.propagator import PauliPropagator

# All 2-qubit Pauli labels (16 total)
LABELS_2Q = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]

def pauli_matrix(label: str) -> np.ndarray:
    """Convert Pauli label to matrix representation."""
    return Pauli(label).to_matrix()

def swap_matrix(q1: int, q2: int, n: int) -> np.ndarray:
    """Generate SWAP gate matrix for given qubits in n-qubit system."""
    qc = QuantumCircuit(n)
    qc.swap(q1, q2)
    return Operator(qc).data

def iswap_matrix(q1: int, q2: int, n: int) -> np.ndarray:
    """Generate iSWAP gate matrix for given qubits in n-qubit system."""
    qc = QuantumCircuit(n)
    qc.iswap(q1, q2)
    return Operator(qc).data

def pauli_terms_to_matrix(terms: list, n: int) -> np.ndarray:
    """Convert list of PauliTerm objects to their matrix sum representation."""
    total_matrix = np.zeros((2**n, 2**n), dtype=complex)
    for term in terms:
        pauli = decode_pauli(term.key, term.n)
        total_matrix += term.coeff * pauli.to_matrix()
    return total_matrix

@pytest.mark.parametrize("q1,q2", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_swap_rule(q1, q2, label):
    """Test SWAP gate conjugation for all 2-qubit Paulis."""
    U      = swap_matrix(q1, q2, 2)
    kernel = QuantumGate.get("swap")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, q1, q2)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for SWAP({q1},{q2}) on P={label}"

@pytest.mark.parametrize("q1,q2", [(0, 1), (1, 0)])
@pytest.mark.parametrize("label", LABELS_2Q)
def test_iswap_rule(q1, q2, label):
    """Test iSWAP gate conjugation for all 2-qubit Paulis."""
    U      = iswap_matrix(q1, q2, 2)
    kernel = QuantumGate.get("iswap")
    key    = encode_pauli(Pauli(label))
    
    # Create input PauliTerm
    input_term = PauliTerm(1.0, key, 2)
    
    # Apply gate - returns List[PauliTerm]
    output_terms = kernel(input_term, q1, q2)
    
    # Convert output list to matrix
    matsum = pauli_terms_to_matrix(output_terms, 2)
    
    expected = U.conj().T @ pauli_matrix(label) @ U
    assert np.allclose(matsum, expected), f"Mismatch for iSWAP({q1},{q2}) on P={label}"

# Test specific cases to verify the transformation rules
@pytest.mark.parametrize("label", ["XI", "IX", "XX", "YY", "ZZ"])
def test_swap_specific_cases(label):
    """Test SWAP on specific Pauli cases to verify transformation rules."""
    q1, q2 = 0, 1
    U      = swap_matrix(q1, q2, 2)
    kernel = QuantumGate.get("swap")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, q1, q2)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"SWAP specific case failed for P={label}"

@pytest.mark.parametrize("label", ["XI", "IX", "XX", "YY", "ZZ"])
def test_iswap_specific_cases(label):
    """Test iSWAP on specific Pauli cases to verify transformation rules."""
    q1, q2 = 0, 1
    U      = iswap_matrix(q1, q2, 2)
    kernel = QuantumGate.get("iswap")
    key    = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(1.0, key, 2)
    output_terms = kernel(input_term, q1, q2)
    
    matsum = pauli_terms_to_matrix(output_terms, 2)
    expected = U.conj().T @ pauli_matrix(label) @ U
    
    assert np.allclose(matsum, expected), f"iSWAP specific case failed for P={label}"

# Test that SWAP preserves coefficient magnitude
def test_swap_coefficient_preservation():
    """Test that SWAP preserves the magnitude of coefficients."""
    label = "XY"
    coeff = 2.5 + 1.5j
    key = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(coeff, key, 2)
    output_terms = QuantumGate.get("swap")(input_term, 0, 1)
    
    # Should have exactly one output term for SWAP
    assert len(output_terms) == 1
    result = output_terms[0]
    
    # Coefficient magnitude should be preserved
    assert abs(abs(result.coeff) - abs(coeff)) < 1e-12, "SWAP coefficient magnitude not preserved"

# Test that iSWAP preserves coefficient magnitude
def test_iswap_coefficient_preservation():
    """Test that iSWAP preserves the magnitude of coefficients."""
    label = "XY"
    coeff = 2.5 + 1.5j
    key = encode_pauli(Pauli(label))
    
    input_term = PauliTerm(coeff, key, 2)
    output_terms = QuantumGate.get("iswap")(input_term, 0, 1)
    
    # Sum of squared magnitudes should be preserved
    input_mag_sq = abs(coeff)**2
    output_mag_sq = sum(abs(term.coeff)**2 for term in output_terms)
    
    assert abs(input_mag_sq - output_mag_sq) < 1e-12, "iSWAP coefficient magnitude not preserved"

# Test edge cases
def test_swap_identity():
    """Test SWAP on identity Pauli."""
    key = encode_pauli(Pauli("II"))
    input_term = PauliTerm(1.0, key, 2)
    output_terms = QuantumGate.get("swap")(input_term, 0, 1)
    
    assert len(output_terms) == 1
    assert output_terms[0].key == key
    assert abs(output_terms[0].coeff - 1.0) < 1e-12

def test_iswap_identity():
    """Test iSWAP on identity Pauli."""
    key = encode_pauli(Pauli("II"))
    input_term = PauliTerm(1.0, key, 2)
    output_terms = QuantumGate.get("iswap")(input_term, 0, 1)
    
    assert len(output_terms) == 1
    assert output_terms[0].key == key
    assert abs(output_terms[0].coeff - 1.0) < 1e-12

def generate_random_pauli_label(n: int) -> str:
    """Generate a random Pauli label of length n."""
    return "".join(np.random.choice(["I", "X", "Y", "Z"], n))

def pauli_from_label(label: str, n: int) -> PauliTerm:
    """Convert Pauli label to PauliTerm."""
    key = 0
    for i, p in enumerate(reversed(label)):
        if p == 'X':
            key |= 1 << i
        elif p == 'Y':
            key |= (1 << i) | (1 << (n + i))
        elif p == 'Z':
            key |= 1 << (n + i)
    return PauliTerm(1.0, key, n)

def apply_gate_via_propagator(qc: QuantumCircuit, pauli_term: PauliTerm) -> list:
    """Apply quantum circuit via propagator."""
    prop = PauliPropagator(qc)
    history = prop.propagate(pauli_term)
    return history[-1]

@pytest.mark.parametrize("trial", range(10))
def test_swap_iswap_random_circuits(trial):
    """Test SWAP, iSWAP gates: compare PauliPropagator expectation vs Qiskit statevector expectation."""
    np.random.seed(trial + 9000)  # Different seed range to avoid conflicts
    
    # Random circuit parameters
    n = np.random.randint(2, 6)  # 2-5 qubits (minimum 2 for swap gates)
    n_gates = np.random.randint(3, 8)  # 3-7 gates
    
    # Random initial state and observable
    state_label = random_state_label(n)
    pauli_label = random_pauli_label(n)
    observable_key = encode_pauli(Pauli(pauli_label))
    observable = PauliTerm(1.0, observable_key, n)
    
    # Available gate types
    gate_types = ['swap', 'iswap']
    
    # Create quantum circuit with random swap gates
    qc = QuantumCircuit(n, name=f"swap_rand_{n}q_{n_gates}g")
    
    for _ in range(n_gates):
        # Choose random gate type and qubits
        gate_type = np.random.choice(gate_types)
        q1, q2 = np.random.choice(n, 2, replace=False)
        
        # Add gate to circuit
        if gate_type == 'swap':
            qc.swap(q1, q2)
        elif gate_type == 'iswap':
            qc.iswap(q1, q2)
    
    # Method 1: PauliPropagator expectation
    prop = PauliPropagator(qc)
    layers = prop.propagate(observable, max_weight=None)
    pauli_expectation = prop.expectation_pauli_sum(layers[-1], state_label)
    
    # Method 2: Qiskit statevector expectation
    from qiskit.quantum_info import Statevector
    initial_state = Statevector.from_label(state_label)
    final_state = initial_state.evolve(qc)
    qiskit_expectation = final_state.expectation_value(Pauli(pauli_label)).real
    
    # Compare results
    assert abs(pauli_expectation - qiskit_expectation) < 1e-10, (
        f"Trial {trial}: expectation mismatch {pauli_expectation} vs {qiskit_expectation} "
        f"on state {state_label}, observable {pauli_label}, circuit {n}q {n_gates}g"
    ) 