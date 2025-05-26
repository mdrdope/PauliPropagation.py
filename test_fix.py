#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify that the monte_carlo_samples_nonzero method 
gives unbiased estimates.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from pauli_pkg.pauli_propagation.utils import random_su4, encode_pauli
from pauli_pkg.pauli_propagation.pauli_term import PauliTerm
from pauli_pkg.pauli_propagation.propagator import PauliPropagator

def create_test_circuit(nx=2, ny=2):
    """Create a simple test circuit with random SU4 gates."""
    n_qubits = nx * ny
    qc = QuantumCircuit(n_qubits)
    
    # Add some random SU4 gates
    for i in range(n_qubits - 1):
        u_matrix = random_su4()
        gate = UnitaryGate(u_matrix, label='su4')
        qc.append(gate, [i, i + 1])
    
    return qc

def test_bias_comparison():
    """Compare monte_carlo_samples vs monte_carlo_samples_nonzero."""
    print("Testing bias in Monte Carlo sampling...")
    
    # Create test circuit
    nx, ny = 2, 2
    qc = create_test_circuit(nx, ny)
    prop_2d = PauliPropagator(qc)
    
    # Create initial term (Z gate on first qubit)
    n_qubits = nx * ny
    init_key = encode_pauli('Z' + 'I' * (n_qubits - 1))
    init_term = PauliTerm(1.0, init_key, n_qubits)
    
    # Product state label for expectation value calculation
    product_label = '0' * n_qubits
    
    # Different sample sizes to test
    sample_sizes = [1000, 5000, 10000]
    
    print(f"Initial observable: Z{'I' * (n_qubits - 1)}")
    print(f"Final state: |{product_label}>")
    print()
    
    # Test original method
    print("=== Original monte_carlo_samples ===")
    for M in sample_sizes:
        sampled_paulis, _, _, _ = prop_2d.monte_carlo_samples(init_term, M=M)
        
        # Calculate expectation value
        expectations = []
        for pauli in sampled_paulis:
            exp_val = prop_2d.expectation_pauli_sum([pauli], product_label)
            expectations.append(exp_val)
        
        mean_expectation = np.mean(expectations)
        print(f"M={M:5d}: <obs> = {mean_expectation:8.4f}")
    
    print()
    
    # Test nonzero method
    print("=== Fixed monte_carlo_samples_nonzero ===")
    for M in sample_sizes:
        sampled_paulis, _, _, _ = prop_2d.monte_carlo_samples_nonzero(init_term, M=M)
        
        # Calculate expectation value
        expectations = []
        for pauli in sampled_paulis:
            exp_val = prop_2d.expectation_pauli_sum([pauli], product_label)
            expectations.append(exp_val)
        
        mean_expectation = np.mean(expectations)
        print(f"M={M:5d}: <obs> = {mean_expectation:8.4f}")

def test_acceptance_rate():
    """Test and display acceptance rates."""
    print("\n=== Testing Acceptance Rates ===")
    
    # Create test circuit
    nx, ny = 2, 2
    qc = create_test_circuit(nx, ny)
    prop_2d = PauliPropagator(qc)
    
    # Create initial term
    n_qubits = nx * ny
    init_key = encode_pauli('Z' + 'I' * (n_qubits - 1))
    init_term = PauliTerm(1.0, init_key, n_qubits)
    
    M = 5000
    sampled_paulis, _, _, _ = prop_2d.monte_carlo_samples_nonzero(init_term, M=M)
    
    # Calculate acceptance rate from the implementation
    # Note: This would require access to internal tries data
    print(f"Successfully sampled {len(sampled_paulis)} non-zero paths")

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    test_bias_comparison()
    test_acceptance_rate() 