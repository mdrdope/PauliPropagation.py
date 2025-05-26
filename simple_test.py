#!/usr/bin/env python3

print("Starting test...")

try:
    import numpy as np
    print("Numpy imported successfully")
    
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    print("Qiskit imported successfully")
    
    from pauli_pkg.pauli_propagation.utils import random_su4, encode_pauli
    from pauli_pkg.pauli_propagation.pauli_term import PauliTerm
    from pauli_pkg.pauli_propagation.propagator import PauliPropagator
    print("Package imported successfully")
    
    # Create a small test
    np.random.seed(42)
    n_qubits = 4
    qc = QuantumCircuit(n_qubits)
    
    # Add a simple gate
    u_matrix = random_su4()
    gate = UnitaryGate(u_matrix, label='su4')
    qc.append(gate, [0, 1])
    
    prop = PauliPropagator(qc)
    print("Propagator created successfully")
    
    # Create initial term
    init_key = encode_pauli('Z' + 'I' * (n_qubits - 1))
    init_term = PauliTerm(1.0, init_key, n_qubits)
    
    # Test both methods with small sample sizes
    print("\n=== Testing with small sample sizes ===")
    
    # Original method
    sampled_original, _, _, _ = prop.monte_carlo_samples(init_term, M=100)
    product_label = '0' * n_qubits
    
    expectations_original = []
    for pauli in sampled_original:
        exp_val = prop.expectation_pauli_sum([pauli], product_label)
        expectations_original.append(exp_val)
    
    mean_original = np.mean(expectations_original)
    print(f"Original method (100 samples): {mean_original:.6f}")
    
    # Nonzero method
    sampled_nonzero, _, _, _ = prop.monte_carlo_samples_nonzero(init_term, M=100)
    
    expectations_nonzero = []
    for pauli in sampled_nonzero:
        exp_val = prop.expectation_pauli_sum([pauli], product_label)
        expectations_nonzero.append(exp_val)
    
    mean_nonzero = np.mean(expectations_nonzero)
    print(f"Nonzero method (100 samples): {mean_nonzero:.6f}")
    
    print(f"Ratio (nonzero/original): {mean_nonzero/mean_original if mean_original != 0 else 'inf':.3f}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 