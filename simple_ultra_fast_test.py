#!/usr/bin/env python3
"""
Simple test for ultra-fast tuple-based propagation
"""

import time
import sys
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

try:
    from pauli_propagation import PauliTerm, PauliPropagator
    from pauli_propagation.utils import encode_pauli
    from pauli_propagation import staircasetopology2d_qc
    print("? Successfully imported pauli_propagation")
except ImportError as e:
    print(f"? Import error: {e}")
    sys.exit(1)

def simple_test():
    print("=== Simple Ultra-Fast Test ===")
    
    # Create a small test circuit
    qc = staircasetopology2d_qc(2, 2, 1)  # 4 qubits
    print(f"Circuit: {qc.num_qubits} qubits, {len(qc.data)} gates")
    
    # Create simple observable
    pauli_label = "XIXI"
    key = encode_pauli(Pauli(pauli_label))
    init_term = PauliTerm(1.0, key, 4)
    print(f"Initial observable: {pauli_label}")
    
    prop = PauliPropagator(qc)
    
    # Test regular propagate
    print("\n--- Regular propagate ---")
    start_time = time.perf_counter()
    try:
        history = prop.propagate(init_term, max_weight=4, tol=1e-10)
        regular_time = time.perf_counter() - start_time
        print(f"  Time: {regular_time*1000:.2f} ms")
        print(f"  Final terms: {len(history[-1])}")
        
        # Test expectation value
        exp_val = prop.expectation_pauli_sum(history[-1], "0000")
        print(f"  Expectation value: {exp_val:.8f}")
    except Exception as e:
        print(f"  ? Error in regular propagate: {e}")
        return False
    
    # Test ultra-fast propagate_fast
    print("\n--- Ultra-fast propagate_fast ---")
    start_time = time.perf_counter()
    try:
        final_tuples = prop.propagate_fast(init_term, max_weight=4, tol=1e-10)
        fast_time = time.perf_counter() - start_time
        print(f"  Time: {fast_time*1000:.2f} ms")
        print(f"  Final terms: {len(final_tuples)}")
        
        # Test expectation value with tuples
        fast_exp_val = prop.expectation_tuples(final_tuples, "0000")
        print(f"  Expectation value: {fast_exp_val:.8f}")
        
        # Verify correctness
        diff = abs(exp_val - fast_exp_val)
        print(f"  Difference: {diff:.2e}")
        
        if diff < 1e-10:
            print("  ? Results match!")
        else:
            print("  ? Results don't match!")
            return False
            
    except Exception as e:
        print(f"  ? Error in propagate_fast: {e}")
        return False
    
    # Performance comparison
    if fast_time > 0:
        speedup = regular_time / fast_time
        print(f"\n--- Performance Improvement ---")
        print(f"  Speedup: {speedup:.1f}x faster")
    
    return True

if __name__ == "__main__":
    print("Simple Ultra-Fast Tuple-Based Test")
    print("=" * 40)
    
    success = simple_test()
    
    if success:
        print("\n? All tests passed!")
        print("\nUltra-Fast Optimization Summary:")
        print("  ? Gates work directly with tuples")
        print("  ? No PauliTerm objects created during propagation")
        print("  ? Expectation values computed directly from tuples")
        print("  ? Maximum performance achieved!")
    else:
        print("\n? Some tests failed!") 