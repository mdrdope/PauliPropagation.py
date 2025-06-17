#!/usr/bin/env python3
"""
Debug test for ultra-fast tuple-based propagation
"""

import time
import traceback
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from pauli_propagation import PauliTerm, PauliPropagator
from pauli_propagation.utils import encode_pauli
from pauli_propagation import staircasetopology2d_qc

def debug_test():
    print("=== Debug Ultra-Fast Test ===")
    
    # Create a simple test circuit
    qc = staircasetopology2d_qc(2, 2, 1)  # 4 qubits
    print(f"Circuit: {qc.num_qubits} qubits, {len(qc.data)} gates")
    
    # Print gate sequence
    print("Gate sequence:")
    for i, instr in enumerate(qc.data):
        qubits = [qc.qubits.index(q) for q in instr.qubits]
        print(f"  {i}: {instr.operation.name} on qubits {qubits}")
    
    # Create simple observable
    pauli_label = "XIXI"
    key = encode_pauli(Pauli(pauli_label))
    init_term = PauliTerm(1.0, key, 4)
    print(f"Initial observable: {pauli_label}")
    print(f"Initial key: {key}")
    
    prop = PauliPropagator(qc)
    
    # Test 1: Regular propagate
    print("\n--- Test 1: Regular propagate ---")
    try:
        start_time = time.perf_counter()
        history = prop.propagate(init_term, max_weight=4, tol=1e-10)
        regular_time = time.perf_counter() - start_time
        print(f"? Time: {regular_time*1000:.2f} ms")
        print(f"? Final terms: {len(history[-1])}")
        
        # Test expectation value
        exp_val = prop.expectation_pauli_sum(history[-1], "0000")
        print(f"? Expectation value: {exp_val:.8f}")
        
        # Print some sample terms
        print("Sample final terms:")
        for i, term in enumerate(history[-1][:3]):
            print(f"  {i}: coeff={term.coeff:.6f}, key={term.key}, weight={term.weight()}")
            
    except Exception as e:
        print(f"? Error in regular propagate: {e}")
        traceback.print_exc()
        return False
    
    # Test 2: Ultra-fast propagate_fast
    print("\n--- Test 2: Ultra-fast propagate_fast ---")
    try:
        start_time = time.perf_counter()
        final_tuples = prop.propagate_fast(init_term, max_weight=4, tol=1e-10)
        fast_time = time.perf_counter() - start_time
        print(f"? Time: {fast_time*1000:.2f} ms")
        print(f"? Final terms: {len(final_tuples)}")
        
        # Print some sample tuples
        print("Sample final tuples:")
        for i, (coeff, key, n) in enumerate(final_tuples[:3]):
            from pauli_propagation.utils import weight_of_key
            weight = weight_of_key(key, n)
            print(f"  {i}: coeff={coeff:.6f}, key={key}, weight={weight}")
        
        # Test expectation value with tuples
        fast_exp_val = prop.expectation_tuples(final_tuples, "0000")
        print(f"? Expectation value: {fast_exp_val:.8f}")
        
        # Verify correctness
        diff = abs(exp_val - fast_exp_val)
        print(f"? Difference: {diff:.2e}")
        
        if diff < 1e-10:
            print("? Results match!")
        else:
            print(f"? Results don't match! Regular: {exp_val:.10f}, Fast: {fast_exp_val:.10f}")
            return False
            
    except Exception as e:
        print(f"? Error in propagate_fast: {e}")
        traceback.print_exc()
        return False
    
    # Test 3: Simple gate test
    print("\n--- Test 3: Simple gate test ---")
    try:
        from pauli_propagation.gates import TupleGate
        
        # Test a simple H gate on tuple
        test_tuple = (1.0, key, 4)
        h_func = TupleGate.get("h")
        result_tuples = h_func(test_tuple, 0)  # Apply H to qubit 0
        
        print(f"H gate on qubit 0:")
        print(f"  Input: {test_tuple}")
        print(f"  Output: {result_tuples}")
        
    except Exception as e:
        print(f"? Error in simple gate test: {e}")
        traceback.print_exc()
        return False
    
    # Performance comparison
    if fast_time > 0:
        speedup = regular_time / fast_time
        print(f"\n--- Performance Improvement ---")
        print(f"? Speedup: {speedup:.1f}x faster")
    
    return True

if __name__ == "__main__":
    print("Debug Ultra-Fast Tuple-Based Test")
    print("=" * 50)
    
    success = debug_test()
    
    if success:
        print("\n? All debug tests passed!")
    else:
        print("\n? Some debug tests failed!") 