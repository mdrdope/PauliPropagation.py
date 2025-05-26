#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from qiskit import QuantumCircuit
from pauli_pkg.pauli_propagation.propagator import PauliPropagator
from pauli_pkg.pauli_propagation.pauli_term import PauliTerm

def create_test_circuit():
    """创建一个简单的测试电路"""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.5, 0)
    qc.ry(0.3, 1)
    return qc

def test_monte_carlo_comparison():
    """比较原始方法和修正后方法的结果"""
    print("创建测试电路...")
    qc = create_test_circuit()
    propagator = PauliPropagator(qc)
    
    # 创建初始Pauli项
    init_term = PauliTerm(1.0, 0b0101, 2)  # ZI
    
    print(f"初始Pauli项: {init_term}")
    
    # 使用原始方法计算期望值
    print("\n使用原始monte_carlo_samples方法...")
    sample_sizes = [1000, 5000, 10000]
    
    for M in sample_sizes:
        print(f"\n样本数: {M}")
        
        # 原始方法
        sampled_paulis, _, _, _ = propagator.monte_carlo_samples(init_term, M)
        expectation_original = propagator.expectation_pauli_sum(sampled_paulis, "00")
        print(f"原始方法估计值: {expectation_original:.6f}")
        
        # 修正前的方法
        sampled_paulis_nonzero, _, _, _ = propagator.monte_carlo_samples_nonzero(init_term, M)
        expectation_nonzero = propagator.expectation_pauli_sum(sampled_paulis_nonzero, "00")
        print(f"修正前方法估计值: {expectation_nonzero:.6f}")
        
        print(f"差异: {abs(expectation_original - expectation_nonzero):.6f}")

if __name__ == "__main__":
    try:
        test_monte_carlo_comparison()
        print("\n测试完成！")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc() 