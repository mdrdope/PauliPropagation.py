#!/usr/bin/env python3

print("开始测试...")

try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("导入模块...")
    import numpy as np
    from qiskit import QuantumCircuit
    from pauli_pkg.pauli_propagation.propagator import PauliPropagator
    from pauli_pkg.pauli_propagation.pauli_term import PauliTerm
    
    print("创建测试电路...")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    propagator = PauliPropagator(qc)
    
    print("创建初始Pauli项...")
    init_term = PauliTerm(1.0, 0b0101, 2)  # ZI
    
    print("运行蒙特卡洛采样...")
    M = 1000
    
    # 原始方法
    print("原始方法...")
    sampled_paulis, _, _, _ = propagator.monte_carlo_samples(init_term, M)
    expectation_original = propagator.expectation_pauli_sum(sampled_paulis, "00")
    print(f"原始方法估计值: {expectation_original}")
    
    # 修正后方法
    print("修正后方法...")
    sampled_paulis_nonzero, _, _, _ = propagator.monte_carlo_samples_nonzero(init_term, M)
    expectation_nonzero = propagator.expectation_pauli_sum(sampled_paulis_nonzero, "00")
    print(f"修正后方法估计值: {expectation_nonzero}")
    
    print(f"差异: {abs(expectation_original - expectation_nonzero)}")
    print("测试完成！")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc() 