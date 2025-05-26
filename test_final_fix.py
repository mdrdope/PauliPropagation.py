#!/usr/bin/env python3

print("��ʼ����...")

try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("����ģ��...")
    import numpy as np
    from qiskit import QuantumCircuit
    from pauli_pkg.pauli_propagation.propagator import PauliPropagator
    from pauli_pkg.pauli_propagation.pauli_term import PauliTerm
    
    print("�������Ե�·...")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    propagator = PauliPropagator(qc)
    
    print("������ʼPauli��...")
    init_term = PauliTerm(1.0, 0b0101, 2)  # ZI
    
    print("�������ؿ������...")
    M = 1000
    
    # ԭʼ����
    print("ԭʼ����...")
    sampled_paulis, _, _, _ = propagator.monte_carlo_samples(init_term, M)
    expectation_original = propagator.expectation_pauli_sum(sampled_paulis, "00")
    print(f"ԭʼ��������ֵ: {expectation_original}")
    
    # �����󷽷�
    print("�����󷽷�...")
    sampled_paulis_nonzero, _, _, _ = propagator.monte_carlo_samples_nonzero(init_term, M)
    expectation_nonzero = propagator.expectation_pauli_sum(sampled_paulis_nonzero, "00")
    print(f"�����󷽷�����ֵ: {expectation_nonzero}")
    
    print(f"����: {abs(expectation_original - expectation_nonzero)}")
    print("������ɣ�")
    
except Exception as e:
    print(f"����: {e}")
    import traceback
    traceback.print_exc() 