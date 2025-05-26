#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from qiskit import QuantumCircuit
from pauli_pkg.pauli_propagation.propagator import PauliPropagator
from pauli_pkg.pauli_propagation.pauli_term import PauliTerm

def create_test_circuit():
    """����һ���򵥵Ĳ��Ե�·"""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.5, 0)
    qc.ry(0.3, 1)
    return qc

def test_monte_carlo_comparison():
    """�Ƚ�ԭʼ�����������󷽷��Ľ��"""
    print("�������Ե�·...")
    qc = create_test_circuit()
    propagator = PauliPropagator(qc)
    
    # ������ʼPauli��
    init_term = PauliTerm(1.0, 0b0101, 2)  # ZI
    
    print(f"��ʼPauli��: {init_term}")
    
    # ʹ��ԭʼ������������ֵ
    print("\nʹ��ԭʼmonte_carlo_samples����...")
    sample_sizes = [1000, 5000, 10000]
    
    for M in sample_sizes:
        print(f"\n������: {M}")
        
        # ԭʼ����
        sampled_paulis, _, _, _ = propagator.monte_carlo_samples(init_term, M)
        expectation_original = propagator.expectation_pauli_sum(sampled_paulis, "00")
        print(f"ԭʼ��������ֵ: {expectation_original:.6f}")
        
        # ����ǰ�ķ���
        sampled_paulis_nonzero, _, _, _ = propagator.monte_carlo_samples_nonzero(init_term, M)
        expectation_nonzero = propagator.expectation_pauli_sum(sampled_paulis_nonzero, "00")
        print(f"����ǰ��������ֵ: {expectation_nonzero:.6f}")
        
        print(f"����: {abs(expectation_original - expectation_nonzero):.6f}")

if __name__ == "__main__":
    try:
        test_monte_carlo_comparison()
        print("\n������ɣ�")
    except Exception as e:
        print(f"����: {e}")
        import traceback
        traceback.print_exc() 