import sys
import os
import numpy as np

# ��ӵ�ǰĿ¼��Python·����
sys.path.append(os.path.abspath('.'))

from pauli_pkg.pauli_propagation import PauliTerm, PauliPropagator
from pauli_pkg.pauli_propagation.utils import encode_pauli, decode_pauli
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

def test_large_qubit_encoding():
    """���Ա���ͽ��볬��63�����ӱ��ص��������"""
    n = 70  # ʹ��70�����ӱ��أ�����int64��63λ����
    pauli_str = "I" * 69 + "Z"
    pauli = Pauli(pauli_str)
    
    print(f"���Ա���ͽ��� {n} �����ӱ��ص��������...")
    
    # �����������
    key = encode_pauli(pauli)
    print(f"������keyֵ: {key}")
    print(f"λ��: {key.bit_length()}")
    
    # Python���������Ͽ��Դ洢����������
    print(f"����Python��ԭ������������Ϊ: {type(key)}")
    
    # ��֤������
    expected = (1 << (n + 69))  # Z�����һ������λ�ã���n+index
    print(f"Ԥ��ֵ: {expected}")
    
    # ���벢��֤
    decoded = decode_pauli(key, n)
    result = decoded.to_label()
    print(f"������: {result}")
    print(f"�Ƿ���ԭʼ�������ƥ��: {result == pauli_str}")
    
    return result == pauli_str

def test_large_qubit_propagation():
    """���Գ���63�����ӱ��صĵ�·����"""
    n = 70
    print(f"\n���� {n} �����ӱ��صĵ�·����...")
    
    # ������·
    qc = QuantumCircuit(n)
    qc.x(0)  # X���ڵ�һ�����ӱ�����
    qc.cx(0, 69)  # CX�Ŵӵ�һ�������һ�����ӱ���
    
    # �����۲��������һ�����ӱ����ϵ�Z
    pauli_str = "I" * 69 + "Z"
    pauli = Pauli(pauli_str)
    key = encode_pauli(pauli)
    obs = PauliTerm(1.0, key, n)
    
    print(f"��ʼ�۲���: {obs}")
    
    # ִ�д���
    prop = PauliPropagator(qc)
    print("��ʼ����...")
    layers = prop.propagate(obs)
    
    # ��֤���
    final_terms = layers[-1]
    print(f"�����������: {len(final_terms)}")
    
    # ����Ԥ�����Ľ��Ӧ�����ڵ�һ�����ӱ����ϵ�Z
    expected_result = "Z" + "I" * 69
    expected_key = encode_pauli(Pauli(expected_result))
    
    # ��֤����а���Ԥ�ڵ���
    found = False
    for term in final_terms:
        print(f"�����: {term}")
        if term.key == expected_key:
            found = True
            print(f"�ҵ�Ԥ����! ϵ��: {term.coeff}")
            break
    
    print(f"�Ƿ��ҵ�Ԥ�ڽ��: {found}")
    return found

if __name__ == "__main__":
    print("=== ���Դ������ӱ��صĴ������� ===")
    encoding_success = test_large_qubit_encoding()
    propagation_success = test_large_qubit_propagation()
    
    if encoding_success and propagation_success:
        print("\n���в��Գɹ�ͨ��! ������Դ�����63�����ӱ��ء�")
    else:
        print("\n����ʧ�ܡ�") 