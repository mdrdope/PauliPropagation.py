import sys
import os
import numpy as np

# 添加当前目录到Python路径中
sys.path.append(os.path.abspath('.'))

from pauli_pkg.pauli_propagation import PauliTerm, PauliPropagator
from pauli_pkg.pauli_propagation.utils import encode_pauli, decode_pauli
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

def test_large_qubit_encoding():
    """测试编码和解码超过63个量子比特的泡利算符"""
    n = 70  # 使用70个量子比特，超过int64的63位限制
    pauli_str = "I" * 69 + "Z"
    pauli = Pauli(pauli_str)
    
    print(f"测试编码和解码 {n} 个量子比特的泡利算符...")
    
    # 编码泡利算符
    key = encode_pauli(pauli)
    print(f"编码后的key值: {key}")
    print(f"位数: {key.bit_length()}")
    
    # Python整数理论上可以存储任意大的整数
    print(f"这是Python的原生整数，类型为: {type(key)}")
    
    # 验证编码结果
    expected = (1 << (n + 69))  # Z在最后一个比特位置，即n+index
    print(f"预期值: {expected}")
    
    # 解码并验证
    decoded = decode_pauli(key, n)
    result = decoded.to_label()
    print(f"解码结果: {result}")
    print(f"是否与原始泡利算符匹配: {result == pauli_str}")
    
    return result == pauli_str

def test_large_qubit_propagation():
    """测试超过63个量子比特的电路传播"""
    n = 70
    print(f"\n测试 {n} 个量子比特的电路传播...")
    
    # 创建电路
    qc = QuantumCircuit(n)
    qc.x(0)  # X门在第一个量子比特上
    qc.cx(0, 69)  # CX门从第一个到最后一个量子比特
    
    # 创建观测量：最后一个量子比特上的Z
    pauli_str = "I" * 69 + "Z"
    pauli = Pauli(pauli_str)
    key = encode_pauli(pauli)
    obs = PauliTerm(1.0, key, n)
    
    print(f"初始观测量: {obs}")
    
    # 执行传播
    prop = PauliPropagator(qc)
    print("开始传播...")
    layers = prop.propagate(obs)
    
    # 验证结果
    final_terms = layers[-1]
    print(f"传播后的项数: {len(final_terms)}")
    
    # 我们预期最后的结果应该是在第一个量子比特上的Z
    expected_result = "Z" + "I" * 69
    expected_key = encode_pauli(Pauli(expected_result))
    
    # 验证结果中包含预期的项
    found = False
    for term in final_terms:
        print(f"结果项: {term}")
        if term.key == expected_key:
            found = True
            print(f"找到预期项! 系数: {term.coeff}")
            break
    
    print(f"是否找到预期结果: {found}")
    return found

if __name__ == "__main__":
    print("=== 测试大量量子比特的处理能力 ===")
    encoding_success = test_large_qubit_encoding()
    propagation_success = test_large_qubit_propagation()
    
    if encoding_success and propagation_success:
        print("\n所有测试成功通过! 代码可以处理超过63个量子比特。")
    else:
        print("\n测试失败。") 