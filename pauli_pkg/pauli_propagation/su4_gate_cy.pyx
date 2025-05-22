# -*- coding: utf-8 -*-
# pauli_pkg/pauli_propagation/su4_gate_cy.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

# 声明NumPy数组类型
cnp.import_array()

# Pauli矩阵定义
_S1 = (
    np.eye(2, dtype=complex),
    np.array([[0,1],[1,0]], dtype=complex),
    np.array([[0,-1j],[1j,0]], dtype=complex),
    np.array([[1,0],[0,-1]], dtype=complex),
)
_P_STACK = np.stack([
    np.kron(_S1[q2], _S1[q1])
    for q2 in range(4) for q1 in range(4)
])

# 位操作辅助表
_CTB = ((0,0), (0,1), (1,1), (1,0))

cdef inline int _code_from_bits(int z, int x) nogil:
    return (z << 1) | x if z == 0 else (2 | (x ^ 1))

cdef inline tuple _bits_from_code(int c):
    return _CTB[c]

cpdef tuple su4_gate_cy(complex coeff,
                        unsigned long long key,
                        int n, int q1, int q2,
                        cnp.ndarray[cnp.complex128_t, ndim=2] mat):
    """
    实现任意2-qubit门的Pauli算符传播。

    参数
    ----------
    coeff : complex
        输入系数
    key : int
        Pauli算符键值
    n : int
        量子比特数量
    q1 : int
        第一个量子比特索引
    q2 : int
        第二个量子比特索引
    mat : np.ndarray
        4x4酉矩阵
    
    返回
    -------
    tuple[int, np.ndarray, np.ndarray]
        项数，系数数组，输出键值数组
    """
    # 提取两个量子比特的X和Z位
    cdef int x1 = (key >>  q1)     & 1
    cdef int z1 = (key >> (n+q1))  & 1
    cdef int x2 = (key >>  q2)     & 1
    cdef int z2 = (key >> (n+q2))  & 1
    
    # 计算Pauli基底索引
    cdef int beta_idx = 4*_code_from_bits(z2, x2) + _code_from_bits(z1, x1)

    # 与酉矩阵共轭
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] conj = mat.conj().T @ _P_STACK[beta_idx] @ mat
    
    # 计算Pauli基底中的系数
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] coeffs = 0.25 * np.einsum(
        'aij,ij->a',
        _P_STACK.conj(),
        conj,
        optimize=True
    )

    # 准备输出数组
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] coeff_out = np.empty(16, dtype=np.complex128)
    cdef cnp.ndarray[object, ndim=1] key_out = np.empty(16, dtype=object)
    
    # 计算所有系数
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] c_arr = coeff * coeffs
    
    # 创建掩码过滤微小值
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] mask = (np.abs(c_arr.real) + np.abs(c_arr.imag)) >= 1e-12
    cdef cnp.ndarray[cnp.int64_t, ndim=1] sig_idx = np.nonzero(mask)[0]

    # 处理有效的系数和键值
    cdef int L = 0
    cdef int alpha, code2a, code1a
    cdef int zb, xb
    cdef unsigned long long new_key
    cdef complex cval

    for i in range(sig_idx.shape[0]):
        alpha = sig_idx[i]
        cval = c_arr[alpha]
        
        code2a, code1a = divmod(alpha, 4)
        new_key = key
        
        # 对q1应用位翻转
        zb, xb = _bits_from_code(code1a)
        if ((new_key >> q1) & 1) != xb:
            new_key ^= 1 << q1
        if ((new_key >> (n+q1)) & 1) != zb:
            new_key ^= 1 << (n+q1)
        
        # 对q2应用位翻转
        zb, xb = _bits_from_code(code2a)
        if ((new_key >> q2) & 1) != xb:
            new_key ^= 1 << q2
        if ((new_key >> (n+q2)) & 1) != zb:
            new_key ^= 1 << (n+q2)

        coeff_out[L] = cval
        key_out[L] = new_key
        L += 1

    return L, coeff_out[:L], key_out[:L]
