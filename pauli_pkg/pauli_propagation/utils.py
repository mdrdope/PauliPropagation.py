# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/utils.py
from __future__ import annotations
import numpy as np
from qiskit.quantum_info import Pauli
__all__ = [
    "encode_pauli",
    "decode_pauli", 
    "weight_of_key",
    "random_su4",
]

def encode_pauli(p: np.typing.NDArray | "Pauli") -> int:
    """
    Pack a q-qubit qiskit.Pauli into one 2q-bit integer key = (Z|X).
    
    Parameters
    ----------
    p : np.typing.NDArray | Pauli
        Input Pauli operator to encode
        
    Returns
    -------
    int
        2q-bit integer key where lower q bits represent X operators
        and upper q bits represent Z operators
    """
    n = len(p.z)
    x_bits = 0
    z_bits = 0
    for i in range(n):
        if p.x[i]:
            x_bits |= 1 << i
        if p.z[i]:
            z_bits |= 1 << i
    return (z_bits << n) | x_bits

def decode_pauli(key: int, n: int) -> "Pauli":
    """
    Unpack integer key back to qiskit.Pauli.
    
    Parameters
    ----------
    key : int
        2q-bit integer key encoding Pauli operator
    n : int
        Number of qubits
        
    Returns
    -------
    Pauli
        Decoded Pauli operator
        
    Notes
    -----
    Delayed import of qiskit.quantum_info.Pauli to avoid circular dependencies
    """
    
    mask   = (1 << n) - 1
    x_bits =  key        & mask
    z_bits = (key >> n)  & mask
    x = np.array([(x_bits >> i) & 1 for i in range(n)], dtype=bool)
    z = np.array([(z_bits >> i) & 1 for i in range(n)], dtype=bool)
    return Pauli((z, x))

# Lookup table for population count (popcount) of 8-bit integers
_POPCOUNT_TABLE = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)

# def _weight_of_key(key: int, n: int) -> int:
#     """
#     Calculate Pauli weight using bit manipulation and lookup table.
    
#     Parameters
#     ----------
#     key : int
#         Encoded Pauli operator
#     n : int
#         Number of qubits
        
#     Returns
#     -------
#     int
#         Number of non-identity Pauli factors
#     """
#     # Combine X and Z bits using bitwise OR
#     mask = (1 << n) - 1
#     bits = (key & mask) | ((key >> n) & mask)
    
#     # Use lookup table for faster population count
#     count = 0
#     while bits:
#         # Process 8 bits at a time using lookup table
#         count += _POPCOUNT_TABLE[bits & 0xFF]
#         bits >>= 8
    
#     return count

def _weight_of_key(key: int, n: int) -> int:
    """
    Calculate Pauli weight using bit manipulation.
    """
    # combine X and Z bits, then use native bit_count
    bits = (key & ((1 << n) - 1)) | (key >> n)
    return bits.bit_count()

def weight_of_key(key: int, n: int) -> int:
    """
    Calculate Pauli weight = number of non-identity factors.
    
    Parameters
    ----------
    key : int
        Encoded Pauli operator
    n : int
        Number of qubits
        
    Returns
    -------
    int
        Number of non-identity Pauli factors
    """
    return _weight_of_key(key, n)

def random_su4() -> np.ndarray:
    """
    Generate a Haar-random 4x4 special-unitary matrix (det = 1).
    
    Returns
    -------
    np.ndarray
        4x4 complex matrix in SU(4)
        
    Notes
    -----
    Uses standard Ginibre -> QR decomposition method
    """
    # Standard Ginibre -> QR trick
    z = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    # Fix phases so R has positive diagonal
    ph = np.diag(r) / np.abs(np.diag(r))
    q  = q * ph
    # Enforce det=1
    q /= np.linalg.det(q) ** 0.25
    return q
