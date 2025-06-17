# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/utils.py
from __future__ import annotations
import numpy as np
import random
from qiskit.quantum_info import Pauli
from functools import lru_cache
__all__ = [
    "encode_pauli",
    "decode_pauli", 
    "weight_of_key",
    "random_su4",
    "random_su2",
    "random_state_label",
    "random_pauli_label",
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


def _weight_of_key(key: int, n: int) -> int:
    """
    Calculate Pauli weight using bit manipulation.
    """
    # combine X and Z bits, then use native bit_count
    bits = (key & ((1 << n) - 1)) | (key >> n)
    return bits.bit_count()

def weight_of_key(key: int, n: int) -> int:
    """
    Calculate the weight (number of non-identity Paulis) of a Pauli key.
    
    Uses optimized bit operations for better performance.
    
    Parameters
    ----------
    key : int
        Encoded Pauli operator
    n : int
        Number of qubits
        
    Returns
    -------
    int
        Weight of the Pauli operator
    """
    # Optimized version using bit manipulation
    # Extract X bits (lower n bits) and Z bits (upper n bits)
    x_bits = key & ((1 << n) - 1)  # Lower n bits
    z_bits = key >> n              # Upper n bits
    
    # Count non-identity Paulis: any qubit with X or Z (or both) set
    non_identity_mask = x_bits | z_bits
    return bin(non_identity_mask).count('1')

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

def random_su2() -> np.ndarray:
    """
    Generate a Haar-random 2x2 special-unitary matrix (det = 1).
    
    Returns
    -------
    np.ndarray
        2x2 complex matrix in SU(2)
        
    Notes
    -----
    Uses random unit quaternion method from "Uniform random rotations" 
    by Shepperd (1987)
    """
    # Generate random unit quaternion using method from 
    # "Uniform random rotations" by Shepperd (1987)
    u = np.random.random(3)
    
    # Convert to quaternion components
    q0 = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
    q1 = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
    q2 = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
    q3 = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
    
    # Convert quaternion to SU(2) matrix
    return np.array([[q3 + 1j*q2, q1 + 1j*q0],
                     [-q1 + 1j*q0, q3 - 1j*q2]], dtype=complex)

# 缓存decode_pauli结果
@lru_cache(maxsize=1024)
def decode_pauli_cached(key: int, n: int) -> 'PauliOp':
    """
    Cached version of decode_pauli for frequently used keys.
    """
    return decode_pauli(key, n)


# Random state and Pauli generation utilities
SYMS_STATE = "01+-rl"
SYMS_PAULI = "IXYZ"


def random_state_label(n):
    """
    Generate a random product state label of length n.
    
    Parameters
    ----------
    n : int
        Length of the state label
        
    Returns
    -------
    str
        Random product state label using symbols from "01+-rl"
        
    Examples
    --------
    >>> label = random_state_label(3)
    >>> len(label)
    3
    >>> all(c in "01+-rl" for c in label)
    True
    """
    return "".join(random.choice(SYMS_STATE) for _ in range(n))


def random_pauli_label(n):
    """
    Generate a random non-identity Pauli label of length n.
    
    Parameters
    ----------
    n : int
        Length of the Pauli label
        
    Returns
    -------
    str
        Random Pauli label using symbols from "IXYZ", guaranteed to be non-identity
        
    Examples
    --------
    >>> label = random_pauli_label(3)
    >>> len(label)
    3
    >>> all(c in "IXYZ" for c in label)
    True
    >>> # Guaranteed to have at least one non-I
    >>> "X" in label or "Y" in label or "Z" in label
    True
    """
    lbl = "".join(random.choice(SYMS_PAULI) for _ in range(n))
    if set(lbl) == {"I"}:
        pos = random.randrange(n)
        lbl = lbl[:pos] + random.choice("XYZ") + lbl[pos+1:]
    return lbl
