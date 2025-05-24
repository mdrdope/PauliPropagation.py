# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/gates.py
"""
Quantum gate implementations for Pauli propagation.

This module provides implementations of various quantum gates and their Pauli propagation rules.
The gates are registered in the QuantumGate class registry and can be accessed by name.

Notes
-----
- All gates implement Pauli propagation rules that transform Pauli operators
- The implementation uses bit manipulation for efficient Pauli operator representation
- Complex coefficients are handled throughout the propagation
- All gates return List[PauliTerm] for consistency
"""

# todo: 
# RX/RY/RZ gates
# CZ gate
# CNOT gate
# SWAP gate
# Toffoli gate
# Fredkin gate
# Ising gate


import numpy as np
np.dtype(np.float64)  
from .pauli_term import PauliTerm
from typing import Tuple, Dict, Callable, Optional, List, Union
from functools import lru_cache
dtype=np.float64




class QuantumGate:
    """
    Registry for quantum gate implementations.
    
    This class maintains a registry of quantum gate implementations that can be
    accessed by name. Each gate implementation must follow the standard interface
    for Pauli propagation using PauliTerm objects.
    
    Attributes
    ----------
    _registry : Dict[str, Callable]
        Dictionary mapping gate names to their implementation functions.
    """
    _registry: Dict[str, Callable] = {}
    _cache: Dict[str, Callable] = {}  # Cache for faster lookup

    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Get the implementation for a named gate.
        
        Parameters
        ----------
        name : str
            Name of the gate to retrieve.
            
        Returns
        -------
        Callable
            The gate implementation function.
            
        Raises
        ------
        NotImplementedError
            If no implementation exists for the requested gate.
        """
        # Check cache first for faster lookup
        if name in cls._cache:
            return cls._cache[name]
            
        # If not in cache, check registry
        if name not in cls._registry:
            raise NotImplementedError(f"No rule for gate '{name}'")
            
        # Add to cache and return
        cls._cache[name] = cls._registry[name]
        return cls._cache[name]


def cx_gate(pauli_term: PauliTerm, ctrl: int, tgt: int) -> List[PauliTerm]:
    """
    Implement CNOT gate Pauli propagation.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    ctrl : int
        Control qubit index
    tgt : int
        Target qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms (always single term for CX)
    """
    coeff = pauli_term.coeff
    key = pauli_term.key
    n = pauli_term.n
    
    # Extract X and Z bits for control and target qubits
    x_c = (key >>  ctrl)     & 1
    z_c = (key >> (n+ctrl))  & 1
    x_t = (key >>  tgt)      & 1
    z_t = (key >> (n+tgt))   & 1
    
    # Compute phase factor
    minus = x_c & z_t & (1 ^ (x_t ^ z_c))
    phase = -1 if minus else +1
    
    # Compute new X and Z bits
    x_tn = x_t ^ x_c
    z_cn = z_c ^ z_t
    
    # Update key with new bits
    outk = key
    if x_tn != x_t: outk ^= 1 << tgt
    if z_cn != z_c: outk ^= 1 << (n+ctrl)
    
    return [PauliTerm(coeff * phase, outk, n)]


def t_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
    """
    Implement T gate Pauli propagation.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms (1 or 2 terms)
    """
    coeff = pauli_term.coeff
    key = pauli_term.key
    n = pauli_term.n
    
    # Extract X and Z bits for target qubit
    x = (key >>  q)     & 1
    z = (key >> (n+q))  & 1
    
    # Z or I Pauli: no change
    if (z and not x) or (not x and not z):
        return [pauli_term]
        
    # X or Y Pauli: splits into two terms
    key2 = key ^ (1 << (n+q))  # Flip Z bit
    c1 = coeff / np.sqrt(2)
    c2 = +c1 if z else -c1
    
    return [PauliTerm(c1, key, n), PauliTerm(c2, key2, n)]


# Pre-compute 2-qubit Pauli matrices (improves SU4 performance)
_SINGLE_P = (
    np.eye(2, dtype=complex),
    np.array([[0,1],[1,0]],      dtype=complex),
    np.array([[0,-1j],[1j,0]],   dtype=complex),
    np.array([[1,0],[0,-1]],     dtype=complex),
)
_P_STACK = np.stack([np.kron(_SINGLE_P[q2], _SINGLE_P[q1])
                     for q2 in range(4) for q1 in range(4)])

# Precompute bit operations
_CODE_TO_BITS = [(0,0), (0,1), (1,1), (1,0)]

def _code_from_bits(z: int, x: int) -> int:
    """Convert Z and X bits to Pauli code."""
    return (z << 1) | x if z == 0 else (2 | (x ^ 1))

def _bits_from_code(c: int) -> Tuple[int, int]:
    """Convert Pauli code to Z and X bits."""
    return _CODE_TO_BITS[c]


def su4_gate(pauli_term: PauliTerm, q1: int, q2: int, mat: np.ndarray) -> List[PauliTerm]:
    """
    Implement arbitrary 2-qubit gate Pauli propagation.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q1 : int
        First qubit index
    q2 : int
        Second qubit index
    mat : np.ndarray
        4x4 unitary matrix
        
    Returns
    -------
    List[PauliTerm]
        List of output Pauli terms
    """
    coeff = pauli_term.coeff
    key = pauli_term.key
    n = pauli_term.n
    
    # Extract X and Z bits for both qubits
    x1 = (key >>  q1)     & 1
    z1 = (key >> (n+q1))  & 1
    x2 = (key >>  q2)     & 1
    z2 = (key >> (n+q2))  & 1
    
    # Calculate index into Pauli basis
    beta_idx = 4*_code_from_bits(z2,x2) + _code_from_bits(z1,x1)

    # Conjugate with unitary matrix
    conj = mat.conj().T @ _P_STACK[beta_idx] @ mat
    
    # Calculate coefficients in Pauli basis
    coeffs = 0.25 * np.einsum('aij,ij->a', _P_STACK.conj(), conj)

    # Prepare output (optimized for performance)
    c_arr = coeff * coeffs                       # vectorized compute of all 16 c values
    mask = (np.abs(c_arr.real) + np.abs(c_arr.imag)) >= 1e-12
    significant_idxs = np.nonzero(mask)[0]       # only non-negligible indices

    result = []
    for alpha in significant_idxs:
        c = c_arr[alpha]
        code2a, code1a = divmod(alpha, 4)
        new_key = key
        
        # apply bit flips for q1
        z1, x1 = _bits_from_code(code1a)
        if ((new_key >> q1) & 1) != x1:
            new_key ^= 1 << q1
        if ((new_key >> (n+q1)) & 1) != z1:
            new_key ^= 1 << (n+q1)
        
        # apply bit flips for q2
        z2, x2 = _bits_from_code(code2a)
        if ((new_key >> q2) & 1) != x2:
            new_key ^= 1 << q2
        if ((new_key >> (n+q2)) & 1) != z2:
            new_key ^= 1 << (n+q2)

        result.append(PauliTerm(c.real, new_key, n)) # .real

    return result


# register
QuantumGate._registry.update({
    "cx" : cx_gate,
    "t"  : t_gate,
    "su4": su4_gate,
})


# Add static methods for direct access
setattr(QuantumGate, "CXgate",  staticmethod(cx_gate))
setattr(QuantumGate, "Tgate",   staticmethod(t_gate))
setattr(QuantumGate, "SU4gate", staticmethod(su4_gate))
