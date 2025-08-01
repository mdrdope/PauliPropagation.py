# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/gates.py
"""
Quantum gate implementations for Pauli propagation.

This module provides implementations of various quantum gates and their Pauli propagation rules.
The gates are registered in the QuantumGate class registry and can be accessed by name.

Notes
-----
- All gates implement Pauli propagation rules that transform Pauli operators as U† P U
- The implementation uses bit manipulation for efficient Pauli operator representation
- Complex coefficients are handled throughout the propagation
- All gates return List[PauliTerm] for consistency
"""

import math
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
    _param_extractors : Dict[str, Callable]
        Dictionary mapping gate names to their parameter extraction functions.
    """
    _registry: Dict[str, Callable] = {}
    _cache: Dict[str, Callable] = {}  # Cache for faster lookup
    _param_extractors: Dict[str, Callable] = {}

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

    @classmethod
    def extract_params(cls, gate_name: str, instruction) -> tuple:
        """
        Extract parameters for a gate from a qiskit instruction.
        
        Parameters
        ----------
        gate_name : str
            Name of the gate
        instruction
            Qiskit instruction object
            
        Returns
        -------
        tuple
            Extracted parameters for the gate
        """
        if gate_name in cls._param_extractors:
            return cls._param_extractors[gate_name](instruction)
        else:
            return ()  # No parameters for this gate

    @classmethod
    def register_param_extractor(cls, gate_name: str, extractor: Callable):
        """
        Register a parameter extractor for a gate.
        
        Parameters
        ----------
        gate_name : str
            Name of the gate
        extractor : Callable
            Function that takes an instruction and returns a tuple of parameters
        """
        cls._param_extractors[gate_name] = extractor

    @staticmethod
    def register_gate(name: str):
        """
        Decorator to register a gate function in the QuantumGate registry.
        
        Parameters
        ----------
        name : str
            Name of the gate to register
            
        Returns
        -------
        Callable
            Decorator function
            
        Example
        -------
        @QuantumGate.register_gate("x")
        def x_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
            # implementation
            pass
        """
        def decorator(func):
            QuantumGate._registry[name] = func
            # Also create a static method on the class for direct access
            setattr(QuantumGate, f"{name.upper()}_gate", staticmethod(func))
            return func
        return decorator


# ------------------------------------------------------------------------------------------------------------------------
# Single-qubit gates
# ------------------------------------------------------------------------------------------------------------------------


@QuantumGate.register_gate("x")
def x_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
    """
    Implements Pauli-X gate propagation: X† P X.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms (single term with possible phase change)
        
    Notes
    -----
    The phase is -1 if the original Pauli term has a Z component (z=1) on the target qubit, +1 otherwise.
    """
    coeff = pauli_term.coeff
    key   = pauli_term.key
    n     = pauli_term.n

    z = (key >> (n+q)) & 1

    phase = -1 if z else +1
    
    return [PauliTerm(coeff * phase, key, n)]


@QuantumGate.register_gate("y")
def y_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
    """
    Implements Pauli-Y gate propagation: Y† P Y.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms (single term with possible phase change)
        
    Notes
    -----
    The phase is -1 if x ^ z == 1, +1 otherwise, where x and z are the X and Z bits of the target qubit.
    """
    coeff = pauli_term.coeff
    key   = pauli_term.key
    n     = pauli_term.n

    x = (key >> q) & 1
    z = (key >> (n+q)) & 1

    phase = -1 if (x ^ z) else +1
    return [PauliTerm(coeff * phase, key, n)]


@QuantumGate.register_gate("z")
def z_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
    """
    Implements Pauli-Z gate propagation: Z† P Z.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms (single term with possible phase change)
        
    Notes
    -----
    The phase is -1 if the original Pauli term has an X component (x=1) on the target qubit, +1 otherwise.
    """
    coeff = pauli_term.coeff
    key   = pauli_term.key
    n     = pauli_term.n

    x = (key >> q) & 1

    phase = -1 if x else +1
    return [PauliTerm(coeff * phase, key, n)]


@QuantumGate.register_gate("h")
def h_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
    """
    Implements Hadamard gate Pauli propagation: H† P H.

    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Qubit index to apply H on

    Returns
    -------
    List[PauliTerm]
        Output Pauli terms (always a single term for H)
    
    Notes
    -----
    The Hadamard gate swaps X and Z on the target qubit. The phase is -1 if the original Pauli term is Y (i.e., x=z=1), +1 otherwise.
    """
    coeff = pauli_term.coeff
    key = pauli_term.key
    n = pauli_term.n

    # Extract X and Z bits at position q
    x = (key >>  q)     & 1
    z = (key >> (n+q))  & 1

    # Phase = -1 exactly when P=Y i.e. x=z=1
    phase = -1 if (x & z) else +1

    # Swap x and z bits
    new_key = key
    if z != x:  # X bit should become old Z
        new_key ^= 1 << q
    if x != z:  # Z bit at n+q becomes old X
        new_key ^= 1 << (n+q)

    return [PauliTerm(coeff * phase, new_key, n)]


@QuantumGate.register_gate("t")
def t_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
    """
    Implements T gate Pauli propagation: T† P T.
    
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
    
    Notes
    -----
    The T gate is equivalent to Rz(pi/4). For X or Y Pauli terms, the propagation splits into two terms with appropriate coefficients.
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


@QuantumGate.register_gate("rx")
def rx_gate(pauli_term: PauliTerm, q: int, theta: float) -> List[PauliTerm]:
    """
    Implements Rx gate Pauli propagation: Rx†(theta) P Rx(theta).
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms (1 or 2 terms depending on input)
        
    Notes
    -----
    Transformation mapping:
      I -> I
      X -> X
      Z -> cos(theta) * Z + sin(theta) * Y
      Y -> cos(theta) * Y - sin(theta) * Z
    """
    coeff, key, n = pauli_term.coeff, pauli_term.key, pauli_term.n
    x = (key >> q)       & 1
    z = (key >> (n+q))   & 1
    c = math.cos(theta)
    s = math.sin(theta)
    mask_x = 1 << q

    # I or X stay single
    if x == 0 and z == 0:
        return [pauli_term]
    if x == 1 and z == 0:
        return [PauliTerm(coeff, key, n)]

    # Z -> cos(theta) * Z + sin(theta) * Y
    if x == 0 and z == 1:
        return [PauliTerm(coeff * c,key,n),PauliTerm(coeff * s,key ^ mask_x, n)]

    # Y -> cos(theta) * Y - sin(theta) * Z
    # bits(1,1)->(0,1) via flipping x
    return [PauliTerm(coeff * c,key,n),PauliTerm(coeff * -s,key ^ mask_x, n)]

# Register parameter extractor for rx gate
QuantumGate.register_param_extractor("rx", lambda instruction: (instruction.operation.params[0],))


@QuantumGate.register_gate("ry")
def ry_gate(pauli_term: PauliTerm, q: int, theta: float) -> List[PauliTerm]:
    """
    Implements Ry gate Pauli propagation: Ry†(theta) P Ry(theta).
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms (1 or 2 terms depending on input)
        
    Notes
    -----
    Transformation mapping:
      I -> I
      Y -> Y
      X -> cos(theta) * X + sin(theta) * Z
      Z -> cos(theta) * Z - sin(theta) * X
    """
    coeff, key, n = pauli_term.coeff, pauli_term.key, pauli_term.n
    x = (key >> q)       & 1
    z = (key >> (n+q))   & 1
    c = math.cos(theta)
    s = math.sin(theta)
    mask_swap = (1 << q) | (1 << (n+q))

    # I or Y stay single
    if x == 0 and z == 0:
        return [pauli_term]
    if x == 1 and z == 1:
        return [PauliTerm(coeff, key, n)]

    # X -> cos(theta) * X + sin(theta) * Z
    if x == 1 and z == 0:
        return [PauliTerm(coeff * c,key,n),PauliTerm(coeff * s,key ^ mask_swap, n)]

    # Z -> cos(theta) * Z - sin(theta) * X
    return [PauliTerm(coeff * c,key,n),PauliTerm(coeff * -s,key ^ mask_swap, n)]

# Register parameter extractor for ry gate
QuantumGate.register_param_extractor("ry", lambda instruction: (instruction.operation.params[0],))


@QuantumGate.register_gate("rz")
def rz_gate(pauli_term: PauliTerm, q: int, theta: float) -> List[PauliTerm]:
    """
    Implements Rz gate Pauli propagation: Rz†(theta) P Rz(theta).
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms (1 or 2 terms depending on input)
        
    Notes
    -----
    Transformation mapping:
      I -> I
      Z -> Z
      X -> cos(theta) * X - sin(theta) * Y
      Y -> cos(theta) * Y + sin(theta) * X
    """
    coeff, key, n = pauli_term.coeff, pauli_term.key, pauli_term.n
    x = (key >> q)       & 1
    z = (key >> (n+q))   & 1
    c = math.cos(theta)
    s = math.sin(theta)
    mask_z = 1 << (n+q)

    # I or Z stay single
    if x == 0 and z == 0:
        return [pauli_term]
    if x == 0 and z == 1:
        return [PauliTerm(coeff, key, n)]

    # X -> cos(theta) * X - sin(theta) * Y
    if x == 1 and z == 0:
        return [PauliTerm(coeff * c,key,n),PauliTerm(coeff * -s,key ^ mask_z, n)]

    # Y -> cos(theta) * Y + sin(theta) * X
    return [PauliTerm(coeff * c,key,n),PauliTerm(coeff * s,key ^ mask_z, n),]

# Register parameter extractor for rz gate
QuantumGate.register_param_extractor("rz", lambda instruction: (instruction.operation.params[0],))


@QuantumGate.register_gate("s")
def s_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
    """
    Implements S (phase) gate Pauli propagation: S† P S (S = Rz(pi/2)).
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms
    """
    return rz_gate(pauli_term, q, math.pi/2)


@QuantumGate.register_gate("sdg")
def sdg_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
    """
    Implements S† (S-dagger) gate Pauli propagation: (S†)† P S† (S† = Rz(-pi/2)).
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms
    """
    return rz_gate(pauli_term, q, -math.pi/2)


@QuantumGate.register_gate("sx")
def sx_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
    """
    Implements SX (sqrt-X) gate Pauli propagation: SX† P SX (SX = Rx(pi/2)).
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms
    """
    return rx_gate(pauli_term, q, math.pi/2)


@QuantumGate.register_gate("sxdg")
def sxdg_gate(pauli_term: PauliTerm, q: int) -> List[PauliTerm]:
    """
    Implements SX† (SX-dagger) gate Pauli propagation: (SX†)† P SX† (SX† = Rx(-pi/2)).
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms
    """
    return rx_gate(pauli_term, q, -math.pi/2)


_SINGLE_PAULI = np.stack([np.eye(2, dtype=complex), # Pre-compute single-qubit Pauli matrices for SU(2) operations
                          np.array([[0, 1], [1, 0]], dtype=complex),  
                          np.array([[0, -1j], [1j, 0]], dtype=complex), 
                          np.array([[1, 0], [0, -1]], dtype=complex)])

@QuantumGate.register_gate("su2")
def su2_gate(pauli_term: PauliTerm, q: int, mat: np.ndarray) -> List[PauliTerm]:
    """
    Implements arbitrary single-qubit gate Pauli propagation: U† P U.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q : int
        Target qubit index
    mat : np.ndarray
        2x2 unitary matrix
        
    Returns
    -------
    List[PauliTerm]
        List of output Pauli terms
        
    Notes
    -----
    This gate implements U† P U for any single-qubit unitary U by expanding
    in the Pauli basis: U† P U = sum_i c_i P_i where P_i in {I,X,Y,Z}.
    """
    coeff = pauli_term.coeff
    key = pauli_term.key
    n = pauli_term.n
    
    # Extract X and Z bits for target qubit
    x = (key >> q) & 1
    z = (key >> (n+q)) & 1
    
    # Calculate index into single-qubit Pauli basis
    # I=0, X=1, Y=2, Z=3 using code_from_bits logic
    beta_idx = _code_from_bits(z, x)
    
    # Conjugate with unitary matrix: U† P U
    conj = mat.conj().T @ _SINGLE_PAULI[beta_idx] @ mat
    
    # Calculate coefficients in Pauli basis using einsum (like su4_gate)
    # c_i = (1/2) * Tr(P_i† * (U† P U))
    coeffs = 0.5 * np.einsum('aij,ij->a', _SINGLE_PAULI.conj(), conj)
    
    # Build output terms
    c_arr = coeff * coeffs
    mask = (np.abs(c_arr.real) + np.abs(c_arr.imag)) >= 1e-12
    significant_idxs = np.nonzero(mask)[0]
    
    result = []
    for alpha in significant_idxs:
        c = c_arr[alpha]
        new_key = key
        
        # Extract new Pauli operator bits
        z_new, x_new = _bits_from_code(alpha)
        
        # Update key with new X and Z bits for target qubit
        if ((new_key >> q) & 1) != x_new:
            new_key ^= 1 << q
        if ((new_key >> (n+q)) & 1) != z_new:
            new_key ^= 1 << (n+q)
        
        # Clean numerical errors in imaginary part
        if abs(c.imag) < 1e-12:
            c = c.real + 0j
            
        result.append(PauliTerm(c, new_key, n))
    
    return result

# Register parameter extractor for su2 gate
QuantumGate.register_param_extractor("su2", lambda instruction: (instruction.operation.to_matrix(),))


# ------------------------------------------------------------------------------------------------------------------------
# Two-qubit gates
# ------------------------------------------------------------------------------------------------------------------------


@QuantumGate.register_gate("cx")
def cx_gate(pauli_term: PauliTerm, ctrl: int, tgt: int) -> List[PauliTerm]:
    """
    Implements CNOT gate Pauli propagation: CX† P CX.
    
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
    
    Notes
    -----
    The CNOT (CX) gate propagates Pauli operators according to the following rules:
    - X on control: X_c ⊗ I_t -> X_c ⊗ X_t
    - Z on target: I_c ⊗ Z_t -> Z_c ⊗ Z_t
    - Other Pauli terms are transformed accordingly, with possible phase changes due to anticommutation.
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


@QuantumGate.register_gate("cy")
def cy_gate(pauli_term: PauliTerm, ctrl: int, tgt: int) -> List[PauliTerm]:
    """
    Implement Controlled-Y gate Pauli propagation: U = CY, return U† P U.

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
        Output Pauli terms (always single term for CY)
    """
    coeff = pauli_term.coeff
    key   = pauli_term.key
    n     = pauli_term.n

    # Extract X and Z bits for control and target
    x_c = (key >>  ctrl)     & 1
    z_c = (key >> (n+ctrl))  & 1
    x_t = (key >>  tgt)      & 1
    z_t = (key >> (n+tgt))   & 1

    # Compute phase factor: -1 in exactly two patterns
    minus = (x_c& ((x_t & (1 ^ z_c) & (1 ^ z_t))| (z_t & z_c & (1 ^ x_t))))
    phase = -1 if minus else +1

    # Compute new bits:
    # control-X unchanged
    # control-Z := z_c XOR x_t XOR z_t
    z_c2 = z_c ^ x_t ^ z_t
    # target-X := x_t XOR x_c
    x_t2 = x_t ^ x_c
    # target-Z := z_t XOR x_c
    z_t2 = z_t ^ x_c

    # Update key with new bits
    outk = key
    if z_c2 != z_c:
        outk ^= 1 << (n+ctrl)
    if x_t2 != x_t:
        outk ^= 1 << tgt
    if z_t2 != z_t:
        outk ^= 1 << (n+tgt)

    return [PauliTerm(coeff * phase, outk, n)]


@QuantumGate.register_gate("cz")
def cz_gate(pauli_term: PauliTerm, ctrl: int, tgt: int) -> List[PauliTerm]:
    """
    Controlled-Z gate Pauli propagation: U = CZ, return U† P U.
    
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
        Output Pauli terms (single term with possible phase change)
        
    Notes
    -----
    CZ gate transformation rules:
    - I, Z operators remain unchanged
    - X_ctrl -> X_ctrl Z_tgt  
    - Y_ctrl -> Y_ctrl Z_tgt
    - X_tgt -> Z_ctrl X_tgt
    - Y_tgt -> Z_ctrl Y_tgt
    
    When both qubits have X components, anticommutation produces additional phase.
    """
    coeff = pauli_term.coeff
    key   = pauli_term.key
    n     = pauli_term.n

    x_c = (key >>  ctrl)     & 1
    z_c = (key >> (n+ctrl))  & 1
    x_t = (key >>  tgt)      & 1
    z_t = (key >> (n+tgt))   & 1

    outk = key
    phase = 1
    
    # If control qubit has X component, add Z to target
    if x_c:
        outk ^= 1 << (n+tgt)
        
    # If target qubit has X component, add Z to control  
    if x_t:
        outk ^= 1 << (n+ctrl)
        
    # Handle anticommutation phase
    # When both control and target have X components, additional phase is produced
    if x_c and x_t:
        # Consider anticommutation of Z_ctrl and X_tgt, and X_ctrl and Z_tgt
        # These phases cancel except in special cases
        
        # For XY -> XZ† YZ† = -XY† ZZ† = -XY (since ZZ† = I)
        # For YX -> YZ† XZ† = -YX† ZZ† = -YX  
        if (x_c and z_t and not z_c) or (x_t and z_c and not z_t):
            phase = -1

    return [PauliTerm(coeff * phase, outk, n)]


@QuantumGate.register_gate("ch")
def ch_gate(pauli_term: PauliTerm, ctrl: int, tgt: int) -> List[PauliTerm]:
    """
    Controlled-H gate Pauli propagation using su4_gate.
    
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
        Output Pauli terms from su4_gate transformation
    """
    # CH gate matrix from Qiskit convention
    sqrt2 = 1/np.sqrt(2)
    ch_mat = np.array([[1, 0, 0, 0],
                       [0, sqrt2, 0, sqrt2],
                       [0, 0, 1, 0],
                       [0, sqrt2, 0, -sqrt2]], dtype=complex)
    
    return su4_gate(pauli_term, ctrl, tgt, ch_mat)


@QuantumGate.register_gate("crx")
def crx_gate(pauli_term: PauliTerm, ctrl: int, tgt: int, theta: float) -> List[PauliTerm]:
    """
    Controlled-Rx gate Pauli propagation.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    ctrl : int
        Control qubit index
    tgt : int
        Target qubit index
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms from su4_gate transformation
        
    Notes
    -----
    Implements U = |0⟩⟨0|_ctrl ⊗ I_tgt + |1⟩⟨1|_ctrl ⊗ Rx(θ)_tgt
    Matrix constructed in (tgt, ctrl) order for su4_gate:
    mat = Rx_tgt ⊗ P1_ctrl + I_tgt ⊗ P0_ctrl
    """
    # 2脳2 basis matrices
    I2 = np.eye(2, dtype=complex)
    X  = np.array([[0,1],[1,0]], dtype=complex)
    # Single-qubit Rx
    c, s = math.cos(theta/2), math.sin(theta/2)
    Rx = c*I2 - 1j*s*X

    # Control projectors
    P0 = np.array([[1,0],[0,0]], dtype=complex)
    P1 = np.array([[0,0],[0,1]], dtype=complex)

    # Construct 4脳4 matrix in (tgt,ctrl) subsystem
    mat = np.kron(Rx, P1) + np.kron(I2, P0)
    return su4_gate(pauli_term, ctrl, tgt, mat)

# Register parameter extractor for crx gate
QuantumGate.register_param_extractor("crx", lambda instruction: (instruction.operation.params[0],))


@QuantumGate.register_gate("cry")
def cry_gate(pauli_term: PauliTerm, ctrl: int, tgt: int, theta: float) -> List[PauliTerm]:
    """
    Controlled-Ry gate Pauli propagation.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    ctrl : int
        Control qubit index
    tgt : int
        Target qubit index
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms from su4_gate transformation
        
    Notes
    -----
    Implements U = |0⟩⟨0|_ctrl ⊗ I_tgt + |1⟩⟨1|_ctrl ⊗ Ry(θ)_tgt
    Matrix constructed in (tgt, ctrl) order: mat = Ry_tgt ⊗ P1_ctrl + I_tgt ⊗ P0_ctrl
    """
    I2 = np.eye(2, dtype=complex)
    Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
    c, s = math.cos(theta/2), math.sin(theta/2)
    Ry = c*I2 - 1j*s*Y

    P0 = np.array([[1,0],[0,0]], dtype=complex)
    P1 = np.array([[0,0],[0,1]], dtype=complex)

    mat = np.kron(Ry, P1) + np.kron(I2, P0)
    return su4_gate(pauli_term, ctrl, tgt, mat)

# Register parameter extractor for cry gate
QuantumGate.register_param_extractor("cry", lambda instruction: (instruction.operation.params[0],))


@QuantumGate.register_gate("crz")
def crz_gate(pauli_term: PauliTerm, ctrl: int, tgt: int, theta: float) -> List[PauliTerm]:
    """
    Controlled-Rz gate Pauli propagation.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    ctrl : int
        Control qubit index
    tgt : int
        Target qubit index
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms from su4_gate transformation
        
    Notes
    -----
    Implements U = |0⟩⟨0|_ctrl ⊗ I_tgt + |1⟩⟨1|_ctrl ⊗ Rz(θ)_tgt
    Matrix constructed in (tgt, ctrl) order: mat = Rz_tgt ⊗ P1_ctrl + I_tgt ⊗ P0_ctrl
    """
    I2 = np.eye(2, dtype=complex)
    Z  = np.array([[1,0],[0,-1]], dtype=complex)
    c, s = math.cos(theta/2), math.sin(theta/2)
    Rz = c*I2 - 1j*s*Z

    P0 = np.array([[1,0],[0,0]], dtype=complex)
    P1 = np.array([[0,0],[0,1]], dtype=complex)

    mat = np.kron(Rz, P1) + np.kron(I2, P0)
    return su4_gate(pauli_term, ctrl, tgt, mat)

# Register parameter extractor for crz gate
QuantumGate.register_param_extractor("crz", lambda instruction: (instruction.operation.params[0],))


# Pre-compute 2-qubit Pauli matrices (improves SU4 performance)
_SINGLE_P = (np.eye(2, dtype=complex),
             np.array([[0,1],[1,0]],dtype=complex),
             np.array([[0,-1j],[1j,0]],dtype=complex),
             np.array([[1,0],[0,-1]],dtype=complex))

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


@QuantumGate.register_gate("rxx")
def rxx_gate(pauli_term: PauliTerm, q1: int, q2: int, theta: float) -> List[PauliTerm]:
    """Bit-mask Pauli propagation rule for the two-qubit rotation RXX(θ)=exp(-i θ/2 X⊗X).

    The conjugation formula for any Pauli operator P that *anti-commutes* with X⊗X is
        U† P U = cos(θ)·P + i·sin(θ)·(X⊗X)P .
    If P commutes with X⊗X it is left unchanged.
    This implementation uses only integer bit operations and produces at most two ``PauliTerm`` objects.
    """
    coeff, key, n = pauli_term.coeff, pauli_term.key, pauli_term.n

    # Determine anti-commutation parity: odd number of Z components on the two qubits
    z1 = (key >> (n + q1)) & 1
    z2 = (key >> (n + q2)) & 1
    antic = z1 ^ z2
    if antic == 0:
        return [pauli_term]  # commutes ⇒ unchanged

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    terms: List[PauliTerm] = []
    if abs(cos_t) > 1e-12:
        terms.append(PauliTerm(coeff * cos_t, key, n))

    # --- construct (X⊗X)·P ---
    # Toggle X bits on q1 and q2
    mask_x = (1 << q1) | (1 << q2)
    key_axp = key ^ mask_x

    # Compute the phase arising from multiplication X*P on each qubit
    phase = 1 + 0j
    for q in (q1, q2):
        xp = (key >> q) & 1
        zp = (key >> (n + q)) & 1
        if zp:
            phase *= (1j if xp else -1j)
    coeff2 = coeff * 1j * sin_t * phase
    if abs(coeff2.real) > 1e-12 or abs(coeff2.imag) > 1e-12:
        terms.append(PauliTerm(coeff2, key_axp, n))
    return terms

# Register simple parameter extractors for these rotation gates
QuantumGate.register_param_extractor("rxx", lambda instruction: (instruction.operation.params[0],))


@QuantumGate.register_gate("ryy")
def ryy_gate(pauli_term: PauliTerm, q1: int, q2: int, theta: float) -> List[PauliTerm]:
    """Bit-mask Pauli propagation for RYY(θ)=exp(-i θ/2 Y⊗Y)."""
    coeff, key, n = pauli_term.coeff, pauli_term.key, pauli_term.n

    xp1, zp1 = (key >> q1) & 1, (key >> (n + q1)) & 1
    xp2, zp2 = (key >> q2) & 1, (key >> (n + q2)) & 1
    antic = (xp1 ^ zp1) ^ (xp2 ^ zp2)  # parity of (X⊕Z)
    if antic == 0:
        return [pauli_term]

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    terms: List[PauliTerm] = []
    if abs(cos_t) > 1e-12:
        terms.append(PauliTerm(coeff * cos_t, key, n))

    # Toggle both X and Z bits on the two qubits
    mask_x = (1 << q1) | (1 << q2)
    mask_z = (1 << (n + q1)) | (1 << (n + q2))
    key_axp = key ^ mask_x ^ mask_z

    phase = 1 + 0j
    for xp, zp in ((xp1, zp1), (xp2, zp2)):
        if xp == zp:
            continue  # I or Y ⇒ phase 1
        phase *= (-1j if xp else 1j)  # X ⇒ -i , Z ⇒ +i

    coeff2 = coeff * 1j * sin_t * phase
    if abs(coeff2.real) > 1e-12 or abs(coeff2.imag) > 1e-12:
        terms.append(PauliTerm(coeff2, key_axp, n))
    return terms

# Register simple parameter extractor for this rotation gate
QuantumGate.register_param_extractor("ryy", lambda instruction: (instruction.operation.params[0],))


@QuantumGate.register_gate("rzz")
def rzz_gate(pauli_term: PauliTerm, q1: int, q2: int, theta: float) -> List[PauliTerm]:
    """Bit-mask Pauli propagation for RZZ(θ)=exp(-i θ/2 Z⊗Z)."""
    coeff, key, n = pauli_term.coeff, pauli_term.key, pauli_term.n

    x1 = (key >> q1) & 1
    x2 = (key >> q2) & 1
    antic = x1 ^ x2  # anticommute if odd number of X components
    if antic == 0:
        return [pauli_term]

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    terms: List[PauliTerm] = []
    if abs(cos_t) > 1e-12:
        terms.append(PauliTerm(coeff * cos_t, key, n))

    # Toggle Z bits on the two qubits
    mask_z = (1 << (n + q1)) | (1 << (n + q2))
    key_axp = key ^ mask_z

    phase = 1 + 0j
    for xp, zp in ((x1, (key >> (n + q1)) & 1), (x2, (key >> (n + q2)) & 1)):
        if xp == 0:
            continue  # I or Z ⇒ phase 1
        phase *= (1j if zp == 0 else -1j)  # X ⇒ +i , Y ⇒ -i

    coeff2 = coeff * 1j * sin_t * phase
    if abs(coeff2.real) > 1e-12 or abs(coeff2.imag) > 1e-12:
        terms.append(PauliTerm(coeff2, key_axp, n))
    return terms

# Register simple parameter extractor for this rotation gate
QuantumGate.register_param_extractor("rzz", lambda instruction: (instruction.operation.params[0],))


@QuantumGate.register_gate("su4")
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
    c_arr = coeff * coeffs
    mask = (np.abs(c_arr.real) + np.abs(c_arr.imag)) >= 1e-12
    significant_idxs = np.nonzero(mask)[0]

    result = []
    for alpha in significant_idxs:
        c = c_arr[alpha]
        code2a, code1a = divmod(alpha, 4)
        new_key = key
        
        # apply bit flips for q1
        z1_new, x1_new = _bits_from_code(code1a)
        if ((new_key >> q1) & 1) != x1_new:
            new_key ^= 1 << q1
        if ((new_key >> (n+q1)) & 1) != z1_new:
            new_key ^= 1 << (n+q1)
        
        # apply bit flips for q2
        z2_new, x2_new = _bits_from_code(code2a)
        if ((new_key >> q2) & 1) != x2_new:
            new_key ^= 1 << q2
        if ((new_key >> (n+q2)) & 1) != z2_new:
            new_key ^= 1 << (n+q2)

        # Clean small imaginary parts
        if abs(c.imag) < 1e-12:
            c = c.real + 0j
        result.append(PauliTerm(c, new_key, n))

    return result

# Register parameter extractor for su4 gate
QuantumGate.register_param_extractor("su4", lambda instruction: (instruction.operation.to_matrix(),))

@QuantumGate.register_gate("swap")
def swap_gate(pauli_term: PauliTerm, q1: int, q2: int) -> List[PauliTerm]:
    """
    SWAP gate Pauli propagation (pure bit operations, no phase).
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    q1 : int
        First qubit index
    q2 : int
        Second qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms (single term with swapped qubit operators)
        
    Notes
    -----
    Implements U = SWAP(q1,q2), returns U† P U
    """
    coeff, key, n = pauli_term.coeff, pauli_term.key, pauli_term.n
    if q1 == q2:
        return [pauli_term]

    # Extract X/Z bits
    x1, x2 = (key >> q1) & 1, (key >> q2) & 1
    z1, z2 = (key >> (n+q1)) & 1, (key >> (n+q2)) & 1

    new_key = key
    # Swap X bits
    if x1 != x2:
        new_key ^= (1 << q1) | (1 << q2)
    # Swap Z bits
    if z1 != z2:
        new_key ^= (1 << (n+q1)) | (1 << (n+q2))

    return [PauliTerm(coeff, new_key, n)]


@QuantumGate.register_gate("iswap")
def iswap_gate(pauli_term: PauliTerm, q1: int, q2: int) -> List[PauliTerm]:
    """
    iSWAP gate Pauli propagation using su4_gate.
    """
    iswap_mat = np.array([[1, 0, 0, 0],
                          [0, 0, 1j, 0],
                          [0, 1j, 0, 0],
                          [0, 0, 0, 1]],dtype=complex)
        
    return su4_gate(pauli_term, q1, q2, iswap_mat)


# ------------------------------------------------------------------------------------------------------------------------
# Three-qubit gates
# ------------------------------------------------------------------------------------------------------------------------


@QuantumGate.register_gate("ccx")
def ccx_gate(pauli_term: PauliTerm,
             ctrl1: int,
             ctrl2: int,
             tgt:   int) -> List[PauliTerm]:
    """
    Toffoli (CCX) gate Pauli propagation: U = CCX, return U† P U.
    
    Parameters
    ----------
    pauli_term : PauliTerm
        Input Pauli term
    ctrl1 : int
        First control qubit index
    ctrl2 : int
        Second control qubit index
    tgt : int
        Target qubit index
        
    Returns
    -------
    List[PauliTerm]
        Output Pauli terms from gate decomposition
        
    Notes
    -----
    Uses classical decomposition from Blueqat:
        h tgt;
        cx ctrl2, tgt; tdg tgt;
        cx ctrl1, tgt; t tgt;
        cx ctrl2, tgt; tdg tgt;
        cx ctrl1, tgt;
        t ctrl2; t tgt;
        h tgt;
        cx ctrl1, ctrl2; t ctrl1; tdg ctrl2; cx ctrl1, ctrl2;
    """
    
    # Initialize with single term
    terms: List[PauliTerm] = [pauli_term]

    # Forward operation sequence (in execution order)
    ops = [lambda t: h_gate(t, tgt),
            lambda t: cx_gate(t, ctrl2, tgt),
            lambda t: rz_gate(t, tgt, -math.pi/4),  # tdg tgt
            lambda t: cx_gate(t, ctrl1, tgt),
            lambda t: rz_gate(t, tgt,  math.pi/4),  # t tgt
            lambda t: cx_gate(t, ctrl2, tgt),
            lambda t: rz_gate(t, tgt, -math.pi/4),  # tdg tgt
            lambda t: cx_gate(t, ctrl1, tgt),
            lambda t: rz_gate(t, ctrl2,  math.pi/4),# t ctrl2
            lambda t: rz_gate(t, tgt,   math.pi/4),# t tgt
            lambda t: h_gate(t, tgt),
            lambda t: cx_gate(t, ctrl1, ctrl2),
            lambda t: rz_gate(t, ctrl1, math.pi/4),# t ctrl1
            lambda t: rz_gate(t, ctrl2,-math.pi/4),# tdg ctrl2
            lambda t: cx_gate(t, ctrl1, ctrl2)]

    # Backward propagation: apply each step's U† P U in reverse order
    for op in reversed(ops):
        next_terms: List[PauliTerm] = []
        for trm in terms:
            next_terms.extend(op(trm))
        terms = next_terms

    return terms

