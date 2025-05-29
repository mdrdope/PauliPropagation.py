# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/decomposition.py
"""
Quantum gate decomposition utilities for Pauli propagation.

This module provides KAK decomposition for SU(4) gates and utilities to replace
SU(4) gates in quantum circuits with their decomposed forms.
"""

import numpy as np
from qiskit.synthesis import TwoQubitWeylDecomposition, OneQubitEulerDecomposer
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, RXXGate, RYYGate, RZZGate, RZGate, RYGate
from typing import Tuple, List
import warnings
from .utils import random_su4
import scipy.linalg

__all__ = [
    "su4_kak_decomp",
    "su4_kak_reconstruct",
]


def su2_to_euler(U: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a 2x2 SU(2) matrix to Euler angles (ZYZ decomposition).
    
    Decomposes U = Rz(alpha) * Ry(beta) * Rz(gamma)
    
    Parameters
    ----------
    U : np.ndarray
        2x2 SU(2) matrix
        
    Returns
    -------
    Tuple[float, float, float]
        (alpha, beta, gamma) Euler angles
    """
    # Extract angles from SU(2) matrix
    # U = [[cos(beta/2)*exp(i*(alpha+gamma)/2), -sin(beta/2)*exp(i*(alpha-gamma)/2)],
    #      [sin(beta/2)*exp(-i*(alpha-gamma)/2), cos(beta/2)*exp(-i*(alpha+gamma)/2)]]
    
    # Get beta from the magnitude of off-diagonal elements
    beta = 2 * np.arctan2(np.abs(U[1, 0]), np.abs(U[0, 0]))
    
    if np.abs(np.sin(beta/2)) < 1e-10:
        # Special case: beta ¡Ö 0, U ¡Ö exp(i*alpha)*I
        alpha = np.angle(U[0, 0]) * 2
        gamma = 0
    else:
        # General case
        alpha = np.angle(U[0, 0]) + np.angle(U[1, 1])
        gamma = np.angle(U[0, 0]) - np.angle(U[1, 1])
    
    return alpha, beta, gamma


def su4_kak_decomp(U: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                                         Tuple[float, float, float], float]:
    """
    Perform KAK decomposition of a 4x4 SU(4) matrix with proper phase handling.
    
    This function decomposes a 4x4 unitary matrix U into the KAK form:
    U = exp(i*phi) * (K1l ? K1r) @ exp(i*a*XX + i*b*YY + i*c*ZZ) @ (K2l ? K2r)
    
    where K1l, K1r, K2l, K2r are 2x2 SU(2) matrices and a, b, c are Weyl coordinates.
    
    Parameters
    ----------
    U : np.ndarray
        4x4 complex unitary matrix to decompose
        
    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[float, float, float], float]
        A tuple containing:
        - (K1l, K1r, K2l, K2r): Four 2x2 SU(2) matrices
        - (a, b, c): Weyl coordinates
        - phi: Global phase
        
    Notes
    -----
    Uses Qiskit's TwoQubitWeylDecomposition which gives:
    U = exp(i*phi) * (K1l ? K1r) @ exp(i*a*XX + i*b*YY + i*c*ZZ) @ (K2l ? K2r)
    """
    # Perform Weyl (KAK) decomposition using Qiskit
    decomp = TwoQubitWeylDecomposition(U)
    
    # Get decomposition components
    K1l, K1r, K2l, K2r = decomp.K1l, decomp.K1r, decomp.K2l, decomp.K2r
    a, b, c = decomp.a, decomp.b, decomp.c
    phi = decomp.global_phase

    # Verify reconstruction with Qiskit's form
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Qiskit uses exp(i*theta*Pauli) form
    XX = np.kron(sigma_x, sigma_x)
    YY = np.kron(sigma_y, sigma_y)
    ZZ = np.kron(sigma_z, sigma_z)
    
    # Compute exp(i*a*XX + i*b*YY + i*c*ZZ)
    exponent = 1j * (a * XX + b * YY + c * ZZ)
    exp_term = scipy.linalg.expm(exponent)
    
    # Verify reconstruction: U = exp(i*phi) * (K1l ? K1r) @ exp_term @ (K2l ? K2r)
    U_rec = np.exp(1j * phi) * np.kron(K1l, K1r) @ exp_term @ np.kron(K2l, K2r)
    
    if not np.allclose(U_rec, U, atol=1e-10):
        # If reconstruction fails, give warning
        warnings.warn(f"KAK reconstruction error: {np.max(np.abs(U_rec - U)):.2e}")

    return (K1l, K1r, K2l, K2r), (a, b, c), phi


def su4_kak_reconstruct(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Replace all SU(4) gates in a quantum circuit with their KAK decomposition.
    
    This function takes a quantum circuit containing SU(4) gates (registered as "su4" 
    gates in the QuantumGate registry) and replaces each SU(4) gate with its KAK 
    decomposition consisting of single-qubit rotations and two-qubit rotations (Rxx, Ryy, Rzz).
    
    Parameters
    ----------
    qc : QuantumCircuit
        Input quantum circuit that may contain SU(4) gates
        
    Returns
    -------
    QuantumCircuit
        New quantum circuit with SU(4) gates replaced by their KAK decomposition
        
    Notes
    -----
    - Only processes gates with name "su4" acting on exactly 2 qubits
    - Skips gates that are not named "su4"
    - Each SU(4) gate is replaced with Qiskit's own KAK decomposition circuit
    - Uses qiskit.synthesis.TwoQubitWeylDecomposition.circuit() for implementation
    """
    # Create new quantum circuit
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    
    # Iterate through all instructions in the original circuit
    for instruction in qc.data:
        gate = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        
        # Check if this is an "su4" gate acting on exactly 2 qubits
        if (hasattr(gate, 'name') and gate.name == "su4" and len(qubits) == 2):
            
            # Extract SU(4) matrix and perform KAK decomposition
            U = gate.to_matrix()
            decomp = TwoQubitWeylDecomposition(U)
            
            # Get Qiskit's own circuit decomposition
            kak_circuit = decomp.circuit()
            
            q1, q2 = qubits[0], qubits[1]
            
            # Add the KAK circuit to our new circuit
            # Map the decomposition circuit qubits to our target qubits
            for instr in kak_circuit.data:
                gate_op = instr.operation
                gate_qubits = instr.qubits
                
                # Map qubits: decomposition circuit uses qubits [0, 1], we need [q1, q2]
                mapped_qubits = []
                for qubit in gate_qubits:
                    qubit_index = kak_circuit.find_bit(qubit)[0]
                    if qubit_index == 0:
                        mapped_qubits.append(q1)
                    elif qubit_index == 1:
                        mapped_qubits.append(q2)
                    else:
                        raise ValueError(f"Unexpected qubit index {qubit_index}")
                
                new_qc.append(gate_op, mapped_qubits)
            
            # Add global phase
            new_qc.global_phase += kak_circuit.global_phase
        else:
            # For non-su4 gates, copy directly
            new_qc.append(gate, qubits, clbits)
    
    return new_qc 