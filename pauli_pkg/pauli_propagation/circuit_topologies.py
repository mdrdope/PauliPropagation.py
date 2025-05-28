# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/circuit_topologies.py
"""
Quantum Circuit Topologies

This module provides functions to generate quantum circuits with specific topologies.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from .utils import random_su4


def _staircase_edges(nx, ny):
    """
    Return the ordered list of 1-based index pairs of the staircase walk.
    
    Parameters
    ----------
    nx : int
        Number of columns in the grid
    ny : int
        Number of rows in the grid
        
    Returns
    -------
    List[Tuple[int, int]]
        List of 1-based index pairs representing edges in the staircase topology
    """
    next_inds, temp_inds, edges = [1], [], []
    while next_inds:
        for ind in next_inds:
            if ind % nx != 0:                      # step right
                nxt = ind + 1
                edges.append((ind, nxt)); temp_inds.append(nxt)
            if ((ind-1)//nx + 1) < ny:             # step down
                nxt = ind + nx
                edges.append((ind, nxt)); temp_inds.append(nxt)
        next_inds, temp_inds = temp_inds, []
    seen, uniq = set(), []
    for e in edges:                                # preserve order, dedup
        if e not in seen:
            seen.add(e); uniq.append(e)
    return uniq


def staircasetopology2d_qc(nx, ny, L):
    """
    Build a QuantumCircuit that places a fresh SU(4) gate on every edge
    of the 2-D staircase topology for an nx × ny grid (row-major indexing).
    
    Parameters
    ----------
    nx : int
        Number of columns in the grid
    ny : int
        Number of rows in the grid
    L : int
        Number of layers (depth) of the circuit
        
    Returns
    -------
    QuantumCircuit
        A quantum circuit with SU(4) gates arranged in staircase topology
        
    Examples
    --------
    >>> # Create a 3x3 grid with 2 layers
    >>> qc = staircasetopology2d_qc(3, 3, 2)
    >>> print(f"Circuit has {qc.num_qubits} qubits and {len(qc.data)} gates")
    """
    nqubits = nx * ny
    qc = QuantumCircuit(nqubits)
    for _ in range(L):
        for k, (q1, q2) in enumerate(_staircase_edges(nx, ny)):  # 1-based
            mat = random_su4()
            gate = UnitaryGate(mat, label=f"SU4_{k}")   # create the gate
            gate._name = "su4"                          # override its name
            qc.append(gate, [q1-1, q2-1])               # 0-based
    return qc


def get_staircase_edges(nx, ny):
    """
    Get the edges of a staircase topology for visualization or analysis.
    
    Parameters
    ----------
    nx : int
        Number of columns in the grid
    ny : int
        Number of rows in the grid
        
    Returns
    -------
    List[Tuple[int, int]]
        List of 1-based index pairs representing edges
    """
    return _staircase_edges(nx, ny)


def print_staircase_info(nx, ny):
    """
    Print information about the staircase topology.
    
    Parameters
    ----------
    nx : int
        Number of columns in the grid
    ny : int
        Number of rows in the grid
    """
    edges = _staircase_edges(nx, ny)
    nqubits = nx * ny
    
    print(f"Staircase topology for {nx}×{ny} grid:")
    print(f"  Total qubits: {nqubits}")
    print(f"  Total edges: {len(edges)}")
    print(f"  Edges (1-based indexing): {edges}")
    
    # Convert to 0-based for clarity
    edges_0based = [(q1-1, q2-1) for q1, q2 in edges]
    print(f"  Edges (0-based indexing): {edges_0based}")
