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

__all__ = [
    "staircasetopology2d_qc",
    "get_staircase_edges",
    "print_staircase_info",
    "ibm_eagle_topology",
    "tfi_trotter_circuit",
]

# IBM Eagle topology - 127 qubits connectivity
ibm_eagle_topology = [
    (1, 2), (1, 15), (2, 3), (3, 4), (4, 5), (5, 6), (5, 16), (6, 7), (7, 8), (8, 9), (9, 10), (9, 17),
    (10, 11), (11, 12), (12, 13), (13, 14), (13, 18), (15, 19), (16, 23), (17, 27), (18, 31), (19, 20),
    (20, 21), (21, 22), (21, 34), (22, 23), (23, 24), (24, 25), (25, 26), (25, 35), (26, 27), (27, 28),
    (28, 29), (29, 30), (29, 36), (30, 31), (31, 32), (32, 33), (33, 37), (34, 40), (35, 44), (36, 48),
    (37, 52), (38, 39), (38, 53), (39, 40), (40, 41), (41, 42), (42, 43), (42, 54), (43, 44), (44, 45),
    (45, 46), (46, 47), (46, 55), (47, 48), (48, 49), (49, 50), (50, 51), (50, 56), (51, 52), (53, 57),
    (54, 61), (55, 65), (56, 69), (57, 58), (58, 59), (59, 60), (59, 72), (60, 61), (61, 62), (62, 63),
    (63, 64), (63, 73), (64, 65), (65, 66), (66, 67), (67, 68), (67, 74), (68, 69), (69, 70), (70, 71),
    (71, 75), (72, 78), (73, 82), (74, 86), (75, 90), (76, 77), (76, 91), (77, 78), (78, 79), (79, 80),
    (80, 81), (80, 92), (81, 82), (82, 83), (83, 84), (84, 85), (84, 93), (85, 86), (86, 87), (87, 88),
    (88, 89), (88, 94), (89, 90), (91, 95), (92, 99), (93, 103), (94, 107), (95, 96), (96, 97), (97, 98),
    (97, 110), (98, 99), (99, 100), (100, 101), (101, 102), (101, 111), (102, 103), (103, 104), (104, 105),
    (105, 106), (105, 112), (106, 107), (107, 108), (108, 109), (109, 113), (110, 115), (111, 119), (112, 123),
    (113, 127), (114, 115), (115, 116), (116, 117), (117, 118), (118, 119), (119, 120), (120, 121), (121, 122),
    (122, 123), (123, 124), (124, 125), (125, 126), (126, 127)
]


def tfi_trotter_circuit(nqubits, nlayers, topology=None, start_with_ZZ=True, rx_theta=None, rzz_theta=None):
    """
    Build a TFI (Transverse Field Ising) Trotter circuit.
    
    This function creates a quantum circuit that simulates the time evolution
    under the Transverse Field Ising Hamiltonian using Trotter decomposition.
    
    Parameters
    ----------
    nqubits : int
        Number of qubits in the circuit
    nlayers : int
        Number of Trotter layers
    topology : List[Tuple[int, int]], optional
        List of 1-based qubit pairs for ZZ interactions.
        If None, uses IBM Eagle topology
    start_with_ZZ : bool, optional
        If True, starts with RZZ layer. If False, starts with RX layer.
        Default is True
    rx_theta : float, optional
        Rotation angle for RX gates. If None, uses ��/2
    rzz_theta : float, optional
        Rotation angle for RZZ gates. If None, uses ��/4
        
    Returns
    -------
    QuantumCircuit
        TFI Trotter quantum circuit
        
    Examples
    --------
    >>> # Create a TFI circuit with IBM Eagle topology (default)
    >>> nq = 127  # IBM Eagle has 127 qubits
    >>> nl = 5    # 5 layers
    >>> circuit = tfi_trotter_circuit(nq, nl)
    
    >>> # Create a TFI circuit with custom topology
    >>> custom_topology = [(1, 2), (2, 3), (3, 4)]
    >>> circuit = tfi_trotter_circuit(4, 3, topology=custom_topology)
    """
    if topology is None:
        topology = ibm_eagle_topology
    
    if rx_theta is None:
        rx_theta = np.pi / 2
        
    if rzz_theta is None:
        rzz_theta =   np.pi / 4
    
    qc = QuantumCircuit(nqubits)
    
    # If nlayers is 0, return empty circuit
    if nlayers == 0:
        return qc
    
    # First layer
    if start_with_ZZ:
        # Add RZZ layer according to topology
        for q1, q2 in topology:
            # Convert from 1-based to 0-based indexing
            qc.rzz(rzz_theta, q1 - 1, q2 - 1)
    
    # Middle layers
    for _ in range(nlayers - 1):
        # Add RX layer to all qubits
        for qubit in range(nqubits):
            qc.rx(rx_theta, qubit)
        
        # Add RZZ layer according to topology
        for q1, q2 in topology:
            # Convert from 1-based to 0-based indexing
            qc.rzz(rzz_theta, q1 - 1, q2 - 1)
    
    # Final RX layer
    for qubit in range(nqubits):
        qc.rx(rx_theta, qubit)
    
    # Final RZZ layer if we didn't start with ZZ
    if not start_with_ZZ:
        # Add RZZ layer according to topology
        for q1, q2 in topology:
            # Convert from 1-based to 0-based indexing
            qc.rzz(rzz_theta, q1 - 1, q2 - 1)
    
    return qc

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


def staircasetopology2d_qc(nx, ny, L=1, seed=42):
    """
    Build a QuantumCircuit that places a fresh SU(4) gate on every edge
    of the 2-D staircase topology for an nx x ny grid (row-major indexing).
    
    Parameters
    ----------
    nx : int
        Number of columns in the grid
    ny : int
        Number of rows in the grid
    L : int, optional
        Number of layers (depth) of the circuit, default is 1
    seed : int, optional
        Random seed for reproducible gate generation
        
    Returns
    -------
    QuantumCircuit
        A quantum circuit with SU(4) gates arranged in staircase topology
        
    Examples
    --------
    >>> # Create a 3x3 grid with 2 layers
    >>> qc = staircasetopology2d_qc(3, 3, L=2)
    >>> print(f"Circuit has {qc.num_qubits} qubits and {len(qc.data)} gates")
    
    >>> # Create a reproducible circuit
    >>> qc1 = staircasetopology2d_qc(2, 2, seed=42)
    >>> qc2 = staircasetopology2d_qc(2, 2, seed=42)
    >>> # qc1 and qc2 will have identical gates
    """
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        
    nqubits = nx * ny
    qc = QuantumCircuit(nqubits)
    for layer in range(L):
        for k, (q1, q2) in enumerate(_staircase_edges(nx, ny)):  # 1-based
            mat = random_su4()
            gate = UnitaryGate(mat, label=f"SU4")   # create the gate
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
