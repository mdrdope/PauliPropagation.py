# -*- coding: utf-8 -*-

import random
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Pauli, Statevector

from pauli_propagation import PauliTerm, PauliPropagator
from pauli_propagation.utils import encode_pauli, random_su4, random_su2, random_state_label, random_pauli_label

SYMS_STATE = "01+-rl"
SYMS_PAULI = "IXYZ"

def _add_random_su4(qc, ctrl, tgt):
    """Add a random SU(4) gate to the circuit on qubits ctrl and tgt."""
    U = random_su4()
    gate = UnitaryGate(U, label="randSU4")
    gate._name = "su4"
    qc.append(gate, [ctrl, tgt])

def _add_random_su2(qc, q):
    """Add a random SU(2) gate to the circuit on qubit q."""
    U = random_su2()
    gate = UnitaryGate(U, label="randSU2")
    gate._name = "su2"
    qc.append(gate, [q])

def _add_random_rotation(qc, q, gate_type):
    """Add a random rotation gate to the circuit."""
    theta = random.uniform(0, 2*np.pi)
    if gate_type == "rx":
        qc.rx(theta, q)
    elif gate_type == "ry":
        qc.ry(theta, q)
    elif gate_type == "rz":
        qc.rz(theta, q)

def _add_random_two_qubit_rotation(qc, q1, q2, gate_type):
    """Add a random two-qubit rotation gate to the circuit."""
    theta = random.uniform(0, 2*np.pi)
    if gate_type == "rxx":
        qc.rxx(theta, q1, q2)
    elif gate_type == "ryy":
        qc.ryy(theta, q1, q2)
    elif gate_type == "rzz":
        qc.rzz(theta, q1, q2)

def _add_random_controlled_rotation(qc, ctrl, tgt, gate_type):
    """Add a random controlled rotation gate to the circuit."""
    theta = random.uniform(0, 2*np.pi)
    if gate_type == "crx":
        qc.crx(theta, ctrl, tgt)
    elif gate_type == "cry":
        qc.cry(theta, ctrl, tgt)
    elif gate_type == "crz":
        qc.crz(theta, ctrl, tgt)

def build_random_circuit(n, gate_count):
    """Build a random circuit on n qubits with gate_count random gates."""
    qc = QuantumCircuit(n, name=f"rand_{n}q_{gate_count}g")
    
    # Define gate operations
    single_qubit_gates = [lambda q: qc.x(q),
                          lambda q: qc.y(q), 
                          lambda q: qc.z(q),
                          lambda q: qc.t(q), 
                          lambda q: qc.h(q),
                          lambda q: qc.s(q),
                          lambda q: qc.sdg(q),
                          lambda q: qc.sx(q),
                          lambda q: qc.sxdg(q),
                          lambda q: _add_random_rotation(qc, q, "rx"),
                          lambda q: _add_random_rotation(qc, q, "ry"),
                          lambda q: _add_random_rotation(qc, q, "rz"),
                          lambda q: _add_random_su2(qc, q)]
    
    two_qubit_gates = [lambda ctrl, tgt: qc.cx(ctrl, tgt),
                       lambda ctrl, tgt: qc.cz(ctrl, tgt),
                       lambda ctrl, tgt: qc.cy(ctrl, tgt),
                       lambda ctrl, tgt: qc.ch(ctrl, tgt),
                       lambda ctrl, tgt: qc.swap(ctrl, tgt),
                       lambda ctrl, tgt: qc.iswap(ctrl, tgt),
                       lambda ctrl, tgt: _add_random_su4(qc, ctrl, tgt),
                       lambda ctrl, tgt: _add_random_two_qubit_rotation(qc, ctrl, tgt, "rxx"),
                       lambda ctrl, tgt: _add_random_two_qubit_rotation(qc, ctrl, tgt, "ryy"),
                       lambda ctrl, tgt: _add_random_two_qubit_rotation(qc, ctrl, tgt, "rzz"),
                       lambda ctrl, tgt: _add_random_controlled_rotation(qc, ctrl, tgt, "crx"),
                       lambda ctrl, tgt: _add_random_controlled_rotation(qc, ctrl, tgt, "cry"),
                       lambda ctrl, tgt: _add_random_controlled_rotation(qc, ctrl, tgt, "crz")]
    
    three_qubit_gates = [lambda ctrl1, ctrl2, tgt: qc.ccx(ctrl1, ctrl2, tgt)]
    
    for _ in range(gate_count):
        # Choose between single-qubit, two-qubit, and three-qubit gates
        gate_type = random.choices([1, 2, 3], weights=[0.4, 0.5, 0.1])[0]  # 40% single, 50% two, 10% three
        
        if gate_type == 1:  # single-qubit gate
            gate_func = random.choice(single_qubit_gates)
            qubit = random.randrange(n)
            gate_func(qubit)
        elif gate_type == 2:  # two-qubit gate
            gate_func = random.choice(two_qubit_gates)
            ctrl, tgt = random.sample(range(n), 2)
            gate_func(ctrl, tgt)
        else:  # three-qubit gate (only if n >= 3)
            if n >= 3:
                gate_func = random.choice(three_qubit_gates)
                ctrl1, ctrl2, tgt = random.sample(range(n), 3)
                gate_func(ctrl1, ctrl2, tgt)
            else:
                # Fall back to two-qubit gate if n < 3
                gate_func = random.choice(two_qubit_gates)
                ctrl, tgt = random.sample(range(n), 2)
                gate_func(ctrl, tgt)
    
    return qc

@pytest.mark.parametrize("trial", range(200))
def test_random_consistency(trial):
    """
    Compare PauliPropagator expectation vs full statevector expectation
    over a random small circuit and random product state + Pauli.
    """
    n = random.randint(3, 12)
    gate_count = random.randint(3, 12)
    qc = build_random_circuit(n, gate_count)

    state_lbl = random_state_label(n)
    pauli_lbl = random_pauli_label(n)
    key = encode_pauli(Pauli(pauli_lbl))
    obs = PauliTerm(1.0, key, n)

    prop = PauliPropagator(qc)
    layers = prop.propagate(obs, max_weight=None, use_parallel=False)
    prop_ev = prop.expectation_pauli_sum(layers[-1], state_lbl)

    sv_ev = Statevector.from_label(state_lbl).evolve(qc).expectation_value(Pauli(pauli_lbl)).real
    assert abs(prop_ev - sv_ev) < 1e-10, (f"Trial {trial}: discrepancy {prop_ev} vs {sv_ev} "
                                          f"on state {state_lbl}, Pauli {pauli_lbl}")
