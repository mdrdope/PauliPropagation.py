import random
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from pauli_propagation import PauliTerm, PauliPropagator
from pauli_propagation.utils import encode_pauli

SYMS_STATE = "01+-rl"
SYMS_PAULI = "IXYZ"

def random_state_label(n):
    """Generate a random product‐state label of length n."""
    return "".join(random.choice(SYMS_STATE) for _ in range(n))

def random_pauli_label(n):
    """Generate a random non‐identity Pauli label of length n."""
    lbl = "".join(random.choice(SYMS_PAULI) for _ in range(n))
    if set(lbl) == {"I"}:
        pos = random.randrange(n)
        lbl = lbl[:pos] + random.choice("XYZ") + lbl[pos+1:]
    return lbl

def build_random_circuit(n, gate_count):
    """Build a random circuit on n qubits with gate_count random T/CX gates."""
    qc = QuantumCircuit(n, name=f"rand_{n}q_{gate_count}g")
    for _ in range(gate_count):
        if random.choice(("t", "cx")) == "t":
            qc.t(random.randrange(n))
        else:
            ctrl, tgt = random.sample(range(n), 2)
            qc.cx(ctrl, tgt)
    return qc

@pytest.mark.parametrize("trial", range(50))
def test_random_consistency(trial):
    """
    Compare PauliPropagator expectation vs full statevector expectation
    over a random small circuit and random product state + Pauli.
    """
    n = random.randint(3, 10)
    gate_count = random.randint(3, 12)
    qc = build_random_circuit(n, gate_count)

    state_lbl = random_state_label(n)
    pauli_lbl = random_pauli_label(n)
    key = encode_pauli(Pauli(pauli_lbl))
    obs = PauliTerm(1.0, key, n)

    prop = PauliPropagator(qc)
    layers = prop.propagate(obs)
    prop_ev = prop.expectation_pauli_sum(layers[-1], state_lbl)

    sv_ev = Statevector.from_label(state_lbl).evolve(qc).expectation_value(Pauli(pauli_lbl)).real
    assert abs(prop_ev - sv_ev) < 1e-10, (
        f"Trial {trial}: discrepancy {prop_ev} vs {sv_ev} "
        f"on state {state_lbl}, Pauli {pauli_lbl}"
    )
