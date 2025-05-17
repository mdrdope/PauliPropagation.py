# pauli_pkg/tests/test_propagator_random.py

# Monte-Carlo sanity check:  PauliPropagator vs full-statevector ?P?.

import random, numpy as np, pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from pauli_propagation import PauliTerm, PauliPropagator

SYMS_STATE = "01+-rl"
SYMS_PAULI = "IXYZ"

def random_state_label(n):
    return "".join(random.choice(SYMS_STATE) for _ in range(n))

def random_pauli_label(n):
    lbl = "".join(random.choice(SYMS_PAULI) for _ in range(n))
    if set(lbl) == {"I"}:
        pos = random.randrange(n)
        lbl = lbl[:pos] + random.choice("XYZ") + lbl[pos+1:]
    return lbl

def build_random_circuit(n, gate_count):
    qc = QuantumCircuit(n, name=f"rand_{n}q_{gate_count}g")
    for _ in range(gate_count):
        if random.choice(("t","cx")) == "t":
            qc.t(random.randrange(n))
        else:
            ctrl, tgt = random.sample(range(n), 2)
            qc.cx(ctrl, tgt)
    return qc

@pytest.mark.parametrize("trial", range(200))
def test_random_consistency(trial):
    n          = random.randint(3, 10)
    gate_count = random.randint(3, 12)
    qc         = build_random_circuit(n, gate_count)

    state_lbl  = random_state_label(n)
    pauli_lbl  = random_pauli_label(n)
    obs        = Pauli(pauli_lbl)

    prop = PauliPropagator(qc)
    layers = prop.propagate(PauliTerm(1.0, obs))
    prop_ev = prop.expectation_pauli_sum(layers[-1], state_lbl)

    sv_ev = Statevector.from_label(state_lbl).evolve(qc).expectation_value(obs).real
    assert abs(prop_ev - sv_ev) < 1e-10
