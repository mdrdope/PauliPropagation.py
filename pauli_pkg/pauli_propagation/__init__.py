# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/__init__.py
"""
Pauli Propagation Package

This package provides tools for Pauli-based back-propagation of observables through noiseless quantum circuits.
"""

from .pauli_term   import PauliTerm
from .gates        import QuantumGate
from .propagator   import PauliPropagator
from .utils        import (
    encode_pauli,
    decode_pauli,
    weight_of_key,
    random_su4,
)

__all__ = [
    "PauliTerm",
    "QuantumGate",
    "PauliPropagator",
    "encode_pauli",
    "decode_pauli",
    "weight_of_key",
    "random_su4",
]

# Version
__version__ = "0.1.0"
