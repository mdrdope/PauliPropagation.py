

# pauli_pkg/pauli_propagation/utils.py

# Utility helpers that are not part of the core propagation logic.

from __future__ import annotations
import numpy as np

__all__ = ["random_su4"]

def random_su4() -> np.ndarray:
    """Return a Haar-random 4×4 special-unitary (det = 1)."""
    z = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
    q, r = np.linalg.qr(z)                       # QR → unitary up to phase
    ph = np.diag(r) / np.abs(np.diag(r))         # normalise columns
    q = q * ph
    q /= np.linalg.det(q) ** 0.25                # enforce det = 1
    return q

