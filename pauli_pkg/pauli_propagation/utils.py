from __future__ import annotations
import numpy as np
from numba import njit

__all__ = [
    "encode_pauli",
    "decode_pauli",
    "weight_of_key",
    "random_su4",
]

# ─── Z‖X  <->  int ──────────────────────────────────────────────
def encode_pauli(p: np.typing.NDArray | "Pauli") -> int:
    """Pack a q-qubit qiskit.Pauli into one 2q-bit integer key = (Z‖X)."""
    n = len(p.z)
    x_bits = 0
    z_bits = 0
    for i in range(n):
        if p.x[i]:
            x_bits |= 1 << i
        if p.z[i]:
            z_bits |= 1 << i
    return (z_bits << n) | x_bits

def decode_pauli(key: int, n: int) -> "Pauli":
    """Unpack integer key back to qiskit.Pauli."""
    from qiskit.quantum_info import Pauli  # 延迟导入避免循环依赖
    mask   = (1 << n) - 1
    x_bits =  key        & mask
    z_bits = (key >> n)  & mask
    x = np.array([(x_bits >> i) & 1 for i in range(n)], dtype=bool)
    z = np.array([(z_bits >> i) & 1 for i in range(n)], dtype=bool)
    return Pauli((z, x))

# ─── weight (popcount of Z∨X) ───────────────────────────────────
@njit(cache=True, fastmath=True)
def _weight_of_key_nb(key: int, n: int) -> int:
    mask = (1 << n) - 1
    bits = (key & mask) | (key >> n)
    cnt  = 0
    while bits:
        bits &= bits - 1
        cnt  += 1
    return cnt

def weight_of_key(key: int, n: int) -> int:
    """Pauli weight = number of non-identity factors."""
    return _weight_of_key_nb(key, n)

# ─── Haar-random SU(4) generator ─────────────────────────────────
def random_su4() -> np.ndarray:
    """Return a Haar-random 4×4 special-unitary (det = 1)."""
    # standard Ginibre → QR trick
    z = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    # fix phases so R has positive diagonal
    ph = np.diag(r) / np.abs(np.diag(r))
    q  = q * ph
    # enforce det=1
    q /= np.linalg.det(q) ** 0.25
    return q
