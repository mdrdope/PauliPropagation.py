import numpy as np
from numba import njit, int64, complex128
from typing import Tuple, Dict, Callable

# ─── QuantumGate registry ───────────────────────────────────────
class QuantumGate:
    """Stores Numba kernels keyed by gate name."""
    _registry: Dict[str, Callable] = {}

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._registry:
            raise NotImplementedError(f"No rule for gate '{name}'")
        return cls._registry[name]

# ─── CX & T kernels (返回 (L,c1,k1,c2,k2) 紧凑 5-tuple) ──────────
@njit(cache=True)
def _cx_bits_nb(coeff: complex128, key: int64, n: int64,
                ctrl: int64, tgt: int64) -> Tuple[int64,
                                                  complex128, int64,
                                                  complex128, int64]:
    x_c = (key >>  ctrl)     & 1
    z_c = (key >> (n+ctrl))  & 1
    x_t = (key >>  tgt)      & 1
    z_t = (key >> (n+tgt))   & 1
    minus = x_c & z_t & (1 ^ (x_t ^ z_c))
    phase = -1 if minus else +1
    x_tn  = x_t ^ x_c
    z_cn  = z_c ^ z_t
    outk  = key
    if x_tn != x_t: outk ^= 1 << tgt
    if z_cn != z_c: outk ^= 1 << (n+ctrl)
    return 1, coeff * phase, outk, 0j, 0


@njit(cache=True)
def _t_bits_nb(coeff: complex128, key: int64, n: int64, q: int64)\
        -> Tuple[int64, complex128, int64, complex128, int64]:
    x = (key >>  q)     & 1
    z = (key >> (n+q))  & 1
    if (z and not x) or (not x and not z):
        return 1, coeff, key, 0j, 0
    key2 = key ^ (1 << (n+q))
    c1   = coeff / np.sqrt(2)
    c2   = +c1 if z else -c1
    return 2, c1, key, c2, key2

# ─── SU(4) kernel (返回 (L, coeff_arr, key_arr)) ─────────────────
# pre-compute 2-qubit Pauli stack
_SINGLE_P = (
    np.eye(2, dtype=np.complex128),
    np.array([[0,1],[1,0]],      dtype=np.complex128),
    np.array([[0,-1j],[1j,0]],   dtype=np.complex128),
    np.array([[1,0],[0,-1]],     dtype=np.complex128),
)
_P_STACK = np.stack([np.kron(_SINGLE_P[q2], _SINGLE_P[q1])
                     for q2 in range(4) for q1 in range(4)])

@njit(cache=True)
def _code_from_bits(z: int64, x: int64) -> int64:
    return (z << 1) | x if z == 0 else (2 | (x ^ 1))

@njit(cache=True)
def _bits_from_code(c: int64) -> Tuple[int64,int64]:
    if c==0:   return 0,0
    if c==1:   return 0,1
    if c==2:   return 1,1
    return 1,0

@njit(cache=True)
def _su4_bits_nb(coeff: complex128, key: int64, n: int64,
                 q1: int64, q2: int64, mat: np.ndarray):
    x1 = (key >>  q1)     & 1;  z1 = (key >> (n+q1)) & 1
    x2 = (key >>  q2)     & 1;  z2 = (key >> (n+q2)) & 1
    beta_idx = 4*_code_from_bits(z2,x2) + _code_from_bits(z1,x1)

    conj   = mat.conj().T @ _P_STACK[beta_idx] @ mat
    coeffs = 0.25 * np.array([np.trace(_P_STACK[i].conj().T @ conj)
                              for i in range(16)], dtype=np.complex128)

    coeff_out = np.empty(16, dtype=np.complex128)
    key_out   = np.empty(16, dtype=np.int64)
    L = 0
    for alpha in range(16):
        c = coeff * coeffs[alpha]
        if abs(c.real)+abs(c.imag) < 1e-12:
            continue
        code2a, code1a = divmod(alpha, 4)
        new_key = key
        for q, code in ((q1, code1a), (q2, code2a)):
            z,x = _bits_from_code(code)
            if ((new_key >>  q) & 1)     != x: new_key ^= 1 <<  q
            if ((new_key >> (n+q)) & 1) != z: new_key ^= 1 << (n+q)
        coeff_out[L] = c;  key_out[L] = new_key;  L += 1
    return L, coeff_out[:L], key_out[:L]

# ─── register to QuantumGate ───────────────────────────────────────────
QuantumGate._registry.update({
    "cx" : _cx_bits_nb,
    "t"  : _t_bits_nb,
    "su4": _su4_bits_nb,
})

setattr(QuantumGate, "CXgate",  staticmethod(_cx_bits_nb))
setattr(QuantumGate, "Tgate",   staticmethod(_t_bits_nb))
setattr(QuantumGate, "SU4gate", staticmethod(_su4_bits_nb))
