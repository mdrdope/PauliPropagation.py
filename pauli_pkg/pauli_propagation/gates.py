# pauli_pkg/pauli_propagation/gates.py

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List
from qiskit.quantum_info import Pauli
from .pauli_term import PauliTerm
from functools import lru_cache
import itertools
    
# ---------- Central registry of Pauli conjugation rules ----------
class QuantumGate:
    """
    A registry for quantum gate Pauli conjugation rules.
    
    This class serves as a central registry for storing and retrieving
    Pauli conjugation rules for different quantum gates. Each rule defines
    how a Pauli operator transforms when conjugated by a specific gate.
    
    Attributes
    ----------
    _registry : Dict[str, Callable]
        Dictionary mapping gate names to their corresponding conjugation rule functions.
        
    Methods
    -------
    register(name)
        Decorator to register a new conjugation rule under the specified name.
    get(name)
        Retrieve the conjugation rule function for the specified gate name.
        
    Notes
    -----
    When a rule is registered using the `register` decorator, it is also
    attached as a static method to the class with an uppercase name followed
    by "gate" (e.g., "CXgate" for "cx").
    
    Each registered rule should be a pure function that takes a PauliTerm
    and relevant qubit indices as input, and returns a list of PauliTerm
    objects representing the result after conjugation.
    
    Examples
    --------
    >>> @QuantumGate.register("new_gate")
    >>> def _new_gate_rule(term, qubit):
    ...     # Implementation of the rule
    ...     return [transformed_term]
    ...
    >>> # Can be accessed in two ways:
    >>> QuantumGate.get("new_gate")(term, 0)
    >>> QuantumGate.NEW_GATEgate(term, 0)
    """

    # {"cx": some_function, "t": another_function, ...}
    _registry: Dict[str, Callable] = {}

    # -------- registration decorator --------
    @classmethod
    def register(cls, name: str):
        """Decorator: register a new rule under *name* and
        attach it as <NAME>gate attribute for convenience."""
        def wrapper(func: Callable):
            # add name of gate and actual function of gate to class variable _registry
            cls._registry[name] = func 

            # it gives the cls a method named "CXgate" from "_cx_rule"
            setattr(cls, f"{name.upper()}gate", staticmethod(func)) 
            return staticmethod(func)
        return wrapper

    # -------- lookup helper --------
    @classmethod
    def get(cls, name: str) -> Callable:
        '''
        Search for if _registry contains a function associated with "name"
        If can't find it, raise an error
        '''
        try:
            return cls._registry[name]
        except KeyError as exc:
            raise NotImplementedError(f"No rule registered for gate '{name}'") from exc

@QuantumGate.register("cx")
def _cx_rule(term: PauliTerm, ctrl: int, tgt: int) -> List[PauliTerm]:
    r"""
    Apply the hard-coded Heisenberg conjugation rule for a controlled-X
    gate (CX) acting on a *control* qubit ``ctrl`` and a *target* qubit
    ``tgt``.

    The function looks up the transformation of the Pauli symbols on the
    control闂佺偨鍎查幃锟絘rget pair in a pre-computed **16 闁艰揪鎷� 16** table and leaves all
    other qubits unchanged.

    Parameters
    ----------
    term : PauliTerm
        The input Pauli term.  Its Pauli operator is interpreted in
        **big-endian** order (the right-most character corresponds to
        qubit 0, as in Qiskit).
    ctrl : int
        Index of the control qubit.
    tgt : int
        Index of the target qubit.

    Returns
    -------
    List[PauliTerm]
        A single-element list containing the updated :class:`PauliTerm`
        with the conjugated Pauli operator and the accumulated phase.

    Notes
    -----
    The update rule is

    .. math::

        X_c^a \, Z_c^b \, X_t^c \, Z_t^d
            \\;\xrightarrow{\\text{CX}(c, t)}\\;
        \\pm X_c^{a'} Z_c^{b'} X_t^{c'} Z_t^{d'},

    where :math:`a,b,c,d,a',b',c',d' \\in \\{0,1\\}`.  All 16 possible
    input pairs are encoded explicitly in the table ``TABLE`` together
    with the sign of the phase.

    """
    # ---------- 16 look-up table ----------
    
    TABLE = { # (phase, new_ctrl, new_tgt)
        ("I", "I"): (+1, "I", "I"),
        ("I", "X"): (+1, "I", "X"),
        ("I", "Y"): (+1, "Z", "Y"),
        ("I", "Z"): (+1, "Z", "Z"),

        ("X", "I"): (+1, "X", "X"),
        ("X", "X"): (+1, "X", "I"),
        ("X", "Y"): (+1, "Y", "Z"),  
        ("X", "Z"): (-1, "Y", "Y"),

        ("Y", "I"): (+1, "Y", "X"),
        ("Y", "X"): (+1, "Y", "I"),
        ("Y", "Y"): (-1, "X", "Z"),
        ("Y", "Z"): (+1, "X", "Y"),

        ("Z", "I"): (+1, "Z", "I"),
        ("Z", "X"): (+1, "Z", "X"),
        ("Z", "Y"): (+1, "I", "Y"),
        ("Z", "Z"): (+1, "I", "Z")}
    # ------------------------------------------
    # 1) Extract the original big-endian label string, e.g. 'ZXI...'
    label_list = list(term.pauli.to_label())      # right-most char is qubit 0
    sym_c = label_list[-1 - ctrl]                # symbol on control qubit
    sym_t = label_list[-1 - tgt]                 # symbol on target  qubit

    # 2) Look up (phase, new_control_symbol, new_target_symbol)
    phase, new_c, new_t = TABLE[(sym_c, sym_t)]

    # 3) Write back the updated symbols
    label_list[-1 - ctrl] = new_c
    label_list[-1 - tgt]  = new_t
    new_label = "".join(label_list)

    # 4) Construct and return the new PauliTerm
    new_pauli = Pauli(new_label)
    return [PauliTerm(term.coeff * phase, new_pauli)]

@QuantumGate.register("t")
def _t_rule(term: PauliTerm, q: int) -> List[PauliTerm]:
    """Conjugate a Pauli operator by a T gate (闁挎粣鎷�/4 phase on |1?).
    
    This function implements the transformation rules for Pauli operators
    when conjugated by a T gate (T? P T). The T gate applies a 闁挎粣鎷�/4 phase
    to the |1? state.
    
    The transformation rules are:
    - Z 闂佺�规嫹闁跨噦鎷� Z (unchanged)
    - X 闂佺�规嫹闁跨噦鎷� (X - Y)/闂佺�规嫹闁跨噦鎷�2
    - Y 闂佺�规嫹闁跨噦鎷� (X + Y)/闂佺�规嫹闁跨噦鎷�2
    - I 闂佺�规嫹闁跨噦鎷� I (unchanged)
    
    Parameters
    ----------
    term : PauliTerm
        The Pauli term to be transformed.
    q : int
        Index of the qubit on which the T gate acts.
        
    Returns
    -------
    List[PauliTerm]
        A list of PauliTerm objects representing the transformed operator.
        For Z and I operators, returns a single-element list with the original term.
        For X and Y operators, returns a two-element list with the transformed terms.
        
    Notes
    -----
    The T gate is a single-qubit phase gate that applies a 闁挎粣鎷�/4 phase shift.
    It is represented by the matrix:
    T = [[1, 0], [0, exp(i闁挎粣鎷�/4)]]
    
    This implementation uses the binary representation of Pauli operators where
    each operator is encoded by two binary arrays (z, x).
    """
    z, x = term.pauli.z.copy(), term.pauli.x.copy()

    # Z  闂佺�规嫹闁跨噦鎷� unchanged
    if z[q] and not x[q]:
        return [term]

    # X  闂佺�规嫹闁跨噦鎷� (X ? Y)/闂佺�规嫹闁跨噦鎷�2
    if x[q] and not z[q]:
        p1 = PauliTerm(term.coeff / np.sqrt(2), term.pauli)            # +X
        z2 = z.copy(); z2[q] = True                                    # +Y
        p2 = PauliTerm(-term.coeff / np.sqrt(2), Pauli((z2, x)))       # ?Y
        return [p1, p2]

    # Y  闂佺�规嫹闁跨噦鎷� (X + Y)/闂佺�规嫹闁跨噦鎷�2
    if x[q] and z[q]:
        p1 = PauliTerm(term.coeff / np.sqrt(2), term.pauli)            # +Y
        z2 = z.copy(); z2[q] = False                                   # +X
        p2 = PauliTerm(term.coeff / np.sqrt(2), Pauli((z2, x)))        # +X
        return [p1, p2]

    # I on that qubit
    return [term]

# @QuantumGate.register("su4")
# def _su4_rule(term: PauliTerm,
#               q1: int,
#               q2: int,
#               mat: np.ndarray) -> List[PauliTerm]:
#     """
#     Conjugate a PauliTerm by an SU(4) unitary acting on qubits q1 and q2
#     (big-endian: right-most label char is qubit-0).
#     """
#     LABELS_2Q = (
#         "II","IX","IY","IZ",
#         "XI","XX","XY","XZ",
#         "YI","YX","YY","YZ",
#         "ZI","ZX","ZY","ZZ"
#     )

#     @lru_cache(maxsize=None)
#     def _pauli_matrix(label: str) -> np.ndarray:
#         """Return the 4脳4 matrix of a two-qubit Pauli label."""
#         _single = {
#             "I": np.eye(2, dtype=complex),
#             "X": np.array([[0, 1], [1, 0]], dtype=complex),
#             "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
#             "Z": np.array([[1, 0], [0, -1]], dtype=complex)
#         }
#         return np.kron(_single[label[0]], _single[label[1]])

#     # 1) extract the two-qubit substring on (q1, q2)
#     full_label = list(term.pauli.to_label())          # big-endian
#     two_label  = full_label[-1 - q2] + full_label[-1 - q1]

#     # 2) build / fetch transfer matrix R
#     # --- NEW robust cache key: matrix bytes + qubit ordering -----------
#     cache_key = (mat.tobytes(), q1, q2)
#     # -------------------------------------------------------------------
#     if not hasattr(_su4_rule, "_cache"):
#         _su4_rule._cache = {}
#     if cache_key not in _su4_rule._cache:
#         R = np.zeros((16, 16), dtype=complex)
#         mat_dag = mat.conj().T
#         for beta, P_beta in enumerate(LABELS_2Q):
#             conj = mat_dag @ _pauli_matrix(P_beta) @ mat
#             for alpha, P_alpha in enumerate(LABELS_2Q):
#                 R[alpha, beta] = 0.25 * np.trace(_pauli_matrix(P_alpha) @ conj)
#         _su4_rule._cache[cache_key] = R
#     else:
#         R = _su4_rule._cache[cache_key]

#     # 3) column 尾 gives new coefficients
#     beta_idx = LABELS_2Q.index(two_label)
#     coeffs   = R[:, beta_idx] * term.coeff

#     # 4) assemble new PauliTerm list
#     terms: List[PauliTerm] = []
#     for alpha_idx, c in enumerate(coeffs):
#         if abs(c) < 1e-12:
#             continue
#         alpha_label = LABELS_2Q[alpha_idx]
#         new_label = full_label.copy()
#         new_label[-1 - q2] = alpha_label[0]   # 伪鈧€ 鈫� q2
#         new_label[-1 - q1] = alpha_label[1]   # 伪鈧� 鈫� q1
#         terms.append(PauliTerm(c, Pauli("".join(new_label))))
#     return terms

@QuantumGate.register("su4")
def _su4_rule(term: PauliTerm,
              q1: int,
              q2: int,
              mat: np.ndarray) -> List[PauliTerm]:
    """
    Conjugate a PauliTerm by an SU(4) unitary on qubits q1, q2 (big-endian).
    Vectorised implementation: ~10× faster than the naive 16×16 loop.
    """
    # ── static pre-computed 16 two-qubit Pauli matrices ────────────────
    if not hasattr(_su4_rule, "_LABELS_2Q"):
        _su4_rule._LABELS_2Q = tuple(
            "II IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ".split()
        )
        _single = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }
        _su4_rule._STACK = np.stack([
            np.kron(_single[l[0]], _single[l[1]])
            for l in _su4_rule._LABELS_2Q
        ])                                           # shape (16,4,4)

    LABELS_2Q = _su4_rule._LABELS_2Q
    P_STACK    = _su4_rule._STACK                   # (16,4,4)

    # 1) extract two-qubit label (big-endian)
    full = list(term.pauli.to_label())
    two_label = full[-1 - q2] + full[-1 - q1]
    beta_idx  = LABELS_2Q.index(two_label)

    # 2) cache on (matrix, beta_idx)  – computes only once per (U, P_beta)
    cache_key = (mat.tobytes(), beta_idx)
    if not hasattr(_su4_rule, "_coeff_cache"):
        _su4_rule._coeff_cache = {}
    if cache_key in _su4_rule._coeff_cache:
        coeffs = _su4_rule._coeff_cache[cache_key] * term.coeff
    else:
        # conj = U† P_beta U
        mat_dag = mat.conj().T
        conj = mat_dag @ P_STACK[beta_idx] @ mat               # (4,4)
        # coeffs[alpha] = ¼ Tr(P_alpha · conj)
        coeffs = 0.25 * np.tensordot(P_STACK.conj(), conj, axes=([1, 2], [0, 1]))
        _su4_rule._coeff_cache[cache_key] = coeffs
        coeffs = coeffs * term.coeff

    # 3) assemble output PauliTerm list (skip ~0 entries)
    new_terms: List[PauliTerm] = []
    for alpha_idx, c in enumerate(coeffs):
        if abs(c) < 1e-12:
            continue
        a_lbl = LABELS_2Q[alpha_idx]
        new_lbl = full.copy()
        new_lbl[-1 - q2] = a_lbl[0]
        new_lbl[-1 - q1] = a_lbl[1]
        new_terms.append(PauliTerm(c, Pauli("".join(new_lbl))))
    return new_terms

