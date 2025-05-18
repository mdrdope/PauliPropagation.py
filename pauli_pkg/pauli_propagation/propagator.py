import numpy as np
from numba import int64, complex128, float64, njit
from typing import List, Dict, Tuple
from qiskit import QuantumCircuit

from .pauli_term  import PauliTerm
from .utils       import weight_of_key
from .gates       import QuantumGate

class PauliPropagator:
    """
    Bit-mask based back-propagation of Pauli observables.
    """

    # -------- expectation value table --------
    _STATE_IDX = {'0':0,'1':1,'+':2,'-':3,'r':4,'l':5}
    _EXP_TABLE = np.zeros((6,2,2), dtype=np.float64)
    _EXP_TABLE[_STATE_IDX['0'],:] = [[1,0],[1,0]]
    _EXP_TABLE[_STATE_IDX['1'],:] = [[1,0],[-1,0]]
    _EXP_TABLE[_STATE_IDX['+'],:] = [[1,1],[0,0]]
    _EXP_TABLE[_STATE_IDX['-'],:] = [[1,-1],[0,0]]
    _EXP_TABLE[_STATE_IDX['r'],:] = [[1,0],[0,1]]
    _EXP_TABLE[_STATE_IDX['l'],:] = [[1,0],[0,-1]]

    @staticmethod
    @njit(cache=True)
    def _expect_keys(coeffs: complex128[:], keys: int64[:],
                     state_idxs: int64[:], n: int64,
                     exp_table: float64[:, :, :]) -> float64:
        total = 0.0
        for t in range(coeffs.size):
            key   = keys[t]
            alpha = coeffs[t]
            prod  = 1.0
            for q in range(n):
                x = (key >>  q)     & 1
                z = (key >> (n+q))  & 1
                prod *= exp_table[state_idxs[q], z, x]
                if prod == 0.0:
                    break
            total += alpha.real * prod
        return total

    # --------------------------------------------------
    def __init__(self, qc: QuantumCircuit):
        self.qc  = qc
        self.n   = qc.num_qubits
        self.q2i = {q: i for i, q in enumerate(qc.qubits)}

    # --------------------------------------------------
    def propagate(self,
                  observable: PauliTerm,
                  max_weight: int | None = None,
                  tol: float = 1e-12
                 ) -> List[List[PauliTerm]]:

        if observable.n != self.n:
            raise ValueError("Observable qubit count mismatch")

        paths: Dict[int, complex] = {observable.key: observable.coeff}
        history: List[List[PauliTerm]] = [[observable]]

        for instr in reversed(self.qc.data):
            gname = instr.operation.name
            rule  = QuantumGate.get(gname)
            qidx  = tuple(self.q2i[q] for q in instr.qubits)

            extra_args: Tuple = ()
            if gname == "su4" and hasattr(instr.operation, "to_matrix"):
                extra_args = (instr.operation.to_matrix().astype(np.complex128),)

            next_paths: Dict[int, complex] = {}

            # for key_in, coeff_in in paths.items():
            #     L, coeff_arr, key_arr = rule(coeff_in, key_in,
            #                                  self.n, *qidx, *extra_args)
            #     for j in range(int(L)):
            #         k2 = int(key_arr[j])
            #         c2 = coeff_arr[j]
            #         if max_weight is not None and \
            #            weight_of_key(k2, self.n) > max_weight:
            #             continue
            #         next_paths[k2] = next_paths.get(k2, 0.0) + c2
            
            for key_in, coeff_in in paths.items():
                out = rule(coeff_in, key_in, self.n, *qidx, *extra_args)

                # ---- adapt to kernel output format ----
                if len(out) == 5:                      # CX / T  ¡ú 5-tuple
                    L, c1, k1, c2, k2 = out
                    coeff_arr = np.array([c1, c2], dtype=np.complex128)[:L]
                    key_arr   = np.array([k1, k2], dtype=np.int64)[:L]
                else:                                  # SU4 ¡ú (L, coeff_vec, key_vec)
                    L, coeff_arr, key_arr = out

                for j in range(int(L)):
                    k2 = int(key_arr[j])
                    c2 = coeff_arr[j]
                    if max_weight is not None and \
                       weight_of_key(k2, self.n) > max_weight:
                        continue
                    next_paths[k2] = next_paths.get(k2, 0.0) + c2


            paths = {k: c for k, c in next_paths.items() if abs(c) > tol}
            history.append([PauliTerm(c, k, self.n) for k, c in paths.items()])

        return history

    # --------------------------------------------------
    def expectation_pauli_sum(self,
                              pauli_sum: List[PauliTerm],
                              product_label: str) -> float:
        if len(product_label) != self.n:
            raise ValueError("Label length mismatch")

        state_idxs = np.fromiter(
            (self._STATE_IDX[ch] for ch in product_label[::-1]),
            dtype=np.int64, count=self.n)

        m = len(pauli_sum)
        coeffs = np.empty(m, dtype=np.complex128)
        keys   = np.empty(m, dtype=np.int64)
        for i, term in enumerate(pauli_sum):
            coeffs[i] = term.coeff
            keys[i]   = term.key

        return float(self._expect_keys(coeffs, keys,
                                       state_idxs, self.n,
                                       self._EXP_TABLE))
