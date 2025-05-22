# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/propagator.py
import numpy as np
from typing import List, Dict, Tuple, Set
from qiskit import QuantumCircuit
import math
from .pauli_term  import PauliTerm
from .utils       import weight_of_key
from .gates       import QuantumGate
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# Threshold for parallel processing and maximum number of worker processes
_PARALLEL_THRESHOLD = 2000
_MAX_WORKERS = 8  # or os.cpu_count()

def _apply_rule(args):
    """
    Helper function for ProcessPoolExecutor to apply a single gate rule and flatten its output.
    
    This function processes a single gate application and formats its output into a standardized
    list of (key, coefficient) pairs.
    
    Parameters
    ----------
    args : tuple
        Contains (rule, coeff_in, key_in, n, qidx, extra_args)
        - rule: Gate rule function to apply
        - coeff_in: Input coefficient
        - key_in: Input Pauli key
        - n: Number of qubits
        - qidx: Qubit indices
        - extra_args: Additional arguments for the gate
        
    Returns
    -------
    List[Tuple[int, complex]]
        List of (key, coefficient) pairs representing the gate output
    """
    rule, coeff_in, key_in, n, qidx, extra_args = args
    out = rule(coeff_in, key_in, n, *qidx, *extra_args)
    # Handle CX/T gate outputs (format: L, c1, k1, c2, k2)
    if len(out) == 5:
        L, c1, k1, c2, k2 = out
        if L == 1:
            return [(int(k1), c1)]
        else:
            return [(int(k1), c1), (int(k2), c2)]
    # Handle SU4 gate outputs (format: L, coeffs, keys)
    L, coeffs, keys = out
    return [(int(keys[i]), coeffs[i]) for i in range(L)]


class PauliPropagator:
    """
    Bit-mask based back-propagation of Pauli observables through quantum circuits.
    
    This class implements efficient back-propagation of Pauli observables through
    quantum circuits using bit-mask representations. It supports both exact propagation
    and Monte Carlo sampling of Pauli paths.
    
    Attributes
    ----------
    qc : QuantumCircuit
        The quantum circuit to propagate through
    n : int
        Number of qubits in the circuit
    q2i : Dict
        Mapping from qubit objects to indices
    """

    # -------- Expectation value lookup tables --------
    # State indices mapping for different basis states
    _STATE_IDX = {'0':0,'1':1,'+':2,'-':3,'r':4,'l':5}
    # Pre-computed expectation values for different states and Pauli operators
    _EXP_TABLE = np.zeros((6,2,2), dtype=float)
    _EXP_TABLE[_STATE_IDX['0'],:] = [[1,0],[1,0]]  # |0⟩ state
    _EXP_TABLE[_STATE_IDX['1'],:] = [[1,0],[-1,0]] # |1⟩ state
    _EXP_TABLE[_STATE_IDX['+'],:] = [[1,1],[0,0]]  # |+⟩ state
    _EXP_TABLE[_STATE_IDX['-'],:] = [[1,-1],[0,0]] # |-⟩ state
    _EXP_TABLE[_STATE_IDX['r'],:] = [[1,0],[0,1]]  # |r⟩ state
    _EXP_TABLE[_STATE_IDX['l'],:] = [[1,0],[0,-1]] # |l⟩ state

    @staticmethod
    def _expect_keys(coeffs: np.ndarray, keys: np.ndarray, 
                     state_idxs: np.ndarray, n: int,
                     exp_table: np.ndarray) -> float:
        """
        Calculate expectation value for a set of Pauli terms.
        
        This method efficiently computes the expectation value of a sum of Pauli terms
        using pre-computed lookup tables and bit-mask operations.
        
        Parameters
        ----------
        coeffs : np.ndarray
            Complex coefficients of Pauli terms
        keys : np.ndarray
            Bit-mask representations of Pauli operators
        state_idxs : np.ndarray
            State indices for each qubit
        n : int
            Number of qubits
        exp_table : np.ndarray
            Expectation value lookup table
            
        Returns
        -------
        float
            Total expectation value
        """
        total = 0.0
        n_terms = len(coeffs)
        
        # Process each term
        for t in range(n_terms):
            key = keys[t]
            alpha = coeffs[t]
            prod = 1.0
            
            # Compute product of expectation values for each qubit
            for q in range(n):
                x = (key >> q) & 1  # Extract X bit
                z = (key >> (n+q)) & 1  # Extract Z bit
                val = exp_table[state_idxs[q], z, x]
                if val == 0.0:  # Early termination if product becomes zero
                    prod = 0.0
                    break
                prod *= val
                
            total += alpha.real * prod
            
        return total

    def __init__(self, qc: QuantumCircuit):
        """
        Initialize propagator with a quantum circuit.
        
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit to propagate through
        """
        self.qc  = qc
        self.n   = qc.num_qubits
        self.q2i = {q: i for i, q in enumerate(qc.qubits)}

    def propagate(self,
                  observable: PauliTerm,
                  max_weight: int | None = None,
                  use_parallel: bool = False,
                  tol: float = 1e-10,
                 ) -> List[List[PauliTerm]]:
        """
        Propagate a Pauli observable through the circuit.
        
        This method performs exact propagation of a Pauli observable through the circuit,
        discarding imaginary parts by using only real coefficients. It supports parallel
        processing for large circuits and weight-based filtering.
        
        Parameters
        ----------
        observable : PauliTerm
            Initial Pauli observable to propagate
        max_weight : int | None
            Maximum weight of Pauli terms to keep (None for no limit)
        use_parallel : bool
            Whether to use parallel processing
        tol : float
            Tolerance for discarding small coefficients
            
        Returns
        -------
        List[List[PauliTerm]]
            History of Pauli terms at each circuit layer
        """
        if observable.n != self.n:
            raise ValueError("Observable qubit count mismatch")

        # Initialize with real coefficient
        paths: Dict[int, float] = {observable.key: float(observable.coeff.real)}
        history: List[List[PauliTerm]] = [[PauliTerm(paths[observable.key], observable.key, self.n)]]

        # Prepare reverse circuit operations
        ops = []
        for instr in reversed(self.qc.data):
            rule = QuantumGate.get(instr.operation.name)
            qidx = tuple(self.q2i[q] for q in instr.qubits)
            extra = ()
            if instr.operation.name == "su4" and hasattr(instr.operation, "to_matrix"):
                extra = (instr.operation.to_matrix(),)
            ops.append((rule, qidx, extra))

        # Set up parallel processing if requested
        executor = ProcessPoolExecutor(max_workers=_MAX_WORKERS) if use_parallel else None
        try:
            for rule, qidx, extra_args in tqdm(ops, desc="Propagating", total=len(ops)):
                next_paths: Dict[int, float] = {}
                items = [(rule, coeff, key, self.n, qidx, extra_args) for key, coeff in paths.items()]

                # Choose between parallel and sequential processing
                if use_parallel and len(items) > _PARALLEL_THRESHOLD:
                    chunksize = max(1, len(items) // _MAX_WORKERS)
                    results_iter = executor.map(_apply_rule, items, chunksize=chunksize)
                else:
                    results_iter = map(_apply_rule, items)

                # Process results and update paths
                for term_list in results_iter:
                    for k2, c2 in term_list:
                        c2_real = c2.real
                        # Apply filters
                        if max_weight is not None and weight_of_key(k2, self.n) > max_weight:
                            continue
                        if abs(c2_real) <= tol:
                            continue
                        next_paths[k2] = next_paths.get(k2, 0.0) + c2_real

                # Prepare for next layer
                paths = {k: c for k, c in next_paths.items() if abs(c) > tol}
                history.append([PauliTerm(c, k, self.n) for k, c in paths.items()])

        finally:
            if executor:
                executor.shutdown()

        return history

    def expectation_pauli_sum(self,
                              pauli_sum: List[PauliTerm],
                              product_label: str) -> float:
        """
        Calculate expectation value of a sum of Pauli terms.
        
        Parameters
        ----------
        pauli_sum : List[PauliTerm]
            List of Pauli terms to evaluate
        product_label : str
            Product state label (e.g. '0+1-')
            
        Returns
        -------
        float
            Expectation value
            
        Raises
        ------
        ValueError
            If label length doesn't match qubit count
        """
        if len(product_label) != self.n:
            raise ValueError("Label length mismatch")

        # Pre-compute state indices once
        state_idxs = np.array([self._STATE_IDX[ch] for ch in product_label[::-1]])
        
        # Pre-allocate arrays for vectorized operations
        m = len(pauli_sum)
        coeffs = np.empty(m, dtype=complex)
        keys = np.empty(m, dtype=object)
        
        # Fill arrays
        for i, term in enumerate(pauli_sum):
            coeffs[i] = term.coeff
            keys[i] = term.key

        # Calculate expectation value
        return float(self._expect_keys(coeffs, keys,
                                     state_idxs, self.n,
                                     self._EXP_TABLE))


    def sample_pauli_path(self, observable: PauliTerm, M: int):
        """
        Perform M full-depth samplings for the given observable and return complete propagation paths.
        
        Algorithm description:
        For m=1 to M repeat:
        1. Initialize s_j = Observable, c_j = 1, w_j = |s_j|
        2. For each layer j (from last to first):
           a. Expand U_j† s_j = ∑ a_j,i P_j,i
           b. Sample an index i_j according to Pr[i] ∝ |a_j,i|²
           c. Update s_(j-1) = P_j,i_j, c_(j-1) = c_j × a_j,i_j, w_(j-1) = |s_(j-1)|
        3. Record complete path {(s_j, c_j, w_j)} from initial to final state
        
        Parameters
        ----------
        observable : PauliTerm
            Initial observable
        M : int
            Number of samples
            
        Returns
        -------
        List[List[PauliTerm]]
            List of M sampling paths, each containing complete PauliTerm sequence
        """
        if observable.n != self.n:
            raise ValueError("Observable qubit count mismatch")

        # Prepare reverse operation list (from last layer to first)
        ops = []
        for instr in reversed(self.qc.data):
            rule = QuantumGate.get(instr.operation.name)
            qidx = tuple(self.q2i[q] for q in instr.qubits)
            extra = ()
            if instr.operation.name == "su4" and hasattr(instr.operation, "to_matrix"):
                extra = (instr.operation.to_matrix(),)
            ops.append((rule, qidx, extra))

        # Sample for each sample
        all_samples = []
        for _ in range(M):
            # Initialize: s_j = Observable, c_j = 1, w_j = |s_j|
            path = []
            current_key = observable.key
            current_coeff = float(observable.coeff.real)  # Keep only real part
            current_weight = weight_of_key(current_key, self.n)
            
            # Record initial state
            path.append(PauliTerm(current_coeff, current_key, self.n))
            
            # Sample for each layer
            for i, (rule, qidx, extra_args) in enumerate(ops):
                # Expand U_j† s_j = ∑ a_j,i P_j,i
                out = rule(current_coeff, current_key, self.n, *qidx, *extra_args)
                
                # Process output format, get expansion terms
                branches = []
                if len(out) == 5:  # CX/T format
                    L, c1, k1, c2, k2 = out
                    if L == 1:
                        branches = [(int(k1), c1)]
                    else:
                        branches = [(int(k1), c1), (int(k2), c2)]
                else:  # SU4 format
                    L, coeffs, keys = out
                    if L > 0:
                        branches = [(int(keys[i]), coeffs[i]) for i in range(L)]
                
                # If no branches, keep current state unchanged
                if not branches:
                    # Add same state
                    path.append(PauliTerm(current_coeff, current_key, self.n))
                    continue
                
                # Calculate sampling probabilities Pr[i] ∝ |a_j,i|²
                probs = np.array([abs(c)**2 for (_, c) in branches], float)
                if probs.sum() == 0:
                    # Handle case where probability sum is 0 (floating point precision issue)
                    probs = np.ones(len(branches)) / len(branches)
                else:
                    probs /= probs.sum()
                
                idx = np.random.choice(len(branches), p=probs) # Sample an index i_j according to probability
                
                # Update state: s_(j-1) = P_j,i_j, c_(j-1) = c_j × a_j,i_j, w_(j-1) = |s_(j-1)|
                current_key, amp = branches[idx]
                current_coeff = float((current_coeff * amp).real)  # Keep only real part
                current_weight = weight_of_key(current_key, self.n)
                
                # Record current state
                path.append(PauliTerm(current_coeff, current_key, self.n))
            
            # Add complete path to sample collection
            all_samples.append(path)
        
        return all_samples

    # def sample_pauli_path(self,
    #                       observable: PauliTerm,
    #                       M: int
    #                      ) -> List[List[PauliTerm]]:
    #     """
    #     Perform M full-depth Monte Carlo Pauli-path samples (no truncation).
    #     Returns a list of M paths; each path is a list of 5 PauliTerm objects
    #     for layers j=4,3,2,1,0.
    #     """
    #     # 1) 构建反向操作序列
    #     ops = []
    #     for instr in reversed(self.qc.data):
    #         rule = QuantumGate.get(instr.operation.name)
    #         qidx = tuple(self.q2i[q] for q in instr.qubits)
    #         extra = ()
    #         if instr.operation.name == "su4" and hasattr(instr.operation, "to_matrix"):
    #             extra = (instr.operation.to_matrix(),)
    #         ops.append((rule, qidx, extra))

    #     all_paths: List[List[PauliTerm]] = []
    #     for _ in range(M):
    #         path: List[PauliTerm] = []

    #         # 初始层 j=4：PauliTerm(coeff, key, n)
    #         init_c = observable.coeff.real
    #         # 对数＆相位跟踪
    #         log_abs = math.log(abs(init_c))
    #         phase   = observable.coeff / init_c  # unit-modulus complex
    #         key     = observable.key
    #         # 重建成浮点系数
    #         coeff   = phase * math.exp(log_abs)
    #         path.append(PauliTerm(coeff, key, self.n))

    #         # 2) 逐层回退
    #         for (rule, qidx, extra) in ops:
    #             # 始终传入 coeff=1.0，让 _apply_rule 只返回单层分支系数 a_{j,i}
    #             branches = _apply_rule((rule, 1.0, key, self.n, qidx, extra))
    #             # branches: List of (key2, a_j_i)

    #             # 计算采样概率 ∝ |a|^2
    #             amps  = np.array([c2 for (_k2, c2) in branches], complex)
    #             probs = np.abs(amps)**2
    #             s     = probs.sum()
    #             # ? do we have to divided, can we just sample based on an unnormalized probability dist?
    #             probs = probs/s if s>0 else np.ones_like(probs)/len(probs) 

    #             # 抽一个分支
    #             idx = np.random.choice(len(branches), p=probs)
    #             key, a = branches[idx]

    #             # 更新 log_abs, phase
    #             log_abs += math.log(abs(a))
    #             phase   *= (a/abs(a))

    #             # 重建系数并记录
    #             coeff = phase * math.exp(log_abs)
    #             path.append(PauliTerm(coeff, key, self.n))

    #         all_paths.append(path)

    #     return all_paths