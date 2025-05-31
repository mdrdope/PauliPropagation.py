# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/propagator.py
import numpy as np
from typing import List, Dict, Tuple, Set, Union
from qiskit import QuantumCircuit
import math
from .pauli_term  import PauliTerm
from .utils       import weight_of_key
from .gates       import QuantumGate
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Threshold for parallel processing and maximum number of worker processes
_PARALLEL_THRESHOLD = 2000
_MAX_WORKERS = 8 # or os.cpu_count()

# 预编译门函数缓存
_GATE_FUNC_CACHE = {}

def _get_gate_func_cached(gate_name: str):
    """Cache gate functions to reduce dictionary lookup overhead"""
    if gate_name not in _GATE_FUNC_CACHE:
        _GATE_FUNC_CACHE[gate_name] = QuantumGate.get(gate_name)
    return _GATE_FUNC_CACHE[gate_name]

def _apply_gate_kernel_batch_optimized(args):
    """
    Optimized version of gate application function - reduce object creation overhead
    """
    terms_data, gate_name, qidx, extra_args = args
    
    # Use pre-cached gate function
    gate_func = _get_gate_func_cached(gate_name)
    
    results = []
    # Pre-allocate result list size to reduce reallocation
    results_reserve = len(terms_data) * 2  # Estimate branching factor
    
    for coeff, key, n in terms_data:
        # Pass tuple directly instead of creating PauliTerm object (if gate function supports)
        # Keep compatibility here, but can be further optimized
        input_term = PauliTerm(coeff, key, n)
        
        # Apply gate
        if extra_args:
            output_terms = gate_func(input_term, *qidx, *extra_args)
        else:
            output_terms = gate_func(input_term, *qidx)
        
        # Batch add results, reduce list.append calls
        for term in output_terms:
            results.append((term.coeff, term.key, term.n))
    
    return results


class PauliPropagator:
    """
    Bit-mask based back-propagation of Pauli observables through quantum circuits.
    
    This class implements efficient back-propagation of Pauli observables through
    quantum circuits using bit-mask representations. It supports exact propagation
    of Pauli paths.
    
    Attributes
    ----------
    qc : QuantumCircuit
        The quantum circuit to propagate through
    n : int
        Number of qubits in the circuit
    q2i : Dict
        Mapping from qubit objects to indices
    """

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


    # -------- Expectation value lookup tables --------
    # State indices mapping for different basis states
    _STATE_IDX = {'0':0,'1':1,'+':2,'-':3,'r':4,'l':5}
    # Pre-computed expectation values for different states and Pauli operators
    _EXP_TABLE = np.zeros((6,2,2), dtype=float)
    _EXP_TABLE[_STATE_IDX['0'],:] = [[1,0],[1,0]]  # |0 state
    _EXP_TABLE[_STATE_IDX['1'],:] = [[1,0],[-1,0]] # |1 state
    _EXP_TABLE[_STATE_IDX['+'],:] = [[1,1],[0,0]]  # |+ state
    _EXP_TABLE[_STATE_IDX['-'],:] = [[1,-1],[0,0]] # |- state
    _EXP_TABLE[_STATE_IDX['r'],:] = [[1,0],[0,1]]  # |r state
    _EXP_TABLE[_STATE_IDX['l'],:] = [[1,0],[0,-1]] # |l state

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


    def propagate(self,
                  observable: PauliTerm,
                  max_weight: int | None = None,
                  use_parallel: bool = False,
                  tol: float = 1e-10,
                 ) -> List[List[PauliTerm]]:
        """
        Propagate a Pauli observable through the circuit.
        
        This method performs exact propagation of a Pauli observable through the circuit,
        using high-performance internal kernels. It supports parallel processing for 
        large circuits and weight-based filtering.
        
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

        # Use internal representation for performance: (coeff, key, n)
        current_terms_data = [(observable.coeff, observable.key, observable.n)]
        history: List[List[PauliTerm]] = [[observable]]

        # Preprocess operation sequence to reduce loop overhead
        ops = []
        for instr in reversed(self.qc.data):
            gate_name = instr.operation.name
            qidx = tuple(self.q2i[q] for q in instr.qubits)
            extra = QuantumGate.extract_params(gate_name, instr)
            # Pre-cache gate function
            gate_func = _get_gate_func_cached(gate_name)
            ops.append((gate_name, qidx, extra, gate_func))

        # Use numpy arrays for coefficient optimization
        import numpy as np
        
        # Optimized merging logic
        def merge_terms_optimized(terms_data):
            if not terms_data:
                return []
            
            # Use dictionary for fast grouping
            key_to_coeff = {}
            for coeff, key, n in terms_data:
                if key in key_to_coeff:
                    key_to_coeff[key] += coeff
                else:
                    key_to_coeff[key] = coeff
            
            # Batch filtering and creation
            filtered_terms = []
            for key, coeff in key_to_coeff.items():
                if abs(coeff.real) > tol or abs(coeff.imag) > tol:
                    if max_weight is None or weight_of_key(key, n) <= max_weight:
                        filtered_terms.append((coeff, key, n))
            return filtered_terms
        
        # Set up parallel processing if requested
        executor = ProcessPoolExecutor(max_workers=_MAX_WORKERS) if use_parallel else None
        try:
            for gate_name, qidx, extra_args, gate_func in tqdm(ops, desc=f"Propagating, max weight: {max_weight}", total=len(ops)):
                
                # For small numbers of terms or if parallel is disabled, process sequentially
                if not use_parallel or len(current_terms_data) < _PARALLEL_THRESHOLD:
                    next_terms_data = _apply_gate_kernel_batch_optimized((current_terms_data, gate_name, qidx, extra_args))
                else:
                    # Split terms into chunks for parallel processing
                    chunk_size = max(1, len(current_terms_data) // _MAX_WORKERS)
                    chunks = [current_terms_data[i:i+chunk_size] 
                            for i in range(0, len(current_terms_data), chunk_size)]
                    
                    # Process chunks in parallel
                    future_to_chunk = {executor.submit(_apply_gate_kernel_batch_optimized, (chunk, gate_name, qidx, extra_args)): chunk 
                                       for chunk in chunks}
                    
                    next_terms_data = []
                    for future in as_completed(future_to_chunk):
                        chunk_results = future.result()
                        next_terms_data.extend(chunk_results)
                
                # Use optimized merging function
                current_terms_data = merge_terms_optimized(next_terms_data)
                
                # Create PauliTerm objects for history (only when needed)
                current_terms_objects = [PauliTerm(coeff, key, n) for coeff, key, n in current_terms_data]
                history.append(current_terms_objects)

                # Early termination if no terms remain
                if not current_terms_data:
                    break

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

    def analytical_truncation_mse(self,
                                init_term: PauliTerm,
                                product_label: str = None
                                ) -> Dict[str, Union[Dict[int, float], int]]:
        """
        Compute the exact, instantaneous truncation MSE (Mean Squared Error)
        by dynamic programming over Pauli-propagation paths.

        We treat each reverse-propagation path state as a tuple
        (pauli_key, max_weight_so_far), where max_weight_so_far records
        the largest Pauli weight encountered anywhere along the path.

        Parameters
        ----------
        init_term : PauliTerm
            The starting Pauli term (with coeff=1 and its encoded key)
        product_label : str, optional
            Product-state label for expectation (e.g. "000...0"); defaults to all-zeros

        Returns
        -------
        Dict[str, Union[Dict[int, float], int]]
            {
            'mse_cumulative': {k: MSE^(k) = sum_{max_w > k} p * d^2},
            'mse_per_weight': {k: MSE(k) = sum_{max_w == k} p * d^2},
            'max_weight': int   # maximum weight encountered overall
            }
        """
        if init_term.n != self.n:
            raise ValueError("Initial PauliTerm qubit count mismatch")

        # Default to the |0...0> product state
        if product_label is None:
            product_label = "0" * self.n

        # 0) Initialize key and its initial weight
        init_key = init_term.key
        init_wt = init_term.weight()

        # 1) Build the reverse-propagation gate list
        ops: List[Tuple[str, Tuple[int, ...], Tuple]] = []
        for instr in reversed(self.qc.data):
            name = instr.operation.name
            qidx = tuple(self.q2i[q] for q in instr.qubits)
            extra = QuantumGate.extract_params(name, instr)
            ops.append((name, qidx, extra))

        # 2) Probability DP over states (pauli_key, max_weight_so_far)
        # dist[(key, wmax)] = cumulative probability sum |_|^2 of reaching that state
        dist: Dict[Tuple[int, int], float] = {(init_key, init_wt): 1.0}

        for name, qidx, extra in tqdm(ops, desc="DP over states"):
            gate_func = QuantumGate.get(name)
            new_dist: Dict[Tuple[int, int], float] = {}

            for (key, wmax), prob in dist.items():
                term = PauliTerm(1.0, key, self.n)
                # generate all outgoing branches through this gate
                branches = (gate_func(term, *qidx, *extra)
                            if extra else gate_func(term, *qidx))

                for branch in branches:
                    p_branch = prob * (abs(branch.coeff) ** 2)
                    new_wmax = max(wmax, branch.weight())
                    new_key = branch.key
                    new_dist[(new_key, new_wmax)] = new_dist.get((new_key, new_wmax), 0.0) + p_branch

            dist = new_dist

        # 3) Precompute d^2 = (<P, |0^n>>)^2 for each final key
        d2_cache: Dict[int, float] = {}
        for (key, _) in dist.keys():
            if key not in d2_cache:
                tmp_term = PauliTerm(1.0, key, self.n)
                d2_cache[key] = self.expectation_pauli_sum([tmp_term], product_label) ** 2

        max_k = max(w for (_, w) in dist.keys())

        # 3a) Cumulative MSE^(k) = sum_{max_w > k} p * d^2
        mse_cumulative: Dict[int, float] = {}
        for k in range(max_k + 1):
            tot = 0.0
            for (key, wmax), p in dist.items():
                if wmax > k:
                    tot += p * d2_cache[key]
            mse_cumulative[k] = tot

        # 3b) Incremental MSE(k) = sum_{max_w == k} p * d^2
        mse_per_weight: Dict[int, float] = {}
        for k in range(max_k + 1):
            tot = 0.0
            for (key, wmax), p in dist.items():
                if wmax == k:
                    tot += p * d2_cache[key]
            mse_per_weight[k] = tot

        return {'mse_cumulative': mse_cumulative,
                'mse_per_weight': mse_per_weight,
                'max_weight': max_k}


    # def monte_carlo_samples_nonzero(
    #     self,
    #     init_term: PauliTerm,
    #     M: int,
    #     tol: float = 0.0,
    # ) -> Tuple[
    #     List[PauliTerm],
    #     List[List[bool]],
    #     List[int],
    #     List[float],
    # ]:
    #     """
    #     Generate *M* Monte-Carlo back-propagation paths **conditionally**
    #     on the final Pauli term having non-zero expectation in the
    #     computational |000...0 state.

    #     The interface and return types are identical to
    #     `monte_carlo_samples`.

    #     Every estimator is scaled by the empirical acceptance
    #     probability  伪虃 = M / total_tries, guaranteeing that

    #         np.mean(expectations)  锟???  true expectation          (unbiased)

    #     even though all returned terms satisfy  鉄≒_i, |000...0鉄┾煩 锟??? 0.
    #     """
    #     if init_term.n != self.n:
    #         raise ValueError("Initial term qubit count mismatch")

    #     # Pre-compute reverse gate list once
    #     ops: list[tuple[str, Tuple[int, ...], Tuple]] = []
    #     for instr in reversed(self.qc.data):
    #         name = instr.operation.name
    #         qidx = tuple(self.q2i[q] for q in instr.qubits)
    #         extra: Tuple = ()
    #         if name == "su4" and hasattr(instr.operation, "to_matrix"):
    #             extra = (instr.operation.to_matrix(),)
    #         ops.append((name, qidx, extra))

    #     sampled_last_paulis: list[PauliTerm] = []
    #     weight_exceeded_details: list[list[bool]] = []
    #     total_tries = 0

    #     # Helper to test  鉄≒, |000...0鉄┾煩 锟??? 0  (all X bits must be 0)
    #     n = self.n
    #     x_mask = (1 << n) - 1
    #     def _non_zero_expect(key: int) -> bool:
    #         return (key & x_mask) == 0

    #     with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as exe:
    #         # Continue launching batches until we have M accepted samples
    #         while len(sampled_last_paulis) < M:
    #             # Size of the next batch 锟??? grows with remaining demand
    #             remaining = M - len(sampled_last_paulis)
    #             batch_size = max(_PARALLEL_THRESHOLD, remaining)
    #             args_list = [
    #                 (ops, init_term.key, init_term.coeff, n, tol)
    #                 for _ in range(batch_size)
    #             ]
    #             futures = [exe.submit(
    #                 PauliPropagator._sample_one_path, a
    #             ) for a in args_list]

    #             for fut in as_completed(futures):
    #                 total_tries += 1
    #                 coeff, key, n_qubits, flags = fut.result()

    #                 if not _non_zero_expect(key):
    #                     # Reject 锟??? expectation in |000...0 is zero
    #                     continue

    #                 # Accept
    #                 sampled_last_paulis.append(
    #                     PauliTerm(coeff, key, n_qubits)
    #                 )
    #                 weight_exceeded_details.append(flags)

    #                 if len(sampled_last_paulis) == M:
    #                     break  # stop early when enough accepted samples

    #     # Empirical acceptance probability  伪虃
    #     acc_prob = M / total_tries

    #     # Scale each unbiased coefficient by 伪虃  to retain overall
    #     # unbiasedness after conditioning
    #     for pauli in sampled_last_paulis:
    #         pauli.coeff *= acc_prob

    #     last_pauli_weights = [p.weight() for p in sampled_last_paulis]
    #     coeff_sqs = [abs(p.coeff) ** 2 for p in sampled_last_paulis]

    #     return (
    #         sampled_last_paulis,
    #         weight_exceeded_details,
    #         last_pauli_weights,
    #         coeff_sqs,
    #     )


    # @staticmethod
    # def _sample_one_path(args):
    #     """
    #     Helper function for ProcessPoolExecutor to sample one Monte Carlo path.
    #     This staticmethod can be pickled for multiprocessing.
        
    #     Parameters
    #     ----------
    #     args : tuple
    #         Contains (ops, init_key, init_coeff, n, tol)
    #         where ops is a list of (gate_name, qidx, extra) tuples
            
    #     Returns
    #     -------
    #     Tuple[complex, int, int, List[bool]]
    #         (last_coeff, last_key, n, weight_exceeded_flags) 
    #         where weight_exceeded_flags is a list of 6 booleans indicating 
    #         if weight > [0,1,2,3,4,5,6] was encountered during propagation
    #     """
    #     ops, init_key, init_coeff, n, tol = args
        
    #     current_key = init_key
    #     current_coeff = init_coeff
        
    #     # Track if weight exceeded [0,1,2,3,4,5,6] at any point
    #     weight_thresholds = [1, 2, 3, 4, 5, 6]
    #     weight_exceeded_flags = [False] * 6  # Initialize all as False
        
    #     # Check initial weight
    #     init_term = PauliTerm(1.0, current_key, n)
    #     init_weight = init_term.weight()
    #     for i, threshold in enumerate(weight_thresholds):
    #         if init_weight > threshold:
    #             weight_exceeded_flags[i] = True

    #     for gate_name, qidx, extra in ops:
    #         # Get gate function
    #         gate_func = QuantumGate.get(gate_name)
            
    #         # Create input PauliTerm
    #         inp = PauliTerm(1.0, current_key, n)
            
    #         # Apply gate
    #         if extra:
    #             out_terms = gate_func(inp, *qidx, *extra)
    #         else:
    #             out_terms = gate_func(inp, *qidx)

    #         # Filter non-zero amplitude branches
    #         branches = [(t.key, t.coeff) for t in out_terms]
    #         # branches = [(t.key, t.coeff) for t in out_terms if abs(t.coeff) > tol]
            
    #         if not branches:  # No valid branches, terminate path
    #             break
                
    #         probs = np.array([abs(c)**2 for _, c in branches], dtype=float)
    #         probs /= probs.sum()

    #         idx = np.random.choice(len(branches), p=probs)
    #         current_key, amp = branches[idx]
    #         current_coeff *= amp
            
    #         # Check current step weight against all thresholds
    #         current_term = PauliTerm(1.0, current_key, n)
    #         current_weight = current_term.weight()
    #         for i, threshold in enumerate(weight_thresholds):
    #             if current_weight > threshold:
    #                 weight_exceeded_flags[i] = True

    #     return (current_coeff, current_key, n, weight_exceeded_flags)
