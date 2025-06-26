# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/propagator.py
import numpy as np
from typing import List, Dict, Tuple, Set, Union
from qiskit import QuantumCircuit
from .pauli_term  import PauliTerm
from .utils       import weight_of_key
from .gates       import QuantumGate
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Threshold for parallel processing and maximum number of worker processes
_PARALLEL_THRESHOLD = 2000
_MAX_WORKERS = 8 # or os.cpu_count()

def _apply_gate_kernel_batch(args):
    """
    Helper function for ProcessPoolExecutor to apply gates to a batch of terms.
    
    Parameters
    ----------
    args : tuple
        Contains (terms_data, gate_name, qidx, extra_args)
        terms_data: List of (coeff, key, n) tuples
        gate_name: str, name of the gate
        qidx: tuple of qubit indices
        extra_args: tuple of extra parameters
        
    Returns
    -------
    List[Tuple[complex, int, int]]
        List of output tuples from gate application
    """
    terms_data, gate_name, qidx, extra_args = args
    
    # Get the gate function  
    gate_func = QuantumGate.get(gate_name)
    
    results = []
    for term_tuple in terms_data:
        # Convert tuple to PauliTerm, apply gate, then convert back to tuples
        coeff, key, n = term_tuple
        pauli_term = PauliTerm(coeff, key, n)
        
        if extra_args:
            output_terms = gate_func(pauli_term, *qidx, *extra_args)
        else:
            output_terms = gate_func(pauli_term, *qidx)
        
        # Convert back to tuples
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
    _EXP_TABLE[_STATE_IDX['0'],:] = [[1,0],[1,0]]  # |0> state
    _EXP_TABLE[_STATE_IDX['1'],:] = [[1,0],[-1,0]] # |1> state
    _EXP_TABLE[_STATE_IDX['+'],:] = [[1,1],[0,0]]  # |+> state
    _EXP_TABLE[_STATE_IDX['-'],:] = [[1,-1],[0,0]] # |-> state
    _EXP_TABLE[_STATE_IDX['r'],:] = [[1,0],[0,1]]  # |r> state
    _EXP_TABLE[_STATE_IDX['l'],:] = [[1,0],[0,-1]] # |l> state

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

    def _tuples_to_pauli_terms(self, terms_data: List[Tuple[complex, int, int]]) -> List[PauliTerm]:
        """
        Convert list of tuples to PauliTerm objects.
        Only used when we need to return PauliTerm objects to the user.
        
        Parameters
        ----------
        terms_data : List[Tuple[complex, int, int]]
            List of (coefficient, key, n) tuples
            
        Returns
        -------
        List[PauliTerm]
            List of PauliTerm objects
        """
        return [PauliTerm(coeff, key, n) for coeff, key, n in terms_data]

    def propagate(self,
                  observable: PauliTerm,
                  max_weight: int | None = None,
                  use_parallel: bool = False,
                  tol: float = 1e-10,
                 ) -> List[List[PauliTerm]]:
        """
        Propagate a Pauli observable through the circuit.
        
        This method performs exact propagation of a Pauli observable through the circuit.
        
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

        # Work with tuple representation internally
        current_terms_data = [(observable.coeff, observable.key, observable.n)]
        
        # Store history as tuples - only convert to PauliTerm at the very end
        history_tuples: List[List[Tuple[complex, int, int]]] = [[(observable.coeff, observable.key, observable.n)]]

        # Prepare reverse circuit operations
        ops = []
        for instr in reversed(self.qc.data):
            gate_name = instr.operation.name
            qidx = tuple(self.q2i[q] for q in instr.qubits)
            extra = QuantumGate.extract_params(gate_name, instr)
            ops.append((gate_name, qidx, extra))

        # Set up parallel processing if requested
        executor = ProcessPoolExecutor(max_workers=_MAX_WORKERS) if use_parallel else None
        try:
            for gate_name, qidx, extra_args in tqdm(ops, desc=f"Propagating, max weight: {max_weight}", total=len(ops)):
                
                # For small numbers of terms or if parallel is disabled, process sequentially
                if not use_parallel or len(current_terms_data) < _PARALLEL_THRESHOLD:
                    next_terms_data = _apply_gate_kernel_batch((current_terms_data, gate_name, qidx, extra_args))
                else:
                    # Split terms into chunks for parallel processing
                    chunk_size = max(1, len(current_terms_data) // _MAX_WORKERS)
                    chunks = [current_terms_data[i:i+chunk_size] 
                            for i in range(0, len(current_terms_data), chunk_size)]
                    
                    # Process chunks in parallel
                    future_to_chunk = {executor.submit(_apply_gate_kernel_batch, (chunk, gate_name, qidx, extra_args)): chunk 
                                       for chunk in chunks}
                    
                    next_terms_data = []
                    for future in as_completed(future_to_chunk):
                        chunk_results = future.result()
                        next_terms_data.extend(chunk_results)

                # Combine terms with same keys and apply filters - ALL using tuples
                key_to_coeff: Dict[int, complex] = {}
                
                for coeff, key, n in next_terms_data:
                    # Apply filters BEFORE combining (more efficient)
                    if max_weight is not None and weight_of_key(key, self.n) > max_weight:
                        continue
                    if abs(coeff.real) <= tol and abs(coeff.imag) <= tol:
                        continue
                    
                    # Combine terms with same keys
                    key_to_coeff[key] = key_to_coeff.get(key, 0.0) + coeff

                # Filter out small coefficients and rebuild tuple list
                current_terms_data = []
                for key, coeff in key_to_coeff.items():
                    if abs(coeff.real) > tol or abs(coeff.imag) > tol:
                        current_terms_data.append((coeff, key, self.n))

                # Store as tuples in history
                history_tuples.append(current_terms_data.copy())
                
                # Early termination if no terms remain
                if not current_terms_data:
                    break

        finally:
            if executor:
                executor.shutdown()

        # ONLY at the very end: convert all tuple history to PauliTerm objects for user
        # This is the ONLY place where PauliTerm objects are created in the entire propagation
        history: List[List[PauliTerm]] = [self._tuples_to_pauli_terms(layer_tuples) 
                                          for layer_tuples in history_tuples]

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
            Product state label (e.g. '0+1--')
            
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

    # def analytical_truncation_mse(self,
    #                             init_term: PauliTerm,
    #                             product_label: str = None
    #                             ) -> Dict[str, Union[Dict[int, float], int]]:
    #     """
    #     Compute the exact, instantaneous truncation MSE (Mean Squared Error)
    #     by dynamic programming over Pauli-propagation paths.

    #     We treat each reverse-propagation path state as a tuple
    #     (pauli_key, max_weight_so_far), where max_weight_so_far records
    #     the largest Pauli weight encountered anywhere along the path.

    #     Parameters
    #     ----------
    #     init_term : PauliTerm
    #         The starting Pauli term (with coeff=1 and its encoded key)
    #     product_label : str, optional
    #         Product-state label for expectation (e.g. "000...0"); defaults to all-zeros

    #     Returns
    #     -------
    #     Dict[str, Union[Dict[int, float], int]]
    #         Dictionary containing:
    #         - 'mse_cumulative': {k: MSE^(k) = sum_{max_w > k} p * d^2}
    #         - 'mse_per_weight': {k: MSE(k) = sum_{max_w == k} p * d^2}
    #         - 'max_weight': int, maximum weight encountered overall
    #     """
    #     if init_term.n != self.n:
    #         raise ValueError("Initial PauliTerm qubit count mismatch")

    #     # Default to the |0...0> product state
    #     if product_label is None:
    #         product_label = "0" * self.n

    #     # Initialize key and its initial weight
    #     init_key = init_term.key
    #     init_wt = init_term.weight()

    #     # Build the reverse-propagation gate list
    #     ops: List[Tuple[str, Tuple[int, ...], Tuple]] = []
    #     for instr in reversed(self.qc.data):
    #         name = instr.operation.name
    #         qidx = tuple(self.q2i[q] for q in instr.qubits)
    #         extra = QuantumGate.extract_params(name, instr)
    #         ops.append((name, qidx, extra))

    #     # Probability DP over states (pauli_key, max_weight_so_far)
    #     # dist[(key, wmax)] = cumulative probability sum |alpha_i|^2 of reaching that state
    #     dist: Dict[Tuple[int, int], float] = {(init_key, init_wt): 1.0}

    #     for name, qidx, extra in tqdm(ops, desc="DP over states"):
    #         gate_func = QuantumGate.get(name)
    #         new_dist: Dict[Tuple[int, int], float] = {}

    #         for (key, wmax), prob in dist.items():
    #             term = PauliTerm(1.0, key, self.n)
    #             # Generate all outgoing branches through this gate
    #             branches = (gate_func(term, *qidx, *extra)
    #                         if extra else gate_func(term, *qidx))

    #             for branch in branches:
    #                 p_branch = prob * (abs(branch.coeff) ** 2)
    #                 new_wmax = max(wmax, branch.weight())
    #                 new_key = branch.key
    #                 new_dist[(new_key, new_wmax)] = new_dist.get((new_key, new_wmax), 0.0) + p_branch

    #         dist = new_dist

    #     # Precompute d^2 = (<P, |0^n>)^2 for each final key
    #     d2_cache: Dict[int, float] = {}
    #     for (key, _) in dist.keys():
    #         if key not in d2_cache:
    #             tmp_term = PauliTerm(1.0, key, self.n)
    #             d2_cache[key] = self.expectation_pauli_sum([tmp_term], product_label) ** 2

    #     max_k = max(w for (_, w) in dist.keys())

    #     # Cumulative MSE^(k) = sum_{max_w > k} p * d^2
    #     mse_cumulative: Dict[int, float] = {}
    #     for k in range(max_k + 1):
    #         tot = 0.0
    #         for (key, wmax), p in dist.items():
    #             if wmax > k:
    #                 tot += p * d2_cache[key]
    #         mse_cumulative[k] = tot

    #     # Incremental MSE(k) = sum_{max_w == k} p * d^2
    #     mse_per_weight: Dict[int, float] = {}
    #     for k in range(max_k + 1):
    #         tot = 0.0
    #         for (key, wmax), p in dist.items():
    #             if wmax == k:
    #                 tot += p * d2_cache[key]
    #         mse_per_weight[k] = tot

    #     return {'mse_cumulative': mse_cumulative,
    #             'mse_per_weight': mse_per_weight,
    #             'max_weight': max_k}

    # def propagate_fast(self,
    #                    observable: PauliTerm,
    #                    max_weight: int | None = None,
    #                    use_parallel: bool = False,
    #                    tol: float = 1e-10,
    #                   ) -> List[Tuple[complex, int, int]]:
    #     """
    #     Fast propagation that returns only the final layer as tuples.
    #     Use this when you only need the final result and want maximum performance.
        
    #     Parameters
    #     ----------
    #     observable : PauliTerm
    #         Initial Pauli observable to propagate
    #     max_weight : int | None
    #         Maximum weight of Pauli terms to keep (None for no limit)
    #     use_parallel : bool
    #         Whether to use parallel processing
    #     tol : float
    #         Tolerance for discarding small coefficients
            
    #     Returns
    #     -------
    #     List[Tuple[complex, int, int]]
    #         Final Pauli terms as (coefficient, key, n) tuples
    #     """
    #     if observable.n != self.n:
    #         raise ValueError("Observable qubit count mismatch")

    #     # Work with tuples internally for efficiency
    #     current_terms_data = [(observable.coeff, observable.key, observable.n)]

    #     # Prepare reverse circuit operations
    #     ops = []
    #     for instr in reversed(self.qc.data):
    #         gate_name = instr.operation.name
    #         qidx = tuple(self.q2i[q] for q in instr.qubits)
    #         extra = QuantumGate.extract_params(gate_name, instr)
    #         ops.append((gate_name, qidx, extra))

    #     # Set up parallel processing if requested
    #     executor = ProcessPoolExecutor(max_workers=_MAX_WORKERS) if use_parallel else None
    #     try:
    #         for gate_name, qidx, extra_args in tqdm(ops, desc=f"Fast propagating, max weight: {max_weight}", total=len(ops)):
                
    #             # Apply gates
    #             if not use_parallel or len(current_terms_data) < _PARALLEL_THRESHOLD:
    #                 next_terms_data = _apply_gate_kernel_batch((current_terms_data, gate_name, qidx, extra_args))
    #             else:
    #                 # Parallel processing
    #                 chunk_size = max(1, len(current_terms_data) // _MAX_WORKERS)
    #                 chunks = [current_terms_data[i:i+chunk_size] 
    #                         for i in range(0, len(current_terms_data), chunk_size)]
                    
    #                 future_to_chunk = {executor.submit(_apply_gate_kernel_batch, (chunk, gate_name, qidx, extra_args)): chunk 
    #                                    for chunk in chunks}
                    
    #                 next_terms_data = []
    #                 for future in as_completed(future_to_chunk):
    #                     chunk_results = future.result()
    #                     next_terms_data.extend(chunk_results)

    #             # Merge and filter
    #             key_to_coeff: Dict[int, complex] = {}
                
    #             for coeff, key, n in next_terms_data:
    #                 if max_weight is not None and weight_of_key(key, self.n) > max_weight:
    #                     continue
    #                 if abs(coeff.real) <= tol and abs(coeff.imag) <= tol:
    #                     continue
                    
    #                 key_to_coeff[key] = key_to_coeff.get(key, 0.0) + coeff

    #             # Rebuild current terms
    #             current_terms_data = []
    #             for key, coeff in key_to_coeff.items():
    #                 if abs(coeff.real) > tol or abs(coeff.imag) > tol:
    #                     current_terms_data.append((coeff, key, self.n))

    #             # Early termination if no terms remain
    #             if not current_terms_data:
    #                 break

    #     finally:
    #         if executor:
    #             executor.shutdown()

    #     return current_terms_data

    # def expectation_tuples(self,
    #                       terms_data: List[Tuple[complex, int, int]],
    #                       product_label: str) -> float:
    #     """
    #     Calculate expectation value directly from tuple data.
    #     This is the most efficient way to compute expectation values.
        
    #     Parameters
    #     ----------
    #     terms_data : List[Tuple[complex, int, int]]
    #         List of (coefficient, key, n) tuples
    #     product_label : str
    #         Product state label (e.g. '0+1-')
            
    #     Returns
    #     -------
    #     float
    #         Expectation value
            
    #     Raises
    #     ------
    #     ValueError
    #         If label length doesn't match qubit count
    #     """
    #     if len(product_label) != self.n:
    #         raise ValueError("Label length mismatch")

    #     if not terms_data:
    #         return 0.0

    #     # Pre-compute state indices once
    #     state_idxs = np.array([self._STATE_IDX[ch] for ch in product_label[::-1]])
        
    #     # Extract coefficients and keys directly from tuples
    #     coeffs = np.array([coeff for coeff, _, _ in terms_data], dtype=complex)
    #     keys = np.array([key for _, key, _ in terms_data], dtype=object)

    #     # Calculate expectation value using vectorized computation
    #     return float(self._expect_keys(coeffs, keys,
    #                                  state_idxs, self.n,
    #                                  self._EXP_TABLE))