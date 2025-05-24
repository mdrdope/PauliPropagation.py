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


# Threshold for parallel processing and maximum number of worker processes
_PARALLEL_THRESHOLD = 2000
_MAX_WORKERS = 8  # or os.cpu_count()


def _apply_gate_kernel_batch(args):
    """
    Helper function for ProcessPoolExecutor to apply gates to a batch of terms.
    Uses a clean, universal approach that works with any gate type.
    
    Parameters
    ----------
    args : tuple
        Contains (terms_data, gate_name, qidx, extra_args)
        where terms_data is a list of (coeff, key, n) tuples
        
    Returns
    -------
    List[Tuple[complex, int, int]]
        List of (coefficient, key, n) tuples representing output terms
    """
    terms_data, gate_name, qidx, extra_args = args
    
    # Get the gate function
    gate_func = QuantumGate.get(gate_name)
    
    results = []
    for coeff, key, n in terms_data:
        # Create input PauliTerm
        input_term = PauliTerm(coeff, key, n)
        
        # Apply gate (all gates now return List[PauliTerm])
        if extra_args:
            output_terms = gate_func(input_term, *qidx, *extra_args)
        else:
            output_terms = gate_func(input_term, *qidx)
        
        # Convert back to tuple format for efficiency
        for term in output_terms:
            results.append((term.coeff, term.key, term.n))
    
    return results


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

        # Prepare reverse circuit operations
        ops = []
        for instr in reversed(self.qc.data):
            gate_name = instr.operation.name
            qidx = tuple(self.q2i[q] for q in instr.qubits)
            extra = ()
            if gate_name == "su4" and hasattr(instr.operation, "to_matrix"):
                extra = (instr.operation.to_matrix(),)
            ops.append((gate_name, qidx, extra))

        # Set up parallel processing if requested
        executor = ProcessPoolExecutor(max_workers=_MAX_WORKERS) if use_parallel else None
        try:
            for gate_name, qidx, extra_args in tqdm(ops, desc="Propagating", total=len(ops)):
                
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

                # Combine terms with same keys and apply filters
                key_to_coeff: Dict[int, complex] = {}
                
                for coeff, key, n in next_terms_data:
                    # Apply filters
                    if max_weight is not None and weight_of_key(key, self.n) > max_weight:
                        continue
                    if abs(coeff.real) <= tol and abs(coeff.imag) <= tol:
                        continue
                    
                    # Combine terms with same keys
                    key_to_coeff[key] = key_to_coeff.get(key, 0.0) + coeff

                # Filter out small coefficients and create new term list
                current_terms_data = []
                current_terms_objects = []
                for key, coeff in key_to_coeff.items():
                    if abs(coeff.real) > tol or abs(coeff.imag) > tol:
                        current_terms_data.append((coeff, key, self.n))
                        current_terms_objects.append(PauliTerm(coeff, key, self.n))

                history.append(current_terms_objects)

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
    

    @staticmethod
    def _sample_one_path(args):
        """
        Helper function for ProcessPoolExecutor to sample one Monte Carlo path.
        This staticmethod can be pickled for multiprocessing.
        
        Parameters
        ----------
        args : tuple
            Contains (ops, init_key, init_coeff, n, tol)
            where ops is a list of (gate_name, qidx, extra) tuples
            
        Returns
        -------
        Tuple[complex, int, int, List[bool]]
            (last_coeff, last_key, n, weight_exceeded_flags) 
            where weight_exceeded_flags is a list of 6 booleans indicating 
            if weight > [1,2,3,4,5,6] was encountered during propagation
        """
        ops, init_key, init_coeff, n, tol = args
        
        current_key = init_key
        current_coeff = init_coeff
        
        # Track if weight exceeded [1,2,3,4,5,6] at any point
        weight_thresholds = [1, 2, 3, 4, 5, 6]
        weight_exceeded_flags = [False] * 6  # Initialize all as False
        
        # Check initial weight
        init_term = PauliTerm(1.0, current_key, n)
        init_weight = init_term.weight()
        for i, threshold in enumerate(weight_thresholds):
            if init_weight > threshold:
                weight_exceeded_flags[i] = True

        for gate_name, qidx, extra in ops:
            # Get gate function
            gate_func = QuantumGate.get(gate_name)
            
            # Create input PauliTerm
            inp = PauliTerm(1.0, current_key, n)
            
            # Apply gate
            if extra:
                out_terms = gate_func(inp, *qidx, *extra)
            else:
                out_terms = gate_func(inp, *qidx)

            # Filter non-zero amplitude branches
            branches = [(t.key, t.coeff) for t in out_terms]
            # branches = [(t.key, t.coeff) for t in out_terms if abs(t.coeff) > tol]
            
            if not branches:  # No valid branches, terminate path
                break
                
            probs = np.array([abs(c)**2 for _, c in branches], dtype=float)
            probs /= probs.sum()

            idx = np.random.choice(len(branches), p=probs)
            current_key, amp = branches[idx]
            current_coeff *= amp
            
            # Check current step weight against all thresholds
            current_term = PauliTerm(1.0, current_key, n)
            current_weight = current_term.weight()
            for i, threshold in enumerate(weight_thresholds):
                if current_weight > threshold:
                    weight_exceeded_flags[i] = True

        return (current_coeff, current_key, n, weight_exceeded_flags)

    def monte_calro_samples(self,
                          init_term: PauliTerm,
                          M: int,
                          tol: float = 0
                         ) -> Tuple[List[PauliTerm], List[List[bool]], List[int], List[float]]:
        """
        Generate M Monte Carlo backtracking paths, only keeping the final PauliTerms.
        Always uses parallel processing for optimal performance.
        
        Parameters
        ----------
        init_term : PauliTerm
            Initial Pauli term
        M : int
            Number of Monte Carlo paths to generate
        tol : float
            Tolerance for filtering small coefficients
            
        Returns
        -------
        Tuple[List[PauliTerm], List[List[bool]], List[int], List[float]]
            (sampled_last_paulis, weight_exceeded_details, last_pauli_weights, coeff_sqs)
            - sampled_last_paulis: List of final PauliTerms for each path
            - weight_exceeded_details: List of lists, each sublist has 6 booleans indicating 
              if weight > [1,2,3,4,5,6] was encountered during that path's propagation
            - last_pauli_weights: List of weights for each final PauliTerm
            - coeff_sqs: List of |coeff|^2 for each final PauliTerm
        """
        
        if init_term.n != self.n:
            raise ValueError("Initial term qubit count mismatch")
        
        ops = [] # Prepare reverse gate operation sequence
        for instr in reversed(self.qc.data):
            gate_name = instr.operation.name
            qidx = tuple(self.q2i[q] for q in instr.qubits) 
            extra = ()
            if gate_name == "su4" and hasattr(instr.operation, "to_matrix"):
                extra = (instr.operation.to_matrix(),)
            ops.append((gate_name, qidx, extra))

        # Always use parallel processing
        sampled_last_paulis = []
        weight_exceeded_details = []
        
        with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            # Prepare arguments for all paths
            args_list = [(ops, init_term.key, init_term.coeff, self.n, tol) for _ in range(M)]
            
            # Submit all tasks
            futures = [executor.submit(PauliPropagator._sample_one_path, args) for args in args_list]
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=M, desc="MC sampling"):
                coeff, key, n, weight_exceeded_flags = future.result()
                
                # Create final PauliTerm and store results
                last_pauli = PauliTerm(coeff, key, n)
                sampled_last_paulis.append(last_pauli)
                weight_exceeded_details.append(weight_exceeded_flags)

        # Calculate additional required values
        last_pauli_weights = [pauli.weight() for pauli in sampled_last_paulis]
        coeff_sqs = [np.abs(pauli.coeff)**2 for pauli in sampled_last_paulis]

        return sampled_last_paulis, weight_exceeded_details, last_pauli_weights, coeff_sqs
