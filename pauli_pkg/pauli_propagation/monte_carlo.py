# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/monte_carlo.py
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Set, Union, Optional
from qiskit import QuantumCircuit
from .pauli_term  import PauliTerm
from .utils       import weight_of_key
from .gates       import QuantumGate
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Threshold for parallel processing and maximum number of worker processes
_MAX_WORKERS = 8 # or os.cpu_count()

class MonteCarlo:
    """
    Monte Carlo sampling for Pauli observable propagation through quantum circuits.
    
    This class implements Monte Carlo sampling of Pauli paths through quantum circuits
    using bit-mask representations. The sampling results are stored internally and
    not exposed directly to maintain encapsulation.
    
    Attributes
    ----------
    qc : QuantumCircuit
        The quantum circuit to propagate through
    n : int
        Number of qubits in the circuit
    q2i : Dict
        Mapping from qubit objects to indices
    _sampled_last_paulis : List[PauliTerm]
        Internal storage for sampled final Pauli terms
    _weight_exceeded_details : List[List[bool]]
        Internal storage for weight exceeded flags
    _last_pauli_weights : List[int]
        Internal storage for final Pauli weights
    _coeff_sqs : List[float]
        Internal storage for coefficient squares
    """

    def __init__(self, qc: QuantumCircuit):
        """
        Initialize Monte Carlo sampler with a quantum circuit.
        
        Parameters
        ----------
        qc : QuantumCircuit
            The quantum circuit to propagate through
        """
        self.qc  = qc
        self.n   = qc.num_qubits
        self.q2i = {q: i for i, q in enumerate(qc.qubits)}
        
        # Internal storage for sampling results
        self._sampled_last_paulis = []
        self._weight_exceeded_details = []
        self._last_pauli_weights = []
        self._coeff_sqs = []

    @staticmethod
    def _sample_one_path(args): # paper method
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
            (last_coeff_unbiased, last_key, n, weight_exceeded_flags)
            - last_coeff_unbiased: current_coeff / |current_coeff|^2, ensuring E[last_coeff_unbiased * d_current_coeff] = expected current_coeff d_current_coeff
            - last_key: final Pauli key
            - n: number of qubits
            - weight_exceeded_flags: booleans for weight thresholds
        """
        ops, init_key, init_coeff, n, tol = args

        current_key = init_key
        current_coeff = init_coeff


        # Track if weight exceeded [0,1,2,3,4,5,6] at any point
        weight_thresholds = [0, 1, 2, 3, 4, 5, 6]
        weight_exceeded_flags = [False] * 7

        # Check initial weight using more efficient method
        init_weight = weight_of_key(current_key, n)
        for i, threshold in enumerate(weight_thresholds):
            if init_weight > threshold:
                weight_exceeded_flags[i] = True

        # Propagate backwards through the circuit using standard gates
        for gate_name, qidx, extra in ops:
            gate_func = QuantumGate.get(gate_name)
            current_pauli = PauliTerm(1.0, current_key, n)

            if extra:
                out_terms = gate_func(current_pauli, *qidx, *extra)
            else:
                out_terms = gate_func(current_pauli, *qidx)

            branches = [(term.key, term.coeff) for term in out_terms]
            if not branches:
                break

            probs = np.array([abs(c)**2 for _, c in branches], dtype=float)
            probs /= probs.sum()

            idx = np.random.choice(len(branches), p=probs)
            current_key, amp = branches[idx]
            current_coeff *= amp

            # Update weight flags using efficient weight calculation
            current_weight = weight_of_key(current_key, n)
            for i, threshold in enumerate(weight_thresholds):
                if current_weight > threshold:
                    weight_exceeded_flags[i] = True

        # Calculate the unbiased estimator: current_coeff / |current_coeff|^2
        p = abs(current_coeff)**2
        if p > 0:
            last_coeff_unbiased = current_coeff / p
        else:
            last_coeff_unbiased = 0.0

        return (last_coeff_unbiased, current_key, n, weight_exceeded_flags)

    def _save_samples(self, 
                     sample_file: str, 
                     sampled_last_paulis: List[PauliTerm], 
                     weight_exceeded_details: List[List[bool]], 
                     last_pauli_weights: List[int], 
                     coeff_sqs: List[float]) -> None:
        """
        Save Monte Carlo samples to a pickle file.
        
        Parameters
        ----------
        sample_file : str
            Path to the pickle file
        sampled_last_paulis : List[PauliTerm]
            List of final PauliTerms
        weight_exceeded_details : List[List[bool]]
            Weight exceeded flags for each sample
        last_pauli_weights : List[int]
            Final weights for each sample
        coeff_sqs : List[float]
            Coefficient squares for each sample
        """
        sample_data = {'sampled_last_paulis': sampled_last_paulis,
                       'weight_exceeded_details': weight_exceeded_details,
                       'last_pauli_weights': last_pauli_weights,
                       'coeff_sqs': coeff_sqs,
                       'qc_info': {'num_qubits': self.n,
                                   'circuit_depth': len(self.qc.data)}}
        
        # Create directory if needed
        dir_path = os.path.dirname(sample_file)
        if dir_path:  # Only create directory if there's a directory path
            os.makedirs(dir_path, exist_ok=True)
        
        with open(sample_file, 'wb') as f:
            pickle.dump(sample_data, f)

    def _load_samples(self, sample_file: str) -> Tuple[List[PauliTerm], List[List[bool]], List[int], List[float]]:
        """
        Load Monte Carlo samples from a pickle file.
        
        Parameters
        ----------
        sample_file : str
            Path to the pickle file
            
        Returns
        -------
        Tuple[List[PauliTerm], List[List[bool]], List[int], List[float]]
            Loaded sample data in the same format as monte_carlo_samples returns
        """
        if not os.path.exists(sample_file):
            return [], [], [], []
            
        with open(sample_file, 'rb') as f:
            sample_data = pickle.load(f)
            
        # Validate circuit compatibility
        if sample_data['qc_info']['num_qubits'] != self.n:
            raise ValueError(f"Loaded samples have {sample_data['qc_info']['num_qubits']} qubits, "
                           f"but current circuit has {self.n} qubits")
            
        return (sample_data['sampled_last_paulis'],
                sample_data['weight_exceeded_details'],
                sample_data['last_pauli_weights'],
                sample_data['coeff_sqs'])

    def monte_carlo_samples(self,
                          init_term: PauliTerm,
                          M: int,
                          tol: float = 0,
                          sample_file: Optional[str] = None,
                          load_existing: bool = False) -> Tuple[List[PauliTerm], List[List[bool]], List[int], List[float]]:
        """
        Generate M Monte Carlo backtracking paths, only keeping the final PauliTerms.
        Always uses parallel processing for optimal performance.
        
        Supports sample persistence: can save/load samples to/from pickle files for
        incremental sampling across multiple runs.
        
        The results are stored internally and also returned for compatibility.
        
        Parameters
        ----------
        init_term : PauliTerm
            Initial Pauli term
        M : int
            Total number of Monte Carlo paths desired (including loaded samples)
        tol : float
            Tolerance for filtering small coefficients
        sample_file : Optional[str]
            Path to pickle file for saving/loading samples. If None, no persistence.
        load_existing : bool
            Whether to load existing samples from sample_file before generating new ones
            
        Returns
        -------
        Tuple[List[PauliTerm], List[List[bool]], List[int], List[float]]
            (sampled_last_paulis, weight_exceeded_details, last_pauli_weights, coeff_sqs)
            - sampled_last_paulis: List of final PauliTerms for each path
            - weight_exceeded_details: List of lists, each sublist has 7 booleans indicating 
              if weight > [0,1,2,3,4,5,6] was encountered during that path's propagation
            - last_pauli_weights: List of weights for each final PauliTerm
            - coeff_sqs: List of |coeff|^2 for each final PauliTerm
        """
        
        if init_term.n != self.n:
            raise ValueError("Initial term qubit count mismatch")
        
        # Initialize result containers
        sampled_last_paulis = []
        weight_exceeded_details = []
        last_pauli_weights = []
        coeff_sqs = []
        
        # Load existing samples if requested
        if load_existing and sample_file and os.path.exists(sample_file):
            print(f"Loading existing samples from {sample_file}...")
            (loaded_paulis, loaded_weight_details, 
             loaded_weights, loaded_coeff_sqs) = self._load_samples(sample_file)
            
            sampled_last_paulis.extend(loaded_paulis)
            weight_exceeded_details.extend(loaded_weight_details)
            last_pauli_weights.extend(loaded_weights)
            coeff_sqs.extend(loaded_coeff_sqs)
            
            print(f"Loaded {len(loaded_paulis)} existing samples")
        
        # Calculate how many new samples we need
        existing_count = len(sampled_last_paulis)
        new_samples_needed = max(0, M - existing_count)
        
        if new_samples_needed == 0:
            print(f"Already have {existing_count} samples, no new sampling needed")
        else:
            print(f"Generating {new_samples_needed} new samples (total target: {M})")
            
            # Prepare reverse gate operation sequence
            ops = []
            for instr in reversed(self.qc.data):
                gate_name = instr.operation.name
                qidx = tuple(self.q2i[q] for q in instr.qubits) 
                extra = QuantumGate.extract_params(gate_name, instr)
                ops.append((gate_name, qidx, extra))

            # Generate new samples using parallel processing
            new_sampled_paulis = []
            new_weight_details = []
            
            with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as executor:
                # Prepare arguments for new paths
                args_list = [(ops, init_term.key, init_term.coeff, self.n, tol) 
                           for _ in range(new_samples_needed)]
                
                # Submit all tasks
                futures = [executor.submit(MonteCarlo._sample_one_path, args) for args in args_list]
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=new_samples_needed, desc="MC sampling"):
                    coeff, key, n, weight_exceeded_flags = future.result()
                    
                    # Create final PauliTerm and store results
                    last_pauli = PauliTerm(coeff, key, n)
                    new_sampled_paulis.append(last_pauli)
                    new_weight_details.append(weight_exceeded_flags)

            # Add new samples to existing ones
            sampled_last_paulis.extend(new_sampled_paulis)
            weight_exceeded_details.extend(new_weight_details)
            
            # Calculate additional required values for new samples
            new_pauli_weights = [pauli.weight() for pauli in new_sampled_paulis]
            new_coeff_sqs = [np.abs(pauli.coeff)**2 for pauli in new_sampled_paulis]
            
            last_pauli_weights.extend(new_pauli_weights)
            coeff_sqs.extend(new_coeff_sqs)

        # Save samples if requested and new samples were generated
        if sample_file and new_samples_needed > 0:
            print(f"Saving {len(sampled_last_paulis)} samples to {sample_file}...")
            self._save_samples(sample_file, sampled_last_paulis, weight_exceeded_details, 
                             last_pauli_weights, coeff_sqs)

        # Store results internally
        self._sampled_last_paulis = sampled_last_paulis
        self._weight_exceeded_details = weight_exceeded_details
        self._last_pauli_weights = last_pauli_weights
        self._coeff_sqs = coeff_sqs
        self._init_coeff = init_term.coeff

        return sampled_last_paulis, weight_exceeded_details, last_pauli_weights, coeff_sqs

    def estimate_mse_for_truncation(self, 
                                  propagator,
                                  product_label: str,
                                  max_k: int = 6) -> Dict[str, Dict[int, float]]:
        """
        Estimate MSE for truncation using Monte Carlo sampling.
        
        This method computes two types of MSE:
        1. Cumulative MSE: contributions from all paths with weight > k
        2. Layer MSE: contributions from paths with weight == k
        
        Parameters
        ----------
        propagator : object
            Propagator object with expectation_pauli_sum method
        product_label : str
            Product state label for expectation calculation (usually "0"*n)
        max_k : int, optional
            Maximum weight threshold, default is 6
            
        Returns
        -------
        Dict[str, Dict[int, float]]
            Dictionary with two keys:
            - 'cumulative': {k: mse_value} for weight > k cumulative MSE
            - 'layer': {k: mse_value} for weight == k layer MSE
        """
        if not self._sampled_last_paulis:
            raise ValueError("No Monte Carlo sampling data available. Please call monte_carlo_samples method first.")
        
        M = len(self._sampled_last_paulis)
        
        # Pre-compute d_i^2 for each path
        d_list = []
        for pauli_term in self._sampled_last_paulis:
            # Create single-term list for each sampled Pauli term
            single_term = [PauliTerm(1.0, pauli_term.key, pauli_term.n)]
            d_val = propagator.expectation_pauli_sum(single_term, product_label)
            d_list.append(d_val * d_val)
        d_vals = np.array(d_list)
        
        # Convert weight exceeded flags and weight arrays
        flags = np.array(self._weight_exceeded_details, dtype=bool)  # shape = (M, 7)
        weights = np.array(self._last_pauli_weights, dtype=int)      # shape = (M,)
        
        # Calculate MSE and unbiased variance
        results = {'MSE': {}, 'Var': {}}  # Store mean and variance estimators
        # Note: layer MSE can be re-enabled later if needed
        
        # Cumulative MSE and unbiased variance: all paths with weight > k
        z_factor = abs(self._init_coeff)**2   # Z factor ensures unbiasedness

        for k in range(0, max_k + 1):
            if k >= flags.shape[1]:
                # No paths can exceed this weight threshold
                results['MSE'][k] = 0.0
                results['Var'][k] = 0.0
                continue

            # Build the array X_i^{(k)} = Z * d_i^2 * 1_{weight>k}
            X_k = np.where(flags[:, k], d_vals * z_factor, 0.0)

            # Sample mean (unbiased MSE estimator)
            bar_X = X_k.mean()
            results['MSE'][k] = bar_X

            # Unbiased sample variance estimator: 1/(M(M-1)) * Σ (X_i - bar_X)^2
            if M > 1:
                diff_sq_sum = np.sum((X_k - bar_X) ** 2)
                var_unbiased = diff_sq_sum / (M * (M - 1))
            else:
                var_unbiased = 0.0

            results['Var'][k] = var_unbiased

        return results

    # def estimate_Z(args):
    #     ops, init_key, init_coeff, n, tol = args
    #     init_pauli_term = PauliTerm(init_coeff, init_key, n)

        
    #     for gate_name, qidx, extra in ops:
    #         gate_func = QuantumGate.get(gate_name)

    #         if extra:
    #             out_terms = gate_func(init_pauli_term, *qidx, *extra)
    #         else:
    #             out_terms = gate_func(init_pauli_term, *qidx)
    #         # ``out_terms`` is a list of ``PauliTerm`` objects.  The quantity
    #         # ``Z`` is defined as the sum of squared magnitudes of the
    #         # coefficients of all PauliTerms in ``out_terms``.

    #         break

    # def print_mse_summary(self, 
    #                      mse_results: Dict[str, Dict[int, float]], 
    #                      max_k: int = 6) -> None:
    #     """
    #     Print a human-readable summary of the Monte-Carlo estimated
    #     mean-squared-error (MSE).
    #
    #     Parameters
    #     ----------
    #     mse_results : Dict[str, Dict[int, float]]
    #         Output produced by :py:meth:`estimate_mse_for_truncation`.
    #     max_k : int, optional
    #         Largest weight :math:`k` to be shown in the summary. Defaults to
    #         ``6``.
    #     """
    #     print("Monte Carlo MSE summary:")
    #     print("=" * 50)
        
    #     print("\nCumulative MSE (weight > k):")
    #     for k in range(0, max_k + 1):
    #         mse_val = mse_results['cumulative'].get(k, 0.0)
    #         print(f"   weight > {k}: {mse_val:.6e}")
        
    #     print("\nLayer MSE (weight == k):")
    #     for k in range(0, max_k + 1):
    #         mse_val = mse_results['layer'].get(k, 0.0)
    #         print(f"   weight == {k}: {mse_val:.6e}")

    # def get_sample_count(self) -> int:
    #     """
    #     Get the number of stored samples.
        
    #     Returns
    #     -------
    #     int
    #         Number of Monte Carlo samples stored
    #     """
    #     return len(self._sampled_last_paulis)

    # def get_weight_exceeded_statistics(self, threshold: int) -> float:
    #     """
    #     Get the fraction of paths that exceeded a given weight threshold.
        
    #     Parameters
    #     ----------
    #     threshold : int
    #         Weight threshold to check (0-6)
            
    #     Returns
    #     -------
    #     float
    #         Fraction of paths that exceeded the threshold
    #     """
    #     if not self._weight_exceeded_details:
    #         return 0.0
        
    #     if threshold < 0 or threshold >= len(self._weight_exceeded_details[0]):
    #         raise ValueError(f"Threshold must be between 0 and {len(self._weight_exceeded_details[0])-1}")
        
    #     exceeded_count = sum(1 for flags in self._weight_exceeded_details if flags[threshold])
    #     return exceeded_count / len(self._weight_exceeded_details)

    # def get_average_final_weight(self) -> float:
    #     """
    #     Get the average weight of final Pauli terms.
        
    #     Returns
    #     -------
    #     float
    #         Average weight of final Pauli terms
    #     """
    #     if not self._last_pauli_weights:
    #         return 0.0
    #     return np.mean(self._last_pauli_weights)

    # def get_coefficient_statistics(self) -> Dict[str, float]:
    #     """
    #     Get statistics about the coefficient magnitudes.
        
    #     Returns
    #     -------
    #     Dict[str, float]
    #         Dictionary containing mean, std, min, max of |coeff|^2
    #     """
    #     if not self._coeff_sqs:
    #         return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
    #     coeff_array = np.array(self._coeff_sqs)
    #     return {"mean": np.mean(coeff_array),
    #             "std": np.std(coeff_array),
    #             "min": np.min(coeff_array),
    #             "max": np.max(coeff_array)}
            

