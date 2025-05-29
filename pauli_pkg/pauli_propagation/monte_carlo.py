# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/monte_carlo.py
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
            - last_coeff_unbiased: Φ_γ / |Φ_γ|^2, so that E[last_coeff_unbiased * d_γ] = Σ Φ_γ d_γ
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

        # Check initial weight
        init_term = PauliTerm(1.0, current_key, n)
        init_weight = init_term.weight()
        for i, threshold in enumerate(weight_thresholds):
            if init_weight > threshold:
                weight_exceeded_flags[i] = True

        # Propagate backwards through the circuit
        for gate_name, qidx, extra in ops:
            gate_func = QuantumGate.get(gate_name)
            inp = PauliTerm(1.0, current_key, n)

            if extra:
                out_terms = gate_func(inp, *qidx, *extra)
            else:
                out_terms = gate_func(inp, *qidx)

            branches = [(t.key, t.coeff) for t in out_terms]
            if not branches:
                break

            probs = np.array([abs(c)**2 for _, c in branches], dtype=float)
            probs /= probs.sum()

            idx = np.random.choice(len(branches), p=probs)
            current_key, amp = branches[idx]
            current_coeff *= amp

            # Update weight flags
            current_weight = PauliTerm(1.0, current_key, n).weight()
            for i, threshold in enumerate(weight_thresholds):
                if current_weight > threshold:
                    weight_exceeded_flags[i] = True

        # Compute unbiased estimator: Φ_γ / |Φ_γ|^2
        p = abs(current_coeff)**2
        if p > 0:
            last_coeff_unbiased = current_coeff / p
        else:
            last_coeff_unbiased = 0.0

        return (last_coeff_unbiased, current_key, n, weight_exceeded_flags)

    def monte_carlo_samples(self,
                          init_term: PauliTerm,
                          M: int,
                          tol: float = 0
                         ) -> Tuple[List[PauliTerm], List[List[bool]], List[int], List[float]]:
        """
        Generate M Monte Carlo backtracking paths, only keeping the final PauliTerms.
        Always uses parallel processing for optimal performance.
        
        The results are stored internally and also returned for compatibility.
        
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
            - weight_exceeded_details: List of lists, each sublist has 7 booleans indicating 
              if weight > [0,1,2,3,4,5,6] was encountered during that path's propagation
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
            futures = [executor.submit(MonteCarlo._sample_one_path, args) for args in args_list]
            
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

        # Store results internally
        self._sampled_last_paulis = sampled_last_paulis
        self._weight_exceeded_details = weight_exceeded_details
        self._last_pauli_weights = last_pauli_weights
        self._coeff_sqs = coeff_sqs

        return sampled_last_paulis, weight_exceeded_details, last_pauli_weights, coeff_sqs


    def estimate_mse_for_truncation(self, 
                                  propagator,
                                  product_label: str,
                                  max_k: int = 6) -> Dict[str, Dict[int, float]]:
        """
        使用Monte Carlo采样估计截断误差的MSE。
        
        这个方法计算两种类型的MSE：
        1. 累积MSE：权重 > k 的所有路径的贡献
        2. 单层MSE：权重 == k 的路径的贡献
        
        Parameters
        ----------
        propagator : object
            具有expectation_pauli_sum方法的传播器对象
        product_label : str
            计算期望值时使用的产品标签（通常是"0"*n）
        max_k : int, optional
            最大权重阈值，默认为6
            
        Returns
        -------
        Dict[str, Dict[int, float]]
            包含两个键的字典：
            - 'cumulative': {k: mse_value} 权重 > k 的累积MSE
            - 'layer': {k: mse_value} 权重 == k 的单层MSE
        """
        if not self._sampled_last_paulis:
            raise ValueError("没有可用的Monte Carlo采样数据。请先调用monte_carlo_samples方法。")
        
        M = len(self._sampled_last_paulis)
        
        # 1) 预先计算每条路径的 d_i^2
        d_list = []
        for pauli_term in self._sampled_last_paulis:
            # 为每个采样的Pauli项创建单项列表
            single_term = [PauliTerm(1.0, pauli_term.key, pauli_term.n)]
            d_val = propagator.expectation_pauli_sum(single_term, product_label)
            d_list.append(d_val * d_val)
        d_vals = np.array(d_list)
        
        # 2) 转换权重超出详情和权重数组
        flags = np.array(self._weight_exceeded_details, dtype=bool)  # shape = (M, 7)
        weights = np.array(self._last_pauli_weights, dtype=int)      # shape = (M,)
        
        # 3) 计算MSE
        results = {'cumulative': {},  # 累积MSE：权重 > k
                   'layer': {}}     # 单层MSE：权重 == k
        
        # 累积MSE：权重 > k 的所有路径
        for k in range(0, max_k + 1):
            if k < flags.shape[1]:
                mse_cumulative = d_vals[flags[:, k]].sum() / M
                results['cumulative'][k] = mse_cumulative
            else:
                results['cumulative'][k] = 0.0
        
        # 单层MSE：权重 == k 的路径
        for k in range(0, max_k + 1):
            mask_eq = (weights == k)
            mse_layer = d_vals[mask_eq].sum() / M
            results['layer'][k] = mse_layer
        
        return results

    # def print_mse_summary(self, 
    #                      mse_results: Dict[str, Dict[int, float]], 
    #                      max_k: int = 6) -> None:
    #     """
    #     打印MSE结果的摘要。
        
    #     Parameters
    #     ----------
    #     mse_results : Dict[str, Dict[int, float]]
    #         estimate_mse_for_truncation方法返回的结果
    #     max_k : int, optional
    #         最大权重阈值，默认为6
    #     """
    #     print("Monte Carlo MSE 估计结果:")
    #     print("=" * 50)
        
    #     print("\n累积MSE (权重 > k 的路径):")
    #     for k in range(0, max_k + 1):
    #         mse_val = mse_results['cumulative'].get(k, 0.0)
    #         print(f"  权重 > {k}: {mse_val:.6e}")
        
    #     print("\n单层MSE (权重 == k 的路径):")
    #     for k in range(0, max_k + 1):
    #         mse_val = mse_results['layer'].get(k, 0.0)
    #         print(f"  权重 == {k}: {mse_val:.6e}")

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
            

