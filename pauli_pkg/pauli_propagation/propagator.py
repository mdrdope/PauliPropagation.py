# pauli_pkg/pauli_propagation/propagator.py

import numpy as np
from typing import List
from qiskit.quantum_info import Pauli, Statevector
from .pauli_term import PauliTerm
from qiskit import QuantumCircuit
from .gates import QuantumGate
from collections import defaultdict

class PauliPropagator:
    """Back-propagate an observable through a QuantumCircuit.

    This class implements the back-propagation of Pauli observables through a quantum circuit.
    It transforms observables according to the gates in the circuit, working from output to input.
    
    Gate rules are resolved via QuantumGate.get(name).
    Optional argument *max_weight* discards Pauli terms whose
    weight (number of non-identity letters) exceeds max_weight.
    
    Parameters
    ----------
    qc : QuantumCircuit
        The quantum circuit through which to propagate the observable.
        
    Attributes
    ----------
    qc : QuantumCircuit
        The quantum circuit being used for propagation.
    q2i : dict
        Mapping from qubit objects to their indices in the circuit.
    """

    _SINGLE_Q_EV = {"0": {"I":1.0, "X":0.0, "Y":0.0, "Z": 1.0},
                    "1": {"I":1.0, "X":0.0, "Y":0.0, "Z":-1.0},
                    "+": {"I":1.0, "X":1.0, "Y":0.0, "Z": 0.0},
                    "-": {"I":1.0, "X":-1.0,"Y":0.0, "Z": 0.0},
                    "r": {"I":1.0, "X":0.0, "Y":1.0, "Z": 0.0},
                    "l": {"I":1.0, "X":0.0, "Y":-1.0,"Z": 0.0}}

    def __init__(self, qc: QuantumCircuit):
        self.qc  = qc
        self.q2i = {}
        for i, q in enumerate(qc.qubits):
            self.q2i[q] = i

    @staticmethod
    def _pauli_weight(p: Pauli) -> int:
        """Calculate the weight of a Pauli operator.
        
        The weight is defined as the number of non-identity terms in the Pauli string.
        
        Parameters
        ----------
        p : Pauli
            The Pauli operator to calculate the weight for.
            
        Returns
        -------
        int
            The number of non-identity terms in the Pauli operator.
        """
        return int(np.count_nonzero(p.x | p.z))
    
    @staticmethod
    def _merge_like_terms(paths: List[PauliTerm], tol: float = 1e-12) -> List[PauliTerm]:
        """Merge Pauli paths with identical Pauli labels by summing their coefficients.
        
        Parameters
        ----------
        paths : List[PauliTerm]
            The list of Pauli paths to merge.
        tol : float, optional
            Tolerance threshold for filtering out terms with near-zero coefficients,
            by default 1e-12.
            
        Returns
        -------
        List[PauliTerm]
            A new list where identical Pauli operators have been merged by
            summing their coefficients, and terms with coefficients below
            the tolerance threshold have been removed.
        """
        bucket: defaultdict[str, complex] = defaultdict(complex) # 
        for p in paths:
            bucket[p.pauli.to_label()] += p.coeff # merge like terms

        merged: List[PauliTerm] = []
        for label, coeff in bucket.items():
            if abs(coeff) > tol:                          # filter zeros
                merged.append(PauliTerm(coeff, Pauli(label)))

        return merged

    # def propagate(self, observable: PauliTerm, max_weight: int | None = None) -> List[List[PauliTerm]]:
    #     """Propagate an observable backwards through the quantum circuit.
        
    #     This method transforms the observable by applying the inverse of each gate
    #     in the circuit, working from the end of the circuit to the beginning.
        
    #     Parameters
    #     ----------
    #     observable : PauliTerm
    #         The observable to propagate through the circuit.
    #     max_weight : int or None, optional
    #         Maximum weight of Pauli terms to keep during propagation. Terms with
    #         higher weights will be discarded. If None, all terms are kept.
            
    #     Returns
    #     -------
    #     List[List[PauliTerm]]
    #         A list of Pauli path lists, where each inner list represents the
    #         observable at a different layer of the circuit. The first element
    #         corresponds to the original observable, and the last element
    #         corresponds to the observable after full propagation through the circuit.
    #     """
    #     paths: List[PauliTerm] = [observable]
    #     per_layer: List[List[PauliTerm]] = [paths]      # O_N

    #     # walk circuit right �? left
    #     # instr.instr is the name of the gate, 
    #     # we use it to find the corresponding rule in  "QuantumGate"'s registry
    #     # qc.data is a list of quantum gate in the circuit
    #     for instr in reversed(self.qc.data):
    #         gname = instr.operation.name # updated
    #         rule   = QuantumGate.get(instr.operation.name) # obtain the rule for the gate, U^DAGGER p U
            
    #         qidx = [] # the qubits involved in current instr/rule operation
    #         for q in instr.qubits:
    #             qidx.append(self.q2i[q])

    #         # next_paths: List[PauliTerm] = []
    #         # for p in paths:
    #         #     next_paths.extend(rule(p, *qidx)) # *qidx is unpacking the qubit indices for the rule function

    #         # ---------- NEW: auto-append matrix for 'su4' --------------
    #         extra_args = ()
    #         if gname == "su4" and hasattr(instr.operation, "to_matrix"):
    #             extra_args = (instr.operation.to_matrix(),)

    #         next_paths: List[PauliTerm] = []
    #         for p in paths:
    #             next_paths.extend(rule(p, *qidx, *extra_args))

    #         # optional weight truncation
    #         if max_weight is not None:
    #             next_paths = [p for p in next_paths
    #                           if self._pauli_weight(p.pauli) <= max_weight]
                
    #         next_paths = self._merge_like_terms(next_paths) # merge like terms
            
    #         next_paths.sort(key=lambda p: (self._pauli_weight(p.pauli),p.coeff.real < 0)) # tidy ordering
    #         paths = next_paths
    #         per_layer.append(paths)
    #     return per_layer

    def propagate( # updated version, previous version is correct
        self,
        observable: PauliTerm,
        max_weight: int | None = None,
        tol: float = 1e-12,
    ) -> List[List[PauliTerm]]:
        r"""
        Back-propagate *observable* through self.qc and return a per-layer
        history (outer�\most list index grows towards earlier layers).
        Uses an in-place dict {label: coeff} to accumulate terms, avoiding
        list-building + post-merge overhead.
        """
        #  0  initial dict 
        paths_dict: dict[str, complex] = {observable.pauli.to_label(): observable.coeff}
        per_layer: List[List[PauliTerm]] = [[observable]]

        # walk circuit from right to left 
        for instr in reversed(self.qc.data):
            gname = instr.operation.name
            rule  = QuantumGate.get(gname)

            qidx = tuple(self.q2i[q] for q in instr.qubits)      # qubit indices
            extra_args = ()
            if gname == "su4" and hasattr(instr.operation, "to_matrix"):
                extra_args = (instr.operation.to_matrix(),)

            # next_paths accumulates all new terms of this layer
            next_paths: dict[str, complex] = {}

            for label, coeff in paths_dict.items():
                in_term = PauliTerm(coeff, Pauli(label))
                for out_term in rule(in_term, *qidx, *extra_args):
                    # optional weight truncation
                    if max_weight is not None and \
                       self._pauli_weight(out_term.pauli) > max_weight:
                        continue

                    out_label = out_term.pauli.to_label()
                    next_paths[out_label] = (next_paths.get(out_label, 0.0) + out_term.coeff)

            # prune near-zero coefficients
            paths_dict = {l: c for l, c in next_paths.items() if abs(c) > tol}

            # convert to PauliTerm list for user-facing history (unsorted)
            layer_terms = [PauliTerm(c, Pauli(l)) for l, c in paths_dict.items()]
            per_layer.append(layer_terms)

        return per_layer

    def expectation_pauli_sum(self,
                              pauli_sum: list,
                              product_label: str) -> float:
            
            """
            Given a list of PauliTerm objects (each has .coeff, .pauli),
            and a product state specified by `product_label` (e.g. '+0-', etc.),
            return the sum of coefficients times Tr[rho * Pauli].
            and a product state specified by `product_label` (e.g. '+0-', etc.),
            return the sum of coefficients times Tr[rho * Pauli].
            
            This uses the single-qubit dictionary `_SINGLE_Q_EV` to evaluate
            each multi-qubit Pauli operator's expectation under a product state.
            
            Parameters
            ----------
            pauli_sum : list of PauliTerm
                Each PauliTerm has attributes: .coeff (complex) and .pauli (Pauli).
            product_label : str
                A string specifying the single-qubit state of each qubit,
                in the order [q0, q1, ...].
                For example "0+" or "1?r" etc.
            
            Returns
            -------
            float
                The resulting expectation value, i.e. sum_i coeff_i * Tr[rho * P_i].
            """
            
            total_val = 0.0 # We'll sum up alpha_i * (prod of single-qubit expectations).
            
            # Qiskit's Pauli label has qubit-0 as the rightmost character.
            # So we can do reversed or something if we want q0 to be index 0.
            n = len(product_label)  # number of qubits
            for term in pauli_sum: # pauli_sum = [α1 P1, ..., αn Pn], n is number of terms
                
                pauli_str = term.pauli.to_label() # term.pauli.to_label() e.g. "IXZ: qubit-2=I, qubit-1=X, qubit-0=Z
                # We'll accumulate the product of single-qubit expectations
                factor = 1.0
                for qubit_index in range(n):
                    # the letter for this qubit in the Pauli operator:
                    # qubit-0 corresponds to pauli_str[-1], qubit_index �? pauli_str[n-1 - qubit_index]

                    letter = pauli_str[n - 1 - qubit_index]
                    # letter = pauli_str[qubit_index]

                    # state_symbol = product_label[qubit_index] # single-qubit state symbol, e.g. '0' or '+' !!!
                    state_symbol = product_label[n - 1 - qubit_index]
                    
                    factor *= self._SINGLE_Q_EV[state_symbol][letter] # multiply by <state| letter |state>

                    if factor == 0.0:        
                        break                
                    
                total_val += term.coeff * factor
            
            return float(total_val.real)  # or just float(total_val) if real guaranteed
 