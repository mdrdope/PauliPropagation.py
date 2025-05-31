# -*- coding: utf-8 -*-

# pauli_pkg/pauli_propagation/pauli_term.py
from dataclasses import dataclass
from .utils import decode_pauli, weight_of_key, decode_pauli_cached

@dataclass(slots=True)
class PauliTerm:
    """
    A class representing a single Pauli term in a quantum operator.
    
    This class stores a Pauli operator term in a compact bit-mask representation,
    along with its coefficient and the number of qubits it acts on.
    
    Attributes
    ----------
    coeff : complex
        The complex coefficient of the Pauli term
    key : int
        Bit-mask representation of the Pauli operator
        - Lower n bits represent X operators
        - Upper n bits represent Z operators
    n : int
        Number of qubits the operator acts on
        
    Notes
    -----
    The bit-mask representation uses:
    - bit i (0 <= i < n) for X operator on qubit i
    - bit (n+i) for Z operator on qubit i
    - Y operator is represented by both X and Z bits set
    """
    coeff: complex
    key:   int
    n:     int

    def to_label(self, use_cache: bool = True) -> str:
        """
        Convert the Pauli operator to a human-readable string label.
        
        Parameters
        ----------
        use_cache : bool
            Whether to use cached decode function for better performance
        
        Returns
        -------
        str
            String representation of the Pauli operator (e.g. 'IXYZ')
        """
        if use_cache:
            return decode_pauli_cached(self.key, self.n).to_label()
        else:
            return decode_pauli(self.key, self.n).to_label()

    def __repr__(self) -> str:
        """
        String representation of the Pauli term.
        
        Returns
        -------
        str
            String in the format '+c*P' where c is the coefficient
            and P is the Pauli operator label
        """
        return f"{self.coeff:+g}*{self.to_label()}"
    
    def weight(self) -> int:
        """
        Compute the weight of the Pauli term.
        
        The weight is the number of non-identity Pauli operators
        in the term, i.e., the number of qubits on which the operator
        acts non-trivially.
        """
        return weight_of_key(self.key, self.n)  
