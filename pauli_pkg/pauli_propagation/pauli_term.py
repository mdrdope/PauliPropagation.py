# pauli_pkg/pauli_propagation/pauli_term.py

# ----------------------------------------------------------------------
# One term  伪 路 P  in a Pauli expansion (Qiskit鈥檚 label order)
# ----------------------------------------------------------------------
from qiskit.quantum_info import Pauli

class PauliTerm:
    r"""
    One term 伪路P appearing in a Pauli-operator expansion.

    A ``PauliTerm`` stores the complex coefficient (``coeff``) and the
    corresponding multi-qubit Pauli operator (``pauli``) expressed as a
    :class:`qiskit.quantum_info.Pauli` object.

    Parameters
    ----------
    coeff : complex
        Complex prefactor multiplying the Pauli operator.
    pauli : qiskit.quantum_info.Pauli
        Tensor product of single-qubit Pauli matrices in Qiskit label order
        (right-most character = qubit-0).

    Attributes
    ----------
    coeff : complex
        Complex prefactor of the Pauli term.
    pauli : qiskit.quantum_info.Pauli
        Underlying Pauli operator.

    Notes
    -----
    * In Qiskit鈥檚 little-endian convention, label ``'XYZ'`` represents the
      operator :math:`X \\otimes Y \\otimes Z` acting with *Z* on qubit-0,
      *Y* on qubit-1 and *X* on qubit-2.
    * The helper method :py:meth:`label_lsb` returns the label reversed so
      that the left-most character corresponds to qubit-0 (LSB-first
      representation).
    * The :py:meth:`__repr__` string follows the pattern ``'+0.5路IXY'`` for
      readability in debug output.

    Examples
    --------
    >>> from qiskit.quantum_info import Pauli
    >>> term = PauliTerm(0.5, Pauli('IXY'))
    >>> term.coeff
    (0.5+0j)
    >>> term.label_lsb()
    'YXI'
    """
    __slots__ = ("coeff", "pauli")

    def __init__(self, coeff: complex, pauli: Pauli):
        self.coeff = coeff
        self.pauli = pauli

    def label_lsb(self) -> str:
        return self.pauli.to_label()[::-1] 

    def __repr__(self) -> str:                # e.g. "+0.5路IXY"
        # return f"{self.coeff:+g}路{self.pauli.to_label()}"
        # return f"{self.coeff:+g}路{self.label_lsb()}"
        return f"{self.coeff:+g},{self.pauli.to_label()}"