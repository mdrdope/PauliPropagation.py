from dataclasses import dataclass
from .utils import decode_pauli

@dataclass(slots=True)
class PauliTerm:
    """One term α·P  represented by (coeff, bit-mask, n_qubits)."""
    coeff: complex
    key:   int
    n:     int

    # ------ convenience -------
    def to_label(self) -> str:
        return decode_pauli(self.key, self.n).to_label()

    def __repr__(self) -> str:
        return f"{self.coeff:+g}·{self.to_label()}"
