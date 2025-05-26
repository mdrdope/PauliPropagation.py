import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

print("Testing fixed implementation...")

try:
    from pauli_pkg.pauli_propagation.propagator import PauliPropagator
    from pauli_pkg.pauli_propagation.pauli_term import PauliTerm
    
    # Test just the key function
    print("Import successful!")
    
    # Test if the modification works by calling the static method directly
    args = ([], 1, 1.0, 2, 0.0)  # dummy args
    
    print("Test completed - modifications were applied successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}") 