# Package Implementation

## Overview

The Pauli propagation algorithm is implemented as a comprehensive Python package `pauli_pkg` that provides efficient classical simulation of quantum circuits through Heisenberg-picture observable evolution. The package employs bit-mask representations for Pauli operators and supports both exact propagation and Monte Carlo sampling for mean squared error estimation.

## Package Structure

The `pauli_pkg` package is organized into a modular architecture comprising eight core modules:

```plaintext
pauli_pkg/
├── pauli_propagation/
│   ├── __init__.py           # Package initialization and exports
│   ├── pauli_term.py         # Core Pauli operator data structure
│   ├── propagator.py         # Main propagation algorithms
│   ├── monte_carlo.py        # Monte Carlo sampling implementation
│   ├── gates.py              # Quantum gate implementations
│   ├── utils.py              # Utility functions and helpers
│   ├── decomposition.py      # SU(4) KAK decomposition
│   └── circuit_topologies.py # Circuit generation utilities
├── integration_test/         # Integration test suite
├── unit_test/               # Unit test suite
└── pyproject.toml           # Project configuration
```

## Core Data Structures

### PauliTerm Class

The `PauliTerm` class serves as the fundamental data structure for representing Pauli operators in the package. It employs a compact bit-mask encoding that significantly reduces memory overhead compared to traditional matrix representations.

```python
@dataclass(slots=True)
class PauliTerm:
    coeff: complex  # Complex coefficient
    key: int        # Bit-mask representation
    n: int          # Number of qubits
```

The bit-mask encoding uses a $2n$-bit integer where the lower $n$ bits represent $X$ operators and the upper $n$ bits represent $Z$ operators on each qubit. $Y$ operators are encoded by setting both $X$ and $Z$ bits. This representation enables efficient bitwise operations for weight calculations and Pauli operator manipulations.

## Propagation Engine

### PauliPropagator Class

The `PauliPropagator` class implements the core back-propagation algorithm for Heisenberg evolution of observables through quantum circuits. The class provides both exact and truncated propagation methods with optimized performance features.

**Key Features:**
- **Bit-mask optimization**: All Pauli operations use efficient bitwise arithmetic
- **Parallel processing**: Automatic parallelization for large circuits using `ProcessPoolExecutor`
- **Memory efficiency**: Tuple-based internal representation to minimize memory allocation
- **Expectation value computation**: Pre-computed lookup tables for fast expectation value calculations

**Core Methods:**
- `propagate()`: Exact propagation returning full Pauli histories
- `propagate_fast()`: Optimized propagation returning only final terms
- `expectation_pauli_sum()`: Efficient expectation value computation using lookup tables
- `analytical_truncation_mse()`: Analytical MSE estimation for weight truncation

The propagator maintains a registry-based gate system where quantum gates are applied through a unified interface, enabling extensibility and consistent error handling.

## Monte Carlo Sampling

### MonteCarlo Class

The `MonteCarlo` class implements the Monte Carlo sampling algorithm described in Section 5 for estimating truncation errors. The implementation follows the theoretical framework while providing practical optimizations for large-scale simulations.

**Sampling Algorithm Implementation:**
1. **Initial sampling**: Pauli strings are sampled according to their coefficient weights
2. **Iterative back-propagation**: Each layer applies probabilistic branching based on gate decompositions  
3. **Path amplitude tracking**: Unbiased estimators maintain proper normalization throughout propagation
4. **Weight monitoring**: Efficient weight tracking using bit manipulation for threshold detection

**Key Features:**
- **Unbiased MSE estimation**: Implements the theoretical unbiased estimator from Equation~\ref{eq:unbiased_circuit_mse}
- **Parallel sampling**: Multi-process sampling for improved performance
- **Sample persistence**: Automatic saving/loading of sampling results via pickle serialization
- **Certified error bounds**: Statistical confidence intervals based on sample variance

The implementation encapsulates sampling results internally, providing clean interfaces for MSE estimation while maintaining computational efficiency.

## Quantum Gate Library

### QuantumGate Registry

The package implements a comprehensive library of quantum gates through a registry-based architecture. Each gate provides Pauli propagation rules that transform input `PauliTerm` objects into lists of output terms.

**Supported Gate Categories:**
- **Single-qubit gates**: Pauli gates ($X$, $Y$, $Z$), Hadamard ($H$), rotation gates ($R_x$, $R_y$, $R_z$), phase gates ($S$, $T$)
- **Two-qubit gates**: CNOT, controlled rotations (CRX, CRY, CRZ), Pauli rotations (RXX, RYY, RZZ), SWAP, iSWAP
- **Multi-qubit gates**: Toffoli (CCX), arbitrary SU(2) and SU(4) unitaries
- **Parameterized gates**: All rotation gates support arbitrary angles with exact trigonometric implementations

**Implementation Details:**
- **Phase tracking**: Exact phase calculations using complex arithmetic
- **Bit manipulation**: Efficient Pauli transformations using bitwise operations  
- **Caching**: LRU caching for frequently accessed gate operations
- **Parameter extraction**: Automatic parameter extraction from Qiskit instruction objects

The registry pattern enables runtime gate registration and supports both built-in and user-defined gate implementations.

## Utility Functions

### Encoding and Decoding

The `utils` module provides essential functions for Pauli operator manipulation:

**Core Functions:**
- `encode_pauli()`: Converts Qiskit Pauli objects to bit-mask representation
- `decode_pauli()`: Reconstructs Qiskit Pauli objects from bit-masks  
- `weight_of_key()`: Efficient weight calculation using bit counting
- `decode_pauli_cached()`: LRU-cached decoding for performance optimization

**Random Generation:**
- `random_su4()`: Haar-random SU(4) matrix generation using Ginibre decomposition
- `random_su2()`: Haar-random SU(2) matrix generation using unit quaternions
- `random_state_label()`: Random product state generation for testing

## Circuit Decomposition

### SU(4) KAK Decomposition

The `decomposition` module implements the Cartan (KAK) decomposition for arbitrary two-qubit unitaries, enabling the transformation of pathological circuits into more structured forms.

**KAK Decomposition:**
$$U = e^{i\phi} (K_{1l} \otimes K_{1r}) \exp(i a \sigma_x \otimes \sigma_x + i b \sigma_y \otimes \sigma_y + i c \sigma_z \otimes \sigma_z) (K_{2l} \otimes K_{2r})$$

**Implementation Features:**
- **Qiskit integration**: Uses `TwoQubitWeylDecomposition` for numerical stability
- **Phase handling**: Proper global phase tracking and reconstruction verification
- **Circuit reconstruction**: Automatic replacement of SU(4) gates with decomposed forms
- **Error checking**: Reconstruction verification with configurable tolerance

## Circuit Topologies

### Topology Generation

The `circuit_topologies` module provides utilities for generating quantum circuits with specific connectivity patterns relevant to quantum hardware architectures.

**Supported Topologies:**
- **Staircase topology**: 2D grid-based connectivity for pathological circuit generation
- **IBM Eagle topology**: 127-qubit heavy-hexagonal lattice connectivity
- **TFI circuits**: Transverse-field Ising model Trotter evolution circuits

**Key Functions:**
- `staircasetopology2d_qc()`: Generates staircase circuits with random SU(4) gates
- `tfi_trotter_circuit()`: Creates TFI Trotter circuits with configurable parameters
- `get_staircase_edges()`: Computes connectivity patterns for grid topologies

## Performance Optimizations

### Computational Efficiency

The package incorporates several performance optimizations to enable large-scale quantum circuit simulation:

**Memory Optimization:**
- **Bit-mask encoding**: $O(1)$ memory per Pauli operator versus $O(4^n)$ for matrices
- **Tuple representation**: Internal tuple-based storage minimizes object allocation overhead
- **Selective caching**: LRU caches for frequently accessed operations

**Computational Acceleration:**
- **Parallel processing**: Automatic multi-process execution for circuits exceeding threshold sizes
- **Lookup tables**: Pre-computed expectation value tables for common basis states
- **Bit manipulation**: Native bit operations for weight calculations and Pauli transformations
- **Early termination**: Zero-product detection in expectation value calculations

**Scalability Features:**
- **Threshold-based parallelization**: Automatic parallel processing activation for large problems
- **Configurable worker processes**: Adaptive process pool sizing based on system resources
- **Memory-efficient sampling**: Streaming-based Monte Carlo sampling for large sample sizes

## Integration and Testing

### Test Suite Architecture

The package includes comprehensive testing infrastructure organized into unit and integration test categories:

**Unit Tests:** Individual module testing covering:
- Gate implementation correctness for all supported quantum gates
- Pauli operator encoding/decoding consistency
- Utility function validation with edge cases
- Random matrix generation statistical properties

**Integration Tests:** End-to-end validation including:
- Kicked Ising experiment reproduction
- Staircase circuit propagation consistency
- SU(4) KAK decomposition equivalence verification
- Monte Carlo sampling convergence validation

The test suite ensures mathematical correctness while providing performance benchmarks for optimization validation. 