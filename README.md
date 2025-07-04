# Pauli Propagation: Classical Simulation of Quantum Circuits

**Author:** Steven Ma (ym432)  
**Supervisor:** Prof. Hamza Fawzi  
**Date:** July 2025

---

## Overview

This repository provides a Python implementation of the Pauli propagation algorithm for simulating quantum circuits on classical computers. The package enables efficient estimation of observable expectation values in large, deep quantum circuits, pushing the limits of classical simulation and providing a new baseline for benchmarking quantum advantage.

## Scientific Context

Simulating quantum systems is a grand challenge due to the exponential scaling of quantum state space. The Pauli propagation algorithm addresses this by evolving observables (in the Heisenberg picture) as sums of Pauli operators, discarding negligible high-weight terms to control computational cost. This approach enables tractable, certified simulation of circuits that are otherwise intractable for statevector or tensor network methods.

- **Truncation:** The user sets a Pauli weight threshold `k` to balance accuracy and efficiency. Higher `k` increases fidelity at the cost of runtime and memory.
- **Monte Carlo Certification:** The package includes a Monte Carlo procedure to estimate the mean squared error (MSE) introduced by truncation, providing rigorous error bars for all results.

## Key Features
- **Efficient Pauli propagation** for arbitrary quantum circuits and observables
- **Monte Carlo error estimation** for certified simulation accuracy
- **Support for large-scale circuits** (e.g., 36-qubit random SU(4) and 127-qubit IBM Eagle topologies)
- **Benchmarked against statevector, tensor network, and real quantum hardware**

## Repository Structure

### `pauli_pkg/`
```plaintext
pauli_pkg/
├── pauli_propagation/
│   ├── __init__.py
│   ├── pauli_term.py
│   ├── gates.py
│   ├── propagator.py
│   ├── monte_carlo.py
│   ├── circuit_topologies.py
│   ├── decomposition.py
│   └── utils.py
├── unit_test/
│   ├── test_x_y_z.py
│   ├── test_h.py
│   ├── test_rx_ry_rz.py
│   ├── test_cx_cy_cz_ch.py
│   ├── test_rxx_ryy_rzz.py
│   ├── test_crx_cry_crz.py
│   ├── test_swap_iswap.py
│   ├── test_t.py
│   ├── test_ccx.py
│   ├── test_su2.py
│   └── test_su4.py
├── integration_test/
│   ├── test_propagator_random.py
│   ├── test_staircase_random_su4.py
│   ├── test_kicked_ising.py
│   └── test_su4_kak_consistent.py
└── pyproject.toml
```

Description:  
`pauli_pkg` is the core Python package for Pauli propagation simulation.  
- The `pauli_propagation/` directory contains the main library modules:
  - `propagator.py`: Core PauliPropagator class for back-propagation through quantum circuits
  - `monte_carlo.py`: MonteCarlo class for error estimation and sampling
  - `gates.py`: Quantum gate implementations for Pauli operator transformations
  - `circuit_topologies.py`: Functions for generating specific circuit topologies (staircase, IBM Eagle)
  - `decomposition.py`: SU(4) KAK decomposition utilities
  - `utils.py`: Helper functions for Pauli encoding/decoding and random matrix generation
- The `unit_test/` folder contains comprehensive unit tests for individual quantum gates
- The `integration_test/` folder includes integration tests for complete workflows
- The `pyproject.toml` file manages project dependencies and build configurations

### `results/`
```plaintext
results/
├── example/
│   └── mc_samples.pkl
├── kak_6_6/
│   ├── prop_results_L1.json
│   ├── prop_results_L2.json
│   ├── ...
│   ├── kak_mc_results_L1.json
│   ├── kak_mc_results_L2.json
│   ├── ...
│   └── kak_samples_L*.pkl
├── tfi/
│   ├── default_prop_results_L*.json
│   ├── ibm_prop_results_theta_h_*.json
│   ├── tfi_mc_results_L*.json
│   └── kicked_ising_samples_L*.pkl
├── kicked_ising_real/
│   ├── fig4b_bootstrap_mit.txt
│   ├── fig4b_bootstrap_unmit.txt
│   ├── fig4b_experiment_mit.txt
│   ├── fig4b_experiment_unmit.txt
│   ├── fig4b_MPS.txt
│   ├── fig4b_isoTNS.txt
│   └── fig4b_pauli.txt
├── ibm_real_data.json
└── su4_time_expectation_4_4.json
```

Description:  
The `results/` directory contains all simulation outputs and processed experimental data.  
- `example/`: Tutorial notebook sample data
- `kak_6_6/`: Results from 6×6 SU(4) staircase circuits with KAK decomposition
- `tfi/`: Transverse Field Ising model simulation results for various parameters
- `kicked_ising_real/`: Experimental data from IBM Quantum hardware and classical simulations
- JSON files store runtime measurements, expectation values, and error estimates
- PKL files contain Monte Carlo sampling data for error analysis(not uploaded).

### `report/`
```plaintext
report/
├── report_ym432.pdf
└── executive_summary_ym432.pdf
```

Description:  
The `report/` folder contains the complete project documentation.  
- `report_ym432.pdf`: Full technical report with detailed methodology and results
- `executive_summary_ym432.pdf`: Concise summary of key findings and scientific impact

### Jupyter Notebooks
```plaintext
├── Tutorial_notebook.ipynb
├── SU4_staircase_6_6_kak.ipynb
├── TFI_kicked_ising.ipynb
├── SU4_staircase_4_4_benchmark.ipynb
└── TFI_IBM_real_data.ipynb
```

Description:  
The Jupyter notebooks provide interactive demonstrations and reproduce all figures from the report.  
- `Tutorial_notebook.ipynb`: Step-by-step introduction to the Pauli propagation API
- `SU4_staircase_6_6_kak.ipynb`: 36-qubit random SU(4) circuit analysis with KAK decomposition
- `TFI_kicked_ising.ipynb`: 127-qubit IBM Eagle topology simulation and comparison
- `SU4_staircase_4_4_benchmark.ipynb`: Performance benchmarking against statevector methods
- `TFI_IBM_real_data.ipynb`: Analysis of real IBM Quantum experimental data

## Mapping to Report
- **Figure 4:** `Tutorial_notebook.ipynb`
- **Section 7 & Appendix B:** `SU4_staircase_6_6_kak.ipynb`
- **Section 8 & Appendix C:** `TFI_kicked_ising.ipynb`
- **Appendix A:** `SU4_staircase_4_4_benchmark.ipynb`

## Getting Started

1. **Environment Setup:**
   ```bash
   # Create and activate conda environment
   conda create -n dis_qc python=3.10 -y
   conda activate dis_qc
   ```

2. **Install Package:**
   ```bash
   # Navigate to the package directory (where pyproject.toml is located)
   cd pauli_pkg
   
   # Install the package in editable mode
   pip install -e .
   ```

3. **Run Notebooks:**
   - Start with `Tutorial_notebook.ipynb` for a hands-on introduction.

## Running Tests

The package includes comprehensive unit and integration tests using pytest.

```bash
# Navigate to the package directory
cd pauli_pkg

# Run all tests
pytest

# Run unit tests only (individual quantum gates)
pytest unit_test

# Run integration tests only (complete workflows)
pytest integration_test

# Run with verbose output
pytest -v
```

## License

MIT License. See `LICENSE` for details.

## Acknowledgements

ChatGPT assisted with code generation, grammar, and formatting. All scientific ideas and project structure are original to the author.
