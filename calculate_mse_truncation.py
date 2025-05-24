#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate MSE under different truncation levels using Monte Carlo sampling results.

MSE^(k) = (1/M) * 曳|C_i|^2 * <P_i, 老_0>^2 * 1{wt(P_i) > k}
where 老_0 = |0...0?
"""

import numpy as np
from pauli_pkg.pauli_propagation.propagator import PauliPropagator

def calculate_truncation_mse(propagator, sampled_last_paulis, weight_exceeded_details, 
                           coeff_sqs, n_qubits):
    """
    Calculate MSE under different truncation levels.
    
    Parameters
    ----------
    propagator : PauliPropagator
        The propagator object with expectation_pauli_sum method
    sampled_last_paulis : List[PauliTerm]
        Final PauliTerms from Monte Carlo sampling
    weight_exceeded_details : List[List[bool]]
        For each path, 6 booleans indicating if weight > [1,2,3,4,5,6]
    coeff_sqs : List[float]
        |C_i|^2 for each final PauliTerm
    n_qubits : int
        Number of qubits in the system
        
    Returns
    -------
    dict
        Dictionary with truncation levels as keys and MSE values as values
    """
    
    M = len(sampled_last_paulis)
    
    # Create |0...0? product state label
    product_label = '0' * n_qubits
    
    print(f"Computing expectation values for {M} Pauli terms with 老_0 = |{product_label}?...")
    
    # Calculate <P_i, 老_0> for each Pauli term
    expectations = []
    for i, pauli in enumerate(sampled_last_paulis):
        if i % 1000 == 0:
            print(f"Progress: {i}/{M}")
        
        # Calculate expectation value <P_i, 老_0>
        exp_val = propagator.expectation_pauli_sum([pauli], product_label)
        expectations.append(exp_val)
    
    expectations = np.array(expectations)
    coeff_sqs = np.array(coeff_sqs)
    weight_exceeded_details = np.array(weight_exceeded_details, dtype=bool)
    
    print("Computing MSE for different truncation levels...")
    
    # Calculate MSE for each truncation level k = 1, 2, 3, 4, 5, 6
    mse_results = {}
    
    for k in range(1, 7):  # k = 1, 2, 3, 4, 5, 6
        # Use np.where to find indices where weight > k
        # weight_exceeded_details[:, k-1] gives boolean array for weight > k
        truncated_mask = weight_exceeded_details[:, k-1]
        
        # Calculate MSE^(k) = (1/M) * 曳|C_i|^2 * <P_i, 老_0>^2 * 1{wt(P_i) > k}
        # The indicator function 1{wt(P_i) > k} is implemented by the truncated_mask
        mse_terms = coeff_sqs * (expectations ** 2) * truncated_mask
        
        # Sum over all terms and divide by M
        mse_k = np.sum(mse_terms) / M
        
        mse_results[k] = mse_k
        
        n_truncated = np.sum(truncated_mask)
        print(f"Truncation k={k}: {n_truncated}/{M} terms truncated, MSE = {mse_k:.2e}")
    
    return mse_results, expectations

def plot_mse_vs_truncation(mse_results):
    """Plot MSE vs truncation level."""
    import matplotlib.pyplot as plt
    
    truncation_levels = list(mse_results.keys())
    mse_values = list(mse_results.values())
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(truncation_levels, mse_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Truncation Level k')
    plt.ylabel('MSE^(k)')
    plt.title('Mean Square Error vs Truncation Level')
    plt.grid(True, alpha=0.3)
    plt.xticks(truncation_levels)
    
    # Add value labels on points
    for k, mse in zip(truncation_levels, mse_values):
        plt.annotate(f'{mse:.2e}', (k, mse), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return truncation_levels, mse_values

def analyze_truncation_statistics(weight_exceeded_details, last_pauli_weights):
    """Analyze truncation statistics."""
    
    weight_exceeded_details = np.array(weight_exceeded_details, dtype=bool)
    M = len(weight_exceeded_details)
    
    print("=== Truncation Statistics ===")
    for k in range(1, 7):
        n_truncated = np.sum(weight_exceeded_details[:, k-1])
        percentage = n_truncated / M * 100
        print(f"k={k}: {n_truncated:5d}/{M} paths truncated ({percentage:5.1f}%)")
    
    # Also show actual weight distribution
    print("\n=== Actual Weight Distribution ===")
    unique_weights, counts = np.unique(last_pauli_weights, return_counts=True)
    for w, c in zip(unique_weights, counts):
        percentage = c / M * 100
        print(f"Weight {w:2d}: {c:5d} occurrences ({percentage:5.1f}%)")

# Main function to use in your notebook
def compute_mse_analysis(propagator, sampled_last_paulis, weight_exceeded_details, 
                        coeff_sqs, last_pauli_weights, n_qubits):
    """
    Complete MSE analysis function.
    
    Usage in your notebook:
    mse_results, expectations = compute_mse_analysis(
        prop_2d, sampled_last_paulis, weight_exceeded_details, 
        coeff_sqs, last_pauli_weights, nx*ny
    )
    """
    
    print("Starting MSE analysis...")
    
    # Analyze truncation statistics
    analyze_truncation_statistics(weight_exceeded_details, last_pauli_weights)
    
    # Calculate MSE
    mse_results, expectations = calculate_truncation_mse(
        propagator, sampled_last_paulis, weight_exceeded_details, 
        coeff_sqs, n_qubits
    )
    
    # Plot results
    plot_mse_vs_truncation(mse_results)
    
    # Print summary
    print("\n=== MSE Results Summary ===")
    for k, mse in mse_results.items():
        print(f"MSE^({k}) = {mse:.6e}")
    
    return mse_results, expectations

if __name__ == "__main__":
    print("In your notebook, use:")
    print("mse_results, expectations = compute_mse_analysis(")
    print("    prop_2d, sampled_last_paulis, weight_exceeded_details,")
    print("    coeff_sqs, last_pauli_weights, nx*ny)") 