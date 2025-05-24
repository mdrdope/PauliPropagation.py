#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Weight Exceeded Analysis for Monte Carlo Sampling

This example shows how to use the detailed weight exceeded information
returned by monte_calro_samples for statistical analysis and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from pauli_pkg.pauli_propagation.propagator import PauliPropagator
from pauli_pkg.pauli_propagation.pauli_term import PauliTerm
# from qiskit import QuantumCircuit

def analyze_weight_exceeded_data(weight_exceeded_details):
    """
    Analyze the weight exceeded details returned by monte_calro_samples.
    
    Parameters
    ----------
    weight_exceeded_details : List[List[bool]]
        Each sublist has 6 booleans for weight thresholds [1,2,3,4,5,6]
    
    Returns
    -------
    dict
        Dictionary containing various statistics
    """
    
    # Convert to numpy array for easier analysis
    exceeded_array = np.array(weight_exceeded_details, dtype=bool)
    M, num_thresholds = exceeded_array.shape  # M paths, 6 thresholds
    
    # Calculate statistics
    stats = {}
    
    # Percentage of paths exceeding each threshold
    stats['exceed_percentages'] = np.mean(exceeded_array, axis=0) * 100
    
    # Count of paths exceeding each threshold
    stats['exceed_counts'] = np.sum(exceeded_array, axis=0)
    
    # Total number of paths
    stats['total_paths'] = M
    
    # Weight thresholds
    stats['thresholds'] = [1, 2, 3, 4, 5, 6]
    
    # Most common exceeded pattern
    patterns, counts = np.unique(exceeded_array, axis=0, return_counts=True)
    most_common_idx = np.argmax(counts)
    stats['most_common_pattern'] = patterns[most_common_idx]
    stats['most_common_count'] = counts[most_common_idx]
    
    return stats

def plot_weight_analysis(stats, save_path=None):
    """
    Create plots for weight exceeded analysis.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from analyze_weight_exceeded_data
    save_path : str, optional
        Path to save the plot
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Percentage of paths exceeding each weight threshold
    thresholds = stats['thresholds']
    percentages = stats['exceed_percentages']
    
    bars1 = ax1.bar(thresholds, percentages, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Weight Threshold')
    ax1.set_ylabel('Percentage of Paths Exceeded (%)')
    ax1.set_title('Percentage of Monte Carlo Paths\nExceeding Weight Thresholds')
    ax1.set_xticks(thresholds)
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars1, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Number of paths exceeding each weight threshold
    counts = stats['exceed_counts']
    
    bars2 = ax2.bar(thresholds, counts, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Weight Threshold')
    ax2.set_ylabel('Number of Paths Exceeded')
    ax2.set_title('Number of Monte Carlo Paths\nExceeding Weight Thresholds')
    ax2.set_xticks(thresholds)
    ax2.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(count)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def print_summary_statistics(stats):
    """Print a summary of the weight exceeded statistics."""
    
    print("=== Weight Exceeded Analysis Summary ===")
    print(f"Total Monte Carlo paths: {stats['total_paths']}")
    print()
    
    print("Weight Threshold Exceeded Statistics:")
    for i, (threshold, count, pct) in enumerate(zip(
        stats['thresholds'], 
        stats['exceed_counts'], 
        stats['exceed_percentages']
    )):
        print(f"  Weight > {threshold}: {int(count):4d} paths ({pct:5.1f}%)")
    
    print()
    print(f"Most common exceeded pattern: {stats['most_common_pattern']}")
    print(f"  (occurred in {stats['most_common_count']} paths)")
    print()

# Example usage function
def example_usage():
    """
    Example showing how to use the new weight exceeded functionality.
    
    Note: This is a template - you'll need to provide actual QuantumCircuit
    and PauliTerm objects to run this example.
    """
    
    # Create example data (replace with actual monte_calro_samples call)
    print("This is a template showing how to use the weight exceeded analysis.")
    print("To run with real data, you need to:")
    print("1. Create a QuantumCircuit")
    print("2. Create a PauliTerm")
    print("3. Call monte_calro_samples")
    print()
    
    # Example with mock data for demonstration
    # Mock weight_exceeded_details - replace with real data
    np.random.seed(42)
    M = 1000  # number of Monte Carlo samples
    mock_weight_exceeded = []
    
    for _ in range(M):
        # Simulate: higher weight thresholds are less likely to be exceeded
        probs = [0.8, 0.5, 0.3, 0.15, 0.05, 0.01]  # decreasing probabilities
        exceeded = [np.random.random() < p for p in probs]
        mock_weight_exceeded.append(exceeded)
    
    # Analyze the data
    stats = analyze_weight_exceeded_data(mock_weight_exceeded)
    
    # Print summary
    print_summary_statistics(stats)
    
    # Create plots
    plot_weight_analysis(stats, save_path='weight_analysis.png')

if __name__ == "__main__":
    example_usage() 