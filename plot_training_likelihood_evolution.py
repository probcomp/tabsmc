#!/usr/bin/env python
"""Plot the evolution of log-likelihood per data point during training."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import jax.numpy as jnp
import jax

def compute_marginal_likelihood_from_components(batch_log_liks, log_weights, pi_particles):
    """
    Compute marginal log-likelihood per data point from component likelihoods.
    
    batch_log_liks: (P, B, C) - log p(x_i | z_i=c, theta^p) for each particle p, data i, cluster c
    log_weights: (P,) - log weights of particles
    pi_particles: (P, C) - log mixture weights (π) for each particle
    
    Returns: scalar - average log-likelihood per data point
    """
    P, B, C = batch_log_liks.shape
    
    # For each particle, compute marginal likelihood of each data point
    # log p(x_i | theta^p) = log sum_c pi_c^p * p(x_i | z_i=c, theta^p)
    # = log sum_c exp(log_pi[p,c] + batch_log_liks[p,i,c])
    
    # Reshape for broadcasting: pi_particles (P, 1, C) + batch_log_liks (P, B, C)
    log_pi_expanded = pi_particles[:, None, :]  # (P, 1, C)
    
    # Marginal log-likelihood for each data point under each particle
    log_px_given_particle = jax.scipy.special.logsumexp(
        log_pi_expanded + batch_log_liks, axis=2
    )  # (P, B)
    
    # Average over batch for each particle
    log_lik_per_particle = jnp.mean(log_px_given_particle, axis=1)  # (P,)
    
    # Weighted average over particles
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weighted_log_lik = jax.scipy.special.logsumexp(
        log_lik_per_particle + log_weights_normalized
    )
    
    return float(weighted_log_lik)


def main():
    # Load the step 760 checkpoint
    checkpoint_file = Path("results/smc_pums_checkpoint_step_760.pkl")
    print(f"Loading checkpoint: {checkpoint_file}")
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Get final particles to extract π values
    final_particles = checkpoint['particles']
    final_A, final_φ, final_π, final_θ = final_particles
    
    print(f"Final π shape: {final_π.shape}")  # Should be (P, C)
    
    log_likelihoods_history = checkpoint['log_likelihoods']
    config = checkpoint['config']
    
    print(f"Configuration: P={config['n_particles']}, C={config['n_clusters']}, B={config['batch_size']}")
    print(f"Total steps: {len(log_likelihoods_history)}")
    
    # Extract log-likelihood per data point for each step
    steps = []
    ll_per_datapoint = []
    
    # Unfortunately, we only have the final π values, not the historical ones
    # So we'll use these as an approximation
    print("\nNote: Using final π values for all steps (approximation)")
    print("Processing steps...")
    
    for i, entry in enumerate(log_likelihoods_history):
        if i % 50 == 0:
            print(f"  Processing step {i+1}/{len(log_likelihoods_history)}")
        
        step = entry['step']
        batch_log_liks = jnp.array(entry['batch_log_likelihoods'])  # (P, B, C)
        log_weights = jnp.array(entry['log_weights'])  # (P,)
        
        # Use final π values (this is an approximation)
        ll_per_dp = compute_marginal_likelihood_from_components(
            batch_log_liks, log_weights, final_π
        )
        
        steps.append(step)
        ll_per_datapoint.append(ll_per_dp)
    
    # Create the main evolution plot
    plt.figure(figsize=(12, 7))
    plt.plot(steps, ll_per_datapoint, 'b-', linewidth=2)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Log-Likelihood per Data Point', fontsize=12)
    plt.title('SMC Training: Log-Likelihood Evolution (Using Final π)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    improvement = ll_per_datapoint[-1] - ll_per_datapoint[0]
    plt.text(0.02, 0.98, 
            f'Initial: {ll_per_datapoint[0]:.3f}\nFinal: {ll_per_datapoint[-1]:.3f}\nImprovement: {improvement:.3f}', 
            transform=plt.gca().transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add note about approximation
    plt.text(0.98, 0.02, 
            'Note: Using final π values for all steps', 
            transform=plt.gca().transAxes, horizontalalignment='right',
            fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/smc_training_ll_per_datapoint.png', dpi=150)
    print(f"\nPlot saved to figures/smc_training_ll_per_datapoint.png")
    
    # Create a smoothed version
    plt.figure(figsize=(12, 7))
    
    # Apply smoothing
    window_size = 20
    smoothed_ll = np.convolve(ll_per_datapoint, np.ones(window_size)/window_size, mode='valid')
    smoothed_steps = steps[window_size//2:-window_size//2+1]
    
    plt.plot(steps, ll_per_datapoint, 'b-', alpha=0.3, linewidth=1, label='Raw')
    plt.plot(smoothed_steps, smoothed_ll, 'b-', linewidth=2, label=f'Smoothed (window={window_size})')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Log-Likelihood per Data Point', fontsize=12)
    plt.title('SMC Training: Smoothed Log-Likelihood Evolution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/smc_training_ll_smoothed.png', dpi=150)
    print(f"Smoothed plot saved to figures/smc_training_ll_smoothed.png")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Initial LL/dp: {ll_per_datapoint[0]:.4f}")
    print(f"  Final LL/dp: {ll_per_datapoint[-1]:.4f}")
    print(f"  Total improvement: {improvement:.4f}")
    print(f"  Best LL/dp: {max(ll_per_datapoint):.4f} at step {steps[np.argmax(ll_per_datapoint)]}")
    
    # Note about the difference from test evaluation
    print(f"\nNote: These are batch-based estimates during training.")
    print(f"The test evaluation on 2000 samples gave -25.74, which is better.")
    print(f"This difference might be due to:")
    print(f"  1. Using final π values for all steps (approximation)")
    print(f"  2. Different data samples (training batches vs. test set)")
    print(f"  3. Batch effects and variance")


if __name__ == "__main__":
    main()