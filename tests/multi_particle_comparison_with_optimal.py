"""
Compare multi-particle SMC with vs without rejuvenation.
Now with per-datapoint likelihood and optimal baseline.
"""

import jax
import jax.numpy as jnp
from tabsmc.smc import smc_no_rejuvenation
from generate_synthetic_data import generate_mixture_data, create_test_parameters
from compute_optimal_likelihood import compute_optimal_loglik_per_datapoint
import matplotlib.pyplot as plt
import numpy as np
import time


@jax.jit
def compute_test_loglik_vectorized(X_test, π, θ):
    """Vectorized computation of test log-likelihood for a single particle."""
    log_px_given_c = jnp.einsum('ndk,cdk->nc', X_test, θ)
    log_px = jax.scipy.special.logsumexp(π[None, :] + log_px_given_c, axis=1)
    return jnp.sum(log_px)


def compute_weighted_test_loglik(X_test, particles, log_weights):
    """Compute weighted average test log-likelihood."""
    _, _, π, θ = particles
    
    # Compute log-likelihood for each particle
    log_liks = jax.vmap(compute_test_loglik_vectorized, in_axes=(None, 0, 0))(X_test, π, θ)
    
    # Weighted average
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    return jax.scipy.special.logsumexp(log_liks + log_weights_normalized)


def main():
    """Compare multi-particle performance with and without rejuvenation."""
    # Set random seed
    key = jax.random.PRNGKey(42)
    
    # Multi-particle parameters
    P = 20   # Multiple particles
    B = 100  # Batch size
    T = 30   # Number of iterations
    N_train = B * T + 500  # Ensure enough data
    N_test = 200
    D, K, C = 5, 3, 2
    
    print(f"Multi-Particle SMC Comparison (Per-Datapoint Likelihood)")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  P = {P} (MULTIPLE PARTICLES)")
    print(f"  N_train = {N_train:,}, N_test = {N_test:,}")
    print(f"  Timesteps = {T}, Batch size = {B}")
    print(f"  Features = {D}, Categories = {K}, Clusters = {C}")
    
    # Create test parameters
    true_π, true_θ = create_test_parameters(C, D, K)
    print(f"\\nTrue parameters:")
    print(f"  π = {true_π}")
    print(f"  θ shape = {true_θ.shape}")
    
    # Compute optimal log-likelihood per datapoint
    print(f"\\nComputing optimal log-likelihood...")
    optimal_loglik_per_datapoint = compute_optimal_loglik_per_datapoint(true_π, true_θ)
    print(f"  Optimal log-likelihood per datapoint: {optimal_loglik_per_datapoint:.4f}")
    
    # Generate data
    print(f"\\nGenerating data...")
    start_time = time.time()
    key, subkey = jax.random.split(key)
    X_train, assignments_train = generate_mixture_data(subkey, N_train, D, K, true_π, true_θ)
    
    key, subkey = jax.random.split(key)
    X_test, _ = generate_mixture_data(subkey, N_test, D, K, true_π, true_θ)
    data_time = time.time() - start_time
    
    print(f"Data generated in {data_time:.3f}s")
    
    # Verify cluster proportions
    _, counts_train = jnp.unique(assignments_train, return_counts=True)
    empirical_π_train = counts_train / N_train
    print(f"  Training empirical mixing weights: {empirical_π_train}")
    print(f"  Difference from true: {jnp.abs(empirical_π_train - true_π)}")
    
    α_pi = 1.0
    α_theta = 1.0
    
    # Use same initial key for both methods for fair comparison
    base_key = jax.random.PRNGKey(123)
    
    print(f"\\nRunning SMC without rejuvenation (P={P})...")
    start_time = time.time()
    _, _, history_no_rejuv = smc_no_rejuvenation(
        base_key, X_train, T, P, C, B, α_pi, α_theta, rejuvenation=False, return_history=True
    )
    no_rejuv_time = time.time() - start_time
    print(f"Completed in {no_rejuv_time:.2f}s")
    
    print(f"\\nRunning SMC with rejuvenation (P={P})...")
    start_time = time.time()
    _, _, history_rejuv = smc_no_rejuvenation(
        base_key, X_train, T, P, C, B, α_pi, α_theta, rejuvenation=True, return_history=True
    )
    rejuv_time = time.time() - start_time
    print(f"Completed in {rejuv_time:.2f}s")
    
    # Compute test log-likelihoods per datapoint
    print(f"\\nComputing test log-likelihoods per datapoint...")
    test_logliks_no_rejuv = []
    test_logliks_rejuv = []
    
    for t in range(T):
        # Without rejuvenation
        particles_t = (history_no_rejuv['A'][t], history_no_rejuv['phi'][t], 
                       history_no_rejuv['pi'][t], history_no_rejuv['theta'][t])
        weights_t = history_no_rejuv['log_weights'][t]
        loglik_t = compute_weighted_test_loglik(X_test, particles_t, weights_t)
        loglik_per_datapoint = float(loglik_t) / N_test  # Convert to per-datapoint
        test_logliks_no_rejuv.append(loglik_per_datapoint)
        
        # With rejuvenation
        particles_t = (history_rejuv['A'][t], history_rejuv['phi'][t], 
                       history_rejuv['pi'][t], history_rejuv['theta'][t])
        weights_t = history_rejuv['log_weights'][t]
        loglik_t = compute_weighted_test_loglik(X_test, particles_t, weights_t)
        loglik_per_datapoint = float(loglik_t) / N_test  # Convert to per-datapoint
        test_logliks_rejuv.append(loglik_per_datapoint)
        
        if (t + 1) % 5 == 0:  # Print every 5 iterations
            print(f"  Iteration {t+1}: No rejuv = {test_logliks_no_rejuv[-1]:.4f}, "
                  f"With rejuv = {test_logliks_rejuv[-1]:.4f}, "
                  f"Diff = {test_logliks_rejuv[-1] - test_logliks_no_rejuv[-1]:.4f}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    iterations = np.arange(1, T + 1)
    
    plt.plot(iterations, test_logliks_no_rejuv, 'b-', label=f'Without Rejuvenation (P={P})', 
             linewidth=2.5, alpha=0.8)
    plt.plot(iterations, test_logliks_rejuv, 'r-', label=f'With Rejuvenation (P={P})', 
             linewidth=2.5, alpha=0.8)
    
    # Add optimal log-likelihood line
    plt.axhline(y=optimal_loglik_per_datapoint, color='black', linestyle='--', 
                linewidth=2, alpha=0.7, label='Optimal (True Distribution)')
    
    # Add markers every 5 iterations
    marker_indices = np.arange(4, T, 5)  # 5, 10, 15, 20, 25, 30
    plt.plot(iterations[marker_indices], np.array(test_logliks_no_rejuv)[marker_indices], 
             'bo', markersize=8, alpha=0.8)
    plt.plot(iterations[marker_indices], np.array(test_logliks_rejuv)[marker_indices], 
             'rs', markersize=8, alpha=0.8)
    
    plt.xlabel('SMC Iteration', fontsize=14)
    plt.ylabel('Test Log-Likelihood per Datapoint', fontsize=14)
    plt.title(f'Multi-Particle SMC: Rejuvenation vs Weighting Effects', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis to show gap from optimal
    y_min = min(min(test_logliks_no_rejuv), min(test_logliks_rejuv))
    y_max = max(optimal_loglik_per_datapoint + 0.01, max(max(test_logliks_no_rejuv), max(test_logliks_rejuv)))
    plt.ylim(y_min - 0.05, y_max + 0.05)
    
    plt.tight_layout()
    plt.savefig('multi_particle_rejuvenation_with_optimal.png', dpi=150)
    print(f"\\nPlot saved to multi_particle_rejuvenation_with_optimal.png")
    
    # Detailed analysis
    print(f"\\nDetailed Analysis:")
    print(f"  Timing:")
    print(f"    Without rejuvenation: {no_rejuv_time:.2f}s")
    print(f"    With rejuvenation: {rejuv_time:.2f}s")
    print(f"    Rejuvenation overhead: {rejuv_time - no_rejuv_time:.2f}s ({(rejuv_time/no_rejuv_time - 1)*100:.1f}%)")
    
    print(f"\\n  Performance (per datapoint):")
    print(f"    Optimal log-likelihood: {optimal_loglik_per_datapoint:.4f}")
    print(f"\\n    Initial test log-likelihoods:")
    print(f"      Without rejuvenation: {test_logliks_no_rejuv[0]:.4f} (gap: {optimal_loglik_per_datapoint - test_logliks_no_rejuv[0]:.4f})")
    print(f"      With rejuvenation: {test_logliks_rejuv[0]:.4f} (gap: {optimal_loglik_per_datapoint - test_logliks_rejuv[0]:.4f})")
    
    print(f"\\n    Final test log-likelihoods:")
    print(f"      Without rejuvenation: {test_logliks_no_rejuv[-1]:.4f} (gap: {optimal_loglik_per_datapoint - test_logliks_no_rejuv[-1]:.4f})")
    print(f"      With rejuvenation: {test_logliks_rejuv[-1]:.4f} (gap: {optimal_loglik_per_datapoint - test_logliks_rejuv[-1]:.4f})")
    print(f"      Final improvement: {test_logliks_rejuv[-1] - test_logliks_no_rejuv[-1]:.4f}")
    
    # Convergence analysis
    final_10_no_rejuv = np.mean(test_logliks_no_rejuv[-10:])
    final_10_rejuv = np.mean(test_logliks_rejuv[-10:])
    print(f"\\n    Average of last 10 iterations:")
    print(f"      Without rejuvenation: {final_10_no_rejuv:.4f} (gap: {optimal_loglik_per_datapoint - final_10_no_rejuv:.4f})")
    print(f"      With rejuvenation: {final_10_rejuv:.4f} (gap: {optimal_loglik_per_datapoint - final_10_rejuv:.4f})")
    print(f"      Improvement: {final_10_rejuv - final_10_no_rejuv:.4f}")
    
    # Weight analysis
    print(f"\\n  Final weight analysis:")
    final_weights_no_rejuv = history_no_rejuv['log_weights'][-1]
    final_weights_rejuv = history_rejuv['log_weights'][-1]
    
    weight_var_no_rejuv = jnp.var(final_weights_no_rejuv)
    weight_var_rejuv = jnp.var(final_weights_rejuv)
    
    uniform_weight = -jnp.log(P)
    is_uniform_no_rejuv = jnp.allclose(final_weights_no_rejuv, uniform_weight, atol=1e-3)
    is_uniform_rejuv = jnp.allclose(final_weights_rejuv, uniform_weight, atol=1e-3)
    
    print(f"    Weight variance without rejuvenation: {weight_var_no_rejuv:.6f} (uniform: {is_uniform_no_rejuv})")
    print(f"    Weight variance with rejuvenation: {weight_var_rejuv:.6f} (uniform: {is_uniform_rejuv})")
    
    if is_uniform_rejuv and not is_uniform_no_rejuv:
        print(f"    → Rejuvenation triggers more resampling, leading to uniform weights")
    elif is_uniform_no_rejuv and is_uniform_rejuv:
        print(f"    → Both methods have uniform weights (both resampling)")
    else:
        print(f"    → Both methods have diverse weights (neither resampling much)")


if __name__ == "__main__":
    main()