"""
Multi-seed comparison of multi-particle SMC with wall-clock time on x-axis.
Includes three conditions: no rejuvenation, 1 rejuvenation step, 10 rejuvenation steps.
"""

import jax
import jax.numpy as jnp
from tabsmc.smc import smc_step, init_empty, gibbs
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


def run_smc_with_timing(key, X_train, X_test, T, P, C, B, α_pi, α_theta, N_test, rejuvenation_steps=0):
    """Run SMC with timing, excluding compilation time."""
    N, D, K = X_train.shape
    
    # Initialize particles
    keys = jax.random.split(key, P + 1)
    key = keys[0]
    init_keys = keys[1:]
    
    # Vectorized initialization
    init_particle = lambda k: init_empty(k, C, D, K, N, α_pi, α_theta)
    A, φ, π, θ = jax.vmap(init_particle)(init_keys)
    
    # Initialize weights and gammas
    log_weights = jnp.zeros(P)
    log_gammas = jnp.zeros(P)
    
    # Pre-compile by running one step (this won't count toward timing)
    start_idx = 0
    end_idx = B
    I_B = jnp.arange(start_idx, end_idx)
    X_B = X_train[I_B]
    
    # Compilation run
    key, subkey = jax.random.split(key)
    particles = (A, φ, π, θ)
    particles, log_weights, log_gammas = smc_step(
        subkey, particles, log_weights, log_gammas, X_B, I_B, α_pi, α_theta
    )
    A, φ, π, θ = particles
    
    # Perform rejuvenation steps for compilation if needed
    if rejuvenation_steps > 0:
        for _ in range(rejuvenation_steps):
            key, subkey = jax.random.split(key)
            I_rejuv = jax.random.choice(subkey, N, shape=(B,), replace=False)
            X_rejuv = X_train[I_rejuv]
            
            key, subkey = jax.random.split(key)
            keys_rejuv = jax.random.split(subkey, P)
            
            def rejuvenate_particle(p_key, p_A, p_φ, p_π, p_θ):
                A_new, φ_new, π_new, θ_new, _, _ = gibbs(
                    p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, α_pi, α_theta
                )
                return A_new, φ_new, π_new, θ_new
            
            A, φ, π, θ = jax.vmap(rejuvenate_particle)(keys_rejuv, A, φ, π, θ)
    
    # Compute test likelihood for compilation
    particles_test = (A, φ, π, θ)
    _ = compute_weighted_test_loglik(X_test, particles_test, log_weights)
    
    # Reset for actual timing run
    A, φ, π, θ = jax.vmap(init_particle)(init_keys)
    log_weights = jnp.zeros(P)
    log_gammas = jnp.zeros(P)
    
    # Storage for results and timing
    test_logliks = []
    cumulative_times = []
    start_time = time.time()
    
    for t in range(T):
        # Deterministic batch indices
        start_idx = t * B
        end_idx = start_idx + B
        I_B = jnp.arange(start_idx, end_idx)
        X_B = X_train[I_B]
        
        # SMC step
        key, subkey = jax.random.split(key)
        particles = (A, φ, π, θ)
        particles, log_weights, log_gammas = smc_step(
            subkey, particles, log_weights, log_gammas, X_B, I_B, α_pi, α_theta
        )
        A, φ, π, θ = particles
        
        # Rejuvenation steps
        for _ in range(rejuvenation_steps):
            key, subkey = jax.random.split(key)
            I_rejuv = jax.random.choice(subkey, N, shape=(B,), replace=False)
            X_rejuv = X_train[I_rejuv]
            
            key, subkey = jax.random.split(key)
            keys_rejuv = jax.random.split(subkey, P)
            
            def rejuvenate_particle(p_key, p_A, p_φ, p_π, p_θ):
                A_new, φ_new, π_new, θ_new, _, _ = gibbs(
                    p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, α_pi, α_theta
                )
                return A_new, φ_new, π_new, θ_new
            
            A, φ, π, θ = jax.vmap(rejuvenate_particle)(keys_rejuv, A, φ, π, θ)
        
        # Compute test log-likelihood
        particles_test = (A, φ, π, θ)
        loglik_t = compute_weighted_test_loglik(X_test, particles_test, log_weights)
        loglik_per_datapoint = float(loglik_t) / N_test
        test_logliks.append(loglik_per_datapoint)
        
        # Record cumulative time
        elapsed = time.time() - start_time
        cumulative_times.append(elapsed)
    
    return np.array(test_logliks), np.array(cumulative_times)


def run_single_experiment_with_timing(seed, T, B, N_train, N_test, D, K, C, P, α_pi, α_theta):
    """Run a single experiment with a given seed, tracking wall-clock time."""
    key = jax.random.PRNGKey(42 + seed)
    
    # Create test parameters
    true_π, true_θ = create_test_parameters(C, D, K)
    
    # Compute optimal log-likelihood per datapoint
    optimal_loglik_per_datapoint = float(compute_optimal_loglik_per_datapoint(true_π, true_θ))
    
    # Generate data
    key, subkey = jax.random.split(key)
    X_train, _ = generate_mixture_data(subkey, N_train, D, K, true_π, true_θ)
    
    key, subkey = jax.random.split(key)
    X_test, _ = generate_mixture_data(subkey, N_test, D, K, true_π, true_θ)
    
    # Use same initial key for all methods for fair comparison
    base_key = jax.random.PRNGKey(123 + seed)
    
    # Run methods with timing
    results_no_rejuv, times_no_rejuv = run_smc_with_timing(
        base_key, X_train, X_test, T, P, C, B, α_pi, α_theta, N_test, rejuvenation_steps=0
    )
    
    results_rejuv, times_rejuv = run_smc_with_timing(
        base_key, X_train, X_test, T, P, C, B, α_pi, α_theta, N_test, rejuvenation_steps=1
    )
    
    results_rejuv_10, times_rejuv_10 = run_smc_with_timing(
        base_key, X_train, X_test, T, P, C, B, α_pi, α_theta, N_test, rejuvenation_steps=10
    )
    
    return results_no_rejuv, results_rejuv, results_rejuv_10, times_no_rejuv, times_rejuv, times_rejuv_10, optimal_loglik_per_datapoint


def main():
    """Run multi-seed comparison with wall-clock time on x-axis."""
    # Parameters
    n_seeds = 3
    seeds = [0, 1, 2]
    P = 20   # Multiple particles
    B = 100  # Batch size
    T = 30   # Number of iterations
    N_train = B * T + 500  # Ensure enough data
    N_test = 200
    D, K, C = 5, 3, 2
    α_pi = 1.0
    α_theta = 1.0
    
    print(f"Multi-Particle SMC Comparison (Multi-Seed) - Wall-Clock Time")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Number of seeds = {n_seeds}")
    print(f"  P = {P} (MULTIPLE PARTICLES)")
    print(f"  N_train = {N_train:,}, N_test = {N_test:,}")
    print(f"  Timesteps = {T}, Batch size = {B}")
    print(f"  Features = {D}, Categories = {K}, Clusters = {C}")
    
    # Storage for results
    all_logliks_no_rejuv = []
    all_logliks_rejuv = []
    all_logliks_rejuv_10 = []
    all_times_no_rejuv = []
    all_times_rejuv = []
    all_times_rejuv_10 = []
    all_optimal_logliks = []
    
    # Run experiments
    for i, seed in enumerate(seeds):
        print(f"\nRunning experiment {i + 1}/{n_seeds} (seed={seed})...")
        start_time = time.time()
        
        (logliks_no_rejuv, logliks_rejuv, logliks_rejuv_10, 
         times_no_rejuv, times_rejuv, times_rejuv_10, optimal_loglik) = run_single_experiment_with_timing(
            seed, T, B, N_train, N_test, D, K, C, P, α_pi, α_theta
        )
        
        all_logliks_no_rejuv.append(logliks_no_rejuv)
        all_logliks_rejuv.append(logliks_rejuv)
        all_logliks_rejuv_10.append(logliks_rejuv_10)
        all_times_no_rejuv.append(times_no_rejuv)
        all_times_rejuv.append(times_rejuv)
        all_times_rejuv_10.append(times_rejuv_10)
        all_optimal_logliks.append(optimal_loglik)
        
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.2f}s")
        print(f"  Final times: No rejuv = {times_no_rejuv[-1]:.2f}s, "
              f"With 1 rejuv = {times_rejuv[-1]:.2f}s, "
              f"With 10 rejuv = {times_rejuv_10[-1]:.2f}s")
        print(f"  Final logliks: No rejuv = {logliks_no_rejuv[-1]:.4f}, "
              f"With 1 rejuv = {logliks_rejuv[-1]:.4f}, "
              f"With 10 rejuv = {logliks_rejuv_10[-1]:.4f}, "
              f"Optimal = {optimal_loglik:.4f}")
    
    # Convert to arrays
    all_logliks_no_rejuv = np.array(all_logliks_no_rejuv)
    all_logliks_rejuv = np.array(all_logliks_rejuv)
    all_logliks_rejuv_10 = np.array(all_logliks_rejuv_10)
    all_times_no_rejuv = np.array(all_times_no_rejuv)
    all_times_rejuv = np.array(all_times_rejuv)
    all_times_rejuv_10 = np.array(all_times_rejuv_10)
    all_optimal_logliks = np.array(all_optimal_logliks)
    
    # Compute statistics
    median_no_rejuv = np.median(all_logliks_no_rejuv, axis=0)
    median_rejuv = np.median(all_logliks_rejuv, axis=0)
    median_rejuv_10 = np.median(all_logliks_rejuv_10, axis=0)
    median_times_no_rejuv = np.median(all_times_no_rejuv, axis=0)
    median_times_rejuv = np.median(all_times_rejuv, axis=0)
    median_times_rejuv_10 = np.median(all_times_rejuv_10, axis=0)
    median_optimal = np.median(all_optimal_logliks)
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot individual runs with low alpha
    for i in range(n_seeds):
        plt.plot(all_times_no_rejuv[i], all_logliks_no_rejuv[i], 'b-', alpha=0.3, linewidth=1.5)
        plt.plot(all_times_rejuv[i], all_logliks_rejuv[i], 'r-', alpha=0.3, linewidth=1.5)
        plt.plot(all_times_rejuv_10[i], all_logliks_rejuv_10[i], 'g-', alpha=0.3, linewidth=1.5)
    
    # Plot medians with thicker lines
    plt.plot(median_times_no_rejuv, median_no_rejuv, 'b-', 
             label=f'Without Rejuvenation (P={P}, median)', linewidth=3, alpha=0.9)
    plt.plot(median_times_rejuv, median_rejuv, 'r-', 
             label=f'With 1 Rejuvenation Step (P={P}, median)', linewidth=3, alpha=0.9)
    plt.plot(median_times_rejuv_10, median_rejuv_10, 'g-', 
             label=f'With 10 Rejuvenation Steps (P={P}, median)', linewidth=3, alpha=0.9)
    
    # Plot optimal log-likelihood as horizontal line
    plt.axhline(y=median_optimal, color='black', linestyle='--', linewidth=2.5, 
                label=f'Optimal (True Distribution, median = {median_optimal:.3f})', alpha=0.8)
    
    # Add markers at regular time intervals
    max_time = max(np.max(median_times_no_rejuv), np.max(median_times_rejuv), np.max(median_times_rejuv_10))
    time_markers = np.arange(0, max_time, max_time/6)  # 6 markers
    
    # Interpolate to find log-likelihoods at marker times
    for marker_time in time_markers[1:]:  # Skip t=0
        if marker_time <= np.max(median_times_no_rejuv):
            idx = np.argmin(np.abs(median_times_no_rejuv - marker_time))
            plt.plot(marker_time, median_no_rejuv[idx], 'bo', markersize=8)
        
        if marker_time <= np.max(median_times_rejuv):
            idx = np.argmin(np.abs(median_times_rejuv - marker_time))
            plt.plot(marker_time, median_rejuv[idx], 'rs', markersize=8)
        
        if marker_time <= np.max(median_times_rejuv_10):
            idx = np.argmin(np.abs(median_times_rejuv_10 - marker_time))
            plt.plot(marker_time, median_rejuv_10[idx], 'g^', markersize=8)
    
    plt.xlabel('Wall-Clock Time (seconds)', fontsize=14)
    plt.ylabel('Test Log-Likelihood per Datapoint', fontsize=14)
    plt.title(f'Multi-Particle SMC: {n_seeds} Seeds (Wall-Clock Time)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis to show gap from optimal
    y_min = min(np.min(all_logliks_no_rejuv), np.min(all_logliks_rejuv), np.min(all_logliks_rejuv_10))
    y_max = max(median_optimal + 0.01, np.max(all_logliks_no_rejuv), np.max(all_logliks_rejuv), np.max(all_logliks_rejuv_10))
    plt.ylim(y_min - 0.05, y_max + 0.05)
    
    plt.tight_layout()
    plt.savefig('multi_particle_rejuvenation_timing_multiseed.png', dpi=150)
    print(f"\nPlot saved to multi_particle_rejuvenation_timing_multiseed.png")
    
    # Print timing summary
    print(f"\nTiming Summary (across {n_seeds} seeds):")
    print(f"  Final wall-clock times (median):")
    print(f"    Without rejuvenation: {median_times_no_rejuv[-1]:.2f}s")
    print(f"    With 1 rejuvenation step: {median_times_rejuv[-1]:.2f}s")
    print(f"    With 10 rejuvenation steps: {median_times_rejuv_10[-1]:.2f}s")
    print(f"    Slowdown: {median_times_rejuv[-1]/median_times_no_rejuv[-1]:.1f}x (1 rejuv), "
          f"{median_times_rejuv_10[-1]/median_times_no_rejuv[-1]:.1f}x (10 rejuv)")
    
    # Print performance summary
    print(f"\nPerformance Summary (across {n_seeds} seeds):")
    print(f"  Optimal test log-likelihood per datapoint: {median_optimal:.4f}")
    
    print(f"\n  Final test log-likelihoods per datapoint:")
    print(f"    Without rejuvenation: {median_no_rejuv[-1]:.4f}")
    print(f"    With 1 rejuvenation step: {median_rejuv[-1]:.4f}")
    print(f"    With 10 rejuvenation steps: {median_rejuv_10[-1]:.4f}")
    print(f"    Gap from optimal: {median_optimal - median_no_rejuv[-1]:.4f} (no rejuv), "
          f"{median_optimal - median_rejuv[-1]:.4f} (1 rejuv), "
          f"{median_optimal - median_rejuv_10[-1]:.4f} (10 rejuv)")
    
    improvement_1 = median_rejuv[-1] - median_no_rejuv[-1]
    improvement_10 = median_rejuv_10[-1] - median_no_rejuv[-1]
    print(f"\n  Median final improvement over no rejuvenation:")
    print(f"    1 rejuvenation step: {improvement_1:.4f} ({improvement_1/abs(median_no_rejuv[-1])*100:.1f}%)")
    print(f"    10 rejuvenation steps: {improvement_10:.4f} ({improvement_10/abs(median_no_rejuv[-1])*100:.1f}%)")


if __name__ == "__main__":
    main()