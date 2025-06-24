"""Run SMC with 10 rejuvenation steps and save timing results."""

import jax
import jax.numpy as jnp
from tabsmc.smc import smc_step, init_empty, gibbs
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_synthetic_data import generate_mixture_data, create_test_parameters
from compute_optimal_likelihood import compute_optimal_loglik_per_datapoint
import numpy as np
import time
import pickle
from tqdm import tqdm


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


def smc_with_rejuvenation_10(key, X, T, P, C, B, α_pi, α_theta):
    """Run SMC with 10 rejuvenation steps after each SMC step."""
    N, D, K = X.shape
    
    # Initialize particles
    keys = jax.random.split(key, P + 1)
    key = keys[0]
    init_keys = keys[1:]
    
    init_particle = lambda k: init_empty(k, C, D, K, N, α_pi, α_theta)
    A, φ, π, θ = jax.vmap(init_particle)(init_keys)
    
    # Initialize weights and gammas
    log_weights = jnp.zeros(P)
    log_gammas = jnp.zeros(P)
    
    for t in range(T):
        # Deterministic batch indices
        start_idx = t * B
        end_idx = start_idx + B
        I_B = jnp.arange(start_idx, end_idx)
        X_B = X[I_B]
        
        # SMC step
        key, subkey = jax.random.split(key)
        particles = (A, φ, π, θ)
        particles, log_weights, log_gammas = smc_step(
            subkey, particles, log_weights, log_gammas, X_B, I_B, α_pi, α_theta
        )
        A, φ, π, θ = particles
        
        # 10 rejuvenation steps
        for _ in range(10):
            key, subkey = jax.random.split(key)
            I_rejuv = jax.random.choice(subkey, N, shape=(B,), replace=False)
            X_rejuv = X[I_rejuv]
            
            # Run Gibbs step for each particle
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, P)
            
            def rejuvenate_particle(p_key, p_A, p_φ, p_π, p_θ):
                A_new, φ_new, π_new, θ_new, _, _ = gibbs(
                    p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, α_pi, α_theta
                )
                return A_new, φ_new, π_new, θ_new
            
            A, φ, π, θ = jax.vmap(rejuvenate_particle)(keys, A, φ, π, θ)
    
    return (A, φ, π, θ), log_weights


def run_smc_rejuv_10_timing(seed, T, B, N_train, N_test, D, K, C, P, α_pi, α_theta):
    """Run SMC with 10 rejuvenation steps and track timing."""
    print(f"Running SMC with 10 rejuvenation steps (seed={seed})...")
    
    # Generate data with consistent seed
    key = jax.random.PRNGKey(42 + seed)
    true_π, true_θ = create_test_parameters(C, D, K)
    
    key, subkey = jax.random.split(key)
    X_train, _ = generate_mixture_data(subkey, N_train, D, K, true_π, true_θ)
    
    key, subkey = jax.random.split(key)
    X_test, _ = generate_mixture_data(subkey, N_test, D, K, true_π, true_θ)
    
    # Use consistent seed for fair comparison across methods
    method_key = jax.random.PRNGKey(1000 + seed)
    
    # Warmup JIT compilation
    print("  Warming up JIT compilation...")
    warmup_key = jax.random.PRNGKey(999)
    warmup_X = X_train[:100]
    try:
        _ = smc_with_rejuvenation_10(warmup_key, warmup_X, 2, P, C, min(B, 50), α_pi, α_theta)
        # Also warm up test likelihood computation
        init_keys = jax.random.split(warmup_key, P)
        init_particle = lambda k: init_empty(k, C, D, K, N_train, α_pi, α_theta)
        A_temp, φ_temp, π_temp, θ_temp = jax.vmap(init_particle)(init_keys)
        particles_temp = (A_temp, φ_temp, π_temp, θ_temp)
        log_weights_temp = jnp.zeros(P)
        _ = compute_weighted_test_loglik(X_test[:10], particles_temp, log_weights_temp)
    except:
        pass
    
    print("  Starting timed run...")
    start_time = time.time()
    
    # Initialize particles
    keys = jax.random.split(method_key, P + 1)
    key = keys[0]
    init_keys = keys[1:]
    
    init_particle = lambda k: init_empty(k, C, D, K, N_train, α_pi, α_theta)
    A, φ, π, θ = jax.vmap(init_particle)(init_keys)
    
    # Initialize weights
    log_weights = jnp.zeros(P)
    log_gammas = jnp.zeros(P)
    
    # Track results step by step
    logliks = []
    times = []
    
    for t in tqdm(range(T)):
        # Deterministic batch indices
        start_idx = t * B
        end_idx = start_idx + B
        I_B = jnp.arange(start_idx, end_idx)
        X_B = X_train[I_B]
        
        # SMC step
        from tabsmc.smc import smc_step, gibbs
        key, subkey = jax.random.split(key)
        particles = (A, φ, π, θ)
        particles, log_weights, log_gammas = smc_step(
            subkey, particles, log_weights, log_gammas, X_B, I_B, α_pi, α_theta
        )
        A, φ, π, θ = particles
        
        # 10 rejuvenation steps
        for _ in range(10):
            key, subkey = jax.random.split(key)
            I_rejuv = jax.random.choice(subkey, N_train, shape=(B,), replace=False)
            X_rejuv = X_train[I_rejuv]
            
            # Run Gibbs step for each particle
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, P)
            
            def rejuvenate_particle(p_key, p_A, p_φ, p_π, p_θ):
                A_new, φ_new, π_new, θ_new, _, _ = gibbs(
                    p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, α_pi, α_theta
                )
                return A_new, φ_new, π_new, θ_new
            
            A, φ, π, θ = jax.vmap(rejuvenate_particle)(keys, A, φ, π, θ)
        
        # Compute test log-likelihood
        particles_test = (A, φ, π, θ)
        loglik_t = compute_weighted_test_loglik(X_test, particles_test, log_weights)
        
        # Record results
        current_time = time.time() - start_time
        logliks.append(float(loglik_t) / N_test)
        times.append(current_time)
    
    return np.array(logliks), np.array(times)


def main():
    """Run experiments and save results."""
    # Parameters
    n_runs = 11  # Total runs (first will be discarded)
    seeds = list(range(42, 42 + n_runs))
    P = 20
    B = 100
    T = 30
    N_train = B * T + 500
    N_test = 200
    D, K, C = 5, 3, 2
    α_pi = 1.0
    α_theta = 1.0
    
    print("SMC With 10 Rejuvenation Steps - Timing Experiment")
    print("=" * 60)
    print(f"Configuration: P={P}, T={T}, B={B}, Runs={n_runs} (first discarded)")
    
    # Run experiments
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\nRun {i+1}/{n_runs} (seed={seed})...")
        logliks, times = run_smc_rejuv_10_timing(
            seed, T, B, N_train, N_test, D, K, C, P, α_pi, α_theta
        )
        
        if i == 0:
            print("  (This run will be discarded - warmup)")
        else:
            # Discard timestep 0 and adjust times to start from 0
            logliks_trimmed = logliks[1:]
            times_trimmed = times[1:] - times[1]  # Subtract time at step 1 to start from 0
            
            all_results.append({
                'seed': seed,
                'logliks': logliks_trimmed,
                'times': times_trimmed,
                'method': 'rejuv_10'
            })
            print(f"  Seed {seed}: Final loglik = {logliks_trimmed[-1]:.4f} at {times_trimmed[-1]:.2f}s")
    
    # Save results (excluding first run)
    output_file = 'data/smc_rejuv_10_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nResults saved to {output_file} ({len(all_results)} runs after discarding first)")


if __name__ == "__main__":
    main()