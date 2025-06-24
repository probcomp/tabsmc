"""Run SGD baseline for 10x more steps (300 steps) and save timing results."""

import jax
import jax.numpy as jnp
from tabsmc.sgd_baseline import sgd_train, init_params, sgd_step, compute_test_loglik
import optax
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_synthetic_data import generate_mixture_data, create_test_parameters
import numpy as np
import time
import pickle
from tqdm import tqdm


def run_sgd_baseline_timing(seed, T, B, N_train, N_test, D, K, C, learning_rate, α_pi, α_theta):
    """Run SGD baseline and track timing."""
    print(f"Running SGD baseline (seed={seed})...")
    
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
        warmup_params = init_params(warmup_key, C, D, K)
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(warmup_params)
        _, _, _ = sgd_step(warmup_params, opt_state, warmup_X[:10], optimizer, α_pi, α_theta)
        _ = compute_test_loglik(warmup_params, X_test[:10])
    except:
        pass
    
    print("  Starting timed run...")
    start_time = time.time()
    
    # Initialize parameters
    key, subkey = jax.random.split(method_key)
    params = init_params(subkey, C, D, K)
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Track results step by step
    logliks = []
    times = []
    
    for t in tqdm(range(T)):
        # For long runs, use random sampling after exhausting sequential batches
        if t * B < N_train:
            # Deterministic batch indices (same as SMC for initial comparison)
            start_idx = t * B
            end_idx = min(start_idx + B, N_train)
            indices = jnp.arange(start_idx, end_idx)
            # Pad if needed
            if len(indices) < B:
                key, subkey = jax.random.split(key)
                extra_indices = jax.random.choice(subkey, N_train, shape=(B - len(indices),), replace=False)
                indices = jnp.concatenate([indices, extra_indices])
        else:
            # Random sampling after exhausting data
            key, subkey = jax.random.split(key)
            indices = jax.random.choice(subkey, N_train, shape=(B,), replace=False)
        
        X_batch = X_train[indices]
        
        # SGD step
        params, opt_state, loss = sgd_step(params, opt_state, X_batch, optimizer, α_pi, α_theta)
        
        # Compute test log-likelihood
        loglik_t = compute_test_loglik(params, X_test)
        
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
    B = 100
    T = 300  # 10x more steps!
    N_train = 3500  # Same as original
    N_test = 200
    D, K, C = 5, 3, 2
    α_pi = 1.0
    α_theta = 1.0
    learning_rate = 0.1  # Same as before
    
    print("SGD Baseline (Long Run) - Timing Experiment")
    print("=" * 60)
    print(f"Configuration: T={T}, B={B}, LR={learning_rate}, Runs={n_runs} (first discarded)")
    print(f"Total gradient steps: {T} (10x more than standard)")
    
    # Run experiments
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\nRun {i+1}/{n_runs} (seed={seed})...")
        logliks, times = run_sgd_baseline_timing(
            seed, T, B, N_train, N_test, D, K, C, learning_rate, α_pi, α_theta
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
                'method': 'sgd_baseline_long'
            })
            print(f"  Seed {seed}: Final loglik = {logliks_trimmed[-1]:.4f} at {times_trimmed[-1]:.2f}s")
    
    # Save results (excluding first run)
    output_file = 'data/sgd_baseline_long_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nResults saved to {output_file} ({len(all_results)} runs after discarding first)")


if __name__ == "__main__":
    main()