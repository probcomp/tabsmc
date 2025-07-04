"""Run SMC with 10 rejuvenation steps and save Gibbs steps results."""

import jax
import jax.numpy as jnp
from tabsmc.smc import smc_step, init_empty, gibbs
import numpy as np
import time
import pickle
from tqdm import tqdm
import os


def create_test_parameters(C, D, K):
    """Create test parameters for mixture model."""
    if C == 2:
        # Two-cluster case
        true_pi = jnp.array([0.6, 0.4])
        
        # Create distinctive emission patterns
        true_theta = jnp.zeros((C, D, K))
        
        # Cluster 0: prefers early categories (0, 1, ...)
        for d in range(D):
            if K == 3:
                true_theta = true_theta.at[0, d, :].set(jnp.array([0.7, 0.2, 0.1]))
            else:
                weights = jnp.exp(-jnp.arange(K) * 0.5)
                true_theta = true_theta.at[0, d, :].set(weights / jnp.sum(weights))
        
        # Cluster 1: prefers later categories (..., K-2, K-1)
        for d in range(D):
            if K == 3:
                true_theta = true_theta.at[1, d, :].set(jnp.array([0.1, 0.2, 0.7]))
            else:
                weights = jnp.exp(-jnp.arange(K)[::-1] * 0.5)
                true_theta = true_theta.at[1, d, :].set(weights / jnp.sum(weights))
                
        return jnp.log(true_pi), jnp.log(true_theta)
    else:
        raise NotImplementedError(f"C={C} not implemented")


def generate_mixture_data(key, N, D, K, true_pi, true_theta):
    """Generate synthetic data from a mixture model."""
    C = true_pi.shape[0]
    
    # Convert from log space
    pi = jnp.exp(true_pi)
    theta = jnp.exp(true_theta)
    
    # Generate cluster assignments
    key, subkey = jax.random.split(key)
    assignments = jax.random.choice(subkey, C, shape=(N,), p=pi)
    
    # Generate categories for all data points and features at once
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, N * D).reshape(N, D, 2)
    
    def generate_categories_for_point(n_keys, assignment):
        """Generate categories for all features of one data point."""
        theta_n = theta[assignment]  # (D, K)
        
        def sample_category(key_d, probs_d):
            return jax.random.choice(key_d, K, p=probs_d)
        
        categories = jax.vmap(sample_category)(n_keys, theta_n)
        return categories
    
    # Vectorize over all data points
    all_categories = jax.vmap(generate_categories_for_point)(keys, assignments)
    
    # Convert to one-hot encoding
    X = jax.nn.one_hot(all_categories, K)
    
    return X, assignments


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


def run_smc_rejuv_10_gibbs(seed, T, B, N_train, N_test, D, K, C, P, α_pi, α_theta, X_test, true_π, true_θ):
    """Run SMC with 10 rejuvenation steps and track Gibbs steps."""
    print(f"Running SMC with 10 rejuvenation steps (seed={seed})...")
    
    # Generate training data with consistent seed
    key = jax.random.PRNGKey(42 + seed)
    
    key, subkey = jax.random.split(key)
    X_train, _ = generate_mixture_data(subkey, N_train, D, K, true_π, true_θ)
    
    # Use consistent seed for fair comparison across methods
    method_key = jax.random.PRNGKey(1000 + seed)
    
    # Warmup JIT compilation
    print("  Warming up JIT compilation...")
    warmup_key = jax.random.PRNGKey(999)
    try:
        # Initialize for warmup
        init_keys = jax.random.split(warmup_key, P)
        init_particle = lambda k: init_empty(k, C, D, K, 100, α_pi, α_theta)
        A_temp, φ_temp, π_temp, θ_temp = jax.vmap(init_particle)(init_keys)
        particles_temp = (A_temp, φ_temp, π_temp, θ_temp)
        log_weights_temp = jnp.zeros(P)
        _ = compute_weighted_test_loglik(X_test[:10], particles_temp, log_weights_temp)
    except:
        pass
    
    print("  Starting timed run...")
    
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
    gibbs_steps = []
    
    for t in tqdm(range(T)):
        # Deterministic batch indices
        start_idx = t * B
        end_idx = start_idx + B
        I_B = jnp.arange(start_idx, end_idx)
        X_B = X_train[I_B]
        
        # SMC step
        key, subkey = jax.random.split(key)
        particles = (A, φ, π, θ)
        particles, log_weights, log_gammas, batch_log_liks = smc_step(
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
                A_new, φ_new, π_new, θ_new, _, _, _ = gibbs(
                    p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, α_pi, α_theta
                )
                return A_new, φ_new, π_new, θ_new
            
            A, φ, π, θ = jax.vmap(rejuvenate_particle)(keys, A, φ, π, θ)
        
        # Compute test log-likelihood
        particles_test = (A, φ, π, θ)
        loglik_t = compute_weighted_test_loglik(X_test, particles_test, log_weights)
        
        # Record results - 1 SMC step + 10 rejuvenation steps = 11 Gibbs steps
        current_gibbs_steps = (t + 1) * 11
        logliks.append(float(loglik_t) / N_test)
        gibbs_steps.append(current_gibbs_steps)
    
    return np.array(logliks), np.array(gibbs_steps)


def main():
    """Run experiments and save results."""
    # Load shared test set
    with open('data/shared_test_set_gibbs.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    X_test = test_data['X_test']
    true_π = test_data['true_π']
    true_θ = test_data['true_θ']
    N_test = test_data['N_test']
    D = test_data['D']
    K = test_data['K']
    C = test_data['C']
    
    # Parameters
    n_runs = 11  # Total runs (first will be discarded)
    seeds = list(range(42, 42 + n_runs))
    P = 20
    B = 100
    T = 30
    N_train = B * T + 500
    α_pi = 1.0
    α_theta = 1.0
    
    print("SMC With 10 Rejuvenation Steps - Gibbs Steps Experiment")
    print("=" * 60)
    print(f"Configuration: P={P}, T={T}, B={B}, Runs={n_runs} (first discarded)")
    
    # Create data directory if needed
    os.makedirs('data', exist_ok=True)
    
    # Run experiments
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\nRun {i+1}/{n_runs} (seed={seed})...")
        logliks, gibbs_steps = run_smc_rejuv_10_gibbs(
            seed, T, B, N_train, N_test, D, K, C, P, α_pi, α_theta, X_test, true_π, true_θ
        )
        
        if i == 0:
            print("  (This run will be discarded - warmup)")
        else:
            # Discard timestep 0 and adjust gibbs_steps accordingly
            logliks_trimmed = logliks[1:]
            gibbs_steps_trimmed = gibbs_steps[1:]
            
            all_results.append({
                'seed': seed,
                'logliks': logliks_trimmed,
                'gibbs_steps': gibbs_steps_trimmed,
                'method': 'rejuv_10'
            })
            print(f"  Seed {seed}: Final loglik = {logliks_trimmed[-1]:.4f} at {gibbs_steps_trimmed[-1]} Gibbs steps")
    
    # Save results (excluding first run)
    output_file = 'data/smc_rejuv_10_gibbs_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nResults saved to {output_file} ({len(all_results)} runs after discarding first)")


if __name__ == "__main__":
    main()