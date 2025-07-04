"""Final simple SGD vs SMC comparison on PUMS data."""

import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
from tabsmc.sgd_baseline import init_params, sgd_step, compute_test_loglik
from tabsmc.smc import smc_no_rejuvenation
import optax
from tqdm import tqdm


@jax.jit
def compute_test_loglik_vectorized(X_test, π, θ):
    """Vectorized computation of test log-likelihood for a single particle."""
    # Use where to avoid 0 * (-inf) = NaN when computing log probabilities
    # Only include log probabilities for observed categories (where X_test == 1)
    observed_logprobs = jnp.where(
        X_test[:, None, :, :] == 1,  # (N, 1, D, K) - true for observed categories
        θ[None, :, :, :],            # (1, C, D, K) - log probabilities  
        0.0                          # Use 0.0 instead of 0 * (-inf)
    )
    log_px_given_c = jnp.sum(observed_logprobs, axis=(2, 3))  # Sum over D, K -> (N, C)
    log_px = jax.scipy.special.logsumexp(π[None, :] + log_px_given_c, axis=1)
    return jnp.sum(log_px)


def load_and_process_pums_data():
    """Load PUMS data and convert to compatible format."""
    print("Loading and processing PUMS data...")
    
    # Load raw PUMS data - use full training set for SMC requirements
    X_train_raw = np.load('data/pums10000.npy')  # Use full 10000 samples
    X_test_raw = np.load('data/pums_test1000.npy')  # Use full 1000 test samples
    mask = np.load('data/pums_mask.npy')
    
    # Use FULL dimensions - no truncation!
    D, K = X_train_raw.shape[1], X_train_raw.shape[2]  # D=26, K=128
    
    # Convert log probabilities to one-hot
    X_train = np.where(X_train_raw == 0.0, 1.0, 0.0)
    X_test = np.where(X_test_raw == 0.0, 1.0, 0.0)
    
    # Fix any samples that don't sum to 1 per feature using mask
    for n in range(X_train.shape[0]):
        for d in range(D):
            if np.sum(X_train[n, d]) != 1.0:
                # Find first valid category using mask
                valid_cats = np.where(mask[d] > 0)[0]
                X_train[n, d] = 0.0
                if len(valid_cats) > 0:
                    X_train[n, d, valid_cats[0]] = 1.0
    
    for n in range(X_test.shape[0]):
        for d in range(D):
            if np.sum(X_test[n, d]) != 1.0:
                # Find first valid category using mask
                valid_cats = np.where(mask[d] > 0)[0]
                X_test[n, d] = 0.0
                if len(valid_cats) > 0:
                    X_test[n, d, valid_cats[0]] = 1.0
    
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    mask = jnp.array(mask)
    
    print(f"FULL PUMS data shapes: train {X_train.shape}, test {X_test.shape}")
    print(f"Using all {D} features with {K} categories each")
    return X_train, X_test, mask


def run_sgd_experiment(X_train, X_test, mask, seed=42):
    """Run enhanced SGD experiment with SignSGD and more iterations."""
    print(f"Running enhanced SGD experiment with SignSGD (seed={seed})...")
    N_train, D, K = X_train.shape
    N_test = X_test.shape[0]
    C = 2
    T = 1000  # Push SignSGD to maximum performance
    B = 100  # Keep reasonable batch size
    lr = 0.01  # More stable learning rate for SignSGD
    
    key = jax.random.PRNGKey(seed)
    params = init_params(key, C, D, K)
    # Create SignSGD manually - apply sign to gradients then scale
    def sign_sgd(learning_rate):
        def init_fn(params):
            return {}
        
        def update_fn(grads, state, params=None):
            # Apply sign to gradients and scale by learning rate
            updates = jax.tree_map(lambda g: -learning_rate * jnp.sign(g), grads)
            return updates, state
        
        return optax.GradientTransformation(init_fn, update_fn)
    
    optimizer = sign_sgd(lr)
    opt_state = optimizer.init(params)
    
    times = []
    test_logliks = []
    train_logliks = []
    start_time = time.time()
    
    for t in tqdm(range(T), desc="SGD steps"):
        batch_key = jax.random.PRNGKey(seed + t + 1000)  # Include run seed
        batch_indices = jax.random.choice(batch_key, N_train, (B,), replace=False)
        X_batch = X_train[batch_indices]
        
        params, opt_state, _ = sgd_step(params, opt_state, X_batch, optimizer, 1.0, 1.0, mask)
        
        current_time = time.time() - start_time
        
        # Compute test log-likelihood
        current_test_loglik = float(compute_test_loglik(params, X_test, mask))
        test_logliks.append(current_test_loglik / N_test)  # Per datapoint
        
        # Compute train log-likelihood (on full training set)
        current_train_loglik = float(compute_test_loglik(params, X_train, mask))
        train_logliks.append(current_train_loglik / N_train)  # Per datapoint
        
        times.append(current_time)
    
    return times, test_logliks, train_logliks


def run_smc_experiment(X_train, X_test, mask, seed=42):
    """Run SMC experiment with train and test log-likelihood tracking."""
    print(f"Running SMC experiment (seed={seed})...")
    N_train, D, K = X_train.shape
    N_test = X_test.shape[0]
    C = 2
    T = 10  # Significantly reduce for full-scale computation
    P = 5   # Reduce particles for memory/speed
    B = 100 # Increase batch to B*T=1000 < 10000
    
    key = jax.random.PRNGKey(seed)
    
    # Initialize particles manually to track trajectory
    from tabsmc.smc import init_empty, smc_step, gibbs
    keys = jax.random.split(key, P + 1)
    key = keys[0]
    init_keys = keys[1:]
    
    init_particle = lambda k: init_empty(k, C, D, K, N_train, 1.0, 1.0, mask)
    A, φ, π, θ = jax.vmap(init_particle)(init_keys)
    
    # Initialize weights and gammas
    log_weights = jnp.zeros(P)
    log_gammas = jnp.zeros(P)
    
    times = []
    test_logliks = []
    train_logliks = []
    start_time = time.time()
    
    for t in tqdm(range(T), desc="SMC steps"):
        # Deterministic batch indices
        start_idx = t * B
        end_idx = start_idx + B
        I_B = jnp.arange(start_idx, end_idx)
        X_B = X_train[I_B]
        
        # SMC step
        key, subkey = jax.random.split(key)
        particles = (A, φ, π, θ)
        particles, log_weights, log_gammas = smc_step(
            subkey, particles, log_weights, log_gammas, X_B, I_B, 1.0, 1.0, mask
        )
        A, φ, π, θ = particles
        
        # 10 rejuvenation steps
        for _ in range(10):
            key, subkey = jax.random.split(key)
            I_rejuv = jax.random.choice(subkey, N_train, shape=(B,), replace=False)
            X_rejuv = X_train[I_rejuv]
            
            # Run Gibbs step for each particle
            key, subkey = jax.random.split(key)
            keys_gibbs = jax.random.split(subkey, P)
            
            def rejuvenate_particle(p_key, p_A, p_φ, p_π, p_θ):
                A_new, φ_new, π_new, θ_new, _, _ = gibbs(
                    p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, 1.0, 1.0, mask
                )
                return A_new, φ_new, π_new, θ_new
            
            A, φ, π, θ = jax.vmap(rejuvenate_particle)(keys_gibbs, A, φ, π, θ)
        
        # Compute test log-likelihood at this step
        test_log_liks = jax.vmap(compute_test_loglik_vectorized, in_axes=(None, 0, 0))(X_test, π, θ)
        log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
        test_loglik_t = jax.scipy.special.logsumexp(test_log_liks + log_weights_normalized)
        
        # Compute train log-likelihood at this step
        train_log_liks = jax.vmap(compute_test_loglik_vectorized, in_axes=(None, 0, 0))(X_train, π, θ)
        train_loglik_t = jax.scipy.special.logsumexp(train_log_liks + log_weights_normalized)
        
        # Record results (per datapoint)
        current_time = time.time() - start_time
        test_logliks.append(float(test_loglik_t) / N_test)
        train_logliks.append(float(train_loglik_t) / N_train)
        times.append(current_time)
    
    return times, test_logliks, train_logliks


def run_multi_run_benchmark(X_train, X_test, mask, n_runs=5):
    """Run benchmark with multiple runs for each method."""
    print("Running multi-run benchmark...")
    
    sgd_all_results = []
    smc_all_results = []
    
    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")
        run_seed = 42 + run * 100  # Different seeds for each run
        
        # Run SignSGD
        print("Running SignSGD...")
        sgd_times, sgd_test_logliks, sgd_train_logliks = run_sgd_experiment(X_train, X_test, mask, seed=run_seed)
        sgd_all_results.append({
            'times': sgd_times,
            'test_logliks': sgd_test_logliks,
            'train_logliks': sgd_train_logliks,
            'run': run,
            'seed': run_seed
        })
        
        # Run SMC
        print("Running SMC...")
        smc_times, smc_test_logliks, smc_train_logliks = run_smc_experiment(X_train, X_test, mask, seed=run_seed)
        smc_all_results.append({
            'times': smc_times,
            'test_logliks': smc_test_logliks,
            'train_logliks': smc_train_logliks,
            'run': run,
            'seed': run_seed
        })
        
        print(f"Run {run + 1} results:")
        print(f"  SignSGD: {sgd_test_logliks[-1]:.4f} test, {sgd_train_logliks[-1]:.4f} train")
        print(f"  SMC: {smc_test_logliks[-1]:.4f} test, {smc_train_logliks[-1]:.4f} train")
    
    return sgd_all_results, smc_all_results


def main():
    """Run PUMS data multi-run benchmark."""
    print("PUMS Data Multi-Run Benchmark: SignSGD vs SMC")
    print("=" * 50)
    
    # Load data
    X_train, X_test, mask = load_and_process_pums_data()
    
    # Run multi-run benchmark
    sgd_all_results, smc_all_results = run_multi_run_benchmark(X_train, X_test, mask, n_runs=5)
    
    # Discard first run (JIT warmup) for both methods
    sgd_results = sgd_all_results[1:]
    smc_results = smc_all_results[1:]
    
    # Create multi-run wall-clock time vs test log-likelihood plot
    plt.figure(figsize=(12, 8))
    
    # Plot all SignSGD runs with more visible individual runs
    for i, result in enumerate(sgd_results):
        plt.plot(result['times'], result['test_logliks'], 'o-', color='orange', 
                alpha=0.6, linewidth=2, markersize=3, 
                label='SignSGD' if i == 0 else "")
    
    # Plot all SMC runs with more visible individual runs
    for i, result in enumerate(smc_results):
        plt.plot(result['times'], result['test_logliks'], '^-', color='green', 
                alpha=0.6, linewidth=2, markersize=4,
                label='SMC' if i == 0 else "")
    
    plt.xlabel('Wall-Clock Time (seconds)', fontsize=14)
    plt.ylabel('Test Log-Likelihood per Datapoint', fontsize=14)
    plt.title('Multi-Run Benchmark: SignSGD vs SMC on PUMS Data\n(4 runs each, first run discarded for JIT warmup)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Calculate statistics for console output
    sgd_final_scores = [r['test_logliks'][-1] for r in sgd_results]
    smc_final_scores = [r['test_logliks'][-1] for r in smc_results]
    sgd_final_times = [r['times'][-1] for r in sgd_results]
    smc_final_times = [r['times'][-1] for r in smc_results]
    
    plt.tight_layout()
    plt.savefig('figures/pums_multirun_benchmark.png', dpi=150, bbox_inches='tight')
    
    # Print detailed results
    print(f"\n" + "="*60)
    print("MULTI-RUN BENCHMARK RESULTS")
    print("="*60)
    print(f"SignSGD (4 runs): {np.mean(sgd_final_scores):.4f} ± {np.std(sgd_final_scores):.4f} test log-likelihood")
    print(f"                  {np.mean(sgd_final_times):.2f} ± {np.std(sgd_final_times):.2f} seconds")
    print(f"SMC (4 runs):     {np.mean(smc_final_scores):.4f} ± {np.std(smc_final_scores):.4f} test log-likelihood")
    print(f"                  {np.mean(smc_final_times):.2f} ± {np.std(smc_final_times):.2f} seconds")
    print(f"Performance gap:  {np.mean(smc_final_scores) - np.mean(sgd_final_scores):.4f} log-likelihood")
    print(f"Speed advantage:  SignSGD is {np.mean(smc_final_times) / np.mean(sgd_final_times):.2f}x faster")
    print(f"Plot saved to: figures/pums_multirun_benchmark.png")


if __name__ == "__main__":
    main()