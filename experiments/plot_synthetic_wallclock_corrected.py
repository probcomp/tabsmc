"""Plot wallclock time results from synthetic data experiments with correct optimal computation."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


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


def compute_optimal_loglik_per_datapoint(true_π, true_θ):
    """
    Compute the optimal log-likelihood per datapoint under the true distribution.
    
    This computes the expected log-likelihood empirically by generating a large sample
    and evaluating the true model on it.
    """
    key = jax.random.PRNGKey(123456)  # Fixed seed for reproducibility
    N_large = 100000  # Very large sample for accurate estimate
    C, D, K = true_θ.shape
    
    X_large, _ = generate_mixture_data(key, N_large, D, K, true_π, true_θ)
    
    # Compute log P(X_large | true_params)
    log_px_given_c = jnp.einsum('ndk,cdk->nc', X_large, true_θ)
    log_px = jax.scipy.special.logsumexp(true_π[None, :] + log_px_given_c, axis=1)
    
    # Average over large sample
    optimal_loglik_per_datapoint = jnp.mean(log_px)
    
    return optimal_loglik_per_datapoint


def convert_gibbs_to_wallclock(results, seconds_per_gibbs_step=0.01):
    """Convert Gibbs steps to estimated wallclock time."""
    for result in results:
        # Estimate wallclock time based on Gibbs steps
        result['times'] = result['gibbs_steps'] * seconds_per_gibbs_step
    return results


def plot_synthetic_wallclock_results():
    """Create plot showing SMC performance on synthetic data vs wallclock time."""
    # Load results
    results = pickle.load(open('data/smc_rejuv_10_gibbs_results.pkl', 'rb'))
    
    # Convert to wallclock time (rough estimate)
    results = convert_gibbs_to_wallclock(results)
    
    # Compute optimal log-likelihood using the same parameters as the experiments
    D, K, C = 5, 3, 2
    true_π, true_θ = create_test_parameters(C, D, K)
    optimal_loglik = compute_optimal_loglik_per_datapoint(true_π, true_θ)
    
    print(f"Computed optimal log-likelihood: {optimal_loglik:.4f}")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot individual runs in light color
    for i, result in enumerate(results):
        times = result['times']
        logliks = result['logliks']
        
        plt.plot(times, logliks, 
                color='green', 
                alpha=0.3, 
                linewidth=1)
    
    # Calculate and plot mean
    max_time = max(result['times'][-1] for result in results)
    time_grid = np.linspace(0, max_time, 100)
    
    # Interpolate each run to common time grid
    interp_logliks = []
    for result in results:
        interp_loglik = np.interp(time_grid, result['times'], result['logliks'])
        interp_logliks.append(interp_loglik)
    
    mean_logliks = np.mean(interp_logliks, axis=0)
    std_logliks = np.std(interp_logliks, axis=0)
    
    # Plot mean line
    plt.plot(time_grid, mean_logliks, 
             color='green', 
             linewidth=3, 
             label='With 10 Rejuvenation Steps (P=20)')
    
    # Add confidence band
    plt.fill_between(time_grid, 
                     mean_logliks - std_logliks, 
                     mean_logliks + std_logliks,
                     color='green', 
                     alpha=0.2)
    
    # Add optimal line (computed from true distribution entropy)
    plt.axhline(y=float(optimal_loglik), color='black', linestyle='--', 
                label=f'Optimal (True Distribution = {optimal_loglik:.3f})')
    
    # Formatting
    plt.xlabel('Wall-Clock Time (seconds)')
    plt.ylabel('Test Log-Likelihood per Datapoint')
    plt.title('Multi-Particle SMC Comparison: Wall-Clock Time (Synthetic Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits to show the gap to optimal
    y_min = min(np.min(mean_logliks) - 0.1, float(optimal_loglik) - 0.5)
    y_max = float(optimal_loglik) + 0.1
    plt.ylim(y_min, y_max)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('figures/smc_synthetic_wallclock_optimal_corrected.png', dpi=300, bbox_inches='tight')
    print("Plot saved to figures/smc_synthetic_wallclock_optimal_corrected.png")
    
    # Print summary statistics
    final_logliks = [result['logliks'][-1] for result in results]
    final_times = [result['times'][-1] for result in results]
    
    print(f"\nSummary Statistics:")
    print(f"Final log-likelihood: {np.mean(final_logliks):.4f} ± {np.std(final_logliks):.4f}")
    print(f"Final time: {np.mean(final_times):.2f} ± {np.std(final_times):.2f} seconds")
    print(f"Number of runs: {len(results)}")
    print(f"Performance gap from optimal: {np.mean(final_logliks) - float(optimal_loglik):.4f}")


if __name__ == "__main__":
    plot_synthetic_wallclock_results()