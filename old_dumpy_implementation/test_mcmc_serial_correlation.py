import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tabsmc.smc import mcmc_minibatch, gibbs
import tabsmc.dumpy as dp
from tqdm import tqdm


def compute_autocorrelation(x, max_lag=30):
    """Compute autocorrelation function for a time series."""
    x = x - np.mean(x)
    c0 = np.sum(x**2) / len(x)
    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            c_lag = np.sum(x[:-lag] * x[lag:]) / len(x)
            acf.append(c_lag / c0)
    return np.array(acf)


def run_mcmc_chain(key, X, n_samples, C, B, α_pi, α_theta, thinning=1):
    """Run MCMC chain and collect samples."""
    N, D, K = X.shape
    
    # Pre-allocate storage arrays
    π_samples = np.zeros((n_samples, C))
    θ_samples = np.zeros((n_samples, C, D, K))
    cluster_counts = np.zeros((n_samples, C))
    
    # Initialize using init_assignments
    from tabsmc.smc import init_assignments
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_assignments(subkey, X, C, D, K, α_pi, α_theta)
    
    # Run chain
    sample_idx = 0
    for i in tqdm(range(n_samples * thinning)):
        key, subkey = jax.random.split(key)
        I_B = dp.Array(jax.random.choice(subkey, N, shape=(B,), replace=False))
        X_B = dp.Slot()
        X_B["B", "D", "K"] = X[I_B["B"], "D", "K"]
        
        # One Gibbs step
        key, subkey = jax.random.split(key)
        A, φ, π, θ, _, _ = gibbs(
            subkey,
            X_B[:, :, :],
            I_B[:],
            A,
            φ,
            π,
            θ,
            dp.Array(jnp.array(α_pi)),
            dp.Array(jnp.array(α_theta))
        )
        
        # Store samples (with thinning)
        if i % thinning == 0:
            π_samples[sample_idx] = np.array(π)
            θ_samples[sample_idx] = np.array(θ)
            # Count cluster assignments
            counts_slot = dp.Slot()
            counts_slot["C"] = dp.sum(A[:, "C"])
            cluster_counts[sample_idx] = np.array(counts_slot)
            sample_idx += 1
    
    return π_samples, θ_samples, cluster_counts


def test_mcmc_serial_correlation():
    """Test MCMC mixing by analyzing serial correlation."""
    # Set random seed
    key = jax.random.PRNGKey(42)
    
    # Generate synthetic data - 2 clusters
    N, D, K, C = 10000, 5, 3, 2
    
    # True parameters
    true_π = jnp.array([0.3, 0.7])
    true_θ = jnp.zeros((C, D, K))
    # Cluster 0: prefers category 0
    true_θ = true_θ.at[0, :, 0].set(0.8)
    true_θ = true_θ.at[0, :, 1].set(0.1)
    true_θ = true_θ.at[0, :, 2].set(0.1)
    # Cluster 1: prefers category 2
    true_θ = true_θ.at[1, :, 0].set(0.1)
    true_θ = true_θ.at[1, :, 1].set(0.1)
    true_θ = true_θ.at[1, :, 2].set(0.8)
    
    # Generate data
    key, subkey = jax.random.split(key)
    assignments = jax.random.choice(subkey, C, shape=(N,), p=true_π)
    
    X = jnp.zeros((N, D, K))
    for n in range(N):
        for d in range(D):
            key, subkey = jax.random.split(key)
            category = jax.random.choice(subkey, K, p=true_θ[assignments[n], d])
            X = X.at[n, d, category].set(1.0)
    
    X = dp.Array(X)
    
    # Run MCMC
    n_samples = 50  # Reduced from 1000
    B = 1000  # Batch size
    α_pi = jnp.array(1.0)
    α_theta = jnp.array(1.0)
    
    key, subkey = jax.random.split(key)
    π_samples, θ_samples, cluster_counts = run_mcmc_chain(
        subkey, X, n_samples, C, B, α_pi, α_theta, thinning=2  # Reduced from 5
    )
    
    # Compute autocorrelations
    # For π[0] (mixing weight of first cluster)
    π0_acf = compute_autocorrelation(np.exp(π_samples[:, 0]), max_lag=30)
    
    # For cluster counts
    count0_acf = compute_autocorrelation(cluster_counts[:, 0], max_lag=30)
    
    # For θ[0,0,0] (first cluster, first feature, first category)
    θ000_acf = compute_autocorrelation(np.exp(θ_samples[:, 0, 0, 0]), max_lag=30)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Trace plot of π[0]
    axes[0, 0].plot(np.exp(π_samples[:, 0]))
    axes[0, 0].axhline(true_π[0], color='red', linestyle='--', label='True value')
    axes[0, 0].set_title('Trace plot: π[0]')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('π[0]')
    axes[0, 0].legend()
    
    # Plot 2: ACF of π[0]
    axes[0, 1].stem(π0_acf, basefmt=" ")
    axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_title('Autocorrelation: π[0]')
    axes[0, 1].set_xlabel('Lag')
    axes[0, 1].set_ylabel('ACF')
    
    # Plot 3: Trace plot of cluster counts
    axes[1, 0].plot(cluster_counts[:, 0], label='Cluster 0')
    axes[1, 0].plot(cluster_counts[:, 1], label='Cluster 1')
    axes[1, 0].set_title('Trace plot: Cluster counts')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # Plot 4: ACF of cluster 0 count
    axes[1, 1].stem(count0_acf, basefmt=" ")
    axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].set_title('Autocorrelation: Cluster 0 count')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('ACF')
    
    plt.tight_layout()
    plt.savefig('mcmc_serial_correlation.png', dpi=150)
    # plt.show()  # Commented out for non-interactive environment
    
    # Print summary statistics
    print(f"Effective sample size for π[0]: {n_samples / (1 + 2 * np.sum(π0_acf[1:])):.1f}")
    print(f"Effective sample size for cluster 0 count: {n_samples / (1 + 2 * np.sum(count0_acf[1:])):.1f}")
    print(f"Mean π[0]: {np.mean(np.exp(π_samples[:, 0])):.3f} (true: {true_π[0]:.3f})")
    print(f"Std π[0]: {np.std(np.exp(π_samples[:, 0])):.3f}")


if __name__ == "__main__":
    test_mcmc_serial_correlation()