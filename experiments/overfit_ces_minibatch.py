"""Test overfitting ability on a small CES minibatch."""

import jax
import jax.numpy as jnp
from tabsmc.smc import smc_step, init_empty, gibbs
from tabsmc.io import load_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def sample_from_model(key, particles, log_weights, N_samples=1000):
    """Sample from the learned model."""
    _, _, π, θ = particles
    P = π.shape[0]
    
    # Sample particle indices according to weights
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights_normalized = jnp.exp(log_weights_normalized)
    
    key, subkey = jax.random.split(key)
    particle_indices = jax.random.choice(subkey, P, shape=(N_samples,), p=weights_normalized)
    
    # Sample from each chosen particle
    def sample_from_particle(p_key, p_idx):
        π_p = π[p_idx]  # (C,)
        θ_p = θ[p_idx]  # (C, D, K)
        
        # Sample cluster
        p_key, subkey = jax.random.split(p_key)
        cluster = jax.random.choice(subkey, π_p.shape[0], p=jnp.exp(π_p))
        
        # Sample features
        D, K = θ_p.shape[1], θ_p.shape[2]
        p_key, subkey = jax.random.split(p_key)
        feature_keys = jax.random.split(subkey, D)
        
        def sample_feature(f_key, d):
            return jax.random.choice(f_key, K, p=jnp.exp(θ_p[cluster, d]))
        
        categories = jax.vmap(sample_feature)(feature_keys, jnp.arange(D))
        
        # Convert to one-hot
        return jax.nn.one_hot(categories, K)
    
    # Generate samples
    keys = jax.random.split(key, N_samples)
    samples = jax.vmap(sample_from_particle)(keys, particle_indices)
    
    return samples


def compute_marginals(X, mask=None):
    """Compute 1D marginal distributions for each feature."""
    N, D, K = X.shape
    marginals = []
    
    for d in range(D):
        if mask is not None:
            # Only consider valid categories
            valid_cats = int(jnp.sum(mask[d]))
            X_d = X[:, d, :valid_cats]
        else:
            X_d = X[:, d, :]
        
        # Convert from log-space if needed
        if jnp.any(X_d < -10):  # Detect log-space format
            X_d_prob = jnp.exp(X_d)
            X_d_prob = jnp.where(jnp.isfinite(X_d_prob), X_d_prob, 0.0)
        else:
            X_d_prob = X_d
        
        # Compute marginal
        marginal = jnp.mean(X_d_prob, axis=0)
        marginals.append(marginal)
    
    return marginals


def plot_marginal_comparison(data_marginals, model_marginals, mask, save_path, n_features=16):
    """Plot comparison of data vs model marginals for first n_features."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(min(n_features, len(data_marginals))):
        ax = axes[i]
        
        # Get valid categories for this feature
        if mask is not None:
            valid_cats = int(jnp.sum(mask[i]))
            data_marg = data_marginals[i][:valid_cats]
            model_marg = model_marginals[i][:valid_cats]
        else:
            data_marg = data_marginals[i]
            model_marg = model_marginals[i]
        
        x = np.arange(len(data_marg))
        width = 0.35
        
        ax.bar(x - width/2, data_marg, width, label='Data', alpha=0.7, color='blue')
        ax.bar(x + width/2, model_marg, width, label='Model', alpha=0.7, color='red')
        
        ax.set_title(f'Feature {i}')
        ax.set_xlabel('Category')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Marginal comparison plot saved to {save_path}")


def overfit_ces_minibatch():
    """Run overfitting experiment on CES minibatch."""
    print("Loading CES dataset...")
    
    # Load CES data
    train_data, test_data, col_names, mask = load_data("data/lpm/CES")
    print(f"CES data shape: {train_data.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Number of features: {len(col_names)}")
    
    # Take first 1000 samples
    N_batch = 1000
    X_batch = train_data[:N_batch]
    print(f"Using minibatch of size: {X_batch.shape}")
    
    # Convert mask to boolean and JAX array
    mask_bool = jnp.array(mask.astype(bool))
    N, D, K = X_batch.shape
    C = 20  # Number of clusters
    
    print(f"Running SMC overfitting experiment:")
    print(f"  Minibatch size: {N}")
    print(f"  Features: {D}")
    print(f"  Max categories: {K}")
    print(f"  Clusters: {C}")
    
    # Initialize particle
    key = jax.random.PRNGKey(42)
    P = 1  # Single particle for overfitting
    α_pi = 1.0
    α_theta = 1.0
    
    # Initialize
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_empty(subkey, C, D, K, N, α_pi, α_theta, mask_bool)
    
    # Expand for single particle
    A = A[None, :]  # (1, N)
    φ = φ[None, :, :, :]  # (1, C, D, K)
    π = π[None, :]  # (1, C)
    θ = θ[None, :, :, :]  # (1, C, D, K)
    log_weights = jnp.zeros(1)
    log_gammas = jnp.zeros(1)
    
    print("\nRunning 1 SMC step + 100 rejuvenation steps...")
    
    # Run 1 SMC step on entire batch
    key, subkey = jax.random.split(key)
    I_B = jnp.arange(N)
    particles = (A, φ, π, θ)
    particles, log_weights, log_gammas, batch_log_liks = smc_step(
        subkey, particles, log_weights, log_gammas, X_batch, I_B, α_pi, α_theta, mask_bool
    )
    A, φ, π, θ = particles
    
    print("SMC step completed")
    
    # Run 100 rejuvenation steps
    for rejuv_step in tqdm(range(100), desc="Rejuvenation steps"):
        key, subkey = jax.random.split(key)
        I_rejuv = jax.random.choice(subkey, N, shape=(N,), replace=False)  # Use all data
        X_rejuv = X_batch[I_rejuv]
        
        # Run Gibbs step
        key, subkey = jax.random.split(key)
        A_new, φ_new, π_new, θ_new, _, _, _ = gibbs(
            subkey, X_rejuv, I_rejuv, A[0], φ[0], π[0], θ[0], α_pi, α_theta, mask_bool
        )
        
        # Update particle
        A = A_new[None, :]
        φ = φ_new[None, :, :, :]
        π = π_new[None, :]
        θ = θ_new[None, :, :, :]
    
    print("Rejuvenation completed")
    
    # Sample from learned model
    print("Sampling from learned model...")
    key, subkey = jax.random.split(key)
    particles_final = (A, φ, π, θ)
    model_samples = sample_from_model(subkey, particles_final, log_weights, N_samples=N_batch)
    
    # Compute marginals
    print("Computing marginals...")
    data_marginals = compute_marginals(X_batch, mask_bool)
    model_marginals = compute_marginals(model_samples, mask_bool)
    
    # Plot comparison
    os.makedirs('figures', exist_ok=True)
    plot_marginal_comparison(
        data_marginals, model_marginals, mask_bool, 
        'figures/ces_minibatch_overfit_marginals.png'
    )
    
    # Compute total variation distance for first 10 features
    tv_distances = []
    for i in range(min(10, len(data_marginals))):
        if mask_bool is not None:
            valid_cats = int(jnp.sum(mask_bool[i]))
            data_marg = data_marginals[i][:valid_cats]
            model_marg = model_marginals[i][:valid_cats]
        else:
            data_marg = data_marginals[i]
            model_marg = model_marginals[i]
        
        tv_dist = 0.5 * jnp.sum(jnp.abs(data_marg - model_marg))
        tv_distances.append(float(tv_dist))
    
    print(f"\nTotal Variation Distances (first 10 features):")
    for i, tv_dist in enumerate(tv_distances):
        print(f"  Feature {i}: {tv_dist:.4f}")
    
    mean_tv = np.mean(tv_distances)
    print(f"\nMean TV distance: {mean_tv:.4f}")
    print(f"Perfect fit would have TV distance = 0.0")
    print(f"Random model would have TV distance ≈ 1.0")


if __name__ == "__main__":
    overfit_ces_minibatch()