"""Test overfitting ability on a small CES minibatch - FIXED VERSION."""

import jax
import jax.numpy as jnp
from tabsmc.smc import smc_step, init_empty, gibbs
from tabsmc.io import load_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def generate_synthetic_data(key, n_samples, pi, theta, mask=None, batch_size=1000):
    """Generate synthetic data from learned SMC parameters.
    
    NOTE: This expects pi and theta to be in PROBABILITY SPACE, not log space.
    """
    C, D, K = theta.shape
    
    # For large datasets, use smaller batches
    if n_samples > batch_size:
        all_X = []
        all_assignments = []
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            current_batch_size = end_idx - start_idx
            
            key, subkey = jax.random.split(key)
            X_batch, assignments_batch = generate_synthetic_data(
                subkey, current_batch_size, pi, theta, mask, batch_size=batch_size
            )
            
            all_X.append(X_batch)
            all_assignments.append(assignments_batch)
        
        X = jnp.concatenate(all_X, axis=0)
        assignments = jnp.concatenate(all_assignments, axis=0)
        return X, assignments
    
    # Generate cluster assignments
    key, subkey = jax.random.split(key)
    assignments = jax.random.choice(subkey, C, shape=(n_samples,), p=pi)
    
    # Generate categories for all data points and features
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_samples * D).reshape(n_samples, D, 2)
    
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
    
    # Apply mask if provided
    if mask is not None:
        X = X * mask[None, :, :]
    
    return X, assignments


def sample_from_model(key, particles, log_weights, N_samples=1000, mask=None):
    """Sample from the learned model using the existing approach."""
    _, _, π, θ = particles
    P = π.shape[0]
    
    # Sample particle indices according to weights
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights_normalized = jnp.exp(log_weights_normalized)
    
    key, subkey = jax.random.split(key)
    particle_indices = jax.random.choice(subkey, P, shape=(N_samples,), p=weights_normalized)
    
    # Sample from each chosen particle
    def sample_from_particle(p_key, p_idx):
        π_p = π[p_idx]  # (C,) in log space
        θ_p = θ[p_idx]  # (C, D, K) in log space
        
        # Convert to probability space
        π_p_probs = jnp.exp(π_p)
        π_p_probs = π_p_probs / jnp.sum(π_p_probs)  # Normalize
        
        θ_p_probs = jnp.exp(θ_p)
        # Normalize along category dimension for each cluster and feature
        θ_p_probs = θ_p_probs / jnp.sum(θ_p_probs, axis=-1, keepdims=True)
        
        # Generate one sample from this particle
        p_key, subkey = jax.random.split(p_key)
        X_sample, _ = generate_synthetic_data(subkey, 1, π_p_probs, θ_p_probs, mask)
        return X_sample[0]  # Remove batch dimension
    
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
        
        # Only average over valid (non-missing) rows
        # A row is valid if it has exactly one 1 (for one-hot data)
        row_sums = jnp.sum(X_d_prob, axis=1)
        valid_rows = jnp.abs(row_sums - 1.0) < 1e-6  # Allow small numerical errors
        
        if jnp.sum(valid_rows) > 0:
            # Compute marginal only over valid rows
            marginal = jnp.mean(X_d_prob[valid_rows], axis=0)
        else:
            # Fallback if no valid rows found
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
    train_data_log, test_data_log, col_names, mask = load_data("data/lpm/CES")
    print(f"CES data shape: {train_data_log.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Number of features: {len(col_names)}")
    
    # Convert from log-space to proper one-hot encoding
    # In log-space: 0 = log(1) = active category, -inf = log(0) = inactive category
    train_data = (train_data_log == 0.0).astype(np.float32)
    print(f"Converted to one-hot format")
    
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
    
    # Check if we have NaN values after SMC step
    if jnp.any(jnp.isnan(π)) or jnp.any(jnp.isnan(θ)):
        print("ERROR: NaN values detected after SMC step!")
        print(f"π has NaN: {jnp.any(jnp.isnan(π))}")
        print(f"θ has NaN: {jnp.any(jnp.isnan(θ))}")
        return
    
    # Store particles from multiple timesteps (last 10)
    timestep_particles = []
    n_timesteps_to_save = 10
    
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
        
        # Check for NaN values after each rejuvenation step
        if jnp.any(jnp.isnan(π)) or jnp.any(jnp.isnan(θ)):
            print(f"ERROR: NaN values detected after rejuvenation step {rejuv_step}!")
            return
        
        # Save particles from last 10 timesteps
        if rejuv_step >= 100 - n_timesteps_to_save:
            timestep_particles.append((A.copy(), φ.copy(), π.copy(), θ.copy()))
    
    print("Rejuvenation completed")
    
    # Generate samples from multiple timesteps
    print(f"Sampling from last {n_timesteps_to_save} timesteps...")
    all_model_marginals = []
    
    for i, (A_t, φ_t, π_t, θ_t) in enumerate(timestep_particles):
        key, subkey = jax.random.split(key)
        particles_t = (A_t, φ_t, π_t, θ_t)
        model_samples_t = sample_from_model(subkey, particles_t, log_weights, N_samples=N_batch, mask=mask_bool)
        model_marginals_t = compute_marginals(model_samples_t, mask_bool)
        all_model_marginals.append(model_marginals_t)
    
    # Average marginals across timesteps
    print("Averaging marginals across timesteps...")
    n_features = len(all_model_marginals[0])
    averaged_model_marginals = []
    
    for d in range(n_features):
        # Stack marginals for feature d across all timesteps
        feature_marginals = jnp.stack([marginals[d] for marginals in all_model_marginals], axis=0)
        # Average across timesteps
        avg_marginal = jnp.mean(feature_marginals, axis=0)
        averaged_model_marginals.append(avg_marginal)
    
    # Compute data marginals
    print("Computing data marginals...")
    data_marginals = compute_marginals(X_batch, mask_bool)
    
    # Plot comparison
    os.makedirs('figures', exist_ok=True)
    plot_marginal_comparison(
        data_marginals, averaged_model_marginals, mask_bool, 
        'figures/ces_minibatch_overfit_marginals_multistep.png'
    )
    
    # Compute total variation distance for first 10 features
    tv_distances = []
    for i in range(min(10, len(data_marginals))):
        if mask_bool is not None:
            valid_cats = int(jnp.sum(mask_bool[i]))
            data_marg = data_marginals[i][:valid_cats]
            model_marg = averaged_model_marginals[i][:valid_cats]
        else:
            data_marg = data_marginals[i]
            model_marg = averaged_model_marginals[i]
        
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