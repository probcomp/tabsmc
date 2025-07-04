"""Plot marginal evolution across timesteps with error bars."""

import jax
import jax.numpy as jnp
from tabsmc.smc import smc_step, init_empty, gibbs
from tabsmc.io import load_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def generate_synthetic_data(key, n_samples, pi, theta, mask=None, batch_size=1000):
    """Generate synthetic data from learned SMC parameters."""
    C, D, K = theta.shape
    
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


def plot_marginal_evolution(data_marginals, timestep_marginals, mask, save_path, X_data, test_marginals=None, features_to_plot=20):
    """Plot marginal evolution across timesteps with model points overlaid on data bars."""
    n_timesteps = len(timestep_marginals)
    
    # Create subplots - adjust layout for more features
    n_cols = 5
    n_rows = (features_to_plot + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for feat_idx in range(min(features_to_plot, len(data_marginals))):
        ax = axes[feat_idx]
        
        # Get valid categories for this feature
        if mask is not None:
            valid_cats = int(jnp.sum(mask[feat_idx]))
            data_marg = data_marginals[feat_idx][:valid_cats]
        else:
            data_marg = data_marginals[feat_idx]
            valid_cats = len(data_marg)
        
        # Count non-missing datapoints for this feature
        # A datapoint is non-missing if it has exactly one 1 (valid one-hot encoding)
        feature_data = X_data[:, feat_idx, :valid_cats]  # Only consider valid categories
        row_sums = jnp.sum(feature_data, axis=1)
        valid_rows = jnp.abs(row_sums - 1.0) < 1e-6  # Allow small numerical errors
        n_nonmissing = int(jnp.sum(valid_rows))
        
        # Collect marginals across timesteps for this feature
        feature_marginals_across_time = []
        for t in range(n_timesteps):
            timestep_marg = timestep_marginals[t][feat_idx][:valid_cats]
            feature_marginals_across_time.append(timestep_marg)
        
        # Convert to array for easier computation
        feature_marginals_array = jnp.stack(feature_marginals_across_time, axis=0)  # (n_timesteps, n_categories)
        
        # Plot data marginals as bars
        x = np.arange(valid_cats)
        ax.bar(x, data_marg, alpha=0.7, color='blue', label='Train Data', width=0.8)
        
        # Plot test marginals if provided
        if test_marginals is not None:
            test_marg = test_marginals[feat_idx][:valid_cats]
            ax.bar(x, test_marg, alpha=0.5, color='green', label='Test Data', width=0.6)
        
        # Overlay individual timestep points for each category
        for cat in range(valid_cats):
            timestep_values = feature_marginals_array[:, cat]
            # Add small random jitter to x position for visibility
            jittered_x = np.full(n_timesteps, x[cat]) + np.random.normal(0, 0.1, n_timesteps)
            ax.scatter(jittered_x, timestep_values, alpha=0.6, s=30, color='red', 
                      label='Model Points' if cat == 0 else '', zorder=3)
        
        ax.set_title(f'Feature {feat_idx} (N={n_nonmissing})')
        ax.set_xlabel('Category')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3, zorder=1)
        
        # Set y-axis to start from 0
        max_vals = [np.max(data_marg), np.max(feature_marginals_array)]
        if test_marginals is not None:
            max_vals.append(np.max(test_marg))
        max_val = max(max_vals)
        ax.set_ylim(0, max_val * 1.1)
        
        # Set x-axis limits
        ax.set_xlim(-0.5, valid_cats - 0.5)
    
    # Hide unused subplots
    for i in range(features_to_plot, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Marginal evolution plot saved to {save_path}")


def run_marginal_evolution_experiment():
    """Run experiment to track marginal evolution."""
    print("Loading CES dataset...")
    
    # Load CES data
    train_data_log, test_data_log, col_names, mask = load_data("data/lpm/CES")
    
    # Convert from log-space to proper one-hot encoding
    train_data = (train_data_log == 0.0).astype(np.float32)
    test_data = (test_data_log == 0.0).astype(np.float32)
    
    # Take first 1000 samples for training
    N_batch = 1000
    X_batch = train_data[:N_batch]
    
    # Use first 10k samples for test set (or all if less than 10k)
    N_test = min(10000, test_data.shape[0])
    X_test = test_data[:N_test]
    
    # Convert mask to boolean and JAX array
    mask_bool = jnp.array(mask.astype(bool))
    N, D, K = X_batch.shape
    C = 20  # Number of clusters
    
    print(f"Running SMC experiment with timestep tracking...")
    
    # Initialize particle
    key = jax.random.PRNGKey(42)
    P = 1  # Single particle for overfitting
    α_pi = 1e-10  # Extremely small alpha for point posterior
    α_theta = 1e-10  # Extremely small alpha for point posterior
    
    # Initialize
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_empty(subkey, C, D, K, N, α_pi, α_theta, mask_bool)
    
    # Expand for single particle
    A = A[None, :]
    φ = φ[None, :, :, :]
    π = π[None, :]
    θ = θ[None, :, :, :]
    log_weights = jnp.zeros(1)
    log_gammas = jnp.zeros(1)
    
    # Run 1 SMC step
    key, subkey = jax.random.split(key)
    I_B = jnp.arange(N)
    particles = (A, φ, π, θ)
    particles, log_weights, log_gammas, batch_log_liks = smc_step(
        subkey, particles, log_weights, log_gammas, X_batch, I_B, α_pi, α_theta, mask_bool
    )
    A, φ, π, θ = particles
    
    # Store particles from multiple timesteps
    timestep_particles = []
    save_every = 10  # Save every 10 steps
    
    # Run 100 rejuvenation steps
    for rejuv_step in tqdm(range(100), desc="Rejuvenation steps"):
        key, subkey = jax.random.split(key)
        I_rejuv = jax.random.choice(subkey, N, shape=(N,), replace=False)
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
        
        # Save particles at regular intervals
        if rejuv_step % save_every == 0 or rejuv_step >= 90:  # Save last 10 + every 10th
            timestep_particles.append((A.copy(), φ.copy(), π.copy(), θ.copy(), rejuv_step))
    
    print(f"Saved {len(timestep_particles)} timesteps")
    
    # Generate samples from each timestep
    print("Generating samples from each timestep...")
    timestep_marginals = []
    
    for i, (A_t, φ_t, π_t, θ_t, step) in enumerate(timestep_particles):
        key, subkey = jax.random.split(key)
        particles_t = (A_t, φ_t, π_t, θ_t)
        model_samples_t = sample_from_model(subkey, particles_t, log_weights, N_samples=N_batch, mask=mask_bool)
        model_marginals_t = compute_marginals(model_samples_t, mask_bool)
        timestep_marginals.append(model_marginals_t)
        print(f"  Processed timestep {i+1}/{len(timestep_particles)} (step {step})")
    
    # Compute data marginals
    print("Computing train data marginals...")
    data_marginals = compute_marginals(X_batch, mask_bool)
    
    # Compute test data marginals
    print("Computing test data marginals...")
    test_marginals = compute_marginals(X_test, mask_bool)
    
    # Create evolution plot
    os.makedirs('figures', exist_ok=True)
    n_features_to_plot = 20
    plot_marginal_evolution(
        data_marginals, timestep_marginals, mask_bool,
        'figures/ces_marginal_evolution_with_test.png',
        X_batch, test_marginals=test_marginals, features_to_plot=n_features_to_plot
    )
    
    # Print summary statistics
    print(f"\nExperiment completed:")
    print(f"  Train data points: {N_batch}")
    print(f"  Test data points: {N_test}")
    print(f"  Timesteps tracked: {len(timestep_particles)}")
    print(f"  Total features in dataset: {len(data_marginals)}")
    print(f"  Features plotted: {min(n_features_to_plot, len(data_marginals))}")


if __name__ == "__main__":
    run_marginal_evolution_experiment()