"""Generate and save synthetic data from overfitted CES model."""

import jax
import jax.numpy as jnp
from tabsmc.smc import smc_step, init_empty, gibbs
from tabsmc.io import load_data
import numpy as np
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


def sample_from_model(key, particles, log_weights, N_samples=10000, mask=None):
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


def run_overfit_generation():
    """Run experiment to generate synthetic data from overfitted model."""
    print("Loading CES dataset...")
    
    # Load CES data
    train_data_log, test_data_log, col_names, mask = load_data("data/lpm/CES")
    
    # Convert from log-space to proper one-hot encoding
    train_data = (train_data_log == 0.0).astype(np.float32)
    
    # Take first 1000 samples for training
    N_batch = 1000
    X_batch = train_data[:N_batch]
    
    # Convert mask to boolean and JAX array
    mask_bool = jnp.array(mask.astype(bool))
    N, D, K = X_batch.shape
    C = 20  # Number of clusters
    
    print(f"Training on {N_batch} samples")
    print(f"Running SMC experiment with extremely small alphas...")
    
    # Initialize particle with extremely small alphas for point posterior
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
    
    print("Generating synthetic data from overfitted model...")
    # Generate synthetic data from the final model
    key, subkey = jax.random.split(key)
    particles_final = (A, φ, π, θ)
    N_synthetic = 10000
    synthetic_data = sample_from_model(subkey, particles_final, log_weights, N_samples=N_synthetic, mask=mask_bool)
    
    # Save synthetic data
    os.makedirs('synthetic_data', exist_ok=True)
    
    # Convert to numpy for saving
    synthetic_np = np.array(synthetic_data)
    mask_np = np.array(mask_bool)
    
    # Create metadata
    metadata = {
        'dataset': 'CES',
        'model': 'SMC_overfitted',
        'train_samples': N_batch,
        'synthetic_samples': N_synthetic,
        'alpha_pi': α_pi,
        'alpha_theta': α_theta,
        'clusters': C,
        'rejuvenation_steps': 100
    }
    
    # Save data
    save_path = 'synthetic_data/ces_overfitted_synthetic.npz'
    np.savez(save_path, 
             X=synthetic_np, 
             mask=mask_np,
             metadata=metadata)
    
    print(f"\nResults:")
    print(f"  Training samples: {N_batch}")
    print(f"  Synthetic samples generated: {N_synthetic}")
    print(f"  Data shape: {synthetic_np.shape}")
    print(f"  Saved to: {save_path}")
    
    return synthetic_np, mask_np, metadata


if __name__ == "__main__":
    synthetic_data, mask, metadata = run_overfit_generation()