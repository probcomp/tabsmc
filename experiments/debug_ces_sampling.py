"""Debug the CES sampling issue to find why model always outputs category 0."""

import jax
import jax.numpy as jnp
from tabsmc.smc import smc_step, init_empty, gibbs
from tabsmc.io import load_data
import numpy as np


def debug_model_state(particles, log_weights):
    """Debug the learned model state."""
    _, _, π, θ = particles
    
    print("=== MODEL STATE DEBUG ===")
    print(f"π shape: {π.shape}")
    print(f"θ shape: {θ.shape}")
    print(f"log_weights shape: {log_weights.shape}")
    
    # Check π values
    π_probs = jnp.exp(π[0])  # Convert from log space
    print(f"\nMixing weights π (first 10): {π_probs[:10]}")
    print(f"π sum: {jnp.sum(π_probs)}")
    print(f"π max: {jnp.max(π_probs)}")
    print(f"π argmax: {jnp.argmax(π_probs)}")
    
    # Check θ values for first few features and clusters
    print(f"\nθ values for cluster 0, feature 0:")
    θ_00 = jnp.exp(θ[0, 0, 0, :5])  # First 5 categories
    print(f"  Categories 0-4: {θ_00}")
    print(f"  Sum: {jnp.sum(θ_00)}")
    
    print(f"\nθ values for cluster 1, feature 0:")
    θ_10 = jnp.exp(θ[0, 1, 0, :5])
    print(f"  Categories 0-4: {θ_10}")
    print(f"  Sum: {jnp.sum(θ_10)}")
    
    # Check if there are any finite values in θ
    θ_finite = jnp.isfinite(θ[0])
    print(f"\nθ finite values: {jnp.sum(θ_finite)} / {jnp.size(θ_finite)}")
    
    # Check for -inf values (which would become 0 in exp)
    θ_neginf = θ[0] == -jnp.inf
    print(f"θ -inf values: {jnp.sum(θ_neginf)} / {jnp.size(θ_neginf)}")
    
    return π_probs, θ


def debug_sampling_step_by_step(key, particles, log_weights, mask=None):
    """Debug sampling step by step."""
    _, _, π, θ = particles
    
    print("\n=== SAMPLING DEBUG ===")
    
    # Sample particle (should be 0 since we have only 1 particle)
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights_normalized = jnp.exp(log_weights_normalized)
    print(f"Normalized weights: {weights_normalized}")
    
    key, subkey = jax.random.split(key)
    particle_idx = jax.random.choice(subkey, 1, p=weights_normalized)
    print(f"Selected particle: {particle_idx}")
    
    # Get parameters for selected particle
    π_p = π[particle_idx]  # (C,)
    θ_p = θ[particle_idx]  # (C, D, K)
    
    print(f"π_p shape: {π_p.shape}")
    print(f"θ_p shape: {θ_p.shape}")
    
    # Sample cluster
    π_p_probs = jnp.exp(π_p)
    print(f"Cluster probabilities: {π_p_probs[:5]}")
    print(f"Sum of cluster probs: {jnp.sum(π_p_probs)}")
    
    key, subkey = jax.random.split(key)
    cluster = jax.random.choice(subkey, π_p.shape[0], p=π_p_probs)
    print(f"Selected cluster: {cluster}")
    
    # Sample first feature
    θ_cluster_feat0 = θ_p[cluster, 0, :]
    print(f"θ for cluster {cluster}, feature 0: {θ_cluster_feat0[:5]}")
    
    θ_cluster_feat0_probs = jnp.exp(θ_cluster_feat0)
    print(f"Category probabilities for feature 0: {θ_cluster_feat0_probs[:5]}")
    print(f"Sum: {jnp.sum(θ_cluster_feat0_probs)}")
    
    if mask is not None:
        valid_cats = int(jnp.sum(mask[0]))
        print(f"Valid categories for feature 0: {valid_cats}")
        θ_cluster_feat0_valid = θ_cluster_feat0[:valid_cats]
        θ_cluster_feat0_probs_valid = jnp.exp(θ_cluster_feat0_valid)
        print(f"Valid category probs: {θ_cluster_feat0_probs_valid}")
        print(f"Valid sum: {jnp.sum(θ_cluster_feat0_probs_valid)}")
    
    key, subkey = jax.random.split(key)
    category = jax.random.choice(subkey, θ_cluster_feat0.shape[0], p=θ_cluster_feat0_probs)
    print(f"Sampled category for feature 0: {category}")
    
    return cluster, category


def debug_ces_sampling():
    """Debug the CES sampling issue."""
    print("Loading CES dataset...")
    
    # Load CES data
    train_data, test_data, col_names, mask = load_data("data/lpm/CES")
    
    # Take first 100 samples for faster debugging
    N_batch = 100
    X_batch = train_data[:N_batch]
    
    # Convert mask to boolean and JAX array
    mask_bool = jnp.array(mask.astype(bool))
    N, D, K = X_batch.shape
    C = 5  # Fewer clusters for debugging
    
    print(f"Data shape: {X_batch.shape}")
    print(f"First few data points, feature 0: {X_batch[:3, 0, :5]}")
    
    # Initialize particle
    key = jax.random.PRNGKey(42)
    P = 1  # Single particle
    α_pi = 1.0
    α_theta = 1.0
    
    # Initialize
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_empty(subkey, C, D, K, N, α_pi, α_theta, mask_bool)
    
    print(f"\nAfter initialization:")
    print(f"A shape: {A.shape}")
    print(f"φ shape: {φ.shape}")
    print(f"π shape: {π.shape}")  
    print(f"θ shape: {θ.shape}")
    
    # Check initial θ values
    print(f"Initial θ[0,0,:5]: {θ[0, 0, :5]}")
    print(f"Initial θ exp: {jnp.exp(θ[0, 0, :5])}")
    
    # Expand for single particle
    A = A[None, :]
    φ = φ[None, :, :, :]
    π = π[None, :]
    θ = θ[None, :, :, :]
    log_weights = jnp.zeros(1)
    log_gammas = jnp.zeros(1)
    
    print(f"\nAfter expanding for single particle:")
    print(f"π shape: {π.shape}")
    print(f"θ shape: {θ.shape}")
    
    # Run 1 SMC step
    key, subkey = jax.random.split(key)
    I_B = jnp.arange(N)
    particles = (A, φ, π, θ)
    particles, log_weights, log_gammas, batch_log_liks = smc_step(
        subkey, particles, log_weights, log_gammas, X_batch, I_B, α_pi, α_theta, mask_bool
    )
    A, φ, π, θ = particles
    
    print(f"\nAfter SMC step:")
    particles_final = (A, φ, π, θ)
    π_probs, θ_debug = debug_model_state(particles_final, log_weights)
    
    # Debug sampling
    key, subkey = jax.random.split(key)
    cluster, category = debug_sampling_step_by_step(subkey, particles_final, log_weights, mask_bool)


if __name__ == "__main__":
    debug_ces_sampling()