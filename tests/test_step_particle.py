#!/usr/bin/env python3
"""Test for the step_particle function."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import dumpy as dp
from smc import step_particle

def test_step_particle():
    """Basic test for step_particle function."""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Set dimensions
    B = 10  # Number of particles
    C = 3   # Number of clusters/components
    D = 5   # Number of features
    K = 2   # Number of categories per feature
    N = 100 # Total number of data points
    
    # Initialize data
    # X_B: particle data (B x C x D x K) - one-hot encoded features
    X_B_data = jax.random.uniform(key, (B, C, D, K))
    X_B_data = X_B_data / X_B_data.sum(axis=-1, keepdims=True)  # Normalize to make it like one-hot
    X_B = dp.MappedArray(X_B_data, ['B', 'C', 'D', 'K'])  # Convert to MappedArray with named axes
    
    # I_B: indices for particles (B,)
    key, subkey = jax.random.split(key)
    I_B_data = jax.random.choice(subkey, N, (B,), replace=False)
    I_B = dp.MappedArray(I_B_data, ['B'])
    
    # A_B: allocations (B,) - this will be a Slot
    key, subkey = jax.random.split(key)
    A_B = dp.Slot()  # Initialize as Slot as expected by step_particle
    
    # A_one_hot: one-hot allocation matrix (N x C)
    A_one_hot_data = jnp.zeros((N, C))
    A_one_hot = dp.MappedArray(A_one_hot_data, ['N', 'C'])
    
    # φ: sufficient statistics (C x D x K)
    φ_data = jax.random.uniform(key, (C, D, K))
    φ = dp.MappedArray(φ_data, ['C', 'D', 'K'])
    
    # π: mixing weights (C,) - in log space
    key, subkey = jax.random.split(key)
    π_data = jax.random.normal(subkey, (C,))
    π_data = π_data - jax.scipy.special.logsumexp(π_data)  # Normalize in log space
    π = dp.MappedArray(π_data, ['C'])
    
    # θ: emission parameters (C x D x K) - in log space
    key, subkey = jax.random.split(key)
    θ_data = jax.random.normal(subkey, (C, D, K))
    θ_data = θ_data - jax.scipy.special.logsumexp(θ_data, axis=-1, keepdims=True)  # Normalize
    θ = dp.MappedArray(θ_data, ['C', 'D', 'K'])
    
    # Hyperparameters - convert to dumpy arrays
    α_pi = dp.Array(jnp.ones(C) * 0.1)  # Dirichlet prior for mixing weights
    α_theta = dp.Array(jnp.ones(K) * 0.1)  # Dirichlet prior for emissions
    
    print("Testing step_particle...")
    print(f"Dimensions: B={B}, C={C}, D={D}, K={K}, N={N}")
    
    # Call step_particle
    try:
        A_one_hot_new, φ_new, π_new, θ_new, γ, q = step_particle(
            key, X_B, I_B, A_B, A_one_hot, φ, π, θ, α_pi, α_theta
        )
        
        print("✓ step_particle executed successfully!")
        print(f"Output shapes:")
        print(f"  A_one_hot: {A_one_hot_new.shape}")
        print(f"  φ: {φ_new.shape}")
        print(f"  π: {π_new.shape}")
        print(f"  θ: {θ_new.shape}")
        print(f"  γ (log prob): {γ}")
        print(f"  q (proposal): {q}")
        
        # Basic sanity checks
        assert jnp.isfinite(γ), "γ should be finite"
        assert jnp.isfinite(q), "q should be finite"
        assert jnp.allclose(jnp.exp(π_new).sum(), 1.0, atol=1e-5), "π should sum to 1"
        print("\n✓ All sanity checks passed!")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_step_particle()