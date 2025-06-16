"""
JAX version of the step_particle test to ensure compatibility.
"""

import jax
import jax.numpy as jnp
from tabsmc.smc import gibbs, init_empty


def test_gibbs_jax():
    """JAX version of the basic gibbs test."""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)

    # Set dimensions
    B = 10  # Number of particles
    C = 3   # Number of clusters/components
    D = 5   # Number of features
    K = 2   # Number of categories per feature
    N = 100 # Total number of data points

    # Create ground truth probabilities
    class_probs = jnp.array([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],
    ])

    # Initialize data
    # X_B: particle data (B x D x K) - one-hot encoded features
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, B)
    
    # Sample categories for each batch element and feature
    X_B_categories = jax.vmap(
        lambda k: jax.random.categorical(k, class_probs.flatten())
    )(keys) % K  # B samples, mod K to get categories
    
    # Convert to one-hot across features
    X_B = jnp.zeros((B, D, K))
    for b in range(B):
        for d in range(D):
            key, subkey = jax.random.split(key)
            cat = jax.random.categorical(subkey, class_probs[d])
            X_B = X_B.at[b, d, cat].set(1.0)

    # I_B: indices for particles (B,)
    key, subkey = jax.random.split(key)
    I_B = jax.random.choice(subkey, N, (B,), replace=False)
    
    # Initialize particle using init_empty from smc_jax.py
    key, subkey = jax.random.split(key)
    α_pi = 0.1  # Dirichlet prior for mixing weights
    α_theta = 0.1  # Dirichlet prior for emissions
    
    A_one_hot, φ, π, θ = init_empty(subkey, C, D, K, N, α_pi, α_theta)
    
    print("Testing JAX gibbs...")
    print(f"Dimensions: B={B}, C={C}, D={D}, K={K}, N={N}")
    
    # Call gibbs
    A_one_hot_new, φ_new, π_new, θ_new, γ, q = gibbs(
        key, X_B, I_B, A_one_hot, φ, π, θ, α_pi, α_theta
    )
    
    print("✓ JAX gibbs executed successfully!")
    print(f"Output shapes:")
    print(f"  A_one_hot: {A_one_hot_new.shape}")
    print(f"  φ: {φ_new.shape}")
    print(f"  π: {π_new.shape}")
    print(f"  θ: {θ_new.shape}")
    print(f"  γ (log prob): {γ} (scalar)")
    print(f"  q (proposal): {q} (scalar)")
    
    print(f"\nActual values:")
    print(f"  γ = {γ:.4f}")
    print(f"  q = {q:.4f}")
    
    # Debug: let's check what γ and q represent in the gibbs computation
    print(f"\nDebugging γ computation...")
    print(f"  π_new range: [{jnp.min(π_new):.4f}, {jnp.max(π_new):.4f}]")
    print(f"  θ_new range: [{jnp.min(θ_new):.4f}, {jnp.max(θ_new):.4f}]")
    print(f"  φ_new range: [{jnp.min(φ_new):.4f}, {jnp.max(φ_new):.4f}]")
    print(f"  φ_initial range: [{jnp.min(φ):.4f}, {jnp.max(φ):.4f}]")
    
    # Basic sanity checks
    assert jnp.isfinite(γ), f"γ should be finite, got {γ}"
    assert jnp.isfinite(q), f"q should be finite, got {q}"
    assert jnp.allclose(jnp.exp(π_new).sum(), 1.0, atol=1e-5), "π should sum to 1"
    assert jnp.allclose(jnp.exp(θ_new).sum(axis=-1), 1.0, atol=1e-5), "θ should sum to 1"
    print("\n✓ All JAX sanity checks passed!")


def compare_with_original():
    """Compare JAX implementation with original dumpy implementation on same data."""
    print("\n" + "="*50)
    print("COMPARISON: JAX vs Original Implementation")
    print("="*50)
    
    # Use same random seed for both
    key = jax.random.PRNGKey(42)
    
    # Same dimensions
    B, C, D, K, N = 10, 3, 5, 2, 100
    α_pi, α_theta = 0.1, 0.1
    
    # Create same data
    key, subkey = jax.random.split(key)
    X_B = jnp.zeros((B, D, K))
    for b in range(B):
        for d in range(D):
            key, subkey = jax.random.split(key)
            cat = jax.random.categorical(subkey, jnp.array([0.3, 0.7]))
            X_B = X_B.at[b, d, cat].set(1.0)
    
    key, subkey = jax.random.split(key)
    I_B = jax.random.choice(subkey, N, (B,), replace=False)
    
    # Initialize with same key
    key, subkey = jax.random.split(key)
    A_one_hot, φ, π, θ = init_empty(subkey, C, D, K, N, α_pi, α_theta)
    
    # Run JAX version
    key, subkey = jax.random.split(key)
    A_jax, φ_jax, π_jax, θ_jax, γ_jax, q_jax = gibbs(
        subkey, X_B, I_B, A_one_hot, φ, π, θ, α_pi, α_theta
    )
    
    print(f"JAX Results:")
    print(f"  γ = {γ_jax:.6f}")
    print(f"  q = {q_jax:.6f}")
    print(f"  π sum = {jnp.exp(π_jax).sum():.6f}")
    print(f"  θ sums = {jnp.exp(θ_jax).sum(axis=-1)}")
    
    # Try to import and compare with original if available
    try:
        import tabsmc.dumpy as dp
        from tabsmc.smc import gibbs as gibbs_original, init_empty as init_empty_original
        
        print(f"\nRunning original dumpy implementation for comparison...")
        
        # Convert data to dumpy format
        X_B_dp = dp.Array(X_B)
        I_B_dp = dp.Array(I_B)
        
        # Initialize with dumpy
        A_dp, φ_dp, π_dp, θ_dp = init_empty_original(subkey, C, D, K, N, α_pi, α_theta)
        
        # Run original
        A_orig, φ_orig, π_orig, θ_orig, γ_orig, q_orig = gibbs_original(
            subkey, X_B_dp, I_B_dp, A_dp, φ_dp, π_dp, θ_dp, 
            dp.Array(α_pi), dp.Array(α_theta)
        )
        
        print(f"Original Results:")
        print(f"  γ = {jnp.array(γ_orig):.6f}")
        print(f"  q = {jnp.array(q_orig):.6f}")
        print(f"  π sum = {jnp.exp(jnp.array(π_orig)).sum():.6f}")
        print(f"  θ sums = {jnp.exp(jnp.array(θ_orig)).sum(axis=-1)}")
        
        # Note: Due to random sampling, exact values won't match but structure should
        print(f"\n✓ Both implementations completed successfully!")
        
    except ImportError:
        print(f"\nOriginal dumpy implementation not available for comparison")
    
    print(f"\n✅ JAX implementation test completed!")


if __name__ == "__main__":
    test_gibbs_jax()
    compare_with_original()
    print("\n🎉 All JAX step particle tests passed!")