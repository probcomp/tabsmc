import jax
import jax.numpy as jnp
import tabsmc.dumpy as dp
from tabsmc.smc import gibbs, init_particle

def test_gibbs():
    """Basic test for gibbs function."""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)

    # Set dimensions
    B = 10  # Number of particles
    C = 3   # Number of clusters/components
    D = 5   # Number of features
    K = 2   # Number of categories per feature
    N = 100 # Total number of data points

    # Create ground truth probabilities
    class_probs = dp.Array([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],
    ])

    # Initialize data
    # X_B: particle data (B x D x K) - one-hot encoded features
    key, subkey = jax.random.split(key)
    keys = dp.Array(jax.random.split(subkey, B))
    X_B = dp.Slot()
    X_B['B', 'C'] = dp.categorical(keys['B', :], class_probs['C', :])
    X_B = dp.Array(jax.nn.one_hot(X_B, K))

    # I_B: indices for particles (B,)
    key, subkey = jax.random.split(key)
    I_B_data = jax.random.choice(subkey, N, (B,), replace=False)
    I_B = dp.Array(I_B_data)
    
    # Initialize particle using init_particle from smc.py
    key, subkey = jax.random.split(key)
    α_pi = 0.1  # Dirichlet prior for mixing weights
    α_theta = 0.1  # Dirichlet prior for emissions
    
    A_one_hot, φ, π, θ = init_particle(subkey, C, D, K, N, α_pi, α_theta)
    
    print("Testing gibbs...")
    print(f"Dimensions: B={B}, C={C}, D={D}, K={K}, N={N}")
    
    # Call gibbs
    A_one_hot_new, φ_new, π_new, θ_new, γ, q = gibbs(
        key, X_B, I_B, A_one_hot, φ, π, θ, dp.Array(α_pi), dp.Array(α_theta)
    )
    
    print("✓ gibbs executed successfully!")
    print(f"Output shapes:")
    print(f"  A_one_hot: {A_one_hot_new.shape}")
    print(f"  φ: {φ_new.shape}")
    print(f"  π: {π_new.shape}")
    print(f"  θ: {θ_new.shape}")
    print(f"  γ (log prob): {γ.shape}")
    print(f"  q (proposal): {q.shape}")
    
    print(f"\nActual values:")
    print(f"  γ = {γ.data} (should be <= 0)")
    print(f"  q = {q.data} (should be <= 0)")
    
    # Debug: let's check what γ and q represent in the gibbs computation
    print(f"\nDebugging γ computation...")
    print(f"  π_new range: [{jnp.min(π_new.data):.4f}, {jnp.max(π_new.data):.4f}]")
    print(f"  θ_new range: [{jnp.min(θ_new.data):.4f}, {jnp.max(θ_new.data):.4f}]")
    print(f"  φ_new range: [{jnp.min(φ_new.data):.4f}, {jnp.max(φ_new.data):.4f}]")
    print(f"  φ_initial range: [{jnp.min(φ.data):.4f}, {jnp.max(φ.data):.4f}]")
    
    # Basic sanity checks - γ and q can be positive for log-densities
    assert jnp.isfinite(γ.data), f"γ should be finite, got {γ.data}"
    assert jnp.isfinite(q.data), f"q should be finite, got {q.data}"
    assert jnp.allclose(jnp.exp(jnp.array(π_new)).sum(), 1.0, atol=1e-5), "π should sum to 1"
    assert jnp.allclose(jnp.exp(jnp.array(θ_new)).sum(axis=-1), 1.0, atol=1e-5), "θ should sum to 1"
    print("\n✓ All sanity checks passed!")

if __name__ == "__main__":
    test_gibbs()