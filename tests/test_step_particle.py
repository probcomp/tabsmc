import jax
import jax.numpy as jnp
import tabsmc.dumpy as dp
from tabsmc.smc import step_particle

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
    
    # A_B: allocations (B,) - this will be a Slot
    key, subkey = jax.random.split(key)
    A_B = dp.Slot()  # Initialize as Slot as expected by step_particle
    
    # A_one_hot: one-hot allocation matrix (N x C)
    A_one_hot_data = jnp.zeros((N, C))
    A_one_hot = dp.Array(A_one_hot_data)
    
    # φ: sufficient statistics (C x D x K)
    φ_data = jax.random.uniform(key, (C, D, K))
    φ = dp.Array(φ_data)
    
    # π: mixing weights (C,) - in log space
    key, subkey = jax.random.split(key)
    π_data = jax.random.normal(subkey, (C,))
    π_data = π_data - jax.scipy.special.logsumexp(π_data)  # Normalize in log space
    π = dp.Array(π_data)
    
    # θ: emission parameters (C x D x K) - in log space
    key, subkey = jax.random.split(key)
    θ_data = jax.random.normal(subkey, (C, D, K))
    θ_data = θ_data - jax.scipy.special.logsumexp(θ_data, axis=-1, keepdims=True)  # Normalize
    θ = dp.Array(θ_data)
    
    # Hyperparameters - convert to dumpy arrays
    α_pi = dp.Array(0.1)  # Dirichlet prior for mixing weights
    α_theta = dp.Array(0.1)  # Dirichlet prior for emissions
    
    print("Testing step_particle...")
    print(f"Dimensions: B={B}, C={C}, D={D}, K={K}, N={N}")
    
    # Call step_particle
    A_one_hot_new, φ_new, π_new, θ_new, γ, q = step_particle(
        key, X_B, I_B, A_one_hot, φ, π, θ, α_pi, α_theta
    )
    
    print("✓ step_particle executed successfully!")
    print(f"Output shapes:")
    print(f"  A_one_hot: {A_one_hot_new.shape}")
    print(f"  φ: {φ_new.shape}")
    print(f"  π: {π_new.shape}")
    print(f"  θ: {θ_new.shape}")
    print(f"  γ (log prob): {γ.shape}")
    print(f"  q (proposal): {q.shape}")
    
    # Basic sanity checks
    assert γ.data <= 0, "γ should be finite"
    assert q.data <= 0, "q should be finite"
    assert jnp.allclose(jnp.exp(jnp.array(π_new)).sum(), 1.0, atol=1e-5), "π should sum to 1"
    assert jnp.allclose(jnp.exp(jnp.array(θ_new)).sum(axis=-1), 1.0, atol=1e-5), "θ should sum to 1"
    print("\n✓ All sanity checks passed!")

if __name__ == "__main__":
    test_step_particle()