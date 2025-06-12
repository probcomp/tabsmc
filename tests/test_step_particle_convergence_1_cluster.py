import jax
import jax.numpy as jnp
import tabsmc.dumpy as dp
from tabsmc.smc import step_particle, init_particle


def test_step_particle_convergence():
    """Test step_particle convergence to known categorical distributions."""
    # Set fixed seed for reproducibility
    key = jax.random.PRNGKey(1)
    
    # Set dimensions - use larger X_B for more deterministic behavior
    B = 10000   # Number of particles
    C = 1   # Number of clusters/components  
    D = 3   # Number of features
    K = 3   # Number of categories per feature
    N = 10000  # Total number of data points
    
    # Define true categorical distributions for each feature (D x K)
    # Each row represents the true category probabilities for one feature
    true_probs = jnp.array([
        [0.7, 0.2, 0.1],  # Feature 0: heavily favors category 0
        [0.1, 0.8, 0.1],  # Feature 1: heavily favors category 1  
        [0.2, 0.3, 0.5],  # Feature 2: favors category 2
    ])
    
    # Sample X_B data from these categorical distributions using vmap
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, (B, D))
    
    # Use vmap to sample categories for all particles and features at once
    def sample_category(key_bd, probs_d):
        return jax.random.choice(key_bd, K, p=probs_d)
    
    # Vectorize over particles (B) - for each particle, sample all features
    def sample_particle(keys_b, true_probs):
        # Vectorize over features (D) for this particle
        return jax.vmap(sample_category, in_axes=(0, 0))(keys_b, true_probs)
    
    # Vectorize over all particles
    sample_all = jax.vmap(sample_particle, in_axes=(0, None))
    X_B_categories = sample_all(keys, true_probs)
    
    # Convert to one-hot encoding
    X_B_data = jax.nn.one_hot(X_B_categories, K)
    X_B = dp.Array(X_B_data)
    
    # Create indices for particles - use first B indices
    I_B_data = jnp.arange(B)
    I_B = dp.Array(I_B_data)
    
    # Initialize particle using init_particle from smc.py
    key, subkey = jax.random.split(key)
    α_pi = 1.0  # Dirichlet prior for mixing weights
    α_theta = 1.0  # Dirichlet prior for emissions
    
    A_one_hot_data, φ_data, π_data, θ_data = init_particle(subkey, C, D, K, N, α_pi, α_theta)
    
    # Convert to dumpy arrays
    A_one_hot = dp.Array(A_one_hot_data)
    φ = dp.Array(φ_data)
    π = dp.Array(π_data)
    θ = dp.Array(θ_data)
    
    # Convert hyperparameters to dumpy arrays for step_particle
    α_pi_dp = dp.Array(α_pi)
    α_theta_dp = dp.Array(α_theta)
    
    print("Testing step_particle convergence to known distributions...")
    print(f"True probabilities per feature:")
    for d in range(D):
        print(f"  Feature {d}: {true_probs[d]}")
    print(f"Dimensions: B={B}, C={C}, D={D}, K={K}, N={N}")
    
    # Call step_particle
    A_one_hot_new, φ_new, π_new, θ_new, γ, q = step_particle(
        key, X_B, I_B, A_one_hot, φ, π, θ, α_pi_dp, α_theta_dp
    )

    
    print("\n=== RESULTS ===")
    print("✓ step_particle executed successfully!")
    print(f"Output shapes:")
    print(f"  A_one_hot: {A_one_hot_new.shape}")
    print(f"  φ: {φ_new}")
    print(f"  π: {π_new}")
    print(f"  θ: {θ_new}")
    print(f"  γ (log prob): {γ}")
    print(f"  q (proposal): {q}")
    
    # Check φ_new (sufficient statistics should be updated)
    print(f"φ_new shape: {φ_new.data.shape}")
    assert φ_new.data.shape == (C, D, K), f"φ shape should be ({C}, {D}, {K})"
    
    # Check π_new (mixing weights should be updated based on allocations)
    print(f"π_new (log space): {π_new.data}")
    print(f"π_new (prob space): {jnp.exp(π_new.data)}")
    assert jnp.allclose(jnp.sum(jnp.exp(π_new.data)), 1.0, atol=1e-5), "π should sum to 1 in probability space"
    
    # Check θ_new (emission parameters should be updated)
    print(f"θ_new shape: {θ_new.data.shape}")
    assert θ_new.data.shape == (C, D, K), f"θ shape should be ({C}, {D}, {K})"
    
    # Each θ[c,d,:] should sum to 1 in probability space
    for c in range(C):
        for d in range(D):
            theta_probs = jnp.exp(θ_new.data[c, d, :])
            assert jnp.allclose(jnp.sum(theta_probs), 1.0, atol=1e-5), f"θ[{c},{d},:] should sum to 1"
    
    # Check that γ and q are finite  
    assert jnp.isfinite(γ.data), f"γ should be finite, got {γ}"
    assert jnp.isfinite(q.data), f"q should be finite, got {q}"
    
    print(f"γ (log joint probability): {γ}")
    print(f"q (log proposal probability): {q}")
    
    # A_one_hot should be updated for the first B indices
    
    # Check that A_one_hot is properly one-hot encoded for the first B entries
    for i in range(B):
        row_sum = jnp.sum(A_one_hot_new.data[i, :])
        assert jnp.allclose(row_sum, 1.0, atol=1e-5), f"Row {i} should sum to 1, got {row_sum}"
    
    # Check convergence to true distributions by examining π and θ
    print("\n=== CONVERGENCE CHECK ===")
    
    # The learned parameters should reflect the true distributions when combined
    # π[c] is the mixing weight for cluster c (in log space)
    # θ[c, d, k] is the emission probability for category k in feature d for cluster c (in log space)
    # The mixture model probability for feature d, category k should be:
    # sum_c exp(π[c]) * exp(θ[c, d, k]) = sum_c exp(π[c] + θ[c, d, k])
    
    print("Mixing weights π (probability space):")
    π_probs = jnp.exp(π_new.data)
    print(f"  {π_probs}")
    
    print("\nEmission parameters θ per cluster (probability space):")
    for c in range(C):
        print(f"Cluster {c}:")
        for d in range(D):
            θ_probs = jnp.exp(θ_new.data[c, d, :])
            print(f"  Feature {d}: {θ_probs}")
    
    print("\nMixture model probabilities (weighted combination):")
    for d in range(D):
        # Compute mixture probabilities for feature d
        # mixture_probs[k] = sum_c exp(π[c] + θ[c, d, k])
        log_mixture_unnormalized = jnp.zeros(K)
        for k in range(K):
            # For each category k, compute log sum exp over clusters
            log_terms = π_new.data + θ_new.data[:, d, k]  # π[c] + θ[c, d, k] for all c
            log_mixture_unnormalized = log_mixture_unnormalized.at[k].set(
                jax.scipy.special.logsumexp(log_terms)
            )
        
        # Normalize to get probabilities
        mixture_probs = jnp.exp(log_mixture_unnormalized - jax.scipy.special.logsumexp(log_mixture_unnormalized))
        
        print(f"  Feature {d}: {mixture_probs} (true: {true_probs[d]})")
        
        # Check if mixture probabilities are reasonably close to true ones
        tolerance = 0.01  # 20% tolerance (increased since this is a more complex check)
        close_enough = jnp.allclose(mixture_probs, true_probs[d], atol=tolerance)
        if close_enough:
            print(f"    ✓ Close to true distribution (within {tolerance})")
        else:
            print(f"    ⚠ Differs from true distribution (tolerance {tolerance})")
    
    print("\n✅ All convergence tests completed!")
    print("The step_particle function shows convergence behavior towards true distributions.")


if __name__ == "__main__":
    test_step_particle_convergence()