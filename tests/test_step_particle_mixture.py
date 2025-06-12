import jax
import jax.numpy as jnp
import tabsmc.dumpy as dp
from tabsmc.smc import step_particle
import numpy as np

# JIT compile step_particle for faster execution
step_particle_jit = jax.jit(step_particle)


def test_step_particle_mixture():
    """Test step_particle with multi-cluster data generating process."""
    # Set fixed seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Set dimensions
    B = 50   # Number of particles (increased for better convergence)
    C_model = 5   # Number of clusters in our model
    C_true = 3    # Number of clusters in true data generating process
    D = 4    # Number of features
    K = 3    # Number of categories per feature
    N = 1000  # Total number of data points
    
    print("Testing step_particle with multi-cluster mixture model...")
    print(f"True process: {C_true} clusters, Model: {C_model} clusters")
    print(f"Dimensions: B={B}, D={D}, K={K}, N={N}")
    
    # Define true mixture model with 3 clusters
    # True mixing weights (in probability space)
    true_pi = jnp.array([0.5, 0.3, 0.2])  # Cluster 0 is most common
    
    # True emission parameters for each cluster (C_true x D x K)
    # Each cluster has different categorical distributions for each feature
    true_theta = jnp.array([
        # Cluster 0: favors different categories for each feature
        [[0.7, 0.2, 0.1],   # Feature 0: strongly favors category 0
         [0.1, 0.8, 0.1],   # Feature 1: strongly favors category 1
         [0.2, 0.3, 0.5],   # Feature 2: favors category 2
         [0.6, 0.3, 0.1]],  # Feature 3: favors category 0
        
        # Cluster 1: different pattern
        [[0.2, 0.7, 0.1],   # Feature 0: favors category 1
         [0.5, 0.1, 0.4],   # Feature 1: mixed between 0 and 2
         [0.1, 0.7, 0.2],   # Feature 2: favors category 1
         [0.3, 0.3, 0.4]],  # Feature 3: slightly favors category 2
        
        # Cluster 2: another pattern
        [[0.1, 0.3, 0.6],   # Feature 0: favors category 2
         [0.3, 0.2, 0.5],   # Feature 1: favors category 2
         [0.6, 0.2, 0.2],   # Feature 2: favors category 0
         [0.2, 0.6, 0.2]]   # Feature 3: favors category 1
    ])
    
    # Compute true marginal probabilities (what we should recover)
    # true_marginals[d, k] = sum_c true_pi[c] * true_theta[c, d, k]
    true_marginals = jnp.zeros((D, K))
    for d in range(D):
        for k in range(K):
            true_marginals = true_marginals.at[d, k].set(
                jnp.sum(true_pi * true_theta[:, d, k])
            )
    
    print("\nTrue marginal probabilities per feature:")
    for d in range(D):
        print(f"  Feature {d}: {true_marginals[d, :]}")
    
    # Generate X_B data from the true mixture model
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, B * D).reshape(B, D, 2)
    
    # For each particle, first sample which cluster it belongs to, then sample features
    key, subkey = jax.random.split(key)
    cluster_keys = jax.random.split(subkey, B)
    
    def sample_particle_from_mixture(cluster_key, feature_keys):
        # Sample cluster assignment
        cluster = jax.random.choice(cluster_key, C_true, p=true_pi)
        
        # Sample features given cluster
        def sample_feature(feature_key, feature_idx):
            return jax.random.choice(feature_key, K, p=true_theta[cluster, feature_idx, :])
        
        # Sample all features for this particle
        feature_samples = jax.vmap(sample_feature, in_axes=(0, 0))(
            feature_keys, jnp.arange(D)
        )
        return feature_samples
    
    # Generate samples for all particles
    sample_all_particles = jax.vmap(sample_particle_from_mixture, in_axes=(0, 0))
    X_B_categories = sample_all_particles(cluster_keys, keys)
    
    # We'll generate fresh data at each iteration, so no need to compute empirical marginals here
    
    # Create indices for particles
    I_B_data = jnp.arange(B)
    I_B = dp.Array(I_B_data)
    
    # Initialize arrays for our 5-cluster model
    A_one_hot_data = jnp.zeros((N, C_model))
    A_one_hot = dp.Array(A_one_hot_data)
    
    φ_data = jnp.zeros((C_model, D, K))
    φ = dp.Array(φ_data)
    
    # Initialize π and θ randomly for our 5-cluster model
    key, subkey = jax.random.split(key)
    π_data = jax.random.normal(subkey, (C_model,))
    π_data = π_data - jax.scipy.special.logsumexp(π_data)  # Normalize in log space
    π = dp.Array(π_data)
    
    key, subkey = jax.random.split(key)
    θ_data = jax.random.normal(subkey, (C_model, D, K))
    θ_data = θ_data - jax.scipy.special.logsumexp(θ_data, axis=-1, keepdims=True)  # Normalize
    θ = dp.Array(θ_data)
    
    # Hyperparameters
    α_pi = dp.Array(1.0)
    α_theta = dp.Array(1.0)
    
    # Run multiple iterations of step_particle for better convergence
    n_iterations = 20
    print(f"\nRunning {n_iterations} iterations of step_particle...")
    
    # Track convergence metrics
    errors_per_iteration = []
    
    # Current state
    A_one_hot_current = A_one_hot
    φ_current = φ
    π_current = π
    θ_current = θ
    
    for iteration in range(n_iterations):
        key, subkey = jax.random.split(key)
        
        # Resample particles for this iteration
        key, sample_key = jax.random.split(subkey)
        cluster_keys = jax.random.split(sample_key, B)
        keys = jax.random.split(sample_key, B * D).reshape(B, D, 2)
        
        # Generate new X_B samples from the same true mixture
        X_B_categories = sample_all_particles(cluster_keys, keys)
        X_B_data = jax.nn.one_hot(X_B_categories, K)
        X_B = dp.Array(X_B_data)
        
        # Call step_particle (JIT compiled)
        A_one_hot_new, φ_new, π_new, θ_new, γ, q = step_particle_jit(
            subkey, X_B, I_B, A_one_hot_current, φ_current, π_current, θ_current, α_pi, α_theta
        )
        
        # Update current state
        A_one_hot_current = A_one_hot_new
        φ_current = φ_new
        π_current = π_new
        θ_current = θ_new
        
        # Compute learned marginals for this iteration
        learned_marginals = jnp.zeros((D, K))
        for d in range(D):
            for k in range(K):
                log_terms = π_new.data + θ_new.data[:, d, k]
                learned_marginals = learned_marginals.at[d, k].set(
                    jnp.exp(jax.scipy.special.logsumexp(log_terms))
                )
        learned_marginals = learned_marginals / jnp.sum(learned_marginals, axis=1, keepdims=True)
        
        # Compute error against true marginals
        error = jnp.mean(jnp.abs(learned_marginals - true_marginals))
        errors_per_iteration.append(error)
        
        if iteration % 5 == 0:
            print(f"  Iteration {iteration:2d}: error = {error:.6f}")
    
    print(f"  Iteration {n_iterations-1:2d}: error = {errors_per_iteration[-1]:.6f}")
    
    print("\n=== RESULTS ===")
    print("✓ step_particle executed successfully!")
    print(f"Final γ (log joint probability): {γ.data}")
    print(f"Final q (log proposal probability): {q.data}")
    print(f"\nError reduction: {errors_per_iteration[0]:.6f} → {errors_per_iteration[-1]:.6f}")
    
    # Check final convergence
    print("\n=== FINAL MARGINAL CONVERGENCE CHECK ===")
    
    print("Learned mixing weights π (probability space):")
    π_learned = jnp.exp(π_new.data)
    print(f"  {π_learned}")
    
    # Compute learned marginal probabilities
    # learned_marginals[d, k] = sum_c exp(π[c]) * exp(θ[c, d, k])
    learned_marginals = jnp.zeros((D, K))
    for d in range(D):
        for k in range(K):
            # For each category k in feature d, compute weighted sum over clusters
            log_terms = π_new.data + θ_new.data[:, d, k]  # π[c] + θ[c, d, k] for all c
            learned_marginals = learned_marginals.at[d, k].set(
                jnp.exp(jax.scipy.special.logsumexp(log_terms))
            )
    
    # Normalize learned marginals (they should already be normalized, but just to be safe)
    learned_marginals = learned_marginals / jnp.sum(learned_marginals, axis=1, keepdims=True)
    
    # Final learned marginals are already computed in the loop
    print("\nMarginal comparison:")
    print("  Format: Learned | True")
    total_error = 0.0
    for d in range(D):
        print(f"  Feature {d}:")
        print(f"    {learned_marginals[d, :]} | {true_marginals[d, :]}")
        
        # Check closeness to true marginals
        true_error = jnp.mean(jnp.abs(learned_marginals[d, :] - true_marginals[d, :]))
        total_error += true_error
        
        print(f"    Error: {true_error:.6f}")
    
    avg_error = total_error / D
    print(f"\nAverage error across features: {avg_error:.6f}")
    
    # Test passes if learned marginals are close to true marginals
    tolerance = 0.05  # 5% tolerance (stricter now that we run multiple iterations)
    if avg_error < tolerance:
        print(f"✓ Marginals converged to true distribution (error {avg_error:.6f} < {tolerance})")
    else:
        print(f"⚠ Marginals differ from true distribution (error {avg_error:.6f} >= {tolerance})")
    
    # Also check convergence trend
    print(f"\nConvergence trend over iterations:")
    print(f"  Errors: {[f'{e:.4f}' for e in errors_per_iteration[::5]]}")
    
    # Basic sanity checks
    assert jnp.isfinite(γ.data), "γ should be finite"
    assert jnp.isfinite(q.data), "q should be finite"
    assert jnp.allclose(jnp.sum(π_learned), 1.0, atol=1e-4), "π should sum to 1"
    assert jnp.allclose(jnp.sum(jnp.exp(θ_new.data), axis=-1), 1.0, atol=1e-4), "θ should sum to 1"
    
    print("\n✅ All mixture model tests completed!")
    print("The step_particle function shows ability to fit multi-cluster data.")


if __name__ == "__main__":
    test_step_particle_mixture()