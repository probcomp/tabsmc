import jax
import jax.numpy as jnp
import tabsmc.dumpy as dp
from tabsmc.smc import smc_minibatch, init_particle


def test_smc_minibatch():
    """Test the complete SMC algorithm with minibatches."""
    # Set fixed seed for reproducibility
    key = jax.random.PRNGKey(123)
    
    # Set dimensions
    N = 1000  # Total number of data points
    D = 3     # Number of features
    K = 3     # Number of categories per feature
    C_true = 2  # True number of clusters
    C_model = 3  # Model number of clusters
    P = 10    # Number of particles
    T = 50    # Number of iterations
    B = 100   # Minibatch size (renamed from L to B to match new API)
    
    print("Testing SMC minibatch algorithm...")
    print(f"Data size: N={N}, D={D}, K={K}")
    print(f"True clusters: {C_true}, Model clusters: {C_model}")
    print(f"Particles: {P}, Iterations: {T}, Minibatch size: {B}")
    
    # Generate synthetic data from a mixture model
    # True mixing weights
    true_pi = jnp.array([0.6, 0.4])
    
    # True emission parameters (C_true x D x K)
    true_theta = jnp.array([
        # Cluster 0
        [[0.8, 0.1, 0.1],   # Feature 0
         [0.1, 0.8, 0.1],   # Feature 1
         [0.1, 0.1, 0.8]],  # Feature 2
        
        # Cluster 1
        [[0.1, 0.1, 0.8],   # Feature 0
         [0.8, 0.1, 0.1],   # Feature 1
         [0.1, 0.8, 0.1]]   # Feature 2
    ])
    
    # Generate data
    key, subkey = jax.random.split(key)
    cluster_assignments = jax.random.choice(subkey, C_true, shape=(N,), p=true_pi)
    
    # Generate features for each data point
    X = jnp.zeros((N, D, K))
    for n in range(N):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, D)
        cluster = cluster_assignments[n]
        for d in range(D):
            category = jax.random.choice(keys[d], K, p=true_theta[cluster, d])
            X = X.at[n, d, category].set(1.0)
    
    print("\nGenerated synthetic data from 2-cluster mixture model")
    
    # Convert to dumpy array as required by new SMC function
    X_dp = dp.Array(X)
    
    # Hyperparameters (as scalars)
    α_pi = 1.0
    α_theta = 1.0
    
    # Run SMC algorithm (using new streamlined version)
    print("\nRunning SMC algorithm (new dumpy-based version)...")
    key, subkey = jax.random.split(key)
    A, φ, π, θ, w, γ = smc_minibatch(subkey, X_dp, T, P, C_model, B, α_pi, α_theta)
    
    print(f"\nAlgorithm completed!")
    
    # Compute log marginal likelihood estimate from final weights
    log_Z = jax.scipy.special.logsumexp(w.data) - jnp.log(P)
    print(f"Log marginal likelihood estimate: {log_Z}")
    
    # Normalize weights to get probabilities
    w_norm = w.data - jax.scipy.special.logsumexp(w.data)
    w_probs = jnp.exp(w_norm)
    print(f"Final particle weights: {w_probs}")
    print(f"Effective sample size: {1.0 / jnp.sum(w_probs**2):.2f} / {P}")
    
    # Analyze the best particle
    best_p = jnp.argmax(w_probs)
    print(f"\nBest particle (index {best_p}, weight {w_probs[best_p]:.4f}):")
    
    # Get learned parameters for best particle
    π_best = jnp.exp(π.data[best_p])
    θ_best = jnp.exp(θ.data[best_p])
    
    print(f"Learned mixing weights: {π_best}")
    
    # Compute marginal distributions from best particle
    marginals_learned = jnp.zeros((D, K))
    for d in range(D):
        for k in range(K):
            marginals_learned = marginals_learned.at[d, k].set(
                jnp.sum(π_best * θ_best[:, d, k])
            )
    
    # Compute true marginals
    marginals_true = jnp.zeros((D, K))
    for d in range(D):
        for k in range(K):
            marginals_true = marginals_true.at[d, k].set(
                jnp.sum(true_pi * true_theta[:, d, k])
            )
    
    print("\nMarginal distributions comparison:")
    print("Feature | Learned | True")
    for d in range(D):
        print(f"   {d}    | {marginals_learned[d]} | {marginals_true[d]}")
    
    # Compute error
    error = jnp.mean(jnp.abs(marginals_learned - marginals_true))
    print(f"\nAverage marginal error: {error:.4f}")
    
    # Basic sanity checks
    assert jnp.isfinite(log_Z), "Log marginal likelihood should be finite"
    assert jnp.all(w_probs >= 0), "Weights should be non-negative"
    assert jnp.allclose(jnp.sum(w_probs), 1.0, atol=1e-6), "Weights should sum to 1"
    assert error < 0.1, f"Marginal error {error:.4f} should be < 0.1"
    
    print("\n✅ All SMC minibatch tests passed!")


def test_init_particle():
    """Test the particle initialization function."""
    key = jax.random.PRNGKey(42)
    C, D, K, N = 3, 2, 4, 100
    α_pi, α_theta = 1.0, 0.5
    
    A, φ, π, θ = init_particle(key, C, D, K, N, α_pi, α_theta)
    
    # Check shapes
    assert A.shape == (N, C)
    assert φ.shape == (C, D, K)
    assert π.shape == (C,)
    assert θ.shape == (C, D, K)
    
    # Check initialization values
    assert jnp.all(A.data == 0), "Assignments should be initialized to zero"
    assert jnp.all(φ.data == 0), "Sufficient statistics should be zero"
    
    # Check that π sums to 1 in probability space
    π_probs = jnp.exp(π.data)
    assert jnp.allclose(jnp.sum(π_probs), 1.0, atol=1e-6), "π should sum to 1"
    
    # Check that θ sums to 1 in probability space
    θ_probs = jnp.exp(θ.data)
    assert jnp.allclose(jnp.sum(θ_probs, axis=-1), 1.0, atol=1e-6), "θ should sum to 1"
    
    print("✅ init_particle test passed!")


if __name__ == "__main__":
    test_init_particle()
    test_smc_minibatch()