import jax
import jax.numpy as jnp
import tabsmc.dumpy as dp
from tabsmc.smc import smc_minibatch, smc_minibatch_fast, init_smc
import numpy as np


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
    L = 100   # Minibatch size
    
    print("Testing SMC minibatch algorithm...")
    print(f"Data size: N={N}, D={D}, K={K}")
    print(f"True clusters: {C_true}, Model clusters: {C_model}")
    print(f"Particles: {P}, Iterations: {T}, Minibatch size: {L}")
    
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
    
    # Hyperparameters
    α_pi = 1.0
    α_theta = 1.0
    
    # Run SMC algorithm (using fast vectorized version)
    print("\nRunning SMC algorithm (fast version with vmap)...")
    key, subkey = jax.random.split(key)
    particles, log_Z = smc_minibatch_fast(subkey, X, T, P, C_model, L, α_pi, α_theta)
    
    print(f"\nAlgorithm completed!")
    print(f"Log marginal likelihood estimate: {log_Z}")
    print(f"Final particle weights: {particles['w']}")
    print(f"Effective sample size: {1.0 / jnp.sum(particles['w']**2):.2f} / {P}")
    
    # Analyze the best particle
    best_p = jnp.argmax(particles['w'])
    print(f"\nBest particle (index {best_p}, weight {particles['w'][best_p]:.4f}):")
    
    # Get learned parameters for best particle
    π_best = jnp.exp(particles['π'][best_p])
    θ_best = jnp.exp(particles['θ'][best_p])
    
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
    assert jnp.all(particles['w'] >= 0), "Weights should be non-negative"
    assert jnp.allclose(jnp.sum(particles['w']), 1.0, atol=1e-6), "Weights should sum to 1"
    assert error < 0.1, f"Marginal error {error:.4f} should be < 0.1"
    
    print("\n✅ All SMC minibatch tests passed!")


def test_init_smc():
    """Test the initialization function."""
    key = jax.random.PRNGKey(42)
    P, C, D, K, N = 5, 3, 2, 4, 100
    α_pi, α_theta = 1.0, 0.5
    
    particles = init_smc(key, P, C, D, K, N, α_pi, α_theta)
    
    # Check shapes
    assert particles['A'].shape == (P, N, C)
    assert particles['φ'].shape == (P, C, D, K)
    assert particles['π'].shape == (P, C)
    assert particles['θ'].shape == (P, C, D, K)
    assert particles['w'].shape == (P,)
    assert particles['w_log'].shape == (P,)
    
    # Check initialization values
    assert jnp.all(particles['A'] == 0), "Assignments should be initialized to zero"
    assert jnp.all(particles['φ'] == 0), "Sufficient statistics should be zero"
    assert jnp.all(particles['w'] == 1), "Weights should be initialized to 1 (exp(0))"
    assert jnp.all(particles['w_log'] == 0), "Log weights should be zero"
    
    # Check that π sums to 1 in probability space
    π_probs = jnp.exp(particles['π'])
    assert jnp.allclose(jnp.sum(π_probs, axis=1), 1.0, atol=1e-6), "π should sum to 1"
    
    # Check that θ sums to 1 in probability space
    θ_probs = jnp.exp(particles['θ'])
    assert jnp.allclose(jnp.sum(θ_probs, axis=-1), 1.0, atol=1e-6), "θ should sum to 1"
    
    print("✅ init_smc test passed!")


if __name__ == "__main__":
    test_init_smc()
    test_smc_minibatch()