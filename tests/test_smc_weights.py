"""Test SMC weight computations and algorithm."""

import jax
import jax.numpy as jnp
from tabsmc.smc import (
    init_empty, log_dirichlet_score, gibbs, smc_step, smc_no_rejuvenation
)


def test_log_dirichlet_score():
    """Test log_dirichlet_score function."""
    α = jnp.array([2.0, 3.0, 4.0])
    # Create a valid probability vector in log space
    prob = jnp.array([0.2, 0.3, 0.5])
    x = jnp.log(prob)
    score = log_dirichlet_score(α, x)
    
    # Assertions
    assert jnp.isfinite(score), "Score should be finite"
    assert jnp.abs(jnp.sum(prob) - 1.0) < 1e-6, "Probabilities should sum to 1"


def test_smc_weights():
    """Test SMC with weight computations."""
    # Set random seed
    key = jax.random.PRNGKey(42)
    
    # Test with synthetic data
    N, D, K, C = 500, 5, 3, 2  # Increased N to satisfy N >= B*T
    α_pi = 1.0
    α_theta = 1.0
    
    # Generate synthetic data as integer indices
    key, subkey = jax.random.split(key)
    X = jax.random.randint(subkey, (N, D), 0, K)  # Integer indices (N x D)
    
    # Test init_empty
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_empty(subkey, C, D, K, N, α_pi, α_theta)
    
    # Test one gibbs step
    B = 10
    key, subkey = jax.random.split(key)
    I_B = jax.random.choice(subkey, N, shape=(B,), replace=False)
    X_B = X[I_B]
    
    key, subkey = jax.random.split(key)
    A_new, φ_new, π_new, θ_new, γ, q, _ = gibbs(
        subkey, X_B, I_B, A, φ, π, θ, α_pi, α_theta
    )
    
    # Test SMC step with multiple particles
    P = 4  # Number of particles
    
    # Initialize particles
    keys = jax.random.split(key, P + 1)
    key = keys[0]
    init_keys = keys[1:]
    
    init_particle = lambda k: init_empty(k, C, D, K, N, α_pi, α_theta)
    A_particles, φ_particles, π_particles, θ_particles = jax.vmap(init_particle)(init_keys)
    
    log_weights = jnp.zeros(P)
    log_gammas = jnp.zeros(P)
    
    # Run one SMC step
    key, subkey = jax.random.split(key)
    particles = (A_particles, φ_particles, π_particles, θ_particles)
    particles_new, log_weights_new, log_gammas_new, _ = smc_step(
        subkey, particles, log_weights, log_gammas, X_B, I_B, α_pi, α_theta
    )
    
    # Store original P for later assertion
    P_smc_step = P
    
    # Test full SMC algorithm
    T = 10  # Number of iterations
    P = 8   # Number of particles
    B = 20  # Batch size
    
    key, subkey = jax.random.split(key)
    particles_final, log_weights_final = smc_no_rejuvenation(
        subkey, X, T, P, C, B, K, α_pi, α_theta
    )
    
    # Compute log marginal likelihood from weights
    log_ml = jax.scipy.special.logsumexp(log_weights_final) - jnp.log(P)
    
    # Additional test with larger synthetic dataset
    N_large, D_large, K_large, C_large = 1500, 10, 4, 3  # Increased N to satisfy N >= B*T = 50*20 = 1000
    
    # Generate larger synthetic dataset as integer indices
    key, subkey = jax.random.split(key)
    X_large = jax.random.randint(subkey, (N_large, D_large), 0, K_large)  # Integer indices
    
    # Run SMC on larger data
    key, subkey = jax.random.split(key)
    particles_large, weights_large = smc_no_rejuvenation(
        subkey, X_large, T=20, P=16, C=C_large, B=50, K=K_large, α_pi=1.0, α_theta=1.0
    )
    
    # Compute log marginal likelihood from weights
    log_ml_large = jax.scipy.special.logsumexp(weights_large) - jnp.log(16)
    
    # Assertions
    assert X.shape == (N, D), f"Data shape should be ({N}, {D})"
    assert A.shape == (N,), f"Assignment shape should be ({N},)"
    assert φ.shape == (C, D, K), f"Sufficient statistics shape should be ({C}, {D}, {K})"
    from tabsmc.smc import EMPTY_ASSIGNMENT
    assert jnp.all(A == EMPTY_ASSIGNMENT), "Empty initialization should have all assignments as EMPTY_ASSIGNMENT"
    assert jnp.sum(φ) == 0, "Empty initialization should have zero sufficient statistics"
    assert jnp.abs(jnp.sum(jnp.exp(π)) - 1.0) < 1e-6, "Mixing weights should sum to 1"
    
    # Test gibbs step outputs
    assert jnp.isfinite(γ), "Target log probability should be finite"
    assert jnp.isfinite(q), "Proposal log probability should be finite"
    
    # Test SMC step outputs
    assert A_particles.shape == (P_smc_step, N), f"Particle assignments shape should be ({P_smc_step}, {N})"
    assert jnp.all(jnp.isfinite(log_weights_new)), "All weights should be finite"
    assert jnp.all(jnp.isfinite(log_gammas_new)), "All log gammas should be finite"
    
    # Test final SMC outputs
    assert jnp.isfinite(log_ml), "Log marginal likelihood should be finite"
    assert jnp.isfinite(log_ml_large), "Large dataset log marginal likelihood should be finite"
    assert jnp.all(jnp.isfinite(log_weights_final)), "Final weights should be finite"
    assert jnp.all(jnp.isfinite(weights_large)), "Large dataset weights should be finite"
    
    # Test weight normalization
    normalized_weights = jnp.exp(log_weights_final - jax.scipy.special.logsumexp(log_weights_final))
    assert jnp.abs(jnp.sum(normalized_weights) - 1.0) < 1e-6, "Normalized weights should sum to 1"


def run_smc_weights_test():
    """Run SMC weights test as a script (for manual inspection)."""
    # Set random seed
    key = jax.random.PRNGKey(42)

    # Test log_dirichlet_score
    print("Testing log_dirichlet_score...")
    α = jnp.array([2.0, 3.0, 4.0])
    # Create a valid probability vector in log space
    prob = jnp.array([0.2, 0.3, 0.5])
    x = jnp.log(prob)
    score = log_dirichlet_score(α, x)
    print(f"α = {α}")
    print(f"prob = {prob} (sum = {jnp.sum(prob)})")
    print(f"x (log space) = {x}")
    print(f"log_dirichlet_score = {score}")
    print()

    # Test with synthetic data
    print("Testing SMC with synthetic data...")
    N, D, K, C = 500, 5, 3, 2  # Increased N to satisfy N >= B*T
    α_pi = 1.0
    α_theta = 1.0

    # Generate synthetic data as integer indices
    key, subkey = jax.random.split(key)
    X = jax.random.randint(subkey, (N, D), 0, K)  # Integer indices (N x D)
    print(f"Data shape: {X.shape}")

    # Test init_empty
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_empty(subkey, C, D, K, N, α_pi, α_theta)
    print(f"\\nInit empty results:")
    print(f"A shape: {A.shape}, sum: {jnp.sum(A)}")
    print(f"φ shape: {φ.shape}, sum: {jnp.sum(φ)}")
    print(f"π (log space): {π}, exp(π) sum: {jnp.sum(jnp.exp(π))}")
    print(f"θ shape: {θ.shape}")

    # Test one gibbs step
    B = 10
    key, subkey = jax.random.split(key)
    I_B = jax.random.choice(subkey, N, shape=(B,), replace=False)
    X_B = X[I_B]

    key, subkey = jax.random.split(key)
    A_new, φ_new, π_new, θ_new, γ, q, _ = gibbs(
        subkey, X_B, I_B, A, φ, π, θ, α_pi, α_theta
    )

    print(f"\\nGibbs step results:")
    print(f"γ (target log prob): {γ}")
    print(f"q (proposal log prob): {q}")
    print(f"Weight update would be: γ - q = {γ - q}")

    # Test SMC step with multiple particles
    print("\\n\\nTesting SMC step with multiple particles...")
    P = 4  # Number of particles

    # Initialize particles
    keys = jax.random.split(key, P + 1)
    key = keys[0]
    init_keys = keys[1:]

    init_particle = lambda k: init_empty(k, C, D, K, N, α_pi, α_theta)
    A_particles, φ_particles, π_particles, θ_particles = jax.vmap(init_particle)(init_keys)

    log_weights = jnp.zeros(P)
    log_gammas = jnp.zeros(P)

    print(f"Initial particle shapes:")
    print(f"A: {A_particles.shape}")
    print(f"φ: {φ_particles.shape}")
    print(f"π: {π_particles.shape}")
    print(f"θ: {θ_particles.shape}")

    # Run one SMC step
    key, subkey = jax.random.split(key)
    particles = (A_particles, φ_particles, π_particles, θ_particles)
    particles_new, log_weights_new, log_gammas_new, _ = smc_step(
        subkey, particles, log_weights, log_gammas, X_B, I_B, α_pi, α_theta
    )

    print(f"\\nAfter SMC step:")
    print(f"log_weights: {log_weights_new}")
    print(f"normalized weights: {jnp.exp(log_weights_new - jax.scipy.special.logsumexp(log_weights_new))}")
    print(f"log_gammas: {log_gammas_new}")

    # Test full SMC algorithm
    print("\\n\\nTesting full SMC algorithm...")
    T = 10  # Number of iterations
    P = 8   # Number of particles
    B = 20  # Batch size

    key, subkey = jax.random.split(key)
    particles_final, log_weights_final = smc_no_rejuvenation(
        subkey, X, T, P, C, B, K, α_pi, α_theta
    )

    # Compute log marginal likelihood from weights
    log_ml = jax.scipy.special.logsumexp(log_weights_final) - jnp.log(P)
    print(f"Final results after {T} iterations:")
    print(f"Final log weights: {log_weights_final}")
    print(f"Final normalized weights: {jnp.exp(log_weights_final - jax.scipy.special.logsumexp(log_weights_final))}")
    print(f"Log marginal likelihood estimate: {log_ml}")

    # Additional test with larger synthetic dataset
    print("\\n\\nTesting with larger synthetic dataset...")
    N_large, D_large, K_large, C_large = 1500, 10, 4, 3  # Increased N to satisfy N >= B*T = 50*20 = 1000

    # Generate larger synthetic dataset as integer indices
    key, subkey = jax.random.split(key)
    X_large = jax.random.randint(subkey, (N_large, D_large), 0, K_large)  # Integer indices

    # Run SMC on larger data
    key, subkey = jax.random.split(key)
    particles_large, weights_large = smc_no_rejuvenation(
        subkey, X_large, T=20, P=16, C=C_large, B=50, K=K_large, α_pi=1.0, α_theta=1.0
    )

    # Compute log marginal likelihood from weights
    log_ml_large = jax.scipy.special.logsumexp(weights_large) - jnp.log(16)
    print(f"Larger dataset results:")
    print(f"Data shape: {X_large.shape}")
    print(f"Final weights: {jnp.exp(weights_large - jax.scipy.special.logsumexp(weights_large))}")
    print(f"Log marginal likelihood: {log_ml_large}")

    # Test weight update formula
    print("\\n\\nVerifying weight update formula:")
    print("Weight update: w_new = w_old - γ_old + γ_new - q")
    print("This implements p(x_{1:t}) / p(x_{1:t-1}) = p(x_t | x_{1:t-1})")
    print("The log marginal likelihood accumulates these ratios over iterations")


if __name__ == "__main__":
    run_smc_weights_test()