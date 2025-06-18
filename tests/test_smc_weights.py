"""Test SMC weight computations and algorithm."""

import jax
import jax.numpy as jnp
from tabsmc.smc import (
    init_empty, log_dirichlet_score, gibbs, smc_step, smc_no_rejuvenation
)
# Only using synthetic data for this test

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
N, D, K, C = 100, 5, 3, 2
α_pi = 1.0
α_theta = 1.0

# Generate synthetic one-hot data
key, subkey = jax.random.split(key)
data_indices = jax.random.randint(subkey, (N, D), 0, K)
X = jax.nn.one_hot(data_indices, K)
print(f"Data shape: {X.shape}")

# Test init_empty
key, subkey = jax.random.split(key)
A, φ, π, θ = init_empty(subkey, C, D, K, N, α_pi, α_theta)
print(f"\nInit empty results:")
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
A_new, φ_new, π_new, θ_new, γ, q = gibbs(
    subkey, X_B, I_B, A, φ, π, θ, α_pi, α_theta
)

print(f"\nGibbs step results:")
print(f"γ (target log prob): {γ}")
print(f"q (proposal log prob): {q}")
print(f"Weight update would be: γ - q = {γ - q}")

# Test SMC step with multiple particles
print("\n\nTesting SMC step with multiple particles...")
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
particles_new, log_weights_new, log_gammas_new = smc_step(
    subkey, particles, log_weights, log_gammas, X_B, I_B, α_pi, α_theta
)

print(f"\nAfter SMC step:")
print(f"log_weights: {log_weights_new}")
print(f"normalized weights: {jnp.exp(log_weights_new - jax.scipy.special.logsumexp(log_weights_new))}")
print(f"log_gammas: {log_gammas_new}")

# Test full SMC algorithm
print("\n\nTesting full SMC algorithm...")
T = 10  # Number of iterations
P = 8   # Number of particles
B = 20  # Batch size

key, subkey = jax.random.split(key)
particles_final, log_weights_final, log_ml = smc_no_rejuvenation(
    subkey, X, T, P, C, B, α_pi, α_theta
)

print(f"Final results after {T} iterations:")
print(f"Final log weights: {log_weights_final}")
print(f"Final normalized weights: {jnp.exp(log_weights_final - jax.scipy.special.logsumexp(log_weights_final))}")
print(f"Log marginal likelihood estimate: {log_ml}")

# Additional test with larger synthetic dataset
print("\n\nTesting with larger synthetic dataset...")
N_large, D_large, K_large, C_large = 500, 10, 4, 3

# Generate larger synthetic dataset
key, subkey = jax.random.split(key)
data_indices_large = jax.random.randint(subkey, (N_large, D_large), 0, K_large)
X_large = jax.nn.one_hot(data_indices_large, K_large)

# Run SMC on larger data
key, subkey = jax.random.split(key)
particles_large, weights_large, log_ml_large = smc_no_rejuvenation(
    subkey, X_large, T=20, P=16, C=C_large, B=50, α_pi=1.0, α_theta=1.0
)

print(f"Larger dataset results:")
print(f"Data shape: {X_large.shape}")
print(f"Final weights: {jnp.exp(weights_large - jax.scipy.special.logsumexp(weights_large))}")
print(f"Log marginal likelihood: {log_ml_large}")

# Test weight update formula
print("\n\nVerifying weight update formula:")
print("Weight update: w_new = w_old - γ_old + γ_new - q")
print("This implements p(x_{1:t}) / p(x_{1:t-1}) = p(x_t | x_{1:t-1})")
print("The log marginal likelihood accumulates these ratios over iterations")