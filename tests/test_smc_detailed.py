"""Detailed test of SMC weight computations."""

import jax
import jax.numpy as jnp
from tabsmc.smc import (
    init_empty, gibbs, smc_step, smc_no_rejuvenation
)

# Set random seed
key = jax.random.PRNGKey(42)

# Small test case for manual verification
N, D, K, C = 20, 2, 2, 2  # Small dimensions for easy inspection
P = 3  # Few particles
B = 5  # Small batch
α_pi = 1.0
α_theta = 1.0

# Generate simple synthetic data
key, subkey = jax.random.split(key)
data_indices = jax.random.randint(subkey, (N, D), 0, K)
X = jax.nn.one_hot(data_indices, K)

print("Test Case Setup:")
print(f"N={N} data points, D={D} features, K={K} categories, C={C} clusters")
print(f"P={P} particles, B={B} batch size")
print(f"Data shape: {X.shape}")
print()

# Initialize particles
keys = jax.random.split(key, P + 1)
key = keys[0]
init_keys = keys[1:]

init_particle = lambda k: init_empty(k, C, D, K, N, α_pi, α_theta)
A, φ, π, θ = jax.vmap(init_particle)(init_keys)

print("Initial particles:")
print(f"π[0] (log space): {π[0]}, exp(π[0]): {jnp.exp(π[0])}")
print(f"Sum of exp(π[0]): {jnp.sum(jnp.exp(π[0]))}")
print()

# Run one iteration manually to track weight updates
log_weights = jnp.zeros(P)
log_gammas = jnp.zeros(P)

# Select batch
key, subkey = jax.random.split(key)
I_B = jax.random.choice(subkey, N, shape=(B,), replace=False)
X_B = X[I_B]

print(f"Batch indices: {I_B}")
print()

# Before the step, weights should be uniform (log(1/P))
print("Before step:")
print(f"log_weights: {log_weights}")
print(f"log_gammas: {log_gammas}")
print()

# Run one step
key, subkey = jax.random.split(key)
particles = (A, φ, π, θ)
particles_new, log_weights_new, log_gammas_new = smc_step(
    subkey, particles, log_weights, log_gammas, X_B, I_B, α_pi, α_theta
)

print("After step:")
print(f"log_weights_new: {log_weights_new}")
print(f"log_gammas_new: {log_gammas_new}")
print(f"Normalized weights: {jnp.exp(log_weights_new - jax.scipy.special.logsumexp(log_weights_new))}")
print()

# Manually compute weight update for first particle to verify
A_0, φ_0, π_0, θ_0 = A[0], φ[0], π[0], θ[0]
key, subkey = jax.random.split(key)
A_0_new, φ_0_new, π_0_new, θ_0_new, γ_0_new, q_0 = gibbs(
    subkey, X_B, I_B, A_0, φ_0, π_0, θ_0, α_pi, α_theta
)

print("Manual computation for particle 0:")
print(f"γ_new: {γ_0_new}")
print(f"q: {q_0}")
print(f"Weight update should be: 0 - 0 + {γ_0_new} - {q_0} = {γ_0_new - q_0}")
print()

# Test sequential processing
print("\nTesting sequential batch processing:")
batches = jnp.array_split(X, 4)  # Split data into 4 batches
print(f"Number of batches: {len(batches)}")
print(f"Batch sizes: {[b.shape[0] for b in batches]}")

# Run SMC
key, subkey = jax.random.split(key)
particles_final, weights_final, log_ml = smc_no_rejuvenation(
    subkey, X, T=4, P=P, C=C, B=B, α_pi=α_pi, α_theta=α_theta
)

print(f"\nFinal log marginal likelihood: {log_ml}")
print(f"Final weights: {jnp.exp(weights_final - jax.scipy.special.logsumexp(weights_final))}")

# Verify weight normalization
print(f"\nWeight normalization check:")
print(f"Sum of normalized weights: {jnp.sum(jnp.exp(weights_final - jax.scipy.special.logsumexp(weights_final)))}")
print("(Should be 1.0)")