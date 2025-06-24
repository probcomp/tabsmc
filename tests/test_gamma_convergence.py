"""Test gamma/N convergence with different dataset sizes."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tabsmc.smc import init_empty, gibbs
from functools import partial


def track_gamma_per_n(key, X, P, C, B, α_pi, α_theta):
    """Run SMC and track gamma/N values."""
    N, D, K = X.shape
    
    # Initialize particles
    keys = jax.random.split(key, P + 1)
    key = keys[0]
    init_keys = keys[1:]
    
    init_particle = lambda k: init_empty(k, C, D, K, N, α_pi, α_theta)
    A, φ, π, θ = jax.vmap(init_particle)(init_keys)
    
    # Track values
    gamma_per_n_history = []
    n_seen = 0
    
    # Process all data in batches
    n_batches = N // B
    for i in range(n_batches):
        # Get batch
        I_B = jnp.arange(i * B, (i + 1) * B)
        X_B = X[I_B]
        n_seen += B
        
        # Run Gibbs for each particle
        keys = jax.random.split(key, P + 1)
        key = keys[0]
        particle_keys = keys[1:]
        
        def step_particle(p_key, p_A, p_φ, p_π, p_θ):
            return gibbs(p_key, X_B, I_B, p_A, p_φ, p_π, p_θ, α_pi, α_theta)
        
        A, φ, π, θ, γ, q = jax.vmap(step_particle)(particle_keys, A, φ, π, θ)
        
        # Store gamma/N
        gamma_per_n = γ / n_seen
        gamma_per_n_history.append((n_seen, jnp.mean(gamma_per_n), jnp.std(gamma_per_n)))
    
    return gamma_per_n_history


# Test with different dataset sizes
key = jax.random.PRNGKey(42)
D = 5
K = 3
C = 3
P = 20
B = 10
α_pi = 1.0
α_theta = 1.0

# Different dataset sizes
N_values = [100, 200, 500, 1000]
colors = ['blue', 'green', 'red', 'purple']

plt.figure(figsize=(12, 8))

for N, color in zip(N_values, colors):
    # Generate data
    key, subkey = jax.random.split(key)
    data_indices = jax.random.randint(subkey, (N, D), 0, K)
    X = jax.nn.one_hot(data_indices, K)
    
    # Track gamma/N
    key, subkey = jax.random.split(key)
    history = track_gamma_per_n(subkey, X, P, C, B, α_pi, α_theta)
    
    # Extract values
    n_seen_vals = [h[0] for h in history]
    mean_vals = [h[1] for h in history]
    std_vals = [h[2] for h in history]
    
    # Plot mean with error bands
    mean_vals = jnp.array(mean_vals)
    std_vals = jnp.array(std_vals)
    n_seen_vals = jnp.array(n_seen_vals)
    
    plt.plot(n_seen_vals, mean_vals, '-', color=color, label=f'N={N}', linewidth=2)
    plt.fill_between(n_seen_vals, 
                     mean_vals - std_vals, 
                     mean_vals + std_vals,
                     alpha=0.2, color=color)

plt.xlabel('Number of data points seen', fontsize=12)
plt.ylabel('γ / N (average log probability per data point)', fontsize=12)
plt.title('Convergence of γ/N for different dataset sizes', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/gamma_convergence_comparison.png', dpi=150)
plt.show()

# Also create a plot showing the theoretical convergence
plt.figure(figsize=(10, 6))

# For a well-specified model, gamma/N should converge to the negative entropy
# of the true data distribution plus KL divergence from prior to posterior
print("\nFinal γ/N values:")
for N, color in zip(N_values, colors):
    key, subkey = jax.random.split(key)
    data_indices = jax.random.randint(subkey, (N, D), 0, K)
    X = jax.nn.one_hot(data_indices, K)
    
    key, subkey = jax.random.split(key)
    history = track_gamma_per_n(subkey, X, P, C, B, α_pi, α_theta)
    
    final_mean = history[-1][1]
    final_std = history[-1][2]
    print(f"N={N}: {final_mean:.4f} ± {final_std:.4f}")

print("\nNote: γ/N represents the average log probability per data point.")
print("For well-specified models, this should converge as N increases.")
print("The convergence value depends on the true data distribution entropy.")