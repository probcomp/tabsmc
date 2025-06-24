"""Track and plot gamma/N evolution during SMC."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tabsmc.smc import init_empty, gibbs, smc_step
from functools import partial


@partial(jax.jit, static_argnums=(6, 7, 8))
def smc_step_with_history(key, particles, log_weights, log_gammas, X_B, I_B, C, α_pi, α_theta):
    """SMC step that also returns gamma values for tracking."""
    A, φ, π, θ = particles
    P = log_weights.shape[0]
    
    # Update weights by subtracting old gamma
    log_weights = log_weights - log_gammas
    
    # Run Gibbs step for each particle
    keys = jax.random.split(key, P)
    
    def step_particle(p_key, p_A, p_φ, p_π, p_θ):
        return gibbs(p_key, X_B, I_B, p_A, p_φ, p_π, p_θ, α_pi, α_theta)
    
    A_new, φ_new, π_new, θ_new, γ_new, q_new = jax.vmap(step_particle)(keys, A, φ, π, θ)
    
    # Update weights: w = w + γ - q
    log_weights = log_weights + γ_new - q_new
    
    # Normalize weights and compute ESS
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    weights = jnp.exp(log_weights_normalized)
    ess = 1.0 / jnp.sum(weights ** 2)
    
    # Resample if ESS is too low
    resample_threshold = 0.5 * P
    
    def resample(key):
        indices = jax.random.choice(key, P, shape=(P,), p=weights)
        A_resampled = A_new[indices]
        φ_resampled = φ_new[indices]
        π_resampled = π_new[indices]
        θ_resampled = θ_new[indices]
        # Reset weights to uniform after resampling
        log_weights_resampled = jnp.log(jnp.ones(P) / P)
        # Reset gammas to zero
        γ_resampled = jnp.zeros(P)
        return (A_resampled, φ_resampled, π_resampled, θ_resampled), log_weights_resampled, γ_resampled
    
    def no_resample(key):
        return (A_new, φ_new, π_new, θ_new), log_weights, γ_new
    
    key, subkey = jax.random.split(key)
    particles_out, log_weights_out, log_gammas_out = jax.lax.cond(
        ess < resample_threshold,
        resample,
        no_resample,
        subkey
    )
    
    # Return gamma values before potential reset for tracking
    return particles_out, log_weights_out, log_gammas_out, γ_new


def smc_with_gamma_tracking(key, X, T, P, C, B, α_pi, α_theta):
    """Run SMC and track gamma values over iterations."""
    N, D, K = X.shape
    
    # Initialize particles using init_empty
    keys = jax.random.split(key, P + 1)
    key = keys[0]
    init_keys = keys[1:]
    
    # Vectorized initialization
    init_particle = lambda k: init_empty(k, C, D, K, N, α_pi, α_theta)
    A, φ, π, θ = jax.vmap(init_particle)(init_keys)
    
    # Initialize weights and gammas
    log_weights = jnp.zeros(P)
    log_gammas = jnp.zeros(P)
    
    # Track gamma values and data seen
    gamma_history = []
    n_seen_history = []
    n_seen = 0
    
    for t in range(T):
        # Sample batch indices
        key, subkey = jax.random.split(key)
        I_B = jax.random.choice(subkey, N, shape=(B,), replace=False)
        X_B = X[I_B]
        
        # Update count of data seen
        n_seen += B
        
        # SMC step
        key, subkey = jax.random.split(key)
        particles = (A, φ, π, θ)
        particles, log_weights, log_gammas, gammas_current = smc_step_with_history(
            subkey, particles, log_weights, log_gammas, X_B, I_B, C, α_pi, α_theta
        )
        A, φ, π, θ = particles
        
        # Store gamma values and n_seen
        gamma_history.append(gammas_current)
        n_seen_history.append(n_seen)
    
    return gamma_history, n_seen_history, log_weights


# Test with different settings
key = jax.random.PRNGKey(42)

# Settings
N = 200  # Total data points
D = 5    # Features
K = 3    # Categories
C = 3    # Clusters
P = 10   # Particles
B = 10   # Batch size
T = 20   # Iterations
α_pi = 1.0
α_theta = 1.0

# Generate synthetic data
key, subkey = jax.random.split(key)
data_indices = jax.random.randint(subkey, (N, D), 0, K)
X = jax.nn.one_hot(data_indices, K)

print(f"Running SMC with gamma tracking...")
print(f"N={N}, D={D}, K={K}, C={C}, P={P}, B={B}, T={T}")

# Run SMC with tracking
key, subkey = jax.random.split(key)
gamma_history, n_seen_history, final_weights = smc_with_gamma_tracking(
    subkey, X, T, P, C, B, α_pi, α_theta
)

# Convert to arrays for plotting
gamma_history = jnp.array(gamma_history)  # Shape: (T, P)
n_seen_history = jnp.array(n_seen_history)  # Shape: (T,)

# Compute gamma/N for each particle over time
gamma_per_n = gamma_history / n_seen_history[:, None]

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Raw gamma values over iterations
ax1 = axes[0, 0]
for p in range(P):
    ax1.plot(gamma_history[:, p], alpha=0.5, label=f'Particle {p}' if p < 3 else None)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('γ (log target probability)')
ax1.set_title('Raw γ values over iterations')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: gamma/N over iterations
ax2 = axes[0, 1]
for p in range(P):
    ax2.plot(n_seen_history, gamma_per_n[:, p], alpha=0.5, label=f'Particle {p}' if p < 3 else None)
ax2.set_xlabel('Number of data points seen')
ax2.set_ylabel('γ / N')
ax2.set_title('γ/N (average log probability per data point)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Mean and std of gamma/N across particles
ax3 = axes[1, 0]
mean_gamma_per_n = jnp.mean(gamma_per_n, axis=1)
std_gamma_per_n = jnp.std(gamma_per_n, axis=1)
ax3.plot(n_seen_history, mean_gamma_per_n, 'b-', label='Mean')
ax3.fill_between(n_seen_history, 
                  mean_gamma_per_n - std_gamma_per_n,
                  mean_gamma_per_n + std_gamma_per_n,
                  alpha=0.3, label='±1 std')
ax3.set_xlabel('Number of data points seen')
ax3.set_ylabel('γ / N')
ax3.set_title('Mean γ/N across particles')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Final particle weights
ax4 = axes[1, 1]
final_weights_normalized = jnp.exp(final_weights - jax.scipy.special.logsumexp(final_weights))
ax4.bar(range(P), final_weights_normalized)
ax4.set_xlabel('Particle index')
ax4.set_ylabel('Normalized weight')
ax4.set_title('Final particle weights')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../figures/gamma_evolution.png', dpi=150)
plt.show()

# Print summary statistics
print(f"\nSummary statistics:")
print(f"Final mean γ/N: {mean_gamma_per_n[-1]:.4f}")
print(f"Final std γ/N: {std_gamma_per_n[-1]:.4f}")
print(f"Effective sample size: {1.0 / jnp.sum(final_weights_normalized**2):.2f} / {P}")

# Check convergence
print(f"\nConvergence check (last 5 iterations):")
for i in range(-5, 0):
    print(f"  Iteration {T+i}: mean γ/N = {mean_gamma_per_n[i]:.4f}, std = {std_gamma_per_n[i]:.4f}")

# Additional plot: γ/N convergence for individual particles
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(jnp.linspace(0, 1, P))
for p in range(min(P, 5)):  # Show first 5 particles
    plt.plot(n_seen_history, gamma_per_n[:, p], color=colors[p], 
             label=f'Particle {p}', linewidth=2, alpha=0.8)

plt.xlabel('Number of data points seen')
plt.ylabel('γ / N (average log probability per data point)')
plt.title('γ/N evolution for individual particles')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/gamma_per_n_particles.png', dpi=150)
plt.show()

print("\nPlots saved as 'gamma_evolution.png' and 'gamma_per_n_particles.png'")