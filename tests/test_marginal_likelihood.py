"""Test marginal likelihood on held-out data through iterations."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tabsmc.smc import init_empty, gibbs

def compute_marginal_log_likelihood(X_test, π, θ):
    """Compute marginal log likelihood of test data given parameters.
    
    log p(X_test | π, θ) = Σ_n log Σ_c π_c Π_d p(x_nd | θ_cd)
    """
    N_test, D, K = X_test.shape
    C = π.shape[0]
    
    # For each test point, compute log p(x_n | c) for each cluster
    # log p(x_n | c) = Σ_d log p(x_nd | θ_cd) = Σ_d Σ_k x_ndk * log θ_cdk
    log_likelihood_per_cluster = jnp.einsum('ndk,cdk->nc', X_test, θ)  # N_test x C
    
    # Add log π_c to get log of π_c * p(x_n | c)
    log_probs = log_likelihood_per_cluster + π[None, :]  # N_test x C
    
    # Marginalize over clusters: log Σ_c exp(log_probs)
    log_marginal_per_point = jax.scipy.special.logsumexp(log_probs, axis=1)  # N_test,
    
    # Total log likelihood
    total_log_likelihood = jnp.sum(log_marginal_per_point)
    
    # Per datapoint and per feature
    ll_per_point = total_log_likelihood / N_test
    ll_per_feature = total_log_likelihood / (N_test * D)
    
    return total_log_likelihood, ll_per_point, ll_per_feature


def track_parameters_and_likelihood(key, X_train, X_test, P, C, B, α_pi, α_theta, n_iterations=20):
    """Run SMC and track parameters and their likelihood on test data."""
    N, D, K = X_train.shape
    
    # Initialize particles
    keys = jax.random.split(key, P + 1)
    key = keys[0]
    init_keys = keys[1:]
    
    init_particle = lambda k: init_empty(k, C, D, K, N, α_pi, α_theta)
    A, φ, π, θ = jax.vmap(init_particle)(init_keys)
    
    # Track history
    history = []
    n_seen = 0
    
    # Also track likelihood on training data seen so far
    for i in range(n_iterations):
        # Get batch
        I_B = jax.random.choice(key, N, shape=(B,), replace=False)
        key, _ = jax.random.split(key)
        X_B = X_train[I_B]
        n_seen += B
        
        # Store parameters BEFORE update (to see initial random parameters too)
        if i == 0:
            # Compute average parameters across particles
            π_avg = jnp.mean(jax.vmap(lambda p: jnp.exp(p))(π), axis=0)
            π_avg = jnp.log(π_avg / jnp.sum(π_avg))  # Renormalize and convert to log
            
            θ_avg = jnp.mean(jax.vmap(lambda t: jnp.exp(t))(θ), axis=0)
            # Normalize each θ[c,d,:] to sum to 1
            θ_avg = θ_avg / jnp.sum(θ_avg, axis=-1, keepdims=True)
            θ_avg = jnp.log(θ_avg)
            
            # Compute likelihoods
            _, ll_train, ll_train_feat = compute_marginal_log_likelihood(X_train[:n_seen], π_avg, θ_avg)
            _, ll_test, ll_test_feat = compute_marginal_log_likelihood(X_test, π_avg, θ_avg)
            
            history.append({
                'iteration': 0,
                'n_seen': 0,
                'π': π_avg,
                'θ': θ_avg,
                'll_test': ll_test,
                'll_test_per_feature': ll_test_feat,
                'll_train': ll_train,
                'll_train_per_feature': ll_train_feat,
            })
        
        # Run Gibbs for each particle
        keys = jax.random.split(key, P + 1)
        key = keys[0]
        particle_keys = keys[1:]
        
        def step_particle(p_key, p_A, p_φ, p_π, p_θ):
            return gibbs(p_key, X_B, I_B, p_A, p_φ, p_π, p_θ, α_pi, α_theta)
        
        A, φ, π, θ, _, _ = jax.vmap(step_particle)(particle_keys, A, φ, π, θ)
        
        # Compute average parameters across particles
        π_avg = jnp.mean(jax.vmap(lambda p: jnp.exp(p))(π), axis=0)
        π_avg = jnp.log(π_avg / jnp.sum(π_avg))  # Renormalize and convert to log
        
        θ_avg = jnp.mean(jax.vmap(lambda t: jnp.exp(t))(θ), axis=0)
        # Normalize each θ[c,d,:] to sum to 1
        θ_avg = θ_avg / jnp.sum(θ_avg, axis=-1, keepdims=True)
        θ_avg = jnp.log(θ_avg)
        
        # Compute likelihoods on both train (seen so far) and test
        train_subset = X_train[:min(n_seen, N)]
        _, ll_train, ll_train_feat = compute_marginal_log_likelihood(train_subset, π_avg, θ_avg)
        _, ll_test, ll_test_feat = compute_marginal_log_likelihood(X_test, π_avg, θ_avg)
        
        history.append({
            'iteration': i + 1,
            'n_seen': n_seen,
            'π': π_avg,
            'θ': θ_avg,
            'll_test': ll_test,
            'll_test_per_feature': ll_test_feat,
            'll_train': ll_train,
            'll_train_per_feature': ll_train_feat,
        })
    
    return history


# Set up experiment
key = jax.random.PRNGKey(42)
N_train = 500
N_test = 200
D = 5
K = 3
C = 3
P = 20
B = 25
α_pi = 1.0
α_theta = 1.0

# Generate train and test data
key, subkey = jax.random.split(key)
train_indices = jax.random.randint(subkey, (N_train, D), 0, K)
X_train = jax.nn.one_hot(train_indices, K)

key, subkey = jax.random.split(key)
test_indices = jax.random.randint(subkey, (N_test, D), 0, K)
X_test = jax.nn.one_hot(test_indices, K)

# Compute theoretical best likelihood per feature
# For uniform categorical: log p(x) = log(1/K) = -log(K)
theoretical_best_ll_per_feature = jnp.log(1/K)
print(f"Theoretical best log likelihood per feature (uniform over {K} categories): {theoretical_best_ll_per_feature:.4f}")
print(f"This corresponds to entropy H = {-theoretical_best_ll_per_feature:.4f}")
print()

# Run tracking
key, subkey = jax.random.split(key)
history = track_parameters_and_likelihood(
    subkey, X_train, X_test, P, C, B, α_pi, α_theta, n_iterations=20
)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Test likelihood per feature over iterations
ax = axes[0, 0]
iterations = [h['iteration'] for h in history]
test_ll_per_feat = [h['ll_test_per_feature'] for h in history]
ax.plot(iterations, test_ll_per_feat, 'b-', linewidth=2, marker='o')
ax.axhline(y=theoretical_best_ll_per_feature, color='r', linestyle='--', 
           label=f'Best possible = {theoretical_best_ll_per_feature:.3f}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Log likelihood per feature')
ax.set_title('Test Set Likelihood per Feature')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Test likelihood per feature vs data seen
ax = axes[0, 1]
n_seen_vals = [h['n_seen'] for h in history]
ax.plot(n_seen_vals, test_ll_per_feat, 'b-', linewidth=2, marker='o')
ax.axhline(y=theoretical_best_ll_per_feature, color='r', linestyle='--', 
           label=f'Best possible = {theoretical_best_ll_per_feature:.3f}')
ax.set_xlabel('Training data seen')
ax.set_ylabel('Log likelihood per feature')
ax.set_title('Test Likelihood vs Training Data Seen')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Compare train and test likelihood
ax = axes[1, 0]
train_ll_per_feat = [h['ll_train_per_feature'] for h in history[1:]]  # Skip iteration 0
ax.plot(iterations[1:], train_ll_per_feat, 'g-', linewidth=2, marker='s', label='Train')
ax.plot(iterations[1:], test_ll_per_feat[1:], 'b-', linewidth=2, marker='o', label='Test')
ax.axhline(y=theoretical_best_ll_per_feature, color='r', linestyle='--', 
           label=f'Best possible = {theoretical_best_ll_per_feature:.3f}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Log likelihood per feature')
ax.set_title('Train vs Test Likelihood')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Parameter visualization (π values)
ax = axes[1, 1]
for c in range(C):
    π_values = [jnp.exp(h['π'][c]) for h in history]
    ax.plot(iterations, π_values, linewidth=2, marker='o', label=f'π_{c}')
ax.axhline(y=1/C, color='k', linestyle='--', alpha=0.5, label=f'Uniform = {1/C:.3f}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Mixing weight')
ax.set_title('Learned Mixing Weights (π)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/marginal_likelihood_evolution.png', dpi=150)
plt.show()

# Print summary
print("\nSummary of results:")
print(f"Initial test likelihood per feature: {history[0]['ll_test_per_feature']:.4f}")
print(f"Final test likelihood per feature: {history[-1]['ll_test_per_feature']:.4f}")
print(f"Improvement: {history[-1]['ll_test_per_feature'] - history[0]['ll_test_per_feature']:.4f}")
print(f"Gap to theoretical best: {history[-1]['ll_test_per_feature'] - theoretical_best_ll_per_feature:.4f}")
print()
print("Final mixing weights π:")
print(jnp.exp(history[-1]['π']))

# Check if parameters are actually changing
print("\nParameter changes:")
print(f"π distance from uniform: {jnp.linalg.norm(jnp.exp(history[-1]['π']) - 1/C):.4f}")
print(f"π distance from initial: {jnp.linalg.norm(history[-1]['π'] - history[0]['π']):.4f}")