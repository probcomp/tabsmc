"""
Pure JAX implementation of Sequential Monte Carlo for tabular data.

This module provides a high-performance JAX implementation without the dumpy overhead.
"""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def init_empty(key, C, D, K, N, α_pi, α_theta):
    """Initialize a single particle with empty assignments.

    Args:
        key: PRNG key
        C: Number of clusters
        D: Number of features
        K: Number of categories per feature
        N: Total number of data points
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)

    Returns:
        Tuple of (A, φ, π, θ) for a single particle
    """
    # Initialize empty assignments (all zeros)
    A = jnp.zeros((N, C))

    # Initialize zero sufficient statistics
    φ = jnp.zeros((C, D, K))

    # Sample π from Dirichlet prior (in log space)
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, jnp.ones(C) * α_pi))

    # Sample θ from Dirichlet prior (in log space)
    key, subkey = jax.random.split(key)
    keys_theta = jax.random.split(subkey, C * D).reshape(C, D, -1)
    sample_theta = lambda k: jnp.log(jax.random.dirichlet(k, jnp.ones(K) * α_theta))
    θ = jax.vmap(jax.vmap(sample_theta))(keys_theta)

    return A, φ, π, θ


@partial(jax.jit, static_argnums=(2, 3, 4))
def init_assignments(key, X, C, D, K, α_pi, α_theta):
    """Initialize a single particle by sampling assignments and updating parameters.

    Args:
        key: PRNG key
        X: Data array (N x D x K) one-hot encoded
        C: Number of clusters
        D: Number of features
        K: Number of categories per feature
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)

    Returns:
        Tuple of (A, φ, π, θ) for a single particle
    """
    N = X.shape[0]
    
    # Sample π from Dirichlet prior (in log space)
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, jnp.ones(C) * α_pi))
    
    # Sample initial assignments A from categorical(π)
    key, subkey = jax.random.split(key)
    keys_A = jax.random.split(subkey, N)
    A_indices = jax.vmap(lambda k: jax.random.categorical(k, π))(keys_A)
    A = jax.nn.one_hot(A_indices, C)
    
    # Compute sufficient statistics φ
    φ = jnp.einsum('nc,ndk->cdk', A, X)
    
    # Update π using posterior given A
    counts = jnp.sum(A, axis=0)
    α_pi_posterior = α_pi + counts
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, α_pi_posterior))
    
    # Update θ using posterior given φ
    key, subkey = jax.random.split(key)
    keys_theta = jax.random.split(subkey, C * D).reshape(C, D, -1)
    α_theta_posterior = α_theta + φ
    sample_theta = lambda k, α: jnp.log(jax.random.dirichlet(k, α))
    θ = jax.vmap(jax.vmap(sample_theta))(keys_theta, α_theta_posterior)
    
    return A, φ, π, θ


@jax.jit
def log_dirichlet_score(α, x):
    """Compute log probability density of Dirichlet distribution.
    
    Args:
        α: Concentration parameters (shape: [..., K])
        x: Samples in log space (shape: [..., K])
    
    Returns:
        Log probability density
    """
    # Convert log samples to probability space
    prob_x = jnp.exp(x)
    
    # Log normalization constant
    log_norm = jnp.sum(jax.scipy.special.gammaln(α), axis=-1) - jax.scipy.special.gammaln(jnp.sum(α, axis=-1))
    
    # Log density
    log_density = jnp.sum((α - 1) * x, axis=-1) - log_norm
    
    return log_density


@jax.jit
def gibbs(key, X_B, I_B, A_one_hot, φ_old, π_old, θ_old, α_pi, α_theta):
    """One step of Gibbs sampling for particle Gibbs.

    Args:
        key: PRNG key
        X_B: Minibatch data (B x D x K)
        I_B: Batch indices (B,)
        A_one_hot: Current assignments (N x C)
        φ_old: Current sufficient statistics (C x D x K)
        π_old: Current mixing weights in log space (C,)
        θ_old: Current emission parameters in log space (C x D x K)
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)

    Returns:
        Tuple of (A_one_hot, φ, π, θ, γ, q)
    """
    B, D, K = X_B.shape
    C = π_old.shape[0]
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, B)

    # Compute log likelihoods and probabilities
    log_likelihoods = jnp.sum(θ_old[None, :, :, :] * X_B[:, None, :, :], axis=(2, 3))  # B x C
    log_probs = log_likelihoods + π_old[None, :]  # B x C
    
    # Sample new assignments
    A_B_indices = jax.vmap(lambda k, lp: jax.random.categorical(k, lp))(keys, log_probs)
    A_B_one_hot = jax.nn.one_hot(A_B_indices, C)  # B x C
    
    # Compute proposal probability
    A_B_pgibbs = jnp.sum(log_probs * A_B_one_hot, axis=1)  # B,
    
    # Get old assignments for the batch
    A_B_one_hot_old = A_one_hot[I_B]  # B x C
    
    # Update sufficient statistics
    φ_B = jnp.einsum('bc,bdk->cdk', A_B_one_hot, X_B)  # C x D x K
    φ_B_old = jnp.einsum('bc,bdk->cdk', A_B_one_hot_old, X_B)  # C x D x K
    φ = φ_old + φ_B - φ_B_old  # C x D x K
    
    # Update π
    counts = jnp.sum(φ, axis=(1, 2))  # C,
    α_pi_posterior = α_pi + counts
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, α_pi_posterior))
    π_pgibbs = log_dirichlet_score(α_pi_posterior, π)
    
    # Update θ
    key, subkey = jax.random.split(key)
    keys_theta = jax.random.split(subkey, C * D).reshape(C, D, -1)
    α_theta_posterior = α_theta + φ
    sample_theta = lambda k, α: jnp.log(jax.random.dirichlet(k, α))
    θ = jax.vmap(jax.vmap(sample_theta))(keys_theta, α_theta_posterior)
    θ_pgibbs = jnp.sum(jax.vmap(jax.vmap(log_dirichlet_score))(α_theta_posterior, θ))
    
    # Compute proposal log probability
    q = jnp.sum(A_B_pgibbs) + π_pgibbs + θ_pgibbs
    
    # Update assignments
    A_one_hot = A_one_hot.at[I_B].set(A_B_one_hot)
    
    # Compute target log probability
    π_p = log_dirichlet_score(jnp.full(C, α_pi), π)
    θ_p = jnp.sum(jax.vmap(jax.vmap(lambda t: log_dirichlet_score(jnp.full(K, α_theta), t)))(θ))
    A_p = jnp.sum(A_one_hot * π[None, :])
    X_p = jnp.sum(θ * φ)
    
    γ = π_p + θ_p + A_p + X_p
    
    return A_one_hot, φ, π, θ, γ, q


def mcmc_minibatch(key, X, T, C, B, α_pi, α_theta):
    """Run MCMC algorithm with minibatches using Gibbs moves.

    Args:
        key: PRNG key
        X: Full dataset (N x D x K) one-hot encoded
        T: Number of iterations
        C: Number of clusters
        B: Minibatch size
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)

    Returns:
        Final state (A, φ, π, θ)
    """
    N, D, K = X.shape
    
    # Initialize using assignments
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_assignments(subkey, X, C, D, K, α_pi, α_theta)
    
    for t in range(T):
        # Sample random batch indices
        key, subkey = jax.random.split(key)
        I_B = jax.random.choice(subkey, N, shape=(B,), replace=False)
        X_B = X[I_B]
        
        # Run Gibbs sampler on the batch
        key, subkey = jax.random.split(key)
        A, φ, π, θ, _, _ = gibbs(
            subkey, X_B, I_B, A, φ, π, θ, α_pi, α_theta
        )
    
    return A, φ, π, θ