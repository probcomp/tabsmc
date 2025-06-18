"""
Pure JAX implementation of Sequential Monte Carlo for tabular data.

This module provides a high-performance JAX implementation without the dumpy overhead.
"""

import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm


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


@partial(jax.jit, static_argnums=(2, 3))
def init_assignments(key, X, C, α_pi, α_theta):
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
    D, K = X.shape[1:]
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
    # x is in log space, so exp(x) gives the actual probabilities
    # We need to check that the probabilities sum to 1 (within numerical tolerance)
    prob_x = jnp.exp(x)
    
    # Log normalization constant for Dirichlet
    log_norm = jnp.sum(jax.scipy.special.gammaln(α), axis=-1) - jax.scipy.special.gammaln(jnp.sum(α, axis=-1))
    
    # Log density: log p(prob_x | α) = Σ (α - 1) * log(prob_x) - log_norm
    # Since x = log(prob_x), we have: Σ (α - 1) * x - log_norm
    log_density = jnp.sum((α - 1) * x, axis=-1) - log_norm
    
    # Add log Jacobian for the log transformation
    # When transforming from probability space to log space, we need the Jacobian
    # |det(J)| = Π prob_x[i] = exp(Σ log(prob_x[i])) = exp(Σ x[i])
    # So log|det(J)| = Σ x[i] = log(Σ prob_x[i])
    # But since Σ prob_x[i] = 1, log|det(J)| = 0, so no correction needed
    
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


@jax.jit
def smc_step(key, particles, w, γ, X_B, I_B, α_pi, α_theta):
    """One step of SMC without rejuvenation.
    
    Args:
        key: PRNG key
        particles: Tuple of (A, φ, π, θ) arrays with leading particle dimension
        log_weights: Log weights for each particle (P,)
        log_gammas: Previous log target probabilities (P,)
        X_B: New batch of data (B x D x K)
        I_B: Batch indices (B,)
        C: Number of clusters
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)
    
    Returns:
        Updated particles, log weights, and log gammas
    """
    A, φ, π, θ = particles
    P = w.shape[0]
    
    # Run Gibbs step for each particle
    keys = jax.random.split(key, P)
    
    def step_particle(p_key, p_A, p_φ, p_π, p_θ):
        return gibbs(p_key, X_B, I_B, p_A, p_φ, p_π, p_θ, α_pi, α_theta)
    
    A_new, φ_new, π_new, θ_new, γ_new, q_new = jax.vmap(step_particle)(keys, A, φ, π, θ)
    
    # Update weights: w = w + γ - q
    w = w + γ_new - γ - q_new
    
    # Normalize weights and compute ESS
    w_normalized = w - jax.scipy.special.logsumexp(w)
    ess = 1.0 / jnp.sum(jnp.exp(w_normalized) ** 2)
    
    # Resample if ESS is too low
    resample_threshold = 0.5 * P
    
    def resample(key):
        indices = jax.random.choice(key, P, shape=(P,), p=jnp.exp(w_normalized))
        A_resampled = A_new[indices]
        φ_resampled = φ_new[indices]
        π_resampled = π_new[indices]
        θ_resampled = θ_new[indices]
        γ_resampled = γ_new[indices]
        w_resampled = jnp.full(P, -jnp.log(P))  # Uniform weights in log space

        return (A_resampled, φ_resampled, π_resampled, θ_resampled), w_resampled, γ_resampled
    
    def no_resample(key):
        return (A_new, φ_new, π_new, θ_new), w, γ_new
    
    key, subkey = jax.random.split(key)
    particles_out, w_out, γ_out = jax.lax.cond(
        ess < resample_threshold,
        resample,
        no_resample,
        subkey
    )
    
    return particles_out, w_out, γ_out


def smc_no_rejuvenation(key, X, T, P, C, B, α_pi, α_theta, rejuvenation=False, return_history=False):
    """Run SMC with optional rejuvenation on data.
    
    Args:
        key: PRNG key
        X: Full dataset (N x D x K)
        T: Number of iterations
        P: Number of particles
        C: Number of clusters
        B: Batch size
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)
        rejuvenation: If True, perform rejuvenation step after each SMC step
        return_history: If True, return history of particles at each step
    
    Returns:
        If return_history is False:
            Final particles and log weights
        If return_history is True:
            Final particles, log weights, and history
    """
    N, D, K = X.shape
    
    # Ensure we have enough data for all batches
    assert N >= B * T, f"Dataset too small: N={N} must be >= B*T={B*T}"
    
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
    
    # Initialize history if needed
    if return_history:
        # Pre-allocate arrays for history
        history = {
            'A': jnp.zeros((T, P, N, C)),
            'phi': jnp.zeros((T, P, C, D, K)),
            'pi': jnp.zeros((T, P, C)),
            'theta': jnp.zeros((T, P, C, D, K)),
            'log_weights': jnp.zeros((T, P))
        }
    
    for t in tqdm(range(T)):
        # Deterministic batch indices (no wrap-around)
        start_idx = t * B
        end_idx = start_idx + B
        I_B = jnp.arange(start_idx, end_idx)
        X_B = X[I_B]
        
        # SMC step
        key, subkey = jax.random.split(key)
        particles = (A, φ, π, θ)
        particles, log_weights, log_gammas = smc_step(
            subkey, particles, log_weights, log_gammas, X_B, I_B, α_pi, α_theta
        )
        A, φ, π, θ = particles
        
        
        # Rejuvenation step if enabled
        if rejuvenation:
            # Sample a new batch for rejuvenation
            key, subkey = jax.random.split(key)
            I_rejuv = jax.random.choice(subkey, N, shape=(B,), replace=False)
            X_rejuv = X[I_rejuv]
            
            # Run Gibbs step for each particle (updates parameters but not weights/gammas)
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, P)
            
            def rejuvenate_particle(p_key, p_A, p_φ, p_π, p_θ):
                # Run gibbs but only return updated parameters
                A_new, φ_new, π_new, θ_new, _, _ = gibbs(
                    p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, α_pi, α_theta
                )
                return A_new, φ_new, π_new, θ_new
            
            A, φ, π, θ = jax.vmap(rejuvenate_particle)(keys, A, φ, π, θ)
        
        # Store history if requested
        if return_history:
            history['A'] = history['A'].at[t].set(A)
            history['phi'] = history['phi'].at[t].set(φ)
            history['pi'] = history['pi'].at[t].set(π)
            history['theta'] = history['theta'].at[t].set(θ)
            history['log_weights'] = history['log_weights'].at[t].set(log_weights)
    
    if return_history:
        return (A, φ, π, θ), log_weights, history
    else:
        return (A, φ, π, θ), log_weights


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
    A, φ, π, θ = init_assignments(subkey, X, C, α_pi, α_theta)
    
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