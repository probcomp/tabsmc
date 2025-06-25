"""
Pure JAX implementation of Sequential Monte Carlo for tabular data.

This module provides a high-performance JAX implementation without the dumpy overhead.
"""

import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def init_empty(key, C, D, K, N, α_pi, α_theta, mask=None):
    """Initialize a single particle with empty assignments.

    Args:
        key: PRNG key
        C: Number of clusters
        D: Number of features
        K: Number of categories per feature (max across all features if mask provided)
        N: Total number of data points
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)
        mask: Optional boolean mask (D, K) for valid categories per feature

    Returns:
        Tuple of (A, φ, π, θ) for a single particle
    """
    # Initialize empty assignments (all zeros) - use boolean to save memory
    A = jnp.zeros((N, C), dtype=jnp.bool_)

    # Initialize zero sufficient statistics
    φ = jnp.zeros((C, D, K))

    # Sample π from Dirichlet prior (in log space)
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, jnp.ones(C) * α_pi))

    # Sample θ from Dirichlet prior (in log space)
    key, subkey = jax.random.split(key)
    keys_theta = jax.random.split(subkey, C * D).reshape(C, D, -1)
    
    if mask is not None:
        # Use masked sampling for each feature
        def sample_theta_masked(k, d):
            return masked_dirichlet_sample(k, jnp.ones(K) * α_theta, mask[d])
        
        θ = jax.vmap(lambda c_keys: jax.vmap(sample_theta_masked)(c_keys, jnp.arange(D)))(keys_theta)
    else:
        sample_theta = lambda k: jnp.log(jax.random.dirichlet(k, jnp.ones(K) * α_theta))
        θ = jax.vmap(jax.vmap(sample_theta))(keys_theta)

    return A, φ, π, θ


@partial(jax.jit, static_argnums=(2, 3))
def init_assignments(key, X, C, α_pi, α_theta, mask=None):
    """Initialize a single particle by sampling assignments and updating parameters.

    Args:
        key: PRNG key
        X: Data array (N x D x K) one-hot encoded
        C: Number of clusters
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)
        mask: Optional boolean mask (D, K) for valid categories per feature

    Returns:
        Tuple of (A, φ, π, θ) for a single particle
    """
    N, D, K = X.shape
    
    # Sample π from Dirichlet prior (in log space)
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, jnp.ones(C) * α_pi))
    
    # Sample initial assignments A from categorical(π)
    key, subkey = jax.random.split(key)
    keys_A = jax.random.split(subkey, N)
    A_indices = jax.vmap(lambda k: jax.random.categorical(k, π))(keys_A)
    A = jax.nn.one_hot(A_indices, C, dtype=jnp.bool_)
    
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
    
    if mask is not None:
        # Use masked sampling for each feature
        def sample_theta_masked(k, α_post, d):
            return masked_dirichlet_sample(k, α_post, mask[d])
        
        θ = jax.vmap(lambda c_keys, c_α: jax.vmap(sample_theta_masked)(c_keys, c_α, jnp.arange(D)))(keys_theta, α_theta_posterior)
    else:
        sample_theta = lambda k, α: jnp.log(jax.random.dirichlet(k, α))
        θ = jax.vmap(jax.vmap(sample_theta))(keys_theta, α_theta_posterior)
    
    return A, φ, π, θ


@jax.jit
def log_dirichlet_score(α, x, mask=None):
    """Compute log probability density of Dirichlet distribution.
    
    Args:
        α: Concentration parameters (shape: [..., K])
        x: Samples in log space (shape: [..., K])
        mask: Optional boolean mask (shape: [..., K]) for valid categories
    
    Returns:
        Log probability density
    """
    if mask is not None:
        # Use original alpha, but only sum over valid categories
        α_valid = jnp.where(mask, α, 0.0)
        α_sum = jnp.sum(α_valid, axis=-1)
        
        # Log normalization constant (only over valid entries)
        log_norm = jnp.sum(jnp.where(mask, jax.scipy.special.gammaln(α), 0.0), axis=-1) - jax.scipy.special.gammaln(α_sum)
        
        # Log density (only over valid entries)
        log_density = jnp.sum(jnp.where(mask, (α - 1) * x, 0.0), axis=-1) - log_norm
    else:
        # Standard computation without masking
        log_norm = jnp.sum(jax.scipy.special.gammaln(α), axis=-1) - jax.scipy.special.gammaln(jnp.sum(α, axis=-1))
        log_density = jnp.sum((α - 1) * x, axis=-1) - log_norm
    
    return log_density


@jax.jit
def masked_dirichlet_sample(key, α, mask):
    """Sample from Dirichlet distribution with masking.
    
    Args:
        key: PRNG key
        α: Concentration parameters (shape: [..., K])
        mask: Boolean mask (shape: [..., K]) for valid categories
    
    Returns:
        Log probabilities for valid categories, -inf for invalid ones
    """
    # Sample gamma random variables
    gammas = jax.random.gamma(key, α)
    
    # Mask out invalid categories (set to 0)
    gammas_masked = jnp.where(mask, gammas, 0.0)
    
    # Normalize over valid categories only
    gamma_sum = jnp.sum(gammas_masked, axis=-1, keepdims=True)
    
    # Avoid division by zero
    gamma_sum = jnp.where(gamma_sum == 0, 1.0, gamma_sum)
    
    # Compute probabilities for valid categories, 0 for invalid
    probs = jnp.where(mask, gammas_masked / gamma_sum, 0.0)
    
    # Convert to log space, using -inf for invalid categories
    log_probs = jnp.where(mask, jnp.log(jnp.maximum(probs, 1e-10)), -jnp.inf)
    
    return log_probs


@jax.jit
def gibbs(key, X_B, I_B, A_one_hot, φ_old, π_old, θ_old, α_pi, α_theta, mask=None):
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
        mask: Optional boolean mask (D, K) for valid categories per feature

    Returns:
        Tuple of (A_one_hot, φ, π, θ, γ, q)
    """
    B, D, K = X_B.shape
    C = π_old.shape[0]
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, B)

    # Compute log likelihoods and probabilities
    # Handle -inf * 0.0 case: when θ is -inf and X is 0, the result should be 0, not nan
    θ_expanded = θ_old[None, :, :, :]  # 1 x C x D x K
    X_B_expanded = X_B[:, None, :, :]  # B x 1 x D x K
    
    # Use jnp.where to handle -inf * 0.0 case
    products = jnp.where(
        X_B_expanded == 0.0,
        0.0,  # When X is 0, contribution is 0 regardless of θ
        θ_expanded * X_B_expanded  # When X is not 0, use normal multiplication
    )
    
    log_likelihoods = jnp.sum(products, axis=(2, 3))  # B x C
    
    log_probs = log_likelihoods + π_old[None, :]  # B x C
    
    # Sample new assignments
    A_B_indices = jax.vmap(lambda k, lp: jax.random.categorical(k, lp))(keys, log_probs)
    A_B_one_hot = jax.nn.one_hot(A_B_indices, C, dtype=jnp.bool_)  # B x C
    
    # Compute proposal probability
    A_B_pgibbs = jnp.sum(log_probs * A_B_one_hot, axis=1)  # B,
    
    # Get old assignments for the batch
    A_B_one_hot_old = A_one_hot[I_B]  # B x C
    
    # Update sufficient statistics
    φ_B = jnp.einsum('bc,bdk->cdk', A_B_one_hot, X_B)  # C x D x K
    φ_B_old = jnp.einsum('bc,bdk->cdk', A_B_one_hot_old, X_B)  # C x D x K
    φ = φ_old + φ_B - φ_B_old  # C x D x K
    
    # Update π
    if mask is not None:
        # Only count valid categories for cluster counts
        φ_masked = jnp.where(mask[None, :, :], φ, 0.0)
        counts = jnp.sum(φ_masked, axis=(1, 2))  # C,
    else:
        counts = jnp.sum(φ, axis=(1, 2))  # C,
    
    α_pi_posterior = α_pi + counts
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, α_pi_posterior))
    π_pgibbs = log_dirichlet_score(α_pi_posterior, π)
    
    # Update θ
    key, subkey = jax.random.split(key)
    keys_theta = jax.random.split(subkey, C * D).reshape(C, D, -1)
    α_theta_posterior = α_theta + φ
    
    if mask is not None:
        # Use masked sampling for each feature
        def sample_theta_masked(k, α_post, d):
            return masked_dirichlet_sample(k, α_post, mask[d])
        
        θ = jax.vmap(lambda c_keys, c_α: jax.vmap(sample_theta_masked)(c_keys, c_α, jnp.arange(D)))(keys_theta, α_theta_posterior)
        
        # Compute log probability with masking - handle potential -inf values
        def compute_masked_score(c_θ, c_α):
            def score_for_feature(d):
                score = log_dirichlet_score(c_α[d], c_θ[d], mask[d])
                # Replace -inf with large negative number to avoid NaN in sum
                return jnp.where(jnp.isfinite(score), score, -1e10)
            return jnp.sum(jax.vmap(score_for_feature)(jnp.arange(D)))
        
        θ_pgibbs = jnp.sum(jax.vmap(compute_masked_score)(θ, α_theta_posterior))
    else:
        sample_theta = lambda k, α: jnp.log(jax.random.dirichlet(k, α))
        θ = jax.vmap(jax.vmap(sample_theta))(keys_theta, α_theta_posterior)
        θ_pgibbs = jnp.sum(jax.vmap(jax.vmap(log_dirichlet_score))(α_theta_posterior, θ))
    
    # Compute proposal log probability
    q = jnp.sum(A_B_pgibbs) + π_pgibbs + θ_pgibbs
    
    # Update assignments
    A_one_hot = A_one_hot.at[I_B].set(A_B_one_hot)
    
    # Compute target log probability
    π_p = log_dirichlet_score(jnp.full(C, α_pi), π)
    
    if mask is not None:
        # Use masked prior for θ - handle potential -inf values
        def compute_masked_prior_score(c_θ):
            def score_for_feature(d):
                score = log_dirichlet_score(jnp.full(K, α_theta), c_θ[d], mask[d])
                return jnp.where(jnp.isfinite(score), score, -1e10)
            return jnp.sum(jax.vmap(score_for_feature)(jnp.arange(D)))
        
        θ_p = jnp.sum(jax.vmap(compute_masked_prior_score)(θ))
        
        # Only count valid categories in sufficient statistics
        φ_masked_target = jnp.where(mask[None, :, :], φ, 0.0)
        θ_masked_target = jnp.where(mask[None, :, :], θ, 0.0)
        X_p = jnp.sum(θ_masked_target * φ_masked_target)
    else:
        θ_p = jnp.sum(jax.vmap(jax.vmap(lambda t: log_dirichlet_score(jnp.full(K, α_theta), t)))(θ))
        X_p = jnp.sum(θ * φ)
    
    A_p = jnp.sum(A_one_hot * π[None, :])
    
    γ = π_p + θ_p + A_p + X_p
    
    return A_one_hot, φ, π, θ, γ, q


@jax.jit
def smc_step(key, particles, w, γ, X_B, I_B, α_pi, α_theta, mask=None):
    """One step of SMC without rejuvenation.
    
    Args:
        key: PRNG key
        particles: Tuple of (A, φ, π, θ) arrays with leading particle dimension
        log_weights: Log weights for each particle (P,)
        log_gammas: Previous log target probabilities (P,)
        X_B: New batch of data (B x D x K)
        I_B: Batch indices (B,)
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)
        mask: Optional boolean mask (D, K) for valid categories per feature
    
    Returns:
        Updated particles, log weights, and log gammas
    """
    A, φ, π, θ = particles
    P = w.shape[0]
    
    # Run Gibbs step for each particle
    keys = jax.random.split(key, P)
    
    def step_particle(p_key, p_A, p_φ, p_π, p_θ):
        return gibbs(p_key, X_B, I_B, p_A, p_φ, p_π, p_θ, α_pi, α_theta, mask)
    
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


def smc_no_rejuvenation(key, X, T, P, C, B, α_pi, α_theta, rejuvenation=False, return_history=False, mask=None):
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
        mask: Optional boolean mask (D, K) for valid categories per feature
    
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
    init_particle = lambda k: init_empty(k, C, D, K, N, α_pi, α_theta, mask)
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
            subkey, particles, log_weights, log_gammas, X_B, I_B, α_pi, α_theta, mask
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
                    p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, α_pi, α_theta, mask
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


def mcmc_minibatch(key, X, T, C, B, α_pi, α_theta, mask=None):
    """Run MCMC algorithm with minibatches using Gibbs moves.

    Args:
        key: PRNG key
        X: Full dataset (N x D x K) one-hot encoded
        T: Number of iterations
        C: Number of clusters
        B: Minibatch size
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)
        mask: Optional boolean mask (D, K) for valid categories per feature

    Returns:
        Final state (A, φ, π, θ)
    """
    N, D, K = X.shape
    
    # Initialize using assignments
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_assignments(subkey, X, C, α_pi, α_theta, mask)
    
    for t in range(T):
        # Sample random batch indices
        key, subkey = jax.random.split(key)
        I_B = jax.random.choice(subkey, N, shape=(B,), replace=False)
        X_B = X[I_B]
        
        # Run Gibbs sampler on the batch
        key, subkey = jax.random.split(key)
        A, φ, π, θ, _, _ = gibbs(
            subkey, X_B, I_B, A, φ, π, θ, α_pi, α_theta, mask
        )
    
    return A, φ, π, θ