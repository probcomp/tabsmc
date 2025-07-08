"""
Pure JAX implementation of Sequential Monte Carlo for tabular data.

This module provides a high-performance JAX implementation without the dumpy overhead.
"""

import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm

# Define empty assignment constant - use a large but safe value
EMPTY_ASSIGNMENT = 999999  # Large enough to be outside valid cluster range, small enough to avoid overflow

# Define missing value constant for integer indices
MISSING_VALUE = 99999  # Use large positive integer to represent missing values (avoids negative indexing)


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
    # Initialize empty assignments - use uint with max value as empty indicator
    # A[n] = c means data point n is assigned to cluster c, A[n] = EMPTY_ASSIGNMENT means unassigned
    A = jnp.full(N, EMPTY_ASSIGNMENT, dtype=jnp.uint32)  # Shape: (N,) with EMPTY_ASSIGNMENT indicating unassigned

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


@partial(jax.jit, static_argnums=(2, 3, 4))
def init_assignments(key, X, C, K, α_pi, α_theta, mask=None):
    """Initialize a single particle by sampling assignments and updating parameters.

    Args:
        key: PRNG key
        X: Data array (N x D) with integer indices
        C: Number of clusters
        K: Number of categories per feature (max across all features if mask provided)
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)
        mask: Optional boolean mask (D, K) for valid categories per feature

    Returns:
        Tuple of (A, φ, π, θ) for a single particle
    """
    N, D = X.shape
    
    # Sample π from Dirichlet prior (in log space)
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, jnp.ones(C) * α_pi))
    
    # Sample initial assignments A from categorical(π) - store as indices
    key, subkey = jax.random.split(key)
    keys_A = jax.random.split(subkey, N)
    A = jax.vmap(lambda k: jax.random.categorical(k, π))(keys_A).astype(jnp.uint32)  # Shape: (N,)
    
    # Compute sufficient statistics φ using integer indices (vectorized)
    # φ[c,d,k] = sum over n where A[n] == c and X[n,d] == k: 1
    # Handle missing values by not updating φ for missing entries
    
    φ = jnp.zeros((C, D, K))
    
    # Create masks for valid (non-missing, in-bounds) entries
    valid_mask = (X != MISSING_VALUE) & (X >= 0) & (X < K)  # Shape: (N, D)
    
    # Use vectorized approach to avoid conditional statements in JIT
    # Create indices for scatter operation
    n_indices = jnp.arange(N)[:, None]  # Shape: (N, 1)
    d_indices = jnp.arange(D)[None, :]  # Shape: (1, D)
    
    # Broadcast to create full index arrays
    n_expanded = jnp.broadcast_to(n_indices, (N, D))
    d_expanded = jnp.broadcast_to(d_indices, (N, D))
    
    # Get cluster and category indices
    cluster_indices = A[n_expanded]  # Shape: (N, D)
    category_indices = X  # Shape: (N, D)
    
    # Use conditional contributions based on validity mask
    contributions = jnp.where(valid_mask, 1.0, 0.0)
    
    # Use at[].add() with vectorized indices
    φ = φ.at[
        cluster_indices.flatten(),
        d_expanded.flatten(),
        category_indices.flatten()
    ].add(contributions.flatten())
    
    # Update π using posterior given A
    counts = jnp.bincount(A, length=C)  # Count assignments per cluster
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
def gibbs(key, X_B, I_B, A_indices, φ_old, π_old, θ_old, α_pi, α_theta, mask=None):
    """One step of Gibbs sampling for particle Gibbs.

    Args:
        key: PRNG key
        X_B: Minibatch data (B x D) with integer indices
        I_B: Batch indices (B,)
        A_indices: Current assignments as indices (N,)
        φ_old: Current sufficient statistics (C x D x K)
        π_old: Current mixing weights in log space (C,)
        θ_old: Current emission parameters in log space (C x D x K)
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)
        mask: Optional boolean mask (D, K) for valid categories per feature

    Returns:
        Tuple of (A_indices, φ, π, θ, γ, q)
    """
    B, D = X_B.shape
    C = π_old.shape[0]
    K = θ_old.shape[2]  # Get K from θ_old shape
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, B)

    # Compute log likelihoods using integer indices
    # For each data point in batch and each cluster, sum θ[c,d,X_B[b,d]] over d
    # Handle missing values (MISSING_VALUE) by using log-likelihood 0
    
    # Fully vectorized approach - eliminate all loops
    # θ_old shape: (C, D, K)
    # X_B shape: (B, D)
    # We want to compute: log_likelihoods[b, c] = sum_d θ_old[c, d, X_B[b, d]]
    
    # Use vmap to vectorize the .at[].get() operation over all dimensions
    def get_theta_batch_values(theta_k, x_batch):
        """Get theta values for all batch elements for a single (theta_k, x_batch) pair."""
        return jax.vmap(lambda x: theta_k.at[x].get(mode='fill', fill_value=0.0))(x_batch)
    
    # Vectorize over clusters and features
    def get_theta_all_clusters_features(theta_cd, x_batch_d):
        """Get theta values for all clusters and batch elements for all features."""
        return jax.vmap(jax.vmap(get_theta_batch_values, in_axes=(0, None)), in_axes=(0, 0))(theta_cd, x_batch_d)
    
    # Transpose θ_old to (D, C, K) and X_B to (D, B) for easier vectorization
    theta_dck = θ_old.transpose(1, 0, 2)  # Shape: (D, C, K)
    X_B_db = X_B.T  # Shape: (D, B)
    
    # Get θ values for all features, clusters, and batch elements at once
    θ_values_dcb = get_theta_all_clusters_features(theta_dck, X_B_db)  # Shape: (D, C, B)
    
    # Transpose to (B, C, D) and sum over features (d) to get log likelihoods
    θ_values_bcd = θ_values_dcb.transpose(2, 1, 0)  # Shape: (B, C, D)
    log_likelihoods = jnp.sum(θ_values_bcd, axis=2)  # Shape: (B, C)
    
    log_probs = log_likelihoods + π_old[None, :]  # B x C
    
    # Sample new assignments
    A_B_new = jax.vmap(lambda k, lp: jax.random.categorical(k, lp))(keys, log_probs).astype(jnp.uint32)
    
    # Compute proposal probability
    A_B_pgibbs = log_probs[jnp.arange(B), A_B_new]  # B,
    
    # Update sufficient statistics using integer indices
    # Note: φ_old should already have the current batch unassigned before calling gibbs
    # Handle missing values by not updating φ for missing entries
    
    φ_B_new = jnp.zeros((C, D, K))
    
    # Create masks for valid (non-missing, in-bounds) entries
    valid_mask = (X_B != MISSING_VALUE) & (X_B >= 0) & (X_B < K)  # Shape: (B, D)
    
    # Use vectorized approach to avoid conditional statements in JIT
    # Create indices for scatter operation
    b_indices = jnp.arange(B)[:, None]  # Shape: (B, 1)
    d_indices = jnp.arange(D)[None, :]  # Shape: (1, D)
    
    # Broadcast to create full index arrays
    b_expanded = jnp.broadcast_to(b_indices, (B, D))
    d_expanded = jnp.broadcast_to(d_indices, (B, D))
    
    # Get cluster and category indices
    cluster_indices = A_B_new[b_expanded]  # Shape: (B, D)
    category_indices = X_B  # Shape: (B, D)
    
    # Use conditional contributions based on validity mask
    contributions = jnp.where(valid_mask, 1.0, 0.0)
    
    # Use at[].add() with vectorized indices
    φ_B_new = φ_B_new.at[
        cluster_indices.flatten(),
        d_expanded.flatten(),
        category_indices.flatten()
    ].add(contributions.flatten())
    
    φ = φ_old + φ_B_new  # C x D x K (only addition, no subtraction)
    
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
        
        # Compute log probability with masking
        def compute_masked_score(c_θ, c_α):
            def score_for_feature(d):
                score = log_dirichlet_score(c_α[d], c_θ[d], mask[d])
                return jnp.where(jnp.isfinite(score), score, -1e10)
            
            feature_scores = jax.vmap(score_for_feature)(jnp.arange(D))
            return jnp.sum(feature_scores)
        
        cluster_scores = jax.vmap(compute_masked_score)(θ, α_theta_posterior)
        θ_pgibbs = jnp.sum(cluster_scores)
    else:
        sample_theta = lambda k, α: jnp.log(jax.random.dirichlet(k, α))
        θ = jax.vmap(jax.vmap(sample_theta))(keys_theta, α_theta_posterior)
        θ_pgibbs = jnp.sum(jax.vmap(jax.vmap(log_dirichlet_score))(α_theta_posterior, θ))
    
    # Compute proposal log probability
    A_B_sum = jnp.sum(A_B_pgibbs)
    q = A_B_sum + π_pgibbs + θ_pgibbs
    
    # Update assignments
    A_indices = A_indices.at[I_B].set(A_B_new)
    
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
    
    # Compute A_p using integer indices (only for assigned data points)
    assigned_mask = A_indices != EMPTY_ASSIGNMENT
    A_indices_assigned = jnp.where(assigned_mask, A_indices, 0)  # Replace empty with 0 for indexing
    π_values = π[A_indices_assigned]  # Get π values
    A_p = jnp.sum(jnp.where(assigned_mask, π_values, 0.0))  # Only sum assigned points
    
    γ = π_p + θ_p + A_p + X_p
    
    return A_indices, φ, π, θ, γ, q, log_likelihoods


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
        # SMC steps process new data that starts unassigned (EMPTY_ASSIGNMENT)
        # Only unassign data points that are actually assigned before calling gibbs
        C_local = p_π.shape[0]  # Get number of clusters from π
        B_local, D_local = X_B.shape  # Get batch size and feature count
        K_local = p_θ.shape[2]  # Get number of categories
        A_B_old = p_A[I_B]  # Get current assignments for batch
        
        # Only unassign data points that are actually assigned (not EMPTY_ASSIGNMENT)
        assigned_mask = A_B_old != EMPTY_ASSIGNMENT  # Boolean mask for assigned points
        
        # Compute old contributions using integer indices
        φ_B_old = jnp.zeros((C_local, D_local, K_local))
        
        # Create index arrays for assigned points only
        d_indices = jnp.arange(D_local)[None, :]
        
        # Get cluster and category indices, using 0 for unassigned (will be masked out)
        cluster_indices = jnp.where(assigned_mask[:, None], A_B_old[:, None], 0)
        category_indices = X_B
        
        # Create contribution mask
        contribution_mask = assigned_mask[:, None]
        
        # Expand for broadcasting
        cluster_indices_expanded = jnp.broadcast_to(cluster_indices, (B_local, D_local))
        feature_indices_expanded = jnp.broadcast_to(d_indices, (B_local, D_local))
        category_indices_expanded = category_indices
        contribution_mask_expanded = jnp.broadcast_to(contribution_mask, (B_local, D_local))
        
        # Handle missing values: only update φ for valid entries
        valid_data_mask = (X_B != MISSING_VALUE) & (X_B >= 0) & (X_B < K_local)  # Shape: (B_local, D_local)
        
        # Combine assignment validity mask with data validity mask
        final_mask = contribution_mask_expanded & valid_data_mask
        
        # Use at[].add() with conditional contributions
        contributions = jnp.where(final_mask, 1.0, 0.0)
        φ_B_old = φ_B_old.at[
            cluster_indices_expanded.flatten(),
            feature_indices_expanded.flatten(),
            category_indices_expanded.flatten()
        ].add(contributions.flatten())
        φ_B_old = φ_B_old.reshape(C_local, D_local, K_local)
        p_φ_unassigned = p_φ - φ_B_old  # Remove old assignments from φ
        return gibbs(p_key, X_B, I_B, p_A, p_φ_unassigned, p_π, p_θ, α_pi, α_theta, mask)
    
    A_new, φ_new, π_new, θ_new, γ_new, q_new, batch_log_liks = jax.vmap(step_particle)(keys, A, φ, π, θ)
    
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
        batch_log_liks_resampled = batch_log_liks[indices]
        w_resampled = jnp.full(P, -jnp.log(P))  # Uniform weights in log space

        return (A_resampled, φ_resampled, π_resampled, θ_resampled), w_resampled, γ_resampled, batch_log_liks_resampled
    
    def no_resample(key):
        return (A_new, φ_new, π_new, θ_new), w, γ_new, batch_log_liks
    
    key, subkey = jax.random.split(key)
    particles_out, w_out, γ_out, batch_log_liks_out = jax.lax.cond(
        ess < resample_threshold,
        resample,
        no_resample,
        subkey
    )
    
    return particles_out, w_out, γ_out, batch_log_liks_out


def smc_no_rejuvenation(key, X, T, P, C, B, K, α_pi, α_theta, rejuvenation=False, return_history=False, mask=None):
    """Run SMC with optional rejuvenation on data.
    
    Args:
        key: PRNG key
        X: Full dataset (N x D) with integer indices
        T: Number of iterations
        P: Number of particles
        C: Number of clusters
        B: Batch size
        K: Number of categories per feature (max across all features if mask provided)
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
    N, D = X.shape
    
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
            'A': jnp.zeros((T, P, N), dtype=jnp.uint32),  # A is now integer indices
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
        particles, log_weights, log_gammas, _ = smc_step(
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
                # Rejuvenation operates on already-assigned data, so unassign normally
                C_local = p_π.shape[0]  # Get number of clusters from π
                A_rejuv_old = p_A[I_rejuv]  # Get current assignments for rejuv batch
                
                # Compute old contributions using integer indices
                B_local, D_local = X_rejuv.shape  # Get batch size and feature count
                K_local = p_θ.shape[2]  # Get number of categories
                φ_rejuv_old = jnp.zeros((C_local, D_local, K_local))
                
                # Create indices for vectorized update
                d_indices = jnp.arange(D_local)[None, :]
                
                # Expand assignments and data for broadcasting
                A_rejuv_expanded = A_rejuv_old[:, None]  # Shape: (B, 1)
                X_rejuv_values = X_rejuv  # Shape: (B, D)
                
                # Create full index arrays
                cluster_indices = jnp.broadcast_to(A_rejuv_expanded, (B_local, D_local))
                feature_indices = jnp.broadcast_to(d_indices, (B_local, D_local))
                category_indices = X_rejuv_values
                
                # Handle missing values: only update φ for valid entries
                valid_data_mask = (X_rejuv != MISSING_VALUE) & (X_rejuv >= 0) & (X_rejuv < K_local)
                
                # Use at[].add() with vectorized indices, only for valid entries
                contributions = jnp.where(valid_data_mask, 1.0, 0.0)
                φ_rejuv_old = φ_rejuv_old.at[
                    cluster_indices.flatten(),
                    feature_indices.flatten(),
                    category_indices.flatten()
                ].add(contributions.flatten())
                φ_rejuv_old = φ_rejuv_old.reshape(C_local, D_local, K_local)
                p_φ_unassigned = p_φ - φ_rejuv_old  # Remove old assignments from φ
                # Run gibbs but only return updated parameters
                A_new, φ_new, π_new, θ_new, _, _, _ = gibbs(
                    p_key, X_rejuv, I_rejuv, p_A, p_φ_unassigned, p_π, p_θ, α_pi, α_theta, mask
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


def mcmc_minibatch(key, X, T, C, B, K, α_pi, α_theta, mask=None):
    """Run MCMC algorithm with minibatches using Gibbs moves.

    Args:
        key: PRNG key
        X: Full dataset (N x D) with integer indices
        T: Number of iterations
        C: Number of clusters
        B: Minibatch size
        K: Number of categories per feature (max across all features if mask provided)
        α_pi: Dirichlet prior for mixing weights (scalar)
        α_theta: Dirichlet prior for emission parameters (scalar)
        mask: Optional boolean mask (D, K) for valid categories per feature

    Returns:
        Final state (A, φ, π, θ)
    """
    N, _ = X.shape
    
    # Initialize using assignments
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_assignments(subkey, X, C, K, α_pi, α_theta, mask)
    
    for _ in range(T):
        # Sample random batch indices
        key, subkey = jax.random.split(key)
        I_B = jax.random.choice(subkey, N, shape=(B,), replace=False)
        X_B = X[I_B]
        
        # MCMC operates on already-assigned data, so unassign normally
        A_B_old = A[I_B]  # Get current assignments for batch
        
        # Compute old contributions using integer indices
        B_local, D_local = X_B.shape  # Get batch size and feature count
        φ_B_old = jnp.zeros((C, D_local, K))
        
        # Create indices for vectorized update
        d_indices = jnp.arange(D_local)[None, :]
        
        # Expand assignments and data for broadcasting
        A_B_old_expanded = A_B_old[:, None]  # Shape: (B, 1)
        X_B_values = X_B  # Shape: (B, D)
        
        # Create full index arrays
        cluster_indices = jnp.broadcast_to(A_B_old_expanded, (B_local, D_local))
        feature_indices = jnp.broadcast_to(d_indices, (B_local, D_local))
        category_indices = X_B_values
        
        # Handle missing values: only update φ for valid entries
        valid_data_mask = (X_B != MISSING_VALUE) & (X_B >= 0) & (X_B < K)
        
        # Use at[].add() with vectorized indices, only for valid entries
        contributions = jnp.where(valid_data_mask, 1.0, 0.0)
        φ_B_old = φ_B_old.at[
            cluster_indices.flatten(),
            feature_indices.flatten(),
            category_indices.flatten()
        ].add(contributions.flatten())
        φ_B_old = φ_B_old.reshape(C, D_local, K)
        φ_unassigned = φ - φ_B_old  # Remove old assignments from φ
        
        # Run Gibbs sampler on the batch
        key, subkey = jax.random.split(key)
        A, φ, π, θ, _, _, _ = gibbs(
            subkey, X_B, I_B, A, φ_unassigned, π, θ, α_pi, α_theta, mask
        )
    
    return A, φ, π, θ