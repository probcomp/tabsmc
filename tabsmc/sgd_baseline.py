"""
SGD baseline implementation for Dirichlet-Categorical mixture model.

This module provides a stochastic gradient descent baseline for comparison
with the SMC implementation.
"""

import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import Tuple, Optional
from tqdm import tqdm


def softmax_to_log(logits):
    """Convert logits to log probabilities."""
    return jax.nn.log_softmax(logits, axis=-1)


def dirichlet_kl_divergence(log_q, log_prior, prior_alpha):
    """Compute KL divergence from Dirichlet prior to log probabilities.
    
    Uses approximation: KL[q||Dir(α)] ≈ -H[q] + (α-1)∑log(q_i) - log(B(α))
    where H[q] is entropy and B(α) is the beta function.
    """
    # Entropy of q
    q = jnp.exp(log_q)
    entropy = -jnp.sum(q * log_q, axis=-1)
    
    # Prior term
    alpha_sum = jnp.sum(prior_alpha)
    log_beta = jnp.sum(jax.scipy.special.gammaln(prior_alpha)) - jax.scipy.special.gammaln(alpha_sum)
    prior_term = (prior_alpha - 1) @ log_q - log_beta
    
    return -entropy - prior_term


@jax.jit
def elbo_loss(params, X_batch, α_pi, α_theta, mask=None):
    """Compute negative ELBO loss for a batch.
    
    Args:
        params: Dict with 'pi_logits' (C,) and 'theta_logits' (C, D, K)
        X_batch: Data batch (B, D, K)
        α_pi: Dirichlet prior for mixing weights
        α_theta: Dirichlet prior for emission parameters
        mask: Optional boolean mask (D, K) for valid categories
    
    Returns:
        Negative ELBO loss (scalar)
    """
    pi_logits = params['pi_logits']
    theta_logits = params['theta_logits']
    
    # Convert to log probabilities
    log_pi = softmax_to_log(pi_logits)
    
    # Handle masking for theta
    if mask is not None:
        # Apply mask to logits before softmax
        theta_logits_masked = jnp.where(
            mask[None, :, :],
            theta_logits,
            -1e10  # Large negative value for invalid categories
        )
        log_theta = jax.vmap(jax.vmap(softmax_to_log))(theta_logits_masked)
    else:
        log_theta = jax.vmap(jax.vmap(softmax_to_log))(theta_logits)
    
    B, D, K = X_batch.shape
    C = log_pi.shape[0]
    
    # Compute log p(x|c) for each cluster
    # Handle -inf * 0.0 case
    theta_expanded = log_theta[None, :, :, :]  # 1 x C x D x K
    X_expanded = X_batch[:, None, :, :]  # B x 1 x D x K
    
    products = jnp.where(
        X_expanded == 0.0,
        0.0,
        theta_expanded * X_expanded
    )
    
    log_px_given_c = jnp.sum(products, axis=(2, 3))  # B x C
    
    # Compute posterior q(c|x) using Bayes rule
    log_qc_given_x = log_pi[None, :] + log_px_given_c  # B x C
    log_qc_given_x = log_qc_given_x - jax.scipy.special.logsumexp(log_qc_given_x, axis=1, keepdims=True)
    
    # Expected log likelihood
    qc = jnp.exp(log_qc_given_x)
    expected_log_lik = jnp.sum(qc * (log_pi[None, :] + log_px_given_c))
    
    # KL divergence for pi
    kl_pi = dirichlet_kl_divergence(log_pi, log_pi, jnp.ones(C) * α_pi)
    
    # KL divergence for theta
    if mask is not None:
        # Compute KL only for valid categories
        def masked_kl_for_cluster(log_theta_c):
            # log_theta_c has shape (D, K)
            def masked_kl_for_feature(log_theta_cd, d):
                mask_d = mask[d]
                alpha_d = jnp.where(mask_d, α_theta, 1.0)  # Use 1.0 for invalid (won't contribute)
                kl = dirichlet_kl_divergence(log_theta_cd, log_theta_cd, alpha_d)
                # Only return KL if there are valid categories
                return jnp.where(jnp.sum(mask_d) > 0, kl, 0.0)
            
            return jnp.sum(jax.vmap(masked_kl_for_feature, in_axes=(0, 0))(log_theta_c, jnp.arange(D)))
        
        kl_theta = jnp.sum(jax.vmap(masked_kl_for_cluster)(log_theta))
    else:
        kl_theta_per_cd = jax.vmap(jax.vmap(lambda lt: dirichlet_kl_divergence(lt, lt, jnp.ones(K) * α_theta)))(log_theta)
        kl_theta = jnp.sum(kl_theta_per_cd)
    
    # Negative ELBO
    elbo = expected_log_lik - kl_pi - kl_theta
    return -elbo / B  # Normalize by batch size


@partial(jax.jit, static_argnums=(3,))
def sgd_step(params, opt_state, X_batch, optimizer, α_pi, α_theta, mask=None):
    """One SGD step."""
    loss, grads = jax.value_and_grad(elbo_loss)(params, X_batch, α_pi, α_theta, mask)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def init_params(key, C, D, K, scale=0.1):
    """Initialize parameters with small random values."""
    key1, key2 = jax.random.split(key)
    
    pi_logits = jax.random.normal(key1, (C,)) * scale
    theta_logits = jax.random.normal(key2, (C, D, K)) * scale
    
    return {
        'pi_logits': pi_logits,
        'theta_logits': theta_logits
    }


@jax.jit
def compute_test_loglik(params, X_test, mask=None):
    """Compute test log-likelihood."""
    pi_logits = params['pi_logits']
    theta_logits = params['theta_logits']
    
    # Convert to log probabilities
    log_pi = softmax_to_log(pi_logits)
    
    # Handle masking for theta
    if mask is not None:
        theta_logits_masked = jnp.where(
            mask[None, :, :],
            theta_logits,
            -1e10
        )
        log_theta = jax.vmap(jax.vmap(softmax_to_log))(theta_logits_masked)
    else:
        log_theta = jax.vmap(jax.vmap(softmax_to_log))(theta_logits)
    
    # Compute log p(x)
    theta_expanded = log_theta[None, :, :, :]
    X_expanded = X_test[:, None, :, :]
    
    products = jnp.where(
        X_expanded == 0.0,
        0.0,
        theta_expanded * X_expanded
    )
    
    log_px_given_c = jnp.sum(products, axis=(2, 3))
    log_px = jax.scipy.special.logsumexp(log_pi[None, :] + log_px_given_c, axis=1)
    
    return jnp.sum(log_px)


def sgd_train(key, X, T, C, B, α_pi, α_theta, learning_rate=0.01, mask=None):
    """Train model using SGD.
    
    Args:
        key: PRNG key
        X: Full dataset (N x D x K)
        T: Number of iterations
        C: Number of clusters
        B: Batch size
        α_pi: Dirichlet prior for mixing weights
        α_theta: Dirichlet prior for emission parameters
        learning_rate: Learning rate for Adam optimizer
        mask: Optional boolean mask (D, K) for valid categories
        
    Returns:
        Final parameters
    """
    N, D, K = X.shape
    
    # Initialize parameters
    key, subkey = jax.random.split(key)
    params = init_params(subkey, C, D, K)
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Training loop
    for t in range(T):
        # Deterministic batch (same as SMC for fair comparison)
        start_idx = t * B
        end_idx = start_idx + B
        X_batch = X[start_idx:end_idx]
        
        # SGD step
        params, opt_state, loss = sgd_step(params, opt_state, X_batch, optimizer, α_pi, α_theta, mask)
    
    return params


def sgd_train_with_random_batches(key, X, T, C, B, α_pi, α_theta, learning_rate=0.01, return_history=False, mask=None):
    """Train model using SGD with random minibatches.
    
    Args:
        key: PRNG key
        X: Full dataset (N x D x K)
        T: Number of iterations  
        C: Number of clusters
        B: Batch size
        α_pi: Dirichlet prior for mixing weights
        α_theta: Dirichlet prior for emission parameters
        learning_rate: Learning rate for Adam optimizer
        return_history: If True, return history of parameters
        mask: Optional boolean mask (D, K) for valid categories
        
    Returns:
        Final parameters (and optionally history)
    """
    N, D, K = X.shape
    
    # Initialize parameters
    key, subkey = jax.random.split(key)
    params = init_params(subkey, C, D, K)
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    if return_history:
        history = {
            'pi_logits': jnp.zeros((T, C)),
            'theta_logits': jnp.zeros((T, C, D, K)),
            'losses': jnp.zeros(T)
        }
    
    # Training loop
    for t in tqdm(range(T)):
        # Random batch sampling
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(subkey, N, shape=(B,), replace=False)
        X_batch = X[indices]
        
        # SGD step
        params, opt_state, loss = sgd_step(params, opt_state, X_batch, optimizer, α_pi, α_theta, mask)
        
        if return_history:
            history['pi_logits'] = history['pi_logits'].at[t].set(params['pi_logits'])
            history['theta_logits'] = history['theta_logits'].at[t].set(params['theta_logits'])
            history['losses'] = history['losses'].at[t].set(loss)
    
    if return_history:
        return params, history
    return params