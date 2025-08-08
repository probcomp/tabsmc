"""
SGD baseline implementation for Dirichlet-Categorical mixture model.

This module provides a stochastic gradient descent baseline for comparison
with the SMC implementation using simple mixture model MAP estimation.
"""

import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import Tuple, Optional
from tqdm import tqdm


@jax.jit
def mixture_loss(params, X_batch, mask=None):
    """Compute negative log-likelihood for mixture model (MAP estimation).
    
    Args:
        params: Dict with 'pi_logits' (C,) and 'theta_logits' (C, D, K)
        X_batch: Data batch (B, D, K) in one-hot format
        mask: Optional boolean mask (D, K) for valid categories
    
    Returns:
        Negative log-likelihood (scalar)
    """
    pi_logits = params['pi_logits']
    theta_logits = params['theta_logits']
    
    B, D, K = X_batch.shape
    C = pi_logits.shape[0]
    
    # Convert logits to log probabilities
    log_pi = jax.nn.log_softmax(pi_logits)  # (C,)
    
    # Handle masking for theta
    if mask is not None:
        # Apply mask to logits before softmax
        theta_logits_masked = jnp.where(
            mask[None, :, :],
            theta_logits,
            -1e10  # Large negative value for invalid categories
        )
        log_theta = jax.nn.log_softmax(theta_logits_masked, axis=-1)  # (C, D, K)
    else:
        log_theta = jax.nn.log_softmax(theta_logits, axis=-1)  # (C, D, K)
    
    # Compute log p(x_n | z_n=c, theta_c) for each cluster c and datapoint n
    # Using einsum for efficient computation: X is (B,D,K), log_theta is (C,D,K)
    log_cluster_likes = jnp.einsum('bdk,cdk->bc', X_batch, log_theta)  # (B, C)
    
    # Add mixing weights: log p(x_n, z_n=c) = log p(x_n | z_n=c) + log p(z_n=c)
    log_joint = log_cluster_likes + log_pi[None, :]  # (B, C)
    
    # Marginalize over clusters: log p(x_n) = log sum_c p(x_n, z_n=c)
    log_marginal = jax.scipy.special.logsumexp(log_joint, axis=1)  # (B,)
    
    # Return negative log-likelihood (to minimize)
    return -jnp.sum(log_marginal)


@partial(jax.jit, static_argnums=(3,))
def sgd_step(params, opt_state, X_batch, optimizer, mask=None):
    """One SGD step using mixture model MAP estimation."""
    loss, grads = jax.value_and_grad(mixture_loss)(params, X_batch, mask)
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
    """Compute test log-likelihood using mixture model."""
    pi_logits = params['pi_logits']
    theta_logits = params['theta_logits']
    
    # Convert logits to log probabilities
    log_pi = jax.nn.log_softmax(pi_logits)  # (C,)
    
    # Handle masking for theta
    if mask is not None:
        theta_logits_masked = jnp.where(
            mask[None, :, :],
            theta_logits,
            -1e10
        )
        log_theta = jax.nn.log_softmax(theta_logits_masked, axis=-1)  # (C, D, K)
    else:
        log_theta = jax.nn.log_softmax(theta_logits, axis=-1)  # (C, D, K)
    
    # Compute log p(x_n | z_n=c) for each cluster and datapoint
    log_cluster_likes = jnp.einsum('bdk,cdk->bc', X_test, log_theta)  # (B, C)
    
    # Add mixing weights and marginalize
    log_joint = log_cluster_likes + log_pi[None, :]  # (B, C)
    log_px = jax.scipy.special.logsumexp(log_joint, axis=1)  # (B,)
    
    return jnp.sum(log_px)


def sgd_train(key, X, T, C, B, learning_rate=0.01, mask=None):
    """Train model using SGD with MAP estimation.
    
    Args:
        key: PRNG key
        X: Full dataset (N x D x K)
        T: Number of iterations
        C: Number of clusters
        B: Batch size
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
        params, opt_state, loss = sgd_step(params, opt_state, X_batch, optimizer, mask)
    
    return params


def sgd_train_with_random_batches(key, X, T, C, B, learning_rate=0.01, return_history=False, mask=None):
    """Train model using SGD with random minibatches and MAP estimation.
    
    Args:
        key: PRNG key
        X: Full dataset (N x D x K)
        T: Number of iterations  
        C: Number of clusters
        B: Batch size
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
        params, opt_state, loss = sgd_step(params, opt_state, X_batch, optimizer, mask)
        
        if return_history:
            history['pi_logits'] = history['pi_logits'].at[t].set(params['pi_logits'])
            history['theta_logits'] = history['theta_logits'].at[t].set(params['theta_logits'])
            history['losses'] = history['losses'].at[t].set(loss)
    
    if return_history:
        return params, history
    return params