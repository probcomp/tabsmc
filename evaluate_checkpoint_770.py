#!/usr/bin/env python
"""Evaluate the step 770 checkpoint to get proper log-likelihood per data point."""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path
from tabsmc.io import load_data

@jax.jit
def compute_test_loglik_vectorized(X_test, π, θ):
    """Compute test log-likelihood for a single particle."""
    # Use where to avoid 0 * (-inf) = NaN
    observed_logprobs = jnp.where(
        X_test[:, None, :, :] == 1,  # (N, 1, D, K)
        θ[None, :, :, :],            # (1, C, D, K)
        0.0
    )
    log_px_given_c = jnp.sum(observed_logprobs, axis=(2, 3))  # (N, C)
    log_px = jax.scipy.special.logsumexp(π[None, :] + log_px_given_c, axis=1)
    return jnp.sum(log_px)


def compute_weighted_test_loglik(X_test, particles, log_weights):
    """Compute weighted average test log-likelihood."""
    _, _, π, θ = particles
    
    # Compute log-likelihood for each particle
    log_liks = jax.vmap(compute_test_loglik_vectorized, in_axes=(None, 0, 0))(X_test, π, θ)
    
    # Weighted average
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    return jax.scipy.special.logsumexp(log_liks + log_weights_normalized)


def main():
    # Load checkpoint
    checkpoint_file = Path("results/smc_pums_checkpoint_step_770.pkl")
    print(f"Loading checkpoint: {checkpoint_file}")
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    particles = checkpoint['particles']
    log_weights = checkpoint['log_weights']
    config = checkpoint['config']
    
    print(f"\nModel configuration:")
    print(f"  Particles: {config['n_particles']}")
    print(f"  Clusters: {config['n_clusters']}")
    print(f"  Step: {checkpoint['step']}")
    
    # Load data
    print(f"\nLoading PUMS data...")
    train_data_log, test_data_log, col_names, mask = load_data("data/lpm/PUMS")
    
    # Convert from log-space to proper one-hot encoding
    train_data = (train_data_log == 0.0).astype(np.float32)
    test_data = (test_data_log == 0.0).astype(np.float32)
    
    print(f"Data shapes:")
    print(f"  Train: {train_data.shape}")
    print(f"  Test: {test_data.shape}")
    
    # Evaluate on subsets to avoid memory issues
    train_eval_size = 5000
    test_eval_size = 2000
    
    # Training set evaluation
    print(f"\nEvaluating on {train_eval_size} training samples...")
    train_indices = np.random.choice(train_data.shape[0], train_eval_size, replace=False)
    X_train_eval = jnp.array(train_data[train_indices])
    
    train_ll = compute_weighted_test_loglik(X_train_eval, particles, log_weights)
    train_ll_per_point = train_ll / train_eval_size
    
    print(f"  Train log-likelihood: {train_ll:.4f}")
    print(f"  Train LL per data point: {train_ll_per_point:.4f}")
    
    # Test set evaluation
    print(f"\nEvaluating on {test_eval_size} test samples...")
    test_indices = np.random.choice(test_data.shape[0], test_eval_size, replace=False)
    X_test_eval = jnp.array(test_data[test_indices])
    
    test_ll = compute_weighted_test_loglik(X_test_eval, particles, log_weights)
    test_ll_per_point = test_ll / test_eval_size
    
    print(f"  Test log-likelihood: {test_ll:.4f}")
    print(f"  Test LL per data point: {test_ll_per_point:.4f}")
    
    # Additional diagnostics
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    effective_sample_size = 1.0 / jnp.sum(jnp.exp(log_weights_normalized) ** 2)
    print(f"\nDiagnostics:")
    print(f"  Effective sample size: {effective_sample_size:.1f} / {len(log_weights)}")
    print(f"  Log weights range: [{jnp.min(log_weights):.3f}, {jnp.max(log_weights):.3f}]")
    
    # Compare with batch log-likelihoods from training
    if 'log_likelihoods' in checkpoint:
        batch_lls = checkpoint['log_likelihoods'][-1]['batch_log_likelihoods']
        batch_ll_mean = float(np.mean(np.array(batch_lls)))
        print(f"\nTraining batch log-likelihood (last step): {batch_ll_mean:.4f}")
        print(f"Note: This is per-batch, not per-data-point")


if __name__ == "__main__":
    main()