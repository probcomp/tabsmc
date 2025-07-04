#!/usr/bin/env python
"""Evaluate likelihood of trained SMC models."""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

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


def load_data_for_evaluation(dataset_path="data/lpm/PUMS"):
    """Load PUMS data for evaluation."""
    from tabsmc.io import load_data
    import numpy as np
    
    # Load data
    train_data_log, test_data_log, col_names, mask = load_data(dataset_path)
    
    # Convert from log-space to proper one-hot encoding
    train_data = (train_data_log == 0.0).astype(np.float32)
    test_data = (test_data_log == 0.0).astype(np.float32)
    
    # Convert mask to boolean
    mask = (mask > 0).astype(bool)
    
    return train_data, test_data, col_names, mask


def evaluate_model(results_file, dataset_path="data/lpm/PUMS", 
                  train_eval_size=10000, test_eval_size=5000):
    """Evaluate a trained SMC model."""
    
    # Load trained model
    print(f"Loading trained model from {results_file}...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    # Extract final particles and weights
    particles = results['final_particles']
    log_weights = results['final_log_weights']
    config = results['config']
    
    print(f"Model configuration:")
    print(f"  Particles: {config['n_particles']}")
    print(f"  Clusters: {config['n_clusters']}")
    print(f"  Training steps: {config['n_time_steps']}")
    print(f"  Batch size: {config['batch_size']}")
    
    # Load data
    print(f"\nLoading evaluation data...")
    train_data, test_data, col_names, mask = load_data_for_evaluation(dataset_path)
    
    print(f"Data shapes:")
    print(f"  Train: {train_data.shape}")
    print(f"  Test: {test_data.shape}")
    
    # Evaluate on training subset
    print(f"\nEvaluating on training data (subset of {train_eval_size})...")
    train_indices = np.random.choice(train_data.shape[0], 
                                   min(train_eval_size, train_data.shape[0]), 
                                   replace=False)
    X_train_eval = jnp.array(train_data[train_indices])
    train_ll = compute_weighted_test_loglik(X_train_eval, particles, log_weights)
    train_ll_per_point = train_ll / len(train_indices)
    
    print(f"  Train log-likelihood: {train_ll:.4f}")
    print(f"  Train LL per data point: {train_ll_per_point:.4f}")
    
    # Evaluate on test subset
    print(f"\nEvaluating on test data (subset of {test_eval_size})...")
    test_indices = np.random.choice(test_data.shape[0], 
                                  min(test_eval_size, test_data.shape[0]), 
                                  replace=False)
    X_test_eval = jnp.array(test_data[test_indices])
    test_ll = compute_weighted_test_loglik(X_test_eval, particles, log_weights)
    test_ll_per_point = test_ll / len(test_indices)
    
    print(f"  Test log-likelihood: {test_ll:.4f}")
    print(f"  Test LL per data point: {test_ll_per_point:.4f}")
    
    # Compute per-particle likelihoods for diagnostics
    print(f"\nPer-particle diagnostics:")
    _, _, π, θ = particles
    
    # Train diagnostics
    train_lls_per_particle = jax.vmap(compute_test_loglik_vectorized, in_axes=(None, 0, 0))(
        X_train_eval, π, θ
    )
    print(f"  Train LL per particle range: [{jnp.min(train_lls_per_particle):.4f}, {jnp.max(train_lls_per_particle):.4f}]")
    
    # Test diagnostics  
    test_lls_per_particle = jax.vmap(compute_test_loglik_vectorized, in_axes=(None, 0, 0))(
        X_test_eval, π, θ
    )
    print(f"  Test LL per particle range: [{jnp.min(test_lls_per_particle):.4f}, {jnp.max(test_lls_per_particle):.4f}]")
    
    # Weight diagnostics
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    effective_sample_size = 1.0 / jnp.sum(jnp.exp(log_weights_normalized) ** 2)
    print(f"  Effective sample size: {effective_sample_size:.1f} / {len(log_weights)}")
    print(f"  Log weights range: [{jnp.min(log_weights):.3f}, {jnp.max(log_weights):.3f}]")
    
    # Return evaluation results
    evaluation_results = {
        'train_ll': float(train_ll),
        'train_ll_per_point': float(train_ll_per_point),
        'test_ll': float(test_ll),
        'test_ll_per_point': float(test_ll_per_point),
        'train_ll_per_particle': train_lls_per_particle,
        'test_ll_per_particle': test_lls_per_particle,
        'effective_sample_size': float(effective_sample_size),
        'train_eval_size': len(train_indices),
        'test_eval_size': len(test_indices),
    }
    
    return evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SMC model likelihoods")
    parser.add_argument("results_file", type=str, 
                       help="Path to saved SMC results file (.pkl)")
    parser.add_argument("--dataset", type=str, default="data/lpm/PUMS",
                       help="HuggingFace dataset path (default: data/lpm/PUMS)")
    parser.add_argument("--train-eval-size", type=int, default=10000,
                       help="Number of training samples to evaluate (default: 10000)")
    parser.add_argument("--test-eval-size", type=int, default=5000,
                       help="Number of test samples to evaluate (default: 5000)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file to save evaluation results")
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file {args.results_file} not found!")
        return
    
    # Run evaluation
    evaluation_results = evaluate_model(
        args.results_file,
        args.dataset,
        args.train_eval_size,
        args.test_eval_size
    )
    
    # Save evaluation results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(evaluation_results, f)
        
        print(f"\nEvaluation results saved to {output_path}")
    
    print(f"\nEvaluation completed!")


if __name__ == "__main__":
    main()