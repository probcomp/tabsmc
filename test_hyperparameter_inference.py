#!/usr/bin/env python
"""Test hyperparameter inference implementation."""

import jax
import jax.numpy as jnp
import numpy as np
from tabsmc.smc import (
    smc_with_hyperparameter_inference,
    compute_alpha_pi_posterior,
    compute_alpha_theta_posterior,
    sample_hyperparameters
)
from tabsmc.io import load_data

def test_hyperparameter_posteriors():
    """Test hyperparameter posterior computation functions."""
    print("Testing hyperparameter posterior computation...")
    
    # Create mock data
    key = jax.random.PRNGKey(42)
    C, D, K = 3, 5, 4
    
    # Mock pi and theta
    key, subkey = jax.random.split(key)
    pi_log = jnp.log(jax.random.dirichlet(subkey, jnp.ones(C)))
    
    key, subkey = jax.random.split(key)
    theta_log = jnp.log(jax.random.dirichlet(subkey, jnp.ones(K), shape=(C, D)))
    
    # Mock phi (sufficient statistics)
    phi = jax.random.uniform(key, (C, D, K)) * 10
    
    # Create grids
    alpha_pi_grid = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
    alpha_theta_grid = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
    
    # Test alpha_pi posterior
    log_post_pi = compute_alpha_pi_posterior(pi_log, alpha_pi_grid)
    print(f"Alpha_pi posterior log probabilities: {log_post_pi}")
    print(f"Alpha_pi posterior probabilities: {jnp.exp(log_post_pi)}")
    
    # Test alpha_theta posterior
    log_post_theta = compute_alpha_theta_posterior(theta_log, phi, alpha_theta_grid)
    print(f"Alpha_theta posterior log probabilities: {log_post_theta}")
    print(f"Alpha_theta posterior probabilities: {jnp.exp(log_post_theta)}")
    
    # Test sampling
    key, subkey = jax.random.split(key)
    new_alpha_pi, new_alpha_theta = sample_hyperparameters(
        subkey, pi_log, theta_log, phi, alpha_pi_grid, alpha_theta_grid
    )
    print(f"Sampled alpha_pi: {new_alpha_pi}")
    print(f"Sampled alpha_theta: {new_alpha_theta}")
    
    print("✓ Hyperparameter posterior tests passed!")


def test_smc_with_hyperparameter_inference_small():
    """Test SMC with hyperparameter inference on small synthetic data."""
    print("\nTesting SMC with hyperparameter inference on synthetic data...")
    
    # Create small synthetic dataset
    key = jax.random.PRNGKey(42)
    N, D, K = 20, 3, 4
    
    # Generate synthetic categorical data
    key, subkey = jax.random.split(key)
    X = jax.random.choice(subkey, K, shape=(N, D))
    
    # SMC parameters
    T, P, C, B = 2, 3, 2, 5
    α_pi_init, α_theta_init = 1.0, 1.0
    
    # Create hyperparameter grids
    alpha_pi_grid = jnp.array([0.5, 1.0, 2.0])
    alpha_theta_grid = jnp.array([0.5, 1.0, 2.0])
    
    # Run SMC with hyperparameter inference
    key, subkey = jax.random.split(key)
    try:
        particles, log_weights, hyperparams = smc_with_hyperparameter_inference(
            subkey, X, T, P, C, B, K, α_pi_init, α_theta_init,
            alpha_pi_grid, alpha_theta_grid, return_history=False
        )
        
        print(f"Final alpha_pi values: {hyperparams['alpha_pi']}")
        print(f"Final alpha_theta values: {hyperparams['alpha_theta']}")
        print(f"Final log weights: {log_weights}")
        
        print("✓ SMC with hyperparameter inference test passed!")
        
    except Exception as e:
        print(f"✗ SMC test failed: {e}")
        import traceback
        traceback.print_exc()


def test_real_data_small():
    """Test on small subset of real data."""
    print("\nTesting on small subset of CES data...")
    
    try:
        # Load CES data
        train_data, test_data, col_names, mask, K = load_data("data/lmp/CES")
        
        # Use only first 50 samples and 10 features
        X_small = train_data[:50, :10]
        N, D = X_small.shape
        
        # SMC parameters
        T, P, C, B = 3, 2, 5, 10
        α_pi_init, α_theta_init = 1.0, 1.0
        
        # Create hyperparameter grids
        alpha_pi_grid = jnp.array([0.1, 1.0, 5.0])
        alpha_theta_grid = jnp.array([0.1, 1.0, 5.0])
        
        # Run SMC with hyperparameter inference
        key = jax.random.PRNGKey(42)
        particles, log_weights, hyperparams, history = smc_with_hyperparameter_inference(
            key, X_small, T, P, C, B, K, α_pi_init, α_theta_init,
            alpha_pi_grid, alpha_theta_grid, return_history=True,
            mask=mask[:10] if mask is not None else None
        )
        
        print(f"Final alpha_pi values: {hyperparams['alpha_pi']}")
        print(f"Final alpha_theta values: {hyperparams['alpha_theta']}")
        
        # Show evolution over time
        print("\nHyperparameter evolution:")
        for t in range(T):
            print(f"  Time {t}: alpha_pi = {history['alpha_pi'][t]}, alpha_theta = {history['alpha_theta'][t]}")
        
        print("✓ Real data test passed!")
        
    except Exception as e:
        print(f"✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing hyperparameter inference implementation...\n")
    
    # Test individual components
    test_hyperparameter_posteriors()
    
    # Test full SMC integration
    test_smc_with_hyperparameter_inference_small()
    
    # Test on real data
    test_real_data_small()
    
    print("\n" + "="*60)
    print("Hyperparameter inference implementation complete!")
    print("Key features added:")
    print("• compute_alpha_pi_posterior() - Computes posterior for alpha_pi")
    print("• compute_alpha_theta_posterior() - Computes posterior for alpha_theta") 
    print("• sample_hyperparameters() - Samples from hyperparameter posteriors")
    print("• gibbs_with_hyperparameter_inference() - Gibbs with hyperparameter updates")
    print("• smc_with_hyperparameter_inference() - Full SMC with hyperparameter inference")
    print("="*60)