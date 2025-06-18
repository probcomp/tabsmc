"""
Compute the optimal log-likelihood (negative entropy) for the true data distribution.
"""

import jax
import jax.numpy as jnp


def compute_optimal_loglik_per_datapoint(true_π, true_θ):
    """
    Compute the optimal log-likelihood per datapoint under the true distribution.
    
    For a mixture model, this is the expected log-likelihood:
    E[log P(X | true_params)] = sum_c π_c * sum_d sum_k θ_{c,d,k} * log(sum_c' π_{c'} * θ_{c',d,k})
    
    But it's easier to compute empirically by generating a large sample.
    
    Args:
        true_π: True mixing weights (C,)
        true_θ: True emission parameters (C, D, K)
    
    Returns:
        Optimal log-likelihood per datapoint (computed empirically)
    """
    # For mixture models, the theoretical entropy is complex to compute analytically
    # because features are conditionally independent given mixture component,
    # but not marginally independent.
    # 
    # Instead, we'll compute it empirically by generating a very large sample
    # and evaluating the true model on it.
    
    from generate_synthetic_data import generate_mixture_data
    
    key = jax.random.PRNGKey(123456)  # Fixed seed for reproducibility
    N_large = 100000  # Very large sample for accurate estimate
    C, D, K = true_θ.shape
    
    X_large, _ = generate_mixture_data(key, N_large, D, K, true_π, true_θ)
    
    # Compute log P(X_large | true_params)
    log_px_given_c = jnp.einsum('ndk,cdk->nc', X_large, jnp.log(true_θ + 1e-10))
    log_px = jax.scipy.special.logsumexp(jnp.log(true_π + 1e-10)[None, :] + log_px_given_c, axis=1)
    
    # Average over large sample
    optimal_loglik_per_datapoint = jnp.mean(log_px)
    
    return optimal_loglik_per_datapoint


def verify_optimal_loglik_empirically(X_test, true_π, true_θ):
    """
    Verify the optimal log-likelihood computation by evaluating the true model on test data.
    
    Args:
        X_test: Test data (N_test, D, K) - one-hot encoded
        true_π: True mixing weights (C,)
        true_θ: True emission parameters (C, D, K)
    
    Returns:
        Empirical log-likelihood per datapoint using true parameters
    """
    N_test = X_test.shape[0]
    
    # Compute log P(X_test | true_params)
    log_px_given_c = jnp.einsum('ndk,cdk->nc', X_test, jnp.log(true_θ + 1e-10))
    log_px = jax.scipy.special.logsumexp(jnp.log(true_π + 1e-10)[None, :] + log_px_given_c, axis=1)
    
    # Average over test points
    empirical_loglik_per_datapoint = jnp.mean(log_px)
    
    return empirical_loglik_per_datapoint


# Example usage and verification
if __name__ == "__main__":
    from generate_synthetic_data import create_test_parameters, generate_mixture_data
    
    # Test parameters
    C, D, K = 2, 5, 3
    true_π, true_θ = create_test_parameters(C, D, K)
    
    print(f"True parameters:")
    print(f"  π = {true_π}")
    print(f"  θ shape = {true_θ.shape}")
    
    # Compute optimal log-likelihood
    optimal_loglik = compute_optimal_loglik_per_datapoint(true_π, true_θ)
    
    print(f"\nOptimal log-likelihood computation:")
    print(f"  Optimal log-likelihood per datapoint: {optimal_loglik:.4f}")
    
    # Verify empirically
    key = jax.random.PRNGKey(42)
    N_test = 10000  # Large sample for accurate empirical estimate
    X_test, _ = generate_mixture_data(key, N_test, D, K, true_π, true_θ)
    
    empirical_loglik = verify_optimal_loglik_empirically(X_test, true_π, true_θ)
    
    print(f"\nEmpirical verification (N_test = {N_test}):")
    print(f"  Empirical log-likelihood per datapoint: {empirical_loglik:.4f}")
    print(f"  Difference from theoretical: {jnp.abs(empirical_loglik - optimal_loglik):.6f}")
    
    if jnp.abs(empirical_loglik - optimal_loglik) < 0.05:
        print("✅ Theoretical and empirical values match!")
    else:
        print("❌ Mismatch between theoretical and empirical values")