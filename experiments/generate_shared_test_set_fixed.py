"""Generate and save a shared test set for all Gibbs steps experiments."""

import jax
import jax.numpy as jnp
import pickle
import os


def create_test_parameters(C, D, K):
    """Create test parameters for mixture model."""
    if C == 2:
        # Two-cluster case
        true_pi = jnp.array([0.6, 0.4])
        
        # Create distinctive emission patterns
        true_theta = jnp.zeros((C, D, K))
        
        # Cluster 0: prefers early categories (0, 1, ...)
        for d in range(D):
            if K == 3:
                true_theta = true_theta.at[0, d, :].set(jnp.array([0.7, 0.2, 0.1]))
            else:
                weights = jnp.exp(-jnp.arange(K) * 0.5)
                true_theta = true_theta.at[0, d, :].set(weights / jnp.sum(weights))
        
        # Cluster 1: prefers later categories (..., K-2, K-1)
        for d in range(D):
            if K == 3:
                true_theta = true_theta.at[1, d, :].set(jnp.array([0.1, 0.2, 0.7]))
            else:
                weights = jnp.exp(-jnp.arange(K)[::-1] * 0.5)
                true_theta = true_theta.at[1, d, :].set(weights / jnp.sum(weights))
                
        return jnp.log(true_pi), jnp.log(true_theta)
    else:
        raise NotImplementedError(f"C={C} not implemented")


def generate_mixture_data(key, N, D, K, true_pi, true_theta):
    """Generate synthetic data from a mixture model."""
    C = true_pi.shape[0]
    
    # Convert from log space
    pi = jnp.exp(true_pi)
    theta = jnp.exp(true_theta)
    
    # Generate cluster assignments
    key, subkey = jax.random.split(key)
    assignments = jax.random.choice(subkey, C, shape=(N,), p=pi)
    
    # Generate categories for all data points and features at once
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, N * D).reshape(N, D, 2)
    
    def generate_categories_for_point(n_keys, assignment):
        """Generate categories for all features of one data point."""
        theta_n = theta[assignment]  # (D, K)
        
        def sample_category(key_d, probs_d):
            return jax.random.choice(key_d, K, p=probs_d)
        
        categories = jax.vmap(sample_category)(n_keys, theta_n)
        return categories
    
    # Vectorize over all data points
    all_categories = jax.vmap(generate_categories_for_point)(keys, assignments)
    
    # Convert to one-hot encoding
    X = jax.nn.one_hot(all_categories, K)
    
    return X, assignments


def generate_and_save_test_set():
    """Generate and save test set with fixed parameters."""
    # Fixed parameters
    N_test = 200
    D, K, C = 5, 3, 2
    
    # Use a fixed seed for reproducibility
    key = jax.random.PRNGKey(12345)  # Fixed seed for test set
    
    # Create true parameters
    true_π, true_θ = create_test_parameters(C, D, K)
    
    # Generate test data
    X_test, Z_test = generate_mixture_data(key, N_test, D, K, true_π, true_θ)
    
    # Create data directory if needed
    os.makedirs('data', exist_ok=True)
    
    # Save test data and parameters
    test_data = {
        'X_test': X_test,
        'Z_test': Z_test,
        'true_π': true_π,
        'true_θ': true_θ,
        'N_test': N_test,
        'D': D,
        'K': K,
        'C': C
    }
    
    output_file = 'data/shared_test_set_gibbs.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"Generated and saved test set to {output_file}")
    print(f"  N_test={N_test}, D={D}, K={K}, C={C}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  true_π shape: {true_π.shape}")
    print(f"  true_θ shape: {true_θ.shape}")


if __name__ == "__main__":
    generate_and_save_test_set()