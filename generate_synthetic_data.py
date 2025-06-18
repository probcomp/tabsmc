"""
Fast vectorized synthetic data generation for tabular data clustering.
"""

import jax
import jax.numpy as jnp


def generate_mixture_data(key, N, D, K, true_pi, true_theta):
    """Generate synthetic data from a mixture model using vectorized operations.
    
    Args:
        key: PRNG key
        N: Number of data points
        D: Number of features
        K: Number of categories per feature
        true_pi: True mixing weights (C,)
        true_theta: True emission parameters (C, D, K)
    
    Returns:
        X: One-hot encoded data (N, D, K)
        assignments: True cluster assignments (N,)
    """
    C = true_pi.shape[0]
    
    # Generate cluster assignments
    key, subkey = jax.random.split(key)
    assignments = jax.random.choice(subkey, C, shape=(N,), p=true_pi)
    
    # Generate categories for all data points and features at once
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, N * D).reshape(N, D, 2)
    
    def generate_categories_for_point(n_keys, assignment):
        """Generate categories for all features of one data point."""
        theta_n = true_theta[assignment]  # (D, K)
        
        def sample_category(key_d, probs_d):
            return jax.random.choice(key_d, K, p=probs_d)
        
        categories = jax.vmap(sample_category)(n_keys, theta_n)
        return categories
    
    # Vectorize over all data points
    all_categories = jax.vmap(generate_categories_for_point)(keys, assignments)
    
    # Convert to one-hot encoding
    X = jax.nn.one_hot(all_categories, K)
    
    return X, assignments


def create_test_parameters(C, D, K):
    """Create test parameters for mixture model.
    
    Args:
        C: Number of clusters
        D: Number of features  
        K: Number of categories per feature
        
    Returns:
        true_pi: Mixing weights
        true_theta: Emission parameters
    """
    if C == 2:
        # Two-cluster case
        true_pi = jnp.array([0.6, 0.4])
        
        # Create distinctive emission patterns
        true_theta = jnp.zeros((C, D, K))
        
        # Cluster 0: prefers early categories (0, 1, ...)
        for d in range(D):
            prefs = jnp.ones(K) * 0.1
            preferred_cat = d % K
            prefs = prefs.at[preferred_cat].set(0.7)
            # Normalize
            prefs = prefs / jnp.sum(prefs)
            true_theta = true_theta.at[0, d, :].set(prefs)
        
        # Cluster 1: prefers later categories (K-1, K-2, ...)
        for d in range(D):
            prefs = jnp.ones(K) * 0.1
            preferred_cat = (K - 1 - d) % K
            prefs = prefs.at[preferred_cat].set(0.7)
            # Normalize
            prefs = prefs / jnp.sum(prefs)
            true_theta = true_theta.at[1, d, :].set(prefs)
            
    elif C == 3:
        # Three-cluster case
        true_pi = jnp.array([0.4, 0.35, 0.25])
        
        true_theta = jnp.zeros((C, D, K))
        
        # Cluster 0: prefers category 0
        for d in range(D):
            prefs = jnp.array([0.7, 0.2, 0.1][:K])
            if K > 3:
                prefs = jnp.concatenate([prefs, jnp.ones(K-3) * 0.05])
            prefs = prefs / jnp.sum(prefs)
            true_theta = true_theta.at[0, d, :].set(prefs)
        
        # Cluster 1: prefers middle category
        for d in range(D):
            prefs = jnp.array([0.1, 0.7, 0.2][:K])
            if K > 3:
                prefs = jnp.concatenate([prefs, jnp.ones(K-3) * 0.05])
            prefs = prefs / jnp.sum(prefs)
            true_theta = true_theta.at[1, d, :].set(prefs)
            
        # Cluster 2: prefers last category
        for d in range(D):
            if K == 2:
                prefs = jnp.array([0.2, 0.8])
            else:
                prefs = jnp.array([0.1, 0.2, 0.7][:K])
                if K > 3:
                    prefs = jnp.concatenate([prefs, jnp.ones(K-3) * 0.05])
            prefs = prefs / jnp.sum(prefs)
            true_theta = true_theta.at[2, d, :].set(prefs)
    
    else:
        # General case: uniform mixing, random emission patterns
        true_pi = jnp.ones(C) / C
        
        # Create random but distinctive patterns
        key = jax.random.PRNGKey(42)
        true_theta = jax.random.dirichlet(key, jnp.ones(K), shape=(C, D))
    
    return true_pi, true_theta


# Example usage and test
if __name__ == "__main__":
    key = jax.random.PRNGKey(123)
    N, D, K, C = 1000, 5, 3, 2
    
    print(f"Generating {N} data points with {D} features, {K} categories, {C} clusters")
    
    # Create test parameters
    true_pi, true_theta = create_test_parameters(C, D, K)
    
    print(f"True mixing weights: {true_pi}")
    print(f"True theta shape: {true_theta.shape}")
    
    # Generate data
    import time
    start = time.time()
    X, assignments = generate_mixture_data(key, N, D, K, true_pi, true_theta)
    end = time.time()
    
    print(f"Data generation took {end - start:.4f}s")
    print(f"Generated data shape: {X.shape}")
    print(f"Assignments shape: {assignments.shape}")
    
    # Verify cluster proportions
    unique, counts = jnp.unique(assignments, return_counts=True)
    empirical_pi = counts / N
    print(f"Empirical mixing weights: {empirical_pi}")
    print(f"Difference from true: {jnp.abs(empirical_pi - true_pi)}")