"""
Query functions for sampling from SMC models.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def sample_row(
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)
    pi: Array,     # Shape: (n_clusters,)
    key: PRNGKeyArray
) -> Array:
    """Sample a single row from the SMC model.
    
    Args:
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        key: JAX random key
        
    Returns:
        Sampled row of shape (n_features,) with integer category indices
    """
    _, n_features, _ = theta.shape
    
    # Sample cluster assignment using log-space mixture weights
    key_cluster, key_features = jax.random.split(key)
    cluster_idx = jax.random.categorical(key_cluster, pi)
    
    # Get log emission parameters for selected cluster
    cluster_log_theta = theta[cluster_idx]  # (n_features, n_categories)
    
    # Sample features using log-space parameters
    feature_keys = jax.random.split(key_features, n_features)
    
    def sample_feature(feature_key, feature_log_theta):
        """Sample a single feature using log-space parameters."""
        return jax.random.categorical(feature_key, feature_log_theta)
    
    # Vectorized sampling across features
    features = jax.vmap(sample_feature)(feature_keys, cluster_log_theta)
    
    return features


def sample_batch(
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)  
    pi: Array,     # Shape: (n_clusters,)
    key: PRNGKeyArray,
    batch_size: int
) -> Array:
    """Sample a batch of rows from the SMC model.
    
    Args:
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        key: JAX random key
        batch_size: Number of rows to sample
        
    Returns:
        Sampled batch of shape (batch_size, n_features) with integer category indices
    """
    # Generate keys for each sample
    sample_keys = jax.random.split(key, batch_size)
    
    # Vectorized sampling across batch
    batch = jax.vmap(lambda k: sample_row(theta, pi, k))(sample_keys)
    
    return batch


def logprob(
    row: Array,    # Shape: (n_features,) with integer category indices
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)
    pi: Array      # Shape: (n_clusters,)
) -> float:
    """Compute log probability of a row under the model.
    
    Args:
        row: Data row of shape (n_features,) with integer category indices
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        
    Returns:
        Log probability of the row under the mixture model
    """
    _, n_features, _ = theta.shape
    
    # Extract log probabilities for observed categories
    # theta[:, d, row[d]] gives log P(x_d | cluster) for all clusters
    feature_indices = jnp.arange(n_features)
    cluster_log_likelihoods = theta[:, feature_indices, row].sum(axis=1)
    
    # Add log mixture weights to get log P(x, c) = log P(c) + log P(x | c)
    log_joint = pi + cluster_log_likelihoods
    
    # Marginalize over clusters: log P(x) = log sum_c exp(log P(x, c))
    log_prob = jax.scipy.special.logsumexp(log_joint)
    
    return log_prob


def condition_on_row(
    row: Array,    # Shape: (n_features,) with integer category indices or -1 for missing
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)
    pi: Array      # Shape: (n_clusters,)
) -> Array:
    """Compute posterior distribution over clusters given (partial) observations.
    
    Args:
        row: Data row of shape (n_features,) with integer category indices.
             Use -1 to indicate missing values that should be marginalized out.
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        
    Returns:
        Log-space posterior distribution over clusters of shape (n_clusters,)
    """
    n_clusters, n_features, _ = theta.shape
    
    # Create mask for observed features (not -1)
    observed_mask = row != -1
    
    # For each feature, extract log probabilities of observed categories
    # Use where to handle missing values (set their contribution to 0)
    feature_indices = jnp.arange(n_features)
    
    # Extract log probabilities for observed categories
    # theta[:, d, row[d]] gives log P(x_d | cluster) for all clusters
    # When row[d] == -1, we use 0 (first category) as placeholder
    safe_row = jnp.where(observed_mask, row, 0)
    cluster_log_likelihoods = theta[:, feature_indices, safe_row]
    
    # Mask out contributions from missing features (set to 0 in log space)
    cluster_log_likelihoods = jnp.where(
        observed_mask[None, :],  # Broadcast mask
        cluster_log_likelihoods,
        0.0  # No contribution in log space
    )
    
    # Sum across features to get total log likelihood per cluster
    cluster_log_likelihoods = cluster_log_likelihoods.sum(axis=1)
    
    # Compute log posterior: log P(c | x_obs) = log P(c) + log P(x_obs | c) - log P(x_obs)
    log_posterior_unnormalized = pi + cluster_log_likelihoods
    
    # Normalize to get proper posterior distribution
    log_posterior = log_posterior_unnormalized - jax.scipy.special.logsumexp(log_posterior_unnormalized)
    
    return log_posterior


def sample_conditional(
    observed_row: Array,  # Shape: (n_features,) with -1 for missing values
    theta: Array,         # Shape: (n_clusters, n_features, n_categories)
    pi: Array,           # Shape: (n_clusters,)
    key: PRNGKeyArray
) -> Array:
    """Sample missing features conditioned on observed features.
    
    Args:
        observed_row: Data row of shape (n_features,) with integer category indices
                     for observed features and -1 for missing features
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        key: JAX random key
        
    Returns:
        Complete row of shape (n_features,) with all features sampled
    """
    _, n_features, _ = theta.shape
    
    # Compute posterior over clusters given observed features
    log_posterior = condition_on_row(observed_row, theta, pi)
    
    # Sample cluster assignment from posterior
    key_cluster, key_features = jax.random.split(key)
    cluster_idx = jax.random.categorical(key_cluster, log_posterior)
    
    # Get emission parameters for selected cluster
    cluster_log_theta = theta[cluster_idx]  # (n_features, n_categories)
    
    # Create mask for missing features
    missing_mask = observed_row == -1
    
    # Sample missing features
    feature_keys = jax.random.split(key_features, n_features)
    
    def sample_or_keep(feature_idx, feature_key):
        """Sample feature if missing, otherwise keep observed value."""
        is_missing = missing_mask[feature_idx]
        sampled_value = jax.random.categorical(feature_key, cluster_log_theta[feature_idx])
        observed_value = observed_row[feature_idx]
        return jnp.where(is_missing, sampled_value, observed_value)
    
    # Apply sampling to all features
    complete_row = jax.vmap(sample_or_keep)(jnp.arange(n_features), feature_keys)
    
    return complete_row


def argmax_conditional(
    observed_row: Array,  # Shape: (n_features,) with -1 for missing values
    theta: Array,         # Shape: (n_clusters, n_features, n_categories)
    pi: Array,           # Shape: (n_clusters,)
) -> Array:
    """Predict missing features using argmax from marginal distribution conditioned on observed features.
    
    This function computes marginal probabilities for missing features by marginalizing over
    the posterior distribution of clusters given observed features.
    
    Args:
        observed_row: Data row of shape (n_features,) with integer category indices
                     for observed features and -1 for missing features
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        
    Returns:
        Complete row of shape (n_features,) with all features filled using argmax
    """
    n_clusters, n_features, n_categories = theta.shape
    
    # Compute posterior over clusters given observed features
    log_posterior = condition_on_row(observed_row, theta, pi)
    posterior_probs = jnp.exp(log_posterior)  # Convert to probabilities
    
    # Convert theta to probabilities
    theta_probs = jnp.exp(theta)  # Shape: (n_clusters, n_features, n_categories)
    
    # Create mask for missing features
    missing_mask = observed_row == -1
    
    # Compute marginal probabilities for all features using vectorized operations
    # P(X_j = k | observed) = sum_c P(c | observed) * P(X_j = k | c)
    # Shape: (n_features, n_categories)
    marginal_probs = jnp.einsum('c,cjk->jk', posterior_probs, theta_probs)
    
    # Get argmax for each feature
    argmax_values = jnp.argmax(marginal_probs, axis=1)
    
    # Use argmax for missing features, keep observed values for others
    complete_row = jnp.where(missing_mask, argmax_values, observed_row)
    
    return complete_row


def logprob_with_missing(
    row: Array,    # Shape: (n_features,) with -1 for missing values
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)
    pi: Array      # Shape: (n_clusters,)
) -> float:
    """Compute log probability of observed features, marginalizing over missing ones.
    
    Args:
        row: Data row of shape (n_features,) with integer category indices
             for observed features and -1 for missing features
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        
    Returns:
        Log probability of observed features under the mixture model
    """
    _, n_features, _ = theta.shape
    
    # Create mask for observed features
    observed_mask = row != -1
    
    # For each feature, extract log probabilities of observed categories
    feature_indices = jnp.arange(n_features)
    
    # Extract log probabilities for observed categories
    # Use placeholder for missing values
    safe_row = jnp.where(observed_mask, row, 0)
    cluster_log_likelihoods = theta[:, feature_indices, safe_row]
    
    # Mask out contributions from missing features
    cluster_log_likelihoods = jnp.where(
        observed_mask[None, :],  # Broadcast mask
        cluster_log_likelihoods,
        0.0  # No contribution in log space
    )
    
    # Sum across features to get total log likelihood per cluster
    cluster_log_likelihoods = cluster_log_likelihoods.sum(axis=1)
    
    # Add log mixture weights
    log_joint = pi + cluster_log_likelihoods
    
    # Marginalize over clusters
    log_prob = jax.scipy.special.logsumexp(log_joint)
    
    return log_prob


def argmax_row_from_prior(
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)
    pi: Array,     # Shape: (n_clusters,)
) -> Array:
    """Generate a single row using argmax from the marginal distribution.
    
    This function generates samples by:
    1. Computing marginal probabilities for each feature: P(X_j = k) = sum_c P(c) * P(X_j = k | c)
    2. For each feature, selecting the category with highest marginal probability
    
    Args:
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        
    Returns:
        Row of shape (n_features,) with integer category indices
    """
    # Convert to probabilities
    pi_probs = jnp.exp(pi)  # Shape: (n_clusters,)
    theta_probs = jnp.exp(theta)  # Shape: (n_clusters, n_features, n_categories)
    
    # Compute marginal probabilities for each feature using vectorized operations
    # P(X_j = k) = sum_c P(c) * P(X_j = k | c)
    # Shape: (n_features, n_categories)
    marginal_probs = jnp.einsum('c,cjk->jk', pi_probs, theta_probs)
    
    # For each feature, select category with highest marginal probability
    features = jnp.argmax(marginal_probs, axis=1)
    
    return features


def argmax_batch_from_prior(
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)
    pi: Array,     # Shape: (n_clusters,)
    batch_size: int
) -> Array:
    """Generate a batch of rows using argmax from the prior distribution.
    
    Since this is deterministic, all rows will be identical.
    
    Args:
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        batch_size: Number of identical rows to generate
        
    Returns:
        Batch of shape (batch_size, n_features) with integer category indices
    """
    # Generate one row
    single_row = argmax_row_from_prior(theta, pi)
    
    # Repeat for batch
    batch = jnp.tile(single_row[None, :], (batch_size, 1))
    
    return batch


def mutual_information_features(
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)
    pi: Array,     # Shape: (n_clusters,)
    feature_i: int,
    feature_j: int
) -> float:
    """Compute mutual information between two features in the model.
    
    MI(X_i, X_j) = sum_{x_i, x_j} P(x_i, x_j) log(P(x_i, x_j) / (P(x_i) * P(x_j)))
    
    Args:
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        feature_i: Index of first feature
        feature_j: Index of second feature
        
    Returns:
        Mutual information between the two features (non-negative)
    """
    # For vectorized operations, we can't use Python if statements
    # Use JAX operations to ensure consistent ordering
    f_min = jnp.minimum(feature_i, feature_j)
    f_max = jnp.maximum(feature_i, feature_j)
    feature_i = f_min
    feature_j = f_max
    
    # Get shape info (assuming same number of categories for all features)
    _, _, n_categories = theta.shape
    
    # Convert to probabilities
    pi_probs = jnp.exp(pi)  # Shape: (n_clusters,)
    theta_probs = jnp.exp(theta)  # Shape: (n_clusters, n_features, n_categories)
    
    # Compute marginal probabilities for feature i
    # P(X_i = k) = sum_c P(c) * P(X_i = k | c)
    p_xi = jnp.einsum('c,ck->k', pi_probs, theta_probs[:, feature_i, :])
    
    # Compute marginal probabilities for feature j
    # P(X_j = k) = sum_c P(c) * P(X_j = k | c)
    p_xj = jnp.einsum('c,ck->k', pi_probs, theta_probs[:, feature_j, :])
    
    # Compute joint probabilities P(X_i = k, X_j = l)
    # P(X_i = k, X_j = l) = sum_c P(c) * P(X_i = k | c) * P(X_j = l | c)
    # Shape: (n_categories_i, n_categories_j)
    p_xi_xj = jnp.einsum('c,ck,cl->kl', 
                         pi_probs, 
                         theta_probs[:, feature_i, :], 
                         theta_probs[:, feature_j, :])
    
    # Compute product of marginals P(X_i = k) * P(X_j = l)
    # Shape: (n_categories_i, n_categories_j)
    p_xi_p_xj = jnp.outer(p_xi, p_xj)
    
    # Compute MI: sum_{k,l} P(X_i=k, X_j=l) * log(P(X_i=k, X_j=l) / (P(X_i=k) * P(X_j=l)))
    # Use safe log to handle zeros
    epsilon = 1e-10
    ratio = (p_xi_xj + epsilon) / (p_xi_p_xj + epsilon)
    mi_terms = p_xi_xj * jnp.log(ratio)
    
    # Mask out terms where joint probability is effectively zero
    mi_terms = jnp.where(p_xi_xj > epsilon, mi_terms, 0.0)
    
    # Sum over all categories
    mi = jnp.sum(mi_terms)
    
    # Ensure non-negative (handle numerical errors)
    mi = jnp.maximum(mi, 0.0)
    
    return mi


def mutual_information_feature_cluster(
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)
    pi: Array,     # Shape: (n_clusters,)
    feature_idx: int
) -> float:
    """Compute mutual information between a feature and the cluster assignment.
    
    MI(X_i, C) = sum_{x_i, c} P(x_i, c) log(P(x_i, c) / (P(x_i) * P(c)))
    
    Args:
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        feature_idx: Index of the feature
        
    Returns:
        Mutual information between the feature and cluster assignment
    """
    # Convert to probabilities
    pi_probs = jnp.exp(pi)  # Shape: (n_clusters,)
    theta_probs = jnp.exp(theta)  # Shape: (n_clusters, n_features, n_categories)
    
    # P(c) is just pi_probs
    p_c = pi_probs
    
    # Compute marginal probabilities for feature
    # P(X_i = k) = sum_c P(c) * P(X_i = k | c)
    p_xi = jnp.einsum('c,ck->k', pi_probs, theta_probs[:, feature_idx, :])
    
    # Joint probabilities P(X_i = k, C = c) = P(c) * P(X_i = k | c)
    # Shape: (n_categories, n_clusters)
    p_xi_c = theta_probs[:, feature_idx, :].T * pi_probs[None, :]
    
    # Product of marginals P(X_i = k) * P(C = c)
    # Shape: (n_categories, n_clusters)
    p_xi_p_c = jnp.outer(p_xi, p_c)
    
    # Compute MI
    epsilon = 1e-10
    ratio = (p_xi_c + epsilon) / (p_xi_p_c + epsilon)
    mi_terms = p_xi_c * jnp.log(ratio)
    
    # Mask out terms where joint probability is effectively zero
    mi_terms = jnp.where(p_xi_c > epsilon, mi_terms, 0.0)
    
    # Sum over all categories and clusters
    mi = jnp.sum(mi_terms)
    
    # Ensure non-negative (handle numerical errors)
    mi = jnp.maximum(mi, 0.0)
    
    return mi


def mutual_information_matrix(
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)
    pi: Array      # Shape: (n_clusters,)
) -> Array:
    """Compute pairwise mutual information matrix between all features.
    
    This returns a symmetric matrix where element (i,j) is MI(X_i, X_j).
    The diagonal contains the entropy of each feature.
    
    Args:
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        
    Returns:
        Mutual information matrix of shape (n_features, n_features)
    """
    n_features = theta.shape[1]
    
    # Note: entropies computation removed since we call entropy_feature directly in the vectorized function
    
    # Create function to compute MI for a single pair
    def compute_mi_element(i, j):
        return jax.lax.cond(
            i == j,
            lambda _: entropy_feature(theta, pi, i),
            lambda _: mutual_information_features(theta, pi, i, j),
            None
        )
    
    # Create index arrays
    i_indices, j_indices = jnp.meshgrid(jnp.arange(n_features), jnp.arange(n_features), indexing='ij')
    
    # Vectorized computation
    mi_matrix = jax.vmap(jax.vmap(compute_mi_element))(i_indices, j_indices)
    
    # Force symmetry by averaging with transpose
    mi_matrix = (mi_matrix + mi_matrix.T) / 2
    
    return mi_matrix


def entropy_feature(
    theta: Array,  # Shape: (n_clusters, n_features, n_categories)
    pi: Array,     # Shape: (n_clusters,)
    feature_idx: int
) -> float:
    """Compute entropy of a single feature.
    
    H(X_i) = -sum_{x_i} P(x_i) log P(x_i)
    
    Args:
        theta: Log-space emission parameters of shape (n_clusters, n_features, n_categories)
        pi: Log-space mixture weights of shape (n_clusters,)
        feature_idx: Index of the feature
        
    Returns:
        Entropy of the feature (non-negative)
    """
    # Convert to probabilities
    pi_probs = jnp.exp(pi)
    theta_probs = jnp.exp(theta)
    
    # Compute marginal probabilities for feature
    # P(X_i = k) = sum_c P(c) * P(X_i = k | c)
    p_xi = jnp.einsum('c,ck->k', pi_probs, theta_probs[:, feature_idx, :])
    
    # Compute entropy
    epsilon = 1e-10
    # Only compute entropy for non-zero probabilities
    entropy_terms = jnp.where(p_xi > epsilon, -p_xi * jnp.log(p_xi), 0.0)
    entropy = jnp.sum(entropy_terms)
    
    # Ensure non-negative (handle numerical errors)
    entropy = jnp.maximum(entropy, 0.0)
    
    return entropy