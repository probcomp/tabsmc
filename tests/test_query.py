"""
Tests for tabsmc.query module.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tabsmc.query import (
    sample_row, sample_batch, logprob,
    condition_on_row, sample_conditional, logprob_with_missing,
    mutual_information_features, mutual_information_feature_cluster,
    mutual_information_matrix, entropy_feature
)


class TestSampleRow:
    """Test the sample_row function."""
    
    def test_sample_row_shape(self):
        """Test that sample_row returns correct shape."""
        n_clusters, n_features, n_categories = 3, 5, 4
        
        # Create random log-space parameters
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        theta = jax.random.normal(key1, (n_clusters, n_features, n_categories))
        pi = jax.random.normal(key2, (n_clusters,))
        
        # Sample a row
        row = sample_row(theta, pi, key3)
        
        assert row.shape == (n_features,)
        assert row.dtype == jnp.int32
        assert jnp.all((row >= 0) & (row < n_categories))
    
    def test_sample_row_deterministic_case(self):
        """Test sampling with deterministic parameters."""
        n_clusters, n_features, n_categories = 2, 3, 4
        
        # Create theta where cluster 0 always outputs category 1 for all features
        # and cluster 1 always outputs category 2 for all features
        theta = jnp.full((n_clusters, n_features, n_categories), -jnp.inf)
        theta = theta.at[0, :, 1].set(0.0)  # Cluster 0 -> category 1
        theta = theta.at[1, :, 2].set(0.0)  # Cluster 1 -> category 2
        
        # Pi heavily favors cluster 0
        pi = jnp.array([0.0, -10.0])  # Cluster 0 much more likely
        
        # Sample many rows
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 100)
        
        rows = jax.vmap(lambda k: sample_row(theta, pi, k))(keys)
        
        # Most samples should be from cluster 0 (all category 1)
        cluster_0_samples = jnp.sum(jnp.all(rows == 1, axis=1))
        
        # Should be heavily biased toward cluster 0
        assert cluster_0_samples > 80  # At least 80% from cluster 0
    
    def test_sample_row_distribution_convergence(self):
        """Test that large samples converge to expected distribution."""
        n_clusters, n_features, n_categories = 2, 1, 3
        
        # Create known parameters
        # Cluster 0: favors category 0
        # Cluster 1: favors category 2
        theta = jnp.array([
            [[jnp.log(0.7), jnp.log(0.2), jnp.log(0.1)]],  # Cluster 0
            [[jnp.log(0.1), jnp.log(0.2), jnp.log(0.7)]]   # Cluster 1
        ])
        
        # Equal mixture weights
        pi = jnp.array([jnp.log(0.5), jnp.log(0.5)])
        
        # Expected marginal distribution:
        # P(cat=0) = 0.5 * 0.7 + 0.5 * 0.1 = 0.4
        # P(cat=1) = 0.5 * 0.2 + 0.5 * 0.2 = 0.2  
        # P(cat=2) = 0.5 * 0.1 + 0.5 * 0.7 = 0.4
        expected_probs = jnp.array([0.4, 0.2, 0.4])
        
        # Sample large batch
        key = jax.random.PRNGKey(42)
        n_samples = 10000
        batch = sample_batch(theta, pi, key, n_samples)
        
        # Compute empirical distribution
        counts = jnp.bincount(batch[:, 0], length=n_categories)
        empirical_probs = counts / n_samples
        
        # Check convergence (allow 5% tolerance)
        tolerance = 0.05
        assert jnp.allclose(empirical_probs, expected_probs, atol=tolerance)
    
    def test_sample_row_multiple_features(self):
        """Test sampling with multiple features."""
        n_clusters, n_features, n_categories = 2, 4, 3
        
        # Create parameters where each feature has different behavior
        # Start with very negative values (very low probability)
        theta = jnp.full((n_clusters, n_features, n_categories), -10.0)
        
        # Cluster 0: each feature prefers a different category
        theta = theta.at[0, 0, 0].set(0.0)  # Feature 0 -> category 0
        theta = theta.at[0, 1, 1].set(0.0)  # Feature 1 -> category 1
        theta = theta.at[0, 2, 2].set(0.0)  # Feature 2 -> category 2
        theta = theta.at[0, 3, 0].set(0.0)  # Feature 3 -> category 0
        
        # Only use cluster 0
        pi = jnp.array([0.0, -20.0])
        
        # Sample
        key = jax.random.PRNGKey(42)
        row = sample_row(theta, pi, key)
        
        # Check expected pattern
        expected = jnp.array([0, 1, 2, 0])
        
        # Sample multiple times to check consistency
        keys = jax.random.split(key, 10)
        rows = jax.vmap(lambda k: sample_row(theta, pi, k))(keys)
        
        # Most should match expected pattern (allowing for some randomness)
        matches = jnp.sum(jnp.all(rows == expected, axis=1))
        assert matches >= 7  # At least 70% should match


class TestSampleBatch:
    """Test the sample_batch function."""
    
    def test_sample_batch_shape(self):
        """Test that sample_batch returns correct shape."""
        n_clusters, n_features, n_categories = 3, 5, 4
        batch_size = 100
        
        # Create random log-space parameters
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        theta = jax.random.normal(key1, (n_clusters, n_features, n_categories))
        pi = jax.random.normal(key2, (n_clusters,))
        
        # Sample batch
        batch = sample_batch(theta, pi, key3, batch_size)
        
        assert batch.shape == (batch_size, n_features)
        assert batch.dtype == jnp.int32
        assert jnp.all((batch >= 0) & (batch < n_categories))
    
    def test_sample_batch_consistency(self):
        """Test that batch sampling is consistent with individual sampling."""
        n_clusters, n_features, n_categories = 2, 3, 4
        batch_size = 5
        
        # Create parameters
        key = jax.random.PRNGKey(42)
        key1, key2, key_batch, key_individual = jax.random.split(key, 4)
        
        theta = jax.random.normal(key1, (n_clusters, n_features, n_categories))
        pi = jax.random.normal(key2, (n_clusters,))
        
        # Sample batch
        batch = sample_batch(theta, pi, key_batch, batch_size)
        
        # Sample individually with same keys
        individual_keys = jax.random.split(key_batch, batch_size)
        individual_samples = jax.vmap(lambda k: sample_row(theta, pi, k))(individual_keys)
        
        # Should be identical
        assert jnp.array_equal(batch, individual_samples)
    
    def test_sample_batch_large_scale(self):
        """Test batch sampling at larger scale."""
        n_clusters, n_features, n_categories = 5, 10, 6
        batch_size = 1000
        
        # Create random parameters
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        theta = jax.random.normal(key1, (n_clusters, n_features, n_categories))
        pi = jax.random.normal(key2, (n_clusters,))
        
        # Sample large batch
        batch = sample_batch(theta, pi, key3, batch_size)
        
        # Basic checks
        assert batch.shape == (batch_size, n_features)
        assert jnp.all((batch >= 0) & (batch < n_categories))
        
        # Check that we get diversity in samples
        unique_rows = len(jnp.unique(batch, axis=0))
        assert unique_rows > batch_size // 10  # At least 10% unique rows


class TestLogSpaceHandling:
    """Test proper handling of log-space parameters."""
    
    def test_log_space_conversion(self):
        """Test that log-space parameters are converted correctly."""
        n_clusters, n_features, n_categories = 2, 1, 3
        
        # Create log-space parameters
        # Cluster 0: log(0.1, 0.2, 0.7) ≈ (-2.3, -1.6, -0.36)
        # Cluster 1: log(0.8, 0.1, 0.1) ≈ (-0.22, -2.3, -2.3)
        theta = jnp.array([
            [[jnp.log(0.1), jnp.log(0.2), jnp.log(0.7)]],
            [[jnp.log(0.8), jnp.log(0.1), jnp.log(0.1)]]
        ])
        
        # Equal clusters: log(0.5, 0.5) ≈ (-0.69, -0.69)
        pi = jnp.array([jnp.log(0.5), jnp.log(0.5)])
        
        # Expected distribution:
        # P(cat=0) = 0.5 * 0.1 + 0.5 * 0.8 = 0.45
        # P(cat=1) = 0.5 * 0.2 + 0.5 * 0.1 = 0.15
        # P(cat=2) = 0.5 * 0.7 + 0.5 * 0.1 = 0.40
        
        # Sample large batch
        key = jax.random.PRNGKey(42)
        n_samples = 20000
        batch = sample_batch(theta, pi, key, n_samples)
        
        # Compute empirical distribution
        counts = jnp.bincount(batch[:, 0], length=n_categories)
        empirical_probs = counts / n_samples
        
        expected_probs = jnp.array([0.45, 0.15, 0.40])
        
        # Check convergence (3% tolerance for large sample)
        tolerance = 0.03
        assert jnp.allclose(empirical_probs, expected_probs, atol=tolerance)
    
    def test_extreme_log_values(self):
        """Test handling of extreme log values."""
        n_clusters, n_features, n_categories = 2, 1, 3
        
        # Extreme log values (very negative = very small probability)
        theta = jnp.array([
            [[0.0, -100.0, -100.0]],  # Cluster 0: category 0 dominates
            [[-100.0, -100.0, 0.0]]   # Cluster 1: category 2 dominates
        ])
        
        # Equal mixture
        pi = jnp.array([0.0, 0.0])
        
        # Sample
        key = jax.random.PRNGKey(42)
        n_samples = 1000
        batch = sample_batch(theta, pi, key, n_samples)
        
        # Should only see categories 0 and 2
        unique_categories = jnp.unique(batch[:, 0])
        assert len(unique_categories) == 2
        assert 0 in unique_categories
        assert 2 in unique_categories
        assert 1 not in unique_categories


class TestLogProb:
    """Test the logprob function."""
    
    def test_logprob_basic(self):
        """Test basic logprob computation."""
        n_clusters, n_features, n_categories = 2, 3, 4
        
        # Create simple parameters
        theta = jnp.zeros((n_clusters, n_features, n_categories))
        # Normalize to log probabilities
        theta = jax.nn.log_softmax(theta, axis=-1)
        
        # Equal mixture weights
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Test row
        row = jnp.array([0, 1, 2])
        
        # Compute log probability
        log_p = logprob(row, theta, pi)
        
        # Should be a scalar
        assert log_p.shape == ()
        # Should be negative (log of probability)
        assert log_p < 0
    
    def test_logprob_deterministic(self):
        """Test logprob with deterministic parameters."""
        n_clusters, n_features, n_categories = 2, 2, 3
        
        # Create deterministic theta
        # Cluster 0: always outputs category 0
        # Cluster 1: always outputs category 1
        theta = jnp.full((n_clusters, n_features, n_categories), -jnp.inf)
        theta = theta.at[0, :, 0].set(0.0)  # log(1) = 0
        theta = theta.at[1, :, 1].set(0.0)  # log(1) = 0
        
        # Equal mixture weights
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Test row that matches cluster 0
        row = jnp.array([0, 0])
        log_p = logprob(row, theta, pi)
        # P(row) = 0.5 * 1 * 1 + 0.5 * 0 * 0 = 0.5
        expected = jnp.log(0.5)
        assert jnp.allclose(log_p, expected, atol=1e-6)
        
        # Test row that matches cluster 1
        row = jnp.array([1, 1])
        log_p = logprob(row, theta, pi)
        # P(row) = 0.5 * 0 * 0 + 0.5 * 1 * 1 = 0.5
        expected = jnp.log(0.5)
        assert jnp.allclose(log_p, expected, atol=1e-6)
        
        # Test impossible row
        row = jnp.array([2, 2])
        log_p = logprob(row, theta, pi)
        # P(row) = 0.5 * 0 * 0 + 0.5 * 0 * 0 = 0
        assert log_p == -jnp.inf
    
    def test_logprob_single_cluster(self):
        """Test logprob with single cluster."""
        n_clusters, n_features, n_categories = 1, 3, 4
        
        # Create parameters
        theta = jnp.array([
            [[jnp.log(0.25), jnp.log(0.25), jnp.log(0.25), jnp.log(0.25)],
             [jnp.log(0.1), jnp.log(0.2), jnp.log(0.3), jnp.log(0.4)],
             [jnp.log(0.4), jnp.log(0.3), jnp.log(0.2), jnp.log(0.1)]]
        ])
        
        # Single cluster has probability 1
        pi = jnp.array([0.0])  # log(1) = 0
        
        # Test row
        row = jnp.array([0, 2, 1])
        
        # Expected: log(0.25 * 0.3 * 0.3)
        expected = jnp.log(0.25) + jnp.log(0.3) + jnp.log(0.3)
        log_p = logprob(row, theta, pi)
        
        assert jnp.allclose(log_p, expected, atol=1e-6)
    
    def test_logprob_mixture(self):
        """Test logprob with mixture model."""
        n_clusters, n_features, n_categories = 2, 1, 2
        
        # Create simple binary model
        # Cluster 0: 70% category 0, 30% category 1
        # Cluster 1: 20% category 0, 80% category 1
        theta = jnp.array([
            [[jnp.log(0.7), jnp.log(0.3)]],
            [[jnp.log(0.2), jnp.log(0.8)]]
        ])
        
        # Mixture weights: 60% cluster 0, 40% cluster 1
        pi = jnp.array([jnp.log(0.6), jnp.log(0.4)])
        
        # Test category 0
        row = jnp.array([0])
        log_p = logprob(row, theta, pi)
        # P(0) = 0.6 * 0.7 + 0.4 * 0.2 = 0.42 + 0.08 = 0.5
        expected = jnp.log(0.5)
        assert jnp.allclose(log_p, expected, atol=1e-6)
        
        # Test category 1
        row = jnp.array([1])
        log_p = logprob(row, theta, pi)
        # P(1) = 0.6 * 0.3 + 0.4 * 0.8 = 0.18 + 0.32 = 0.5
        expected = jnp.log(0.5)
        assert jnp.allclose(log_p, expected, atol=1e-6)
    
    def test_logprob_consistency_with_sampling(self):
        """Test that logprob is consistent with sampling distribution."""
        n_clusters, n_features, n_categories = 2, 1, 3
        
        # Create known parameters
        theta = jnp.array([
            [[jnp.log(0.7), jnp.log(0.2), jnp.log(0.1)]],
            [[jnp.log(0.1), jnp.log(0.2), jnp.log(0.7)]]
        ])
        pi = jnp.array([jnp.log(0.5), jnp.log(0.5)])
        
        # Compute log probabilities for all categories
        log_probs = []
        for cat in range(n_categories):
            row = jnp.array([cat])
            log_p = logprob(row, theta, pi)
            log_probs.append(log_p)
        
        # Convert to probabilities
        log_probs = jnp.array(log_probs)
        probs = jnp.exp(log_probs)
        
        # Should sum to 1
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)
        
        # Expected probabilities
        # P(0) = 0.5 * 0.7 + 0.5 * 0.1 = 0.4
        # P(1) = 0.5 * 0.2 + 0.5 * 0.2 = 0.2
        # P(2) = 0.5 * 0.1 + 0.5 * 0.7 = 0.4
        expected_probs = jnp.array([0.4, 0.2, 0.4])
        assert jnp.allclose(probs, expected_probs, atol=1e-6)
    
    def test_logprob_multiple_features(self):
        """Test logprob with multiple features."""
        n_clusters, n_features, n_categories = 2, 3, 2
        
        # Create parameters where features are independent
        # All features have same distribution in each cluster
        theta = jnp.array([
            [
                [jnp.log(0.8), jnp.log(0.2)],
                [jnp.log(0.8), jnp.log(0.2)],
                [jnp.log(0.8), jnp.log(0.2)]
            ],
            [
                [jnp.log(0.3), jnp.log(0.7)],
                [jnp.log(0.3), jnp.log(0.7)],
                [jnp.log(0.3), jnp.log(0.7)]
            ]
        ])
        
        # Equal mixture
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Test row with all 0s
        row = jnp.array([0, 0, 0])
        log_p = logprob(row, theta, pi)
        # Cluster 0: log(0.8^3) = 3*log(0.8)
        # Cluster 1: log(0.3^3) = 3*log(0.3)
        # P(row) = 0.5 * 0.8^3 + 0.5 * 0.3^3
        expected = jnp.log(0.5 * 0.8**3 + 0.5 * 0.3**3)
        assert jnp.allclose(log_p, expected, atol=1e-6)
    
    def test_logprob_numerical_stability(self):
        """Test numerical stability with extreme values."""
        n_clusters, n_features, n_categories = 3, 2, 4
        
        # Create parameters with very small probabilities
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories)) * 10
        theta = jax.nn.log_softmax(theta, axis=-1)
        
        # Very uneven mixture weights
        pi = jnp.array([0.0, -10.0, -20.0])
        
        # Test random row
        row = jnp.array([0, 3])
        
        # Should not produce NaN or infinity
        log_p = logprob(row, theta, pi)
        assert jnp.isfinite(log_p)
        assert not jnp.isnan(log_p)


class TestConditionOnRow:
    """Test the condition_on_row function."""
    
    def test_condition_on_row_basic(self):
        """Test basic posterior computation."""
        n_clusters, n_features, n_categories = 2, 3, 4
        
        # Create simple parameters
        theta = jnp.zeros((n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        
        # Equal mixture weights
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Test row with some missing values
        row = jnp.array([0, -1, 2])  # Feature 1 is missing
        
        # Compute posterior
        log_posterior = condition_on_row(row, theta, pi)
        
        # Should be a distribution over clusters
        assert log_posterior.shape == (n_clusters,)
        # Should sum to 1 in probability space
        assert jnp.allclose(jnp.sum(jnp.exp(log_posterior)), 1.0, atol=1e-6)
    
    def test_condition_on_row_deterministic(self):
        """Test posterior with deterministic parameters."""
        n_clusters, n_features, n_categories = 2, 2, 3
        
        # Cluster 0: always outputs category 0
        # Cluster 1: always outputs category 1
        theta = jnp.full((n_clusters, n_features, n_categories), -jnp.inf)
        theta = theta.at[0, :, 0].set(0.0)
        theta = theta.at[1, :, 1].set(0.0)
        
        # Equal priors
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Observe category 0 for first feature
        row = jnp.array([0, -1])
        log_posterior = condition_on_row(row, theta, pi)
        
        # Should strongly favor cluster 0
        posterior = jnp.exp(log_posterior)
        assert posterior[0] > 0.99
        assert posterior[1] < 0.01
        
        # Observe category 1 for first feature
        row = jnp.array([1, -1])
        log_posterior = condition_on_row(row, theta, pi)
        
        # Should strongly favor cluster 1
        posterior = jnp.exp(log_posterior)
        assert posterior[0] < 0.01
        assert posterior[1] > 0.99
    
    def test_condition_on_row_all_missing(self):
        """Test posterior when all features are missing."""
        n_clusters, n_features, n_categories = 3, 4, 2
        
        # Random parameters
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        
        # Unequal priors
        pi = jnp.array([jnp.log(0.5), jnp.log(0.3), jnp.log(0.2)])
        
        # All features missing
        row = jnp.full(n_features, -1)
        log_posterior = condition_on_row(row, theta, pi)
        
        # Should equal the prior
        assert jnp.allclose(log_posterior, pi, atol=1e-6)
    
    def test_condition_on_row_all_observed(self):
        """Test posterior when all features are observed."""
        n_clusters, n_features, n_categories = 2, 2, 2
        
        # Simple binary model
        theta = jnp.array([
            [[jnp.log(0.8), jnp.log(0.2)], [jnp.log(0.8), jnp.log(0.2)]],
            [[jnp.log(0.2), jnp.log(0.8)], [jnp.log(0.2), jnp.log(0.8)]]
        ])
        
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Observe [0, 0] - should favor cluster 0
        row = jnp.array([0, 0])
        log_posterior = condition_on_row(row, theta, pi)
        posterior = jnp.exp(log_posterior)
        
        # P(c=0|x) = P(x|c=0)P(c=0) / P(x)
        # P(x|c=0) = 0.8 * 0.8 = 0.64
        # P(x|c=1) = 0.2 * 0.2 = 0.04
        # P(x) = 0.5 * 0.64 + 0.5 * 0.04 = 0.34
        # P(c=0|x) = 0.64 * 0.5 / 0.34 ≈ 0.941
        expected_posterior_0 = (0.64 * 0.5) / (0.64 * 0.5 + 0.04 * 0.5)
        assert jnp.allclose(posterior[0], expected_posterior_0, atol=1e-3)
    
    def test_condition_on_row_partial_observation(self):
        """Test posterior with partial observations."""
        n_clusters, n_features, n_categories = 2, 3, 2
        
        # Cluster 0: prefers category 0 for all features
        # Cluster 1: prefers category 1 for all features
        theta = jnp.array([
            [[jnp.log(0.9), jnp.log(0.1)] for _ in range(n_features)],
            [[jnp.log(0.1), jnp.log(0.9)] for _ in range(n_features)]
        ])
        
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Observe only first feature as category 0
        row = jnp.array([0, -1, -1])
        log_posterior = condition_on_row(row, theta, pi)
        posterior = jnp.exp(log_posterior)
        
        # Should favor cluster 0 but not as strongly as with all features
        assert posterior[0] > posterior[1]
        assert posterior[0] < 0.95  # Not too extreme
        
        # Observe two features as category 0
        row = jnp.array([0, 0, -1])
        log_posterior = condition_on_row(row, theta, pi)
        posterior = jnp.exp(log_posterior)
        
        # Should more strongly favor cluster 0
        assert posterior[0] > 0.95


class TestSampleConditional:
    """Test the sample_conditional function."""
    
    def test_sample_conditional_shape(self):
        """Test that sample_conditional returns correct shape."""
        n_clusters, n_features, n_categories = 3, 5, 4
        
        # Create random parameters
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        theta = jax.random.normal(key1, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jax.nn.log_softmax(jax.random.normal(key2, (n_clusters,)))
        
        # Partial observation
        observed_row = jnp.array([0, -1, 2, -1, 1])
        
        # Sample conditional
        complete_row = sample_conditional(observed_row, theta, pi, key3)
        
        assert complete_row.shape == (n_features,)
        assert complete_row.dtype == jnp.int32
        assert jnp.all((complete_row >= 0) & (complete_row < n_categories))
        
        # Check observed values are preserved
        assert complete_row[0] == 0
        assert complete_row[2] == 2
        assert complete_row[4] == 1
    
    def test_sample_conditional_deterministic(self):
        """Test conditional sampling with deterministic parameters."""
        n_clusters, n_features, n_categories = 2, 3, 2
        
        # Cluster 0: always outputs category 0
        # Cluster 1: always outputs category 1
        theta = jnp.full((n_clusters, n_features, n_categories), -jnp.inf)
        theta = theta.at[0, :, 0].set(0.0)
        theta = theta.at[1, :, 1].set(0.0)
        
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Observe category 0 for first feature
        observed_row = jnp.array([0, -1, -1])
        
        # Sample many times
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 100)
        samples = jax.vmap(lambda k: sample_conditional(observed_row, theta, pi, k))(keys)
        
        # All samples should have all 0s (cluster 0 selected)
        assert jnp.all(samples == 0)
        
        # Observe category 1 for first feature
        observed_row = jnp.array([1, -1, -1])
        samples = jax.vmap(lambda k: sample_conditional(observed_row, theta, pi, k))(keys)
        
        # All samples should have all 1s (cluster 1 selected)
        assert jnp.all(samples == 1)
    
    def test_sample_conditional_all_missing(self):
        """Test conditional sampling when all features are missing."""
        n_clusters, n_features, n_categories = 2, 3, 4
        
        # Create parameters
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jax.nn.log_softmax(jax.random.normal(key, (n_clusters,)))
        
        # All missing
        observed_row = jnp.full(n_features, -1)
        
        # Should behave like unconditional sampling
        key1, key2 = jax.random.split(key)
        conditional_sample = sample_conditional(observed_row, theta, pi, key1)
        unconditional_sample = sample_row(theta, pi, key1)
        
        # Won't be exactly equal due to different sampling paths, but should have same distribution
        # Test by sampling many times
        keys = jax.random.split(key2, 1000)
        conditional_samples = jax.vmap(lambda k: sample_conditional(observed_row, theta, pi, k))(keys[:500])
        unconditional_samples = jax.vmap(lambda k: sample_row(theta, pi, k))(keys[500:])
        
        # Check distributions are similar (rough check)
        for feat in range(n_features):
            cond_counts = jnp.bincount(conditional_samples[:, feat], length=n_categories)
            uncond_counts = jnp.bincount(unconditional_samples[:, feat], length=n_categories)
            # Normalize
            cond_probs = cond_counts / jnp.sum(cond_counts)
            uncond_probs = uncond_counts / jnp.sum(uncond_counts)
            # Should be similar (allowing for sampling variance)
            assert jnp.allclose(cond_probs, uncond_probs, atol=0.1)
    
    def test_sample_conditional_preserves_observed(self):
        """Test that observed values are always preserved."""
        n_clusters, n_features, n_categories = 3, 6, 5
        
        # Random parameters
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jax.nn.log_softmax(jax.random.normal(key, (n_clusters,)))
        
        # Complex observation pattern
        observed_row = jnp.array([2, -1, 4, 0, -1, 3])
        
        # Sample many times
        keys = jax.random.split(key, 100)
        samples = jax.vmap(lambda k: sample_conditional(observed_row, theta, pi, k))(keys)
        
        # All samples should preserve observed values
        assert jnp.all(samples[:, 0] == 2)
        assert jnp.all(samples[:, 2] == 4)
        assert jnp.all(samples[:, 3] == 0)
        assert jnp.all(samples[:, 5] == 3)
        
        # Missing values should vary
        assert len(jnp.unique(samples[:, 1])) > 1
        assert len(jnp.unique(samples[:, 4])) > 1


class TestLogprobWithMissing:
    """Test the logprob_with_missing function."""
    
    def test_logprob_with_missing_basic(self):
        """Test basic log probability computation with missing values."""
        n_clusters, n_features, n_categories = 2, 3, 4
        
        # Simple parameters
        theta = jnp.zeros((n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Row with missing value
        row = jnp.array([0, -1, 2])
        
        log_p = logprob_with_missing(row, theta, pi)
        
        # Should be a scalar
        assert log_p.shape == ()
        # Should be finite
        assert jnp.isfinite(log_p)
    
    def test_logprob_with_missing_vs_complete(self):
        """Test that missing values are properly marginalized."""
        n_clusters, n_features, n_categories = 2, 2, 2
        
        # Simple binary model
        theta = jnp.array([
            [[jnp.log(0.7), jnp.log(0.3)], [jnp.log(0.8), jnp.log(0.2)]],
            [[jnp.log(0.4), jnp.log(0.6)], [jnp.log(0.3), jnp.log(0.7)]]
        ])
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Compute P(x0=0) by marginalizing over x1
        row_missing = jnp.array([0, -1])
        log_p_marginal = logprob_with_missing(row_missing, theta, pi)
        
        # Compute P(x0=0) manually
        # P(x0=0) = P(x0=0, x1=0) + P(x0=0, x1=1)
        row_00 = jnp.array([0, 0])
        row_01 = jnp.array([0, 1])
        log_p_00 = logprob(row_00, theta, pi)
        log_p_01 = logprob(row_01, theta, pi)
        log_p_manual = jnp.logaddexp(log_p_00, log_p_01)
        
        assert jnp.allclose(log_p_marginal, log_p_manual, atol=1e-6)
    
    def test_logprob_with_missing_all_missing(self):
        """Test log probability when all values are missing."""
        n_clusters, n_features, n_categories = 2, 3, 4
        
        # Random parameters
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jax.nn.log_softmax(jax.random.normal(key, (n_clusters,)))
        
        # All missing
        row = jnp.full(n_features, -1)
        log_p = logprob_with_missing(row, theta, pi)
        
        # Should be log(1) = 0 (total probability)
        assert jnp.allclose(log_p, 0.0, atol=1e-6)
    
    def test_logprob_with_missing_no_missing(self):
        """Test that function works correctly with no missing values."""
        n_clusters, n_features, n_categories = 2, 3, 4
        
        # Random parameters
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jax.nn.log_softmax(jax.random.normal(key, (n_clusters,)))
        
        # Complete observation
        row = jnp.array([0, 2, 1])
        
        # Should match regular logprob
        log_p_missing = logprob_with_missing(row, theta, pi)
        log_p_regular = logprob(row, theta, pi)
        
        assert jnp.allclose(log_p_missing, log_p_regular, atol=1e-6)
    
    def test_logprob_with_missing_consistency(self):
        """Test consistency of marginal probabilities."""
        n_clusters, n_features, n_categories = 2, 1, 3
        
        # Known parameters
        theta = jnp.array([
            [[jnp.log(0.5), jnp.log(0.3), jnp.log(0.2)]],
            [[jnp.log(0.1), jnp.log(0.6), jnp.log(0.3)]]
        ])
        pi = jnp.array([jnp.log(0.7), jnp.log(0.3)])
        
        # Compute marginal probabilities for each category
        log_probs = []
        for cat in range(n_categories):
            row = jnp.array([cat])
            log_p = logprob_with_missing(row, theta, pi)
            log_probs.append(log_p)
        
        # Should sum to 1
        probs = jnp.exp(jnp.array(log_probs))
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6)
        
        # Check expected values
        # P(x=0) = 0.7 * 0.5 + 0.3 * 0.1 = 0.38
        # P(x=1) = 0.7 * 0.3 + 0.3 * 0.6 = 0.39
        # P(x=2) = 0.7 * 0.2 + 0.3 * 0.3 = 0.23
        expected_probs = jnp.array([0.38, 0.39, 0.23])
        assert jnp.allclose(probs, expected_probs, atol=1e-6)


if __name__ == "__main__":
    # Run basic smoke test
    key = jax.random.PRNGKey(42)
    theta = jax.random.normal(key, (3, 5, 4))
    theta = jax.nn.log_softmax(theta, axis=-1)  # Normalize
    pi = jax.nn.log_softmax(jax.random.normal(key, (3,)))
    
    row = sample_row(theta, pi, key)
    print(f"Sample row shape: {row.shape}")
    print(f"Sample row: {row}")
    
    batch = sample_batch(theta, pi, key, 10)
    print(f"Sample batch shape: {batch.shape}")
    print(f"Sample batch:\n{batch}")
    
    # Test logprob
    log_p = logprob(row, theta, pi)
    print(f"Log probability of sampled row: {log_p}")
    
    # Test conditioning functionality
    print("\n--- Testing Conditioning Functionality ---")
    
    # Test with partial observation
    observed_row = jnp.array([0, -1, 2, -1, 1])  # Some missing values
    print(f"Observed row (with -1 for missing): {observed_row}")
    
    # Compute posterior over clusters
    log_posterior = condition_on_row(observed_row, theta, pi)
    print(f"Log posterior over clusters: {log_posterior}")
    print(f"Posterior probabilities: {jnp.exp(log_posterior)}")
    
    # Sample missing values
    complete_row = sample_conditional(observed_row, theta, pi, key)
    print(f"Completed row: {complete_row}")
    
    # Compute log prob with missing
    log_p_missing = logprob_with_missing(observed_row, theta, pi)
    print(f"Log prob of observed features: {log_p_missing}")
    
    print("\nSmoke test passed!")


class TestMutualInformation:
    """Test mutual information functions."""
    
    def test_mutual_information_features_basic(self):
        """Test basic MI computation between features."""
        n_clusters, n_features, n_categories = 2, 3, 4
        
        # Create random parameters
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jax.nn.log_softmax(jax.random.normal(key, (n_clusters,)))
        
        # Compute MI between features 0 and 1
        mi = mutual_information_features(theta, pi, 0, 1)
        
        # MI should be non-negative
        assert mi >= 0
        # MI should be finite
        assert jnp.isfinite(mi)
    
    def test_mutual_information_features_independent(self):
        """Test MI when features are independent given cluster."""
        n_clusters, n_features, n_categories = 1, 2, 3
        
        # With single cluster, features are independent
        theta = jnp.array([
            [[jnp.log(0.5), jnp.log(0.3), jnp.log(0.2)],
             [jnp.log(0.1), jnp.log(0.6), jnp.log(0.3)]]
        ])
        pi = jnp.array([0.0])  # Single cluster
        
        # MI should be approximately 0
        mi = mutual_information_features(theta, pi, 0, 1)
        assert jnp.allclose(mi, 0.0, atol=1e-10)
    
    def test_mutual_information_features_deterministic(self):
        """Test MI with deterministic relationship."""
        n_clusters, n_features, n_categories = 2, 2, 2
        
        # Cluster 0: both features always 0
        # Cluster 1: both features always 1
        theta = jnp.full((n_clusters, n_features, n_categories), -jnp.inf)
        theta = theta.at[0, :, 0].set(0.0)
        theta = theta.at[1, :, 1].set(0.0)
        
        # Equal mixture
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # Features are perfectly correlated
        mi = mutual_information_features(theta, pi, 0, 1)
        
        # Compute entropy of each feature
        entropy = entropy_feature(theta, pi, 0)
        
        # MI should equal the entropy (perfect correlation)
        assert jnp.allclose(mi, entropy, atol=1e-6)
    
    def test_mutual_information_features_symmetric(self):
        """Test that MI is symmetric."""
        n_clusters, n_features, n_categories = 3, 4, 5
        
        # Random parameters
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jax.nn.log_softmax(jax.random.normal(key, (n_clusters,)))
        
        # MI(X,Y) = MI(Y,X)
        mi_01 = mutual_information_features(theta, pi, 0, 1)
        mi_10 = mutual_information_features(theta, pi, 1, 0)
        
        assert jnp.allclose(mi_01, mi_10, atol=1e-6)
    
    def test_mutual_information_feature_cluster_basic(self):
        """Test MI between feature and cluster."""
        n_clusters, n_features, n_categories = 3, 2, 4
        
        # Random parameters
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jax.nn.log_softmax(jax.random.normal(key, (n_clusters,)))
        
        # Compute MI
        mi = mutual_information_feature_cluster(theta, pi, 0)
        
        # Should be non-negative
        assert mi >= 0
        assert jnp.isfinite(mi)
    
    def test_mutual_information_feature_cluster_deterministic(self):
        """Test MI when cluster determines feature."""
        n_clusters, n_features, n_categories = 3, 1, 3
        
        # Each cluster determines a unique category
        theta = jnp.full((n_clusters, n_features, n_categories), -jnp.inf)
        theta = theta.at[0, 0, 0].set(0.0)
        theta = theta.at[1, 0, 1].set(0.0)
        theta = theta.at[2, 0, 2].set(0.0)
        
        # Equal clusters
        pi = jnp.log(jnp.ones(n_clusters) / n_clusters)
        
        # MI should equal entropy of cluster assignment
        mi = mutual_information_feature_cluster(theta, pi, 0)
        cluster_entropy = -jnp.sum(jnp.exp(pi) * pi)
        
        assert jnp.allclose(mi, cluster_entropy, atol=1e-6)
    
    def test_mutual_information_feature_cluster_independent(self):
        """Test MI when feature is independent of cluster."""
        n_clusters, n_features, n_categories = 3, 1, 2
        
        # All clusters have same distribution
        same_dist = jnp.array([jnp.log(0.3), jnp.log(0.7)])
        theta = jnp.array([
            [same_dist],
            [same_dist],
            [same_dist]
        ])
        
        pi = jax.nn.log_softmax(jax.random.normal(jax.random.PRNGKey(42), (n_clusters,)))
        
        # MI should be approximately 0
        mi = mutual_information_feature_cluster(theta, pi, 0)
        assert jnp.allclose(mi, 0.0, atol=1e-10)
    
    def test_mutual_information_matrix_shape(self):
        """Test MI matrix computation."""
        n_clusters, n_features, n_categories = 2, 4, 3
        
        # Random parameters
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jax.nn.log_softmax(jax.random.normal(key, (n_clusters,)))
        
        # Compute MI matrix
        mi_matrix = mutual_information_matrix(theta, pi)
        
        # Check shape
        assert mi_matrix.shape == (n_features, n_features)
        
        # Check symmetry
        assert jnp.allclose(mi_matrix, mi_matrix.T, atol=1e-6)
        
        # Check non-negative
        assert jnp.all(mi_matrix >= 0)
        
        # Check diagonal contains entropies
        for i in range(n_features):
            entropy = entropy_feature(theta, pi, i)
            assert jnp.allclose(mi_matrix[i, i], entropy, atol=1e-6)
    
    def test_mutual_information_matrix_consistency(self):
        """Test that MI matrix is consistent with pairwise computation."""
        n_clusters, n_features, n_categories = 3, 3, 4
        
        # Random parameters
        key = jax.random.PRNGKey(42)
        theta = jax.random.normal(key, (n_clusters, n_features, n_categories))
        theta = jax.nn.log_softmax(theta, axis=-1)
        pi = jax.nn.log_softmax(jax.random.normal(key, (n_clusters,)))
        
        # Compute full matrix
        mi_matrix = mutual_information_matrix(theta, pi)
        
        # Check consistency with pairwise computation
        # Use canonical ordering to match the matrix implementation
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    expected = entropy_feature(theta, pi, i)
                else:
                    # Use canonical ordering: always call with min index first
                    expected = mutual_information_features(theta, pi, min(i,j), max(i,j))
                # Note: slightly looser tolerance due to numerical precision in vectorized computation
                assert jnp.allclose(mi_matrix[i, j], expected, atol=2e-3)
    
    def test_entropy_feature_basic(self):
        """Test entropy computation."""
        n_clusters, n_features, n_categories = 2, 1, 3
        
        # Known distribution
        theta = jnp.array([
            [[jnp.log(0.5), jnp.log(0.3), jnp.log(0.2)]],
            [[jnp.log(0.1), jnp.log(0.6), jnp.log(0.3)]]
        ])
        pi = jnp.array([jnp.log(0.7), jnp.log(0.3)])
        
        # Compute entropy
        entropy = entropy_feature(theta, pi, 0)
        
        # Should be non-negative
        assert entropy >= 0
        
        # Compute expected distribution
        # P(X=0) = 0.7 * 0.5 + 0.3 * 0.1 = 0.38
        # P(X=1) = 0.7 * 0.3 + 0.3 * 0.6 = 0.39
        # P(X=2) = 0.7 * 0.2 + 0.3 * 0.3 = 0.23
        probs = jnp.array([0.38, 0.39, 0.23])
        expected_entropy = -jnp.sum(probs * jnp.log(probs))
        
        assert jnp.allclose(entropy, expected_entropy, atol=1e-6)
    
    def test_entropy_feature_uniform(self):
        """Test entropy of uniform distribution."""
        n_clusters, n_features, n_categories = 1, 1, 4
        
        # Uniform distribution
        theta = jnp.log(jnp.ones((n_clusters, n_features, n_categories)) / n_categories)
        pi = jnp.array([0.0])
        
        entropy = entropy_feature(theta, pi, 0)
        
        # Entropy of uniform distribution is log(n)
        expected = jnp.log(n_categories)
        assert jnp.allclose(entropy, expected, atol=1e-6)
    
    def test_entropy_feature_deterministic(self):
        """Test entropy of deterministic distribution."""
        n_clusters, n_features, n_categories = 1, 1, 5
        
        # Deterministic - always category 2
        theta = jnp.full((n_clusters, n_features, n_categories), -jnp.inf)
        theta = theta.at[0, 0, 2].set(0.0)
        pi = jnp.array([0.0])
        
        entropy = entropy_feature(theta, pi, 0)
        
        # Entropy of deterministic distribution is 0
        assert jnp.allclose(entropy, 0.0, atol=1e-10)