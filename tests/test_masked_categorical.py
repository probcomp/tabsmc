"""Test masked categorical feature system."""

import jax
import jax.numpy as jnp
from tabsmc.smc import (
    masked_dirichlet_sample, log_dirichlet_score, init_empty, init_assignments,
    gibbs, smc_step, smc_no_rejuvenation, mcmc_minibatch
)


def test_masked_dirichlet_sample():
    """Test masked Dirichlet sampling function."""
    key = jax.random.PRNGKey(42)
    
    # Test case: 3 categories, only first 2 are valid
    K = 3
    α = jnp.array([2.0, 3.0, 1.0])
    mask = jnp.array([True, True, False])
    
    log_probs = masked_dirichlet_sample(key, α, mask)
    
    # Test assertions
    assert log_probs.shape == (K,), f"Shape should be ({K},)"
    assert jnp.isfinite(log_probs[0]), "Valid category 0 should have finite log prob"
    assert jnp.isfinite(log_probs[1]), "Valid category 1 should have finite log prob"
    assert log_probs[2] == -jnp.inf, "Invalid category 2 should have -inf log prob"
    
    # Check that valid probabilities sum to 1
    valid_probs = jnp.exp(log_probs[:2])
    assert jnp.abs(jnp.sum(valid_probs) - 1.0) < 1e-6, "Valid probabilities should sum to 1"


def test_log_dirichlet_score_with_mask():
    """Test log_dirichlet_score with masking."""
    α = jnp.array([2.0, 3.0, 1.0])
    x = jnp.array([-1.0, -0.5, -2.0])  # Some log probabilities
    mask = jnp.array([True, True, False])
    
    # Test with mask
    score_masked = log_dirichlet_score(α, x, mask)
    assert jnp.isfinite(score_masked), "Masked score should be finite"
    
    # Test without mask (should be different)
    score_unmasked = log_dirichlet_score(α, x)
    assert score_masked != score_unmasked, "Masked and unmasked scores should differ"


def test_masked_initialization():
    """Test initialization functions with masks."""
    key = jax.random.PRNGKey(42)
    
    # Test parameters
    N, D, C = 100, 3, 2
    K_max = 4
    α_pi, α_theta = 1.0, 1.0
    
    # Create mask: feature 0 has 2 categories, feature 1 has 3, feature 2 has 4
    mask = jnp.array([
        [True, True, False, False],    # Feature 0: 2 categories
        [True, True, True, False],     # Feature 1: 3 categories  
        [True, True, True, True]       # Feature 2: 4 categories
    ])
    
    # Test init_empty
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_empty(subkey, C, D, K_max, N, α_pi, α_theta, mask)
    
    assert A.shape == (N, C), f"A shape should be ({N}, {C})"
    assert φ.shape == (C, D, K_max), f"φ shape should be ({C}, {D}, {K_max})"
    assert π.shape == (C,), f"π shape should be ({C},)"
    assert θ.shape == (C, D, K_max), f"θ shape should be ({C}, {D}, {K_max})"
    
    # Check that invalid categories have -inf log probabilities
    assert jnp.all(θ[:, 0, 2:] == -jnp.inf), "Feature 0 invalid categories should be -inf"
    assert jnp.all(θ[:, 1, 3:] == -jnp.inf), "Feature 1 invalid categories should be -inf"
    # Feature 2 should have all finite values (all categories valid)
    assert jnp.all(jnp.isfinite(θ[:, 2, :])), "Feature 2 all categories should be finite"


def test_masked_gibbs():
    """Test Gibbs sampling with masked features."""
    key = jax.random.PRNGKey(42)
    
    # Test parameters
    N, D, C, B = 50, 2, 2, 10
    K_max = 3
    α_pi, α_theta = 1.0, 1.0
    
    # Create mask: feature 0 has 2 categories, feature 1 has 3
    mask = jnp.array([
        [True, True, False],     # Feature 0: 2 categories
        [True, True, True]       # Feature 1: 3 categories
    ])
    
    # Generate synthetic data respecting the mask
    key, subkey = jax.random.split(key)
    X = jnp.zeros((N, D, K_max))
    
    # For feature 0: only use first 2 categories
    cats_0 = jax.random.choice(subkey, 2, shape=(N,))
    X = X.at[:, 0, :].set(jax.nn.one_hot(cats_0, K_max))
    
    # For feature 1: use all 3 categories
    key, subkey = jax.random.split(key)
    cats_1 = jax.random.choice(subkey, 3, shape=(N,))
    X = X.at[:, 1, :].set(jax.nn.one_hot(cats_1, K_max))
    
    # Initialize particle
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_empty(subkey, C, D, K_max, N, α_pi, α_theta, mask)
    
    # Run one Gibbs step
    key, subkey = jax.random.split(key)
    I_B = jnp.arange(B)
    X_B = X[I_B]
    
    key, subkey = jax.random.split(key)
    A_new, φ_new, π_new, θ_new, γ, q = gibbs(
        subkey, X_B, I_B, A, φ, π, θ, α_pi, α_theta, mask
    )
    
    # Test assertions
    assert jnp.isfinite(γ), "Target log probability should be finite"
    assert jnp.isfinite(q), "Proposal log probability should be finite"
    assert A_new.shape == A.shape, "Assignments shape should be preserved"
    assert φ_new.shape == φ.shape, "Sufficient statistics shape should be preserved"
    assert π_new.shape == π.shape, "Mixing weights shape should be preserved"
    assert θ_new.shape == θ.shape, "Emission parameters shape should be preserved"
    
    # Check that invalid categories still have -inf
    assert jnp.all(θ_new[:, 0, 2] == -jnp.inf), "Feature 0 invalid category should remain -inf"


def test_masked_smc():
    """Test full SMC with masked features."""
    key = jax.random.PRNGKey(42)
    
    # Test parameters
    N, D, C, P, T, B = 100, 2, 2, 4, 5, 20
    K_max = 4
    α_pi, α_theta = 1.0, 1.0
    
    # Create mask: feature 0 has 2 categories, feature 1 has 3
    mask = jnp.array([
        [True, True, False, False],   # Feature 0: 2 categories
        [True, True, True, False]     # Feature 1: 3 categories
    ])
    
    # Generate synthetic data respecting the mask
    key, subkey = jax.random.split(key)
    X = jnp.zeros((N, D, K_max))
    
    # For feature 0: only use first 2 categories
    cats_0 = jax.random.choice(subkey, 2, shape=(N,))
    X = X.at[:, 0, :].set(jax.nn.one_hot(cats_0, K_max))
    
    # For feature 1: use first 3 categories
    key, subkey = jax.random.split(key)
    cats_1 = jax.random.choice(subkey, 3, shape=(N,))
    X = X.at[:, 1, :].set(jax.nn.one_hot(cats_1, K_max))
    
    # Run SMC
    key, subkey = jax.random.split(key)
    particles_final, log_weights_final = smc_no_rejuvenation(
        subkey, X, T, P, C, B, α_pi, α_theta, mask=mask
    )
    
    A_final, φ_final, π_final, θ_final = particles_final
    
    # Test assertions
    assert A_final.shape == (P, N, C), f"Final A shape should be ({P}, {N}, {C})"
    assert φ_final.shape == (P, C, D, K_max), f"Final φ shape should be ({P}, {C}, {D}, {K_max})"
    assert π_final.shape == (P, C), f"Final π shape should be ({P}, {C})"
    assert θ_final.shape == (P, C, D, K_max), f"Final θ shape should be ({P}, {C}, {D}, {K_max})"
    assert log_weights_final.shape == (P,), f"Final weights shape should be ({P},)"
    
    assert jnp.all(jnp.isfinite(log_weights_final)), "All final weights should be finite"
    
    # Check that invalid categories still have -inf across all particles
    assert jnp.all(θ_final[:, :, 0, 2:] == -jnp.inf), "Feature 0 invalid categories should be -inf"
    assert jnp.all(θ_final[:, :, 1, 3:] == -jnp.inf), "Feature 1 invalid categories should be -inf"
    
    # Compute log marginal likelihood
    log_ml = jax.scipy.special.logsumexp(log_weights_final) - jnp.log(P)
    assert jnp.isfinite(log_ml), "Log marginal likelihood should be finite"


def test_masked_mcmc():
    """Test MCMC with masked features."""
    key = jax.random.PRNGKey(42)
    
    # Test parameters  
    N, D, C, T, B = 80, 2, 2, 10, 15
    K_max = 3
    α_pi, α_theta = 1.0, 1.0
    
    # Create mask
    mask = jnp.array([
        [True, True, False],     # Feature 0: 2 categories
        [True, True, True]       # Feature 1: 3 categories
    ])
    
    # Generate synthetic data
    key, subkey = jax.random.split(key)
    X = jnp.zeros((N, D, K_max))
    
    cats_0 = jax.random.choice(subkey, 2, shape=(N,))
    X = X.at[:, 0, :].set(jax.nn.one_hot(cats_0, K_max))
    
    key, subkey = jax.random.split(key)
    cats_1 = jax.random.choice(subkey, 3, shape=(N,))
    X = X.at[:, 1, :].set(jax.nn.one_hot(cats_1, K_max))
    
    # Run MCMC
    key, subkey = jax.random.split(key)
    A, φ, π, θ = mcmc_minibatch(subkey, X, T, C, B, α_pi, α_theta, mask)
    
    # Test assertions
    assert A.shape == (N, C), f"A shape should be ({N}, {C})"
    assert φ.shape == (C, D, K_max), f"φ shape should be ({C}, {D}, {K_max})"
    assert π.shape == (C,), f"π shape should be ({C},)"
    assert θ.shape == (C, D, K_max), f"θ shape should be ({C}, {D}, {K_max})"
    
    # Check valid range
    assert jnp.all(jnp.sum(A, axis=1) == 1), "Each data point should belong to exactly one cluster"
    assert jnp.abs(jnp.sum(jnp.exp(π)) - 1.0) < 1e-6, "Mixing weights should sum to 1"
    
    # Check masking constraints
    assert jnp.all(θ[:, 0, 2] == -jnp.inf), "Feature 0 invalid category should be -inf"


def test_backward_compatibility():
    """Test that existing code works without masks (backward compatibility)."""
    key = jax.random.PRNGKey(42)
    
    # Test parameters (no mask)
    N, D, K, C, P, T, B = 60, 2, 3, 2, 4, 3, 15
    α_pi, α_theta = 1.0, 1.0
    
    # Generate standard data
    key, subkey = jax.random.split(key)
    data_indices = jax.random.randint(subkey, (N, D), 0, K)
    X = jax.nn.one_hot(data_indices, K)
    
    # Test all functions without mask (should work as before)
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_empty(subkey, C, D, K, N, α_pi, α_theta)
    
    key, subkey = jax.random.split(key)
    A2, φ2, π2, θ2 = init_assignments(subkey, X, C, α_pi, α_theta)
    
    key, subkey = jax.random.split(key)
    particles_final, log_weights_final = smc_no_rejuvenation(
        subkey, X, T, P, C, B, α_pi, α_theta
    )
    
    key, subkey = jax.random.split(key)
    A3, φ3, π3, θ3 = mcmc_minibatch(subkey, X, T, C, B, α_pi, α_theta)
    
    # All should work and return finite values
    assert jnp.all(jnp.isfinite(π)), "π should be finite without mask"
    assert jnp.all(jnp.isfinite(θ)), "θ should be finite without mask"
    assert jnp.all(jnp.isfinite(log_weights_final)), "Weights should be finite without mask"


if __name__ == "__main__":
    test_masked_dirichlet_sample()
    test_log_dirichlet_score_with_mask()
    test_masked_initialization()
    test_masked_gibbs()
    test_masked_smc()
    test_masked_mcmc()
    test_backward_compatibility()
    print("All masked categorical tests passed!")