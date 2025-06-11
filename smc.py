"""
Sequential Monte Carlo (SMC) step_particle method extracted from moet.
"""

import jax
import jax.numpy as jnp
import dumpy as dp

def step_particle(key, X_B, I_B, Α_B, A_one_hot, φ, π, θ):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, B)
    log_likelihoods['B', 'C'] = dp.sum(θ['C', :, :] * X_B['B', :, :])
    log_probs['B', 'C'] = log_likelihoods['B', 'C'] + π['C']
    A_B['B'] = dp.categorical(keys['B'], log_probs['B', 'C'])
    A_B_one_hot = jnp.one_hot(A_B['B'], C)
    A_B_pgibbs = dp.sum(log_probs['B', A_B['B']])

    φ_B['C', 'D', 'K'] = dp.sum(A_B_one_hot[:, 'C'] * X_B[:, 'D', 'K'])
    φ['C', 'D', 'K'] = φ['C', 'D', 'K'] + φ_B['C', 'D', 'K']
    counts['C'] = dp.sum(φ['C', :, :])

    key, subkey = jax.random.split(key)
    α_pi['C'] = α + counts['C']
    π['C'] = dp.log_dirichlet(subkey, α_pi['C'])
    π_pgibbs = dp.score_log_dirichlet(α_pi['C'], π['C'])

    keys = jax.random.split(key, (B, C))
    α_theta['C', 'D', 'K'] = α + φ['C', 'D', 'K']
    θ['C', 'D', 'K'] = dp.log_dirichlet(keys['C', 'D'], α_theta['C', 'D', :])
    θ_pgibbs = dp.sum(dp.score_log_dirichlet(α_theta['C', 'D', :], θ['C', 'D', :]))

    q = A_B_pgibbs + π_pgibbs + θ_pgibbs

    A_one_hot[I['B']] = A_B_one_hot['B']
    π_p = dp.score_log_dirichlet(α_pi, π)
    theta_p = dp.score_log_dirichlet(α_theta, θ)
    A_p = dp.sum(A['N', 'C'] * π['C'])
    X_p = dp.sum(θ['C', 'D', 'K'] * φ['C', 'D', 'K'])

    γ =  π_p + theta_p + A_p + X_p

    return A, φ, π, θ, γ, q