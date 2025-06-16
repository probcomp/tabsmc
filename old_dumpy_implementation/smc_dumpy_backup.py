import jax
import jax.numpy as jnp
import tabsmc.dumpy as dp
from functools import partial


@jax.jit
def gibbs(key, X_B, I_B, A_one_hot, φ_old, π_old, θ_old, α_pi, α_theta):
    B, D, K = X_B.shape
    C = π_old.shape[0]
    key, subkey = jax.random.split(key)
    keys = dp.Array(jax.random.split(subkey, B))

    # Initialize dumpy slots
    (
        log_likelihoods,
        log_probs,
        φ,
        φ_B,
        φ_B_old,
        π,
        θ,
        counts,
        α_pi_posterior,
        α_theta_posterior,
        A_B,
        A_B_one_hot_old,
        A_B_pgibbs,
        π_pgibbs,
        θ_pgibbs,
        A_p,
        X_p,
        π_p,
        θ_p,
    ) = [dp.Slot() for _ in range(19)]

    log_likelihoods["B", "C"] = dp.sum(θ_old["C", :, :] * X_B["B", :, :])
    log_probs["B", "C"] = log_likelihoods["B", "C"] + π_old["C"]
    A_B["B"] = dp.categorical(keys["B", :], log_probs["B", :])
    A_B_one_hot = dp.Array(jax.nn.one_hot(A_B, C))
    A_B_pgibbs["B", "C"] = log_probs["B", "C"] * A_B_one_hot["B", "C"]

    A_B_one_hot_old["B", "C"] = A_one_hot[I_B["B"], "C"]
    φ_B["C", "D", "K"] = dp.sum(A_B_one_hot[:, "C"] * X_B[:, "D", "K"])
    φ_B_old["C", "D", "K"] = dp.sum(A_B_one_hot_old[:, "C"] * X_B[:, "D", "K"])
    φ["C", "D", "K"] = φ_old["C", "D", "K"] + φ_B["C", "D", "K"] - φ_B_old["C", "D", "K"]
    counts["C"] = dp.sum(φ["C", :, :])

    key, subkey = jax.random.split(key)
    α_pi_posterior["C"] = α_pi + counts["C"]
    π[:] = dp.log_dirichlet(subkey, α_pi_posterior[:])
    π_pgibbs = dp.score_log_dirichlet(α_pi_posterior[:], π[:])

    keys = dp.Array(jax.random.split(key, (C, D)))
    α_theta_posterior["C", "D", "K"] = α_theta + φ["C", "D", "K"]
    θ["C", "D", :] = dp.log_dirichlet(keys["C", "D", :], α_theta_posterior["C", "D", :])
    θ_pgibbs["C", "D"] = dp.score_log_dirichlet(
        α_theta_posterior["C", "D", :], θ["C", "D", :]
    )

    q = dp.sum(A_B_pgibbs[:, :]) + dp.sum(π_pgibbs) + dp.sum(θ_pgibbs[:, :])

    A_one_hot = dp.Array(
        jnp.array(A_one_hot).at[jnp.array(I_B)].set(jnp.array(A_B_one_hot))
    )
    π_p = dp.score_log_dirichlet(α_pi, π)
    θ_p["C", "D"] = dp.score_log_dirichlet(α_theta, θ["C", "D", :])
    A_p["N", "C"] = A_one_hot["N", "C"] * π["C"]
    X_p["C", "D", "K"] = θ["C", "D", "K"] * φ["C", "D", "K"]

    γ = dp.sum(π_p) + dp.sum(θ_p[:, :]) + dp.sum(A_p[:, :]) + dp.sum(X_p[:, :, :])

    return A_one_hot, φ, π, θ, γ, q


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def init_empty(key, C, D, K, N, α_pi, α_theta):
    """Initialize a single particle with empty assignments.

    Args:
        key: PRNG key
        C: Number of clusters
        D: Number of features
        K: Number of categories per feature
        N: Total number of data points
        α_pi: Dirichlet prior for mixing weights
        α_theta: Dirichlet prior for emission parameters

    Returns:
        Tuple of (A, φ, π, θ) for a single particle
    """
    # Initialize empty assignments (all zeros)
    A = jnp.zeros((N, C))

    # Initialize zero sufficient statistics
    φ = jnp.zeros((C, D, K))

    # Sample π from Dirichlet prior (in log space)
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, jnp.ones(C) * α_pi))

    # Sample θ from Dirichlet prior (in log space)
    key, subkey = jax.random.split(key)
    keys_theta = jax.random.split(subkey, C * D).reshape(C, D, -1)
    sample_theta = lambda k: jnp.log(jax.random.dirichlet(k, jnp.ones(K) * α_theta))
    θ = jax.vmap(jax.vmap(sample_theta))(keys_theta)

    return dp.Array(A), dp.Array(φ), dp.Array(π), dp.Array(θ)


@partial(jax.jit, static_argnums=(2, 3, 4))
def init_assignments(key, X, C, D, K, α_pi, α_theta):
    """Initialize a single particle by sampling assignments and updating parameters.

    Args:
        key: PRNG key
        X: Data array (N x D x K) one-hot encoded
        C: Number of clusters
        D: Number of features
        K: Number of categories per feature
        α_pi: Dirichlet prior for mixing weights
        α_theta: Dirichlet prior for emission parameters

    Returns:
        Tuple of (A, φ, π, θ) for a single particle
    """
    N = X.shape[0]
    
    # Sample π from Dirichlet prior (in log space)
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, jnp.ones(C) * α_pi))
    
    # Sample initial assignments A from categorical(π)
    key, subkey = jax.random.split(key)
    keys_A = jax.random.split(subkey, N)
    A_indices = jax.vmap(lambda k: jax.random.categorical(k, π))(keys_A)
    A = jax.nn.one_hot(A_indices, C)
    
    # Compute sufficient statistics φ
    φ = jnp.einsum('nc,ndk->cdk', A, X)
    
    # Update π using posterior given A
    counts = jnp.sum(A, axis=0)
    α_pi_posterior = α_pi + counts
    key, subkey = jax.random.split(key)
    π = jnp.log(jax.random.dirichlet(subkey, α_pi_posterior))
    
    # Update θ using posterior given φ
    key, subkey = jax.random.split(key)
    keys_theta = jax.random.split(subkey, C * D).reshape(C, D, -1)
    α_theta_posterior = α_theta + φ
    sample_theta = lambda k, α: jnp.log(jax.random.dirichlet(k, α))
    θ = jax.vmap(jax.vmap(sample_theta))(keys_theta, α_theta_posterior)
    
    return dp.Array(A), dp.Array(φ), dp.Array(π), dp.Array(θ)


def smc_minibatch(key, X, T, P, C, B, α_pi, α_theta, ess=0.5):
    """Run SMC algorithm with minibatches.

    Args:
        key: PRNG key
        X: Full dataset (N x D x K) one-hot encoded, dp.Array
        T: Number of iterations
        P: Number of particles
        C: Number of clusters
        L: Minibatch size
        α_pi: Dirichlet prior for mixing weights
        α_theta: Dirichlet prior for emission parameters

    Returns:
        Final particle states and log marginal likelihood estimate
    """
    N, D, K = X.shape

    # Initialize particles
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, P)
    A["P", :, :], φ["P", :, :, :], π["P", :], θ["P", :, :, :] = init_empty(
        keys["P"], C, D, K, N, α_pi[:], α_theta[:]
    )
    w = dp.zeros(P)
    γ = dp.zeros(P)

    for t in range(T):
        I_B = dp.Array(jnp.arange(B*t, B*(t+1)))
        X_B = dp.Slot()
        X_B["B", :, :] = X[I_B["B"]]  # Minibatch data

        w = w - γ

        # Process each particle
        key, subkey = jax.random.split(key)
        A, φ, π, θ, γ, q = gibbs(
            subkey,
            X_B[:, :, :],
            I_B[:],
            A["P", :, :],
            φ["P", :, :, :],
            π["P", :],
            θ["P", :, :, :],
            α_pi,
            α_theta,
        )

        # Compute incremental weights (equation 3 from the algorithm)
        w = w + γ - q

        # Compute effective sample size (ESS) using unnormalized log weights w
        w_norm = w - jax.scipy.special.logsumexp(w)
        w_probs = jnp.exp(w_norm)
        ESS = 1.0 / jnp.sum(w_probs**2)

        # Resample if effective sample size is too low
        key, subkey = jax.random.split(key)
        A, φ, π, θ, w, γ = jax.lax.cond(
            ESS < P * ess, resample, lambda args: args[1:], subkey, A, φ, π, θ, w, γ
        )

    return A, φ, π, θ, w, γ


def resample(key, A, φ, π, θ, w, γ):
    P = w.shape[0]
    K = dp.categorical(key, w, shape=(P,))
    A["P", :, :] = A[K["P"], :, :]
    φ["P", :, :, :] = φ[K["P"], :, :, :]
    π["P", :] = π[K["P"], :]
    θ["P", :, :, :] = θ[K["P"], :, :, :]
    w = dp.logsumexp(w) - dp.log(P)
    γ = dp.zeros(P)

    return A, φ, π, θ, w, γ


def mcmc_minibatch(key, X, T, C, B, α_pi, α_theta):
    """Run MCMC algorithm with minibatches using Gibbs moves.

    Args:
        key: PRNG key
        X: Full dataset (N x D x K) one-hot encoded, dp.Array
        T: Number of iterations
        C: Number of clusters
        B: Minibatch size
        α_pi: Dirichlet prior for mixing weights
        α_theta: Dirichlet prior for emission parameters

    Returns:
        Final state (A, φ, π, θ)
    """
    N, D, K = X.shape
    
    # Initialize using assignments
    key, subkey = jax.random.split(key)
    A, φ, π, θ = init_assignments(subkey, X, C, D, K, α_pi, α_theta)
    
    for t in range(T):
        # Sample random batch indices
        key, subkey = jax.random.split(key)
        I_B = dp.Array(jax.random.choice(subkey, N, shape=(B,), replace=False))
        X_B = dp.Slot()
        X_B["B", :, :] = X[I_B["B"]]  # Minibatch data
        
        # Run Gibbs sampler on the batch
        key, subkey = jax.random.split(key)
        A, φ, π, θ, _, _ = gibbs(
            subkey,
            X_B[:, :, :],
            I_B[:],
            A,
            φ,
            π,
            θ,
            α_pi,
            α_theta
        )
    
    return A, φ, π, θ
