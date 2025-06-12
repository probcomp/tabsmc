import jax
import jax.numpy as jnp
import tabsmc.dumpy as dp
from functools import partial


def step_particle(key, X_B, I_B, A_one_hot, φ_old, π_old, θ_old, α_pi, α_theta):
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
        π,
        θ,
        counts,
        α_pi_posterior,
        α_theta_posterior,
        A_B,
        A_B_pgibbs,
        π_pgibbs,
        θ_pgibbs,
        A_p,
        X_p,
        π_p,
        θ_p,
    ) = [dp.Slot() for _ in range(17)]

    log_likelihoods["B", "C"] = dp.sum(θ_old["C", :, :] * X_B["B", :, :])
    log_probs["B", "C"] = log_likelihoods["B", "C"] + π_old["C"]
    A_B["B"] = dp.categorical(keys["B", :], log_probs["B", :])
    A_B_one_hot = dp.Array(jax.nn.one_hot(A_B, C))
    A_B_pgibbs["B", "C"] = log_probs["B", "C"] * A_B_one_hot["B", "C"]

    φ_B["C", "D", "K"] = dp.sum(A_B_one_hot[:, "C"] * X_B[:, "D", "K"])
    φ["C", "D", "K"] = φ_old["C", "D", "K"] + φ_B["C", "D", "K"]
    counts["C"] = dp.sum(φ["C", :, :])

    key, subkey = jax.random.split(key)
    α_pi_posterior["C"] = α_pi + counts["C"]
    π[:] = dp.log_dirichlet(subkey, α_pi_posterior[:])
    π_pgibbs = dp.score_log_dirichlet(α_pi_posterior[:], π[:])

    keys = dp.Array(jax.random.split(key, (C, D)))
    α_theta_posterior["C", "D", "K"] = α_theta + φ["C", "D", "K"]
    θ["C", "D", :] = dp.log_dirichlet(
        keys["C", "D", :], α_theta_posterior["C", "D", :]
    )
    θ_pgibbs["C", "D"] = dp.score_log_dirichlet(α_theta_posterior["C", "D", :], θ["C", "D", :])

    q = dp.sum(A_B_pgibbs[:, :]) + dp.sum(π_pgibbs) + dp.sum(θ_pgibbs[:, :])

    A_one_hot = dp.Array(jnp.array(A_one_hot).at[jnp.array(I_B)].set(jnp.array(A_B_one_hot)))
    π_p = dp.score_log_dirichlet(α_pi, π)
    θ_p["C", "D"] = dp.score_log_dirichlet(α_theta, θ["C", "D", :])
    A_p["N", "C"] = A_one_hot["N", "C"] * π["C"]
    X_p["C", "D", "K"] = θ["C", "D", "K"] * φ["C", "D", "K"]

    γ = dp.sum(π_p) + dp.sum(θ_p[:, :]) + dp.sum(A_p[:, :]) + dp.sum(X_p[:, :, :])

    return A_one_hot, φ, π, θ, γ, q


# JIT-compiled version of step_particle
step_particle_jit = jax.jit(step_particle)


def smc_minibatch_fast(key, X, T, P, C, L, α_pi, α_theta):
    """Optimized SMC algorithm with minibatches using JIT and vmap.
    
    Args:
        key: PRNG key
        X: Full dataset (N x D x K) one-hot encoded
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
    particles = init_smc(subkey, P, C, D, K, N, α_pi, α_theta)
    
    # Convert to arrays for easier manipulation
    A_all = particles['A']
    φ_all = particles['φ']
    π_all = particles['π']
    θ_all = particles['θ']
    w_log = particles['w_log']
    
    # Track log marginal likelihood
    log_Z = 0.0
    
    # Define single particle update function
    def update_single_particle(key, A_p, φ_p, π_p, θ_p, X_B, I_B):
        # Convert to dumpy arrays
        A_one_hot_p = dp.Array(A_p)
        φ_p_dp = dp.Array(φ_p)
        π_p_dp = dp.Array(π_p)
        θ_p_dp = dp.Array(θ_p)
        
        # Run step_particle
        A_one_hot_new, φ_new, π_new, θ_new, γ, q = step_particle_jit(
            key, X_B, I_B, A_one_hot_p, φ_p_dp, π_p_dp, θ_p_dp,
            dp.Array(α_pi), dp.Array(α_theta)
        )
        
        return (jnp.array(A_one_hot_new), jnp.array(φ_new), 
                jnp.array(π_new), jnp.array(θ_new), γ.data, q.data)
    
    # Vectorize over particles
    update_all_particles = jax.vmap(
        update_single_particle, 
        in_axes=(0, 0, 0, 0, 0, None, None)
    )
    
    # Main SMC loop
    for t in range(T):
        # Sample minibatch
        key, subkey = jax.random.split(key)
        idxs = jax.random.choice(subkey, N, shape=(L,), replace=False)
        X_L = X[idxs]
        
        # Convert to dumpy arrays
        X_B = dp.Array(X_L)
        I_B = dp.Array(idxs)
        
        # Generate keys for all particles
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, P)
        
        # Process all particles in parallel using vmap
        A_new, φ_new, π_new, θ_new, γ_all, q_all = update_all_particles(
            keys, A_all, φ_all, π_all, θ_all, X_B, I_B
        )
        
        # Update particle states
        A_all = A_new
        φ_all = φ_new
        π_all = π_new
        θ_all = θ_new
        
        # Compute incremental weights
        w_inc = γ_all - q_all
        w_log = w_log + w_inc
        
        # Normalize weights
        log_sum_w = jax.scipy.special.logsumexp(w_log)
        w_log = w_log - log_sum_w
        w = jnp.exp(w_log)
        
        # Update log marginal likelihood
        log_Z += log_sum_w - jnp.log(P)
        
        # Resample if ESS is too low
        ESS = 1.0 / jnp.sum(w**2)
        if ESS < P / 2:
            key, subkey = jax.random.split(key)
            parents = jax.random.choice(subkey, P, shape=(P,), p=w, replace=True)
            A_all = A_all[parents]
            φ_all = φ_all[parents]
            π_all = π_all[parents]
            θ_all = θ_all[parents]
            w_log = jnp.zeros(P)
            w = jnp.ones(P) / P
    
    # Return final particles
    particles_final = {
        'A': A_all,
        'φ': φ_all,
        'π': π_all,
        'θ': θ_all,
        'w': w,
        'w_log': w_log,
        'α_pi': α_pi,
        'α_theta': α_theta
    }
    
    return particles_final, log_Z


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def init_particle(key, C, D, K, N, α_pi, α_theta):
    """Initialize a single particle.
    
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
    
    return A, φ, π, θ


def init_smc(key, P, C, D, K, N, α_pi, α_theta):
    """Initialize SMC algorithm with empty assignments and zero sufficient statistics.
    
    Args:
        key: PRNG key
        P: Number of particles
        C: Number of clusters
        D: Number of features
        K: Number of categories per feature
        N: Total number of data points
        α_pi: Dirichlet prior for mixing weights
        α_theta: Dirichlet prior for emission parameters
        
    Returns:
        Dictionary containing initial state
    """
    # Generate keys for all particles
    keys = jax.random.split(key, P)
    
    # Vectorize init_particle over P particles
    init_all_particles = jax.vmap(
        init_particle,
        in_axes=(0, None, None, None, None, None, None)
    )
    
    # Initialize all particles at once
    A, φ, π, θ = init_all_particles(keys, C, D, K, N, α_pi, α_theta)
    
    # Initialize weights to zero (in log space)
    w_log = jnp.zeros(P)
    w = jnp.exp(w_log)
    
    return {
        'A': A,           # Assignments (P x N x C)
        'φ': φ,           # Sufficient statistics (P x C x D x K)
        'π': π,           # Mixing weights (P x C) in log space
        'θ': θ,           # Emission parameters (P x C x D x K) in log space
        'w': w,           # Particle weights (P,)
        'w_log': w_log,   # Log particle weights (P,)
        'α_pi': α_pi,     # Prior for π
        'α_theta': α_theta # Prior for θ
    }


def resample_particles(key, particles, weights):
    """Resample particles according to their weights.
    
    Args:
        key: PRNG key
        particles: Dictionary containing particle states
        weights: Normalized weights (P,)
        
    Returns:
        Resampled particles with uniform weights
    """
    P = weights.shape[0]
    
    # Sample parent indices according to weights
    parents = jax.random.choice(key, P, shape=(P,), p=weights, replace=True)
    
    # Resample all particle components
    resampled = {
        'A': particles['A'][parents],
        'φ': particles['φ'][parents],
        'π': particles['π'][parents],
        'θ': particles['θ'][parents],
        'w': jnp.ones(P) / P,  # Reset to uniform weights
        'w_log': jnp.log(jnp.ones(P) / P),
        'α_pi': particles['α_pi'],
        'α_theta': particles['α_theta']
    }
    
    return resampled


def smc_minibatch(key, X, T, P, C, L, α_pi, α_theta):
    """Run SMC algorithm with minibatches.
    
    Args:
        key: PRNG key
        X: Full dataset (N x D x K) one-hot encoded
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
    particles = init_smc(subkey, P, C, D, K, N, α_pi, α_theta)
    
    # Track log marginal likelihood
    log_Z = 0.0
    
    for t in range(T):
        # Sample minibatch without replacement
        key, subkey = jax.random.split(key)
        idxs = jax.random.choice(subkey, N, shape=(L,), replace=False)
        X_L = X[idxs]  # Minibatch data
        
        # Convert to dumpy arrays for each particle
        X_B = dp.Array(X_L)
        I_B = dp.Array(idxs)
        
        # Initialize arrays for storing results from all particles
        γ_all = []
        q_all = []
        
        # Process each particle
        for p in range(P):
            # Extract particle p's state
            A_one_hot_p = dp.Array(particles['A'][p])
            φ_p = dp.Array(particles['φ'][p])
            π_p = dp.Array(particles['π'][p])
            θ_p = dp.Array(particles['θ'][p])
            
            # Run step_particle (JIT-compiled)
            key, subkey = jax.random.split(key)
            A_one_hot_new, φ_new, π_new, θ_new, γ, q = step_particle_jit(
                subkey, X_B, I_B, A_one_hot_p, φ_p, π_p, θ_p, 
                dp.Array(α_pi), dp.Array(α_theta)
            )
            
            # Store updated state
            particles['A'] = particles['A'].at[p].set(jnp.array(A_one_hot_new))
            particles['φ'] = particles['φ'].at[p].set(jnp.array(φ_new))
            particles['π'] = particles['π'].at[p].set(jnp.array(π_new))
            particles['θ'] = particles['θ'].at[p].set(jnp.array(θ_new))
            
            γ_all.append(γ.data)
            q_all.append(q.data)
        
        # Convert to arrays
        γ_all = jnp.array(γ_all)
        q_all = jnp.array(q_all)
        
        # Compute incremental weights (equation 3 from the algorithm)
        w_inc = γ_all - q_all
        
        # Update log weights
        particles['w_log'] = particles['w_log'] + w_inc
        
        # Normalize weights
        log_sum_w = jax.scipy.special.logsumexp(particles['w_log'])
        particles['w_log'] = particles['w_log'] - log_sum_w
        particles['w'] = jnp.exp(particles['w_log'])
        
        # Update log marginal likelihood estimate
        log_Z += log_sum_w - jnp.log(P)
        
        # Resample if effective sample size is too low
        ESS = 1.0 / jnp.sum(particles['w']**2)
        if ESS < P / 2:
            key, subkey = jax.random.split(key)
            particles = resample_particles(subkey, particles, particles['w'])
    
    return particles, log_Z
