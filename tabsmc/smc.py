import jax
import jax.numpy as jnp
import tabsmc.dumpy as dp


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
