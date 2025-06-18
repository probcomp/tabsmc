"""
Verify that the weight fix is working and understand resampling behavior.
"""

import jax
import jax.numpy as jnp
from tabsmc.smc import smc_no_rejuvenation
from generate_synthetic_data import generate_mixture_data, create_test_parameters


def test_different_scenarios():
    """Test scenarios to see when resampling occurs."""
    
    scenarios = [
        {"name": "Small P (high resampling)", "P": 5, "T": 3, "B": 50},
        {"name": "Large P (low resampling)", "P": 50, "T": 3, "B": 50},
        {"name": "Original long test", "P": 20, "T": 5, "B": 100},
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Testing: {scenario['name']}")
        print(f"P={scenario['P']}, T={scenario['T']}, B={scenario['B']}")
        print(f"{'='*60}")
        
        # Setup
        key = jax.random.PRNGKey(42)
        P, T, B = scenario['P'], scenario['T'], scenario['B']
        N_train = B * T + 100
        N_test = 50
        D, K, C = 3, 2, 2
        
        # Generate data
        true_π, true_θ = create_test_parameters(C, D, K)
        key, subkey = jax.random.split(key)
        X_train, _ = generate_mixture_data(subkey, N_train, D, K, true_π, true_θ)
        key, subkey = jax.random.split(key)
        X_test, _ = generate_mixture_data(subkey, N_test, D, K, true_π, true_θ)
        
        α_pi = 1.0
        α_theta = 1.0
        
        # Test both methods
        for rejuvenation in [False, True]:
            method_name = "With rejuvenation" if rejuvenation else "Without rejuvenation"
            print(f"\n{method_name}:")
            
            base_key = jax.random.PRNGKey(123)
            _, _, history = smc_no_rejuvenation(
                base_key, X_train, T, P, C, B, α_pi, α_theta, 
                rejuvenation=rejuvenation, return_history=True
            )
            
            # Analyze weights at each iteration
            for t in range(T):
                weights = history['log_weights'][t]
                weight_var = jnp.var(weights)
                uniform_weight = -jnp.log(P)
                is_uniform = jnp.allclose(weights, uniform_weight, atol=1e-6)
                
                print(f"  t={t}: weight_var={weight_var:.6f}, uniform={is_uniform}, weights[0]={weights[0]:.4f}")
            
            # Compute final test log-likelihood
            particles_final = (history['A'][-1], history['phi'][-1], 
                             history['pi'][-1], history['theta'][-1])
            weights_final = history['log_weights'][-1]
            
            # Simple unweighted average for comparison
            _, _, π, θ = particles_final
            individual_logliks = []
            for p in range(P):
                π_p, θ_p = π[p], θ[p]
                log_px_given_c = jnp.einsum('ndk,cdk->nc', X_test, θ_p)
                log_px = jax.scipy.special.logsumexp(π_p[None, :] + log_px_given_c, axis=1)
                individual_logliks.append(jnp.sum(log_px))
            
            unweighted_avg = jnp.mean(jnp.array(individual_logliks))
            
            # Weighted average
            log_weights_normalized = weights_final - jax.scipy.special.logsumexp(weights_final)
            weighted_avg = jax.scipy.special.logsumexp(jnp.array(individual_logliks) + log_weights_normalized)
            
            print(f"  Final loglik - Unweighted: {unweighted_avg:.3f}, Weighted: {weighted_avg:.3f}")


if __name__ == "__main__":
    test_different_scenarios()