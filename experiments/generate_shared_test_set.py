"""Generate and save a shared test set for all Gibbs steps experiments."""

import jax
import jax.numpy as jnp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_synthetic_data import generate_mixture_data, create_test_parameters
import pickle


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