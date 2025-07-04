#!/usr/bin/env python
"""Test actual GPU memory usage by disabling JAX preallocation."""

import os
# Disable JAX preallocation BEFORE importing JAX
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import subprocess
import time
import gc
from tabsmc.smc import init_empty, smc_step, gibbs


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(','))
            return used, total
        return None, None
    except:
        return None, None


def print_memory_status(label):
    """Print current memory status."""
    used, total = get_gpu_memory_info()
    if used is not None:
        print(f"{label}: {used}/{total} MB ({used/total*100:.1f}%)")
        return used, total
    else:
        print(f"{label}: GPU memory info not available")
        return None, None


def load_pums_data(batch_size=1000):
    """Load PUMS data."""
    import numpy as np
    
    X_train_raw = np.load('data/pums10000.npy')
    mask = np.load('data/pums_mask.npy')
    
    # Convert to one-hot and take batch
    X_train = np.where(X_train_raw == 0.0, 1.0, 0.0)
    X_batch = jnp.array(X_train[:batch_size])
    mask = jnp.array(mask)
    
    N, D, K = X_batch.shape
    return X_batch, mask, N, D, K


def test_smc_memory_usage():
    """Test actual SMC memory usage with different particle counts."""
    print("=== Testing SMC Memory Usage (No Preallocation) ===")
    
    # Check initial state
    print("JAX preallocation disabled:", os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'))
    print("JAX devices:", jax.devices())
    
    # Load data
    X_batch, mask, N, D, K = load_pums_data(1000)
    C = 10
    alpha_pi = 1.0
    alpha_theta = 1.0
    
    print(f"Data shape: N={N}, D={D}, K={K}, C={C}")
    
    # Test different particle counts
    particle_counts = [1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    
    results = []
    
    for n_particles in particle_counts:
        print(f"\n{'='*50}")
        print(f"Testing {n_particles} particles")
        
        # Clear memory
        gc.collect()
        jax.clear_caches()
        time.sleep(1)
        
        baseline_used, total = print_memory_status("Baseline")
        
        try:
            # Initialize particles
            key = jax.random.PRNGKey(42)
            keys = jax.random.split(key, n_particles + 1)
            key, particle_keys = keys[0], keys[1:]
            
            print("Initializing particles...")
            A, φ, π, θ = jax.vmap(
                lambda k: init_empty(k, C, D, K, N, alpha_pi, alpha_theta, mask)
            )(particle_keys)
            A.block_until_ready()
            
            after_init_used, _ = print_memory_status("After init")
            init_memory = after_init_used - baseline_used if after_init_used and baseline_used else 0
            
            # Initialize SMC state
            log_weights = jnp.zeros(n_particles)
            log_gammas = jnp.zeros(n_particles)
            I_B = jnp.arange(N)
            
            # Run one SMC step
            print("Running SMC step...")
            key, subkey = jax.random.split(key)
            particles = (A, φ, π, θ)
            particles, log_weights, log_gammas = smc_step(
                subkey, particles, log_weights, log_gammas, X_batch, I_B, alpha_pi, alpha_theta, mask
            )
            particles[0].block_until_ready()
            
            after_smc_used, _ = print_memory_status("After SMC step")
            smc_memory = after_smc_used - baseline_used if after_smc_used and baseline_used else 0
            
            # Run 10 rejuvenation steps
            print("Running 10 rejuvenation steps...")
            A, φ, π, θ = particles
            
            for i in range(10):
                key, subkey = jax.random.split(key)
                I_rejuv = jax.random.choice(subkey, N, shape=(N,), replace=False)
                X_rejuv = X_batch[I_rejuv]
                
                key, subkey = jax.random.split(key)
                keys = jax.random.split(subkey, n_particles)
                
                def rejuvenate_particle(p_key, p_A, p_φ, p_π, p_θ):
                    A_new, φ_new, π_new, θ_new, _, _ = gibbs(
                        p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, alpha_pi, alpha_theta, mask
                    )
                    return A_new, φ_new, π_new, θ_new
                
                A, φ, π, θ = jax.vmap(rejuvenate_particle)(keys, A, φ, π, θ)
            
            A.block_until_ready()
            
            after_rejuv_used, _ = print_memory_status("After rejuvenation")
            total_memory = after_rejuv_used - baseline_used if after_rejuv_used and baseline_used else 0
            
            print(f"Memory usage: Init={init_memory}MB, SMC={smc_memory}MB, Total={total_memory}MB")
            
            # Calculate memory per particle
            if total_memory > 0:
                memory_per_particle = total_memory / n_particles
                print(f"Memory per particle: {memory_per_particle:.2f} MB")
            
            results.append({
                'n_particles': n_particles,
                'success': True,
                'init_memory': init_memory,
                'smc_memory': smc_memory,
                'total_memory': total_memory,
                'memory_per_particle': total_memory / n_particles if total_memory > 0 else 0
            })
            
            print(f"✓ SUCCESS with {n_particles} particles")
            
        except Exception as e:
            error_msg = str(e)
            is_oom = "RESOURCE_EXHAUSTED" in error_msg or "out of memory" in error_msg.lower()
            
            if is_oom:
                print(f"✗ OUT OF MEMORY with {n_particles} particles")
                results.append({
                    'n_particles': n_particles,
                    'success': False,
                    'error': 'OOM'
                })
                # Stop testing higher particle counts
                break
            else:
                print(f"✗ ERROR with {n_particles} particles: {error_msg[:100]}...")
                results.append({
                    'n_particles': n_particles,
                    'success': False,
                    'error': error_msg[:100]
                })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Particles':<10} {'Success':<8} {'Init MB':<8} {'SMC MB':<8} {'Total MB':<10} {'MB/Part':<8}")
    print("-" * 60)
    
    max_successful = 0
    for r in results:
        if r['success']:
            max_successful = r['n_particles']
            print(f"{r['n_particles']:<10} {'✓':<8} {r['init_memory']:<8.1f} {r['smc_memory']:<8.1f} "
                  f"{r['total_memory']:<10.1f} {r['memory_per_particle']:<8.2f}")
        else:
            error_type = r.get('error', 'Unknown')
            print(f"{r['n_particles']:<10} {'✗':<8} {error_type:<8} {'':<8} {'':<10} {'':<8}")
    
    print(f"\nMaximum successful particle count: {max_successful}")
    
    # Save results
    import pickle
    with open('data/actual_gpu_memory_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'max_particles': max_successful,
            'config': {
                'batch_size': 1000,
                'rejuvenation_steps': 10,
                'clusters': C,
                'D': D,
                'K': K,
                'preallocation_disabled': True
            }
        }, f)
    
    return results, max_successful


def main():
    print("Testing actual GPU memory usage with JAX preallocation disabled")
    results, max_particles = test_smc_memory_usage()
    print(f"\nResults saved to data/actual_gpu_memory_results.pkl")


if __name__ == "__main__":
    main()