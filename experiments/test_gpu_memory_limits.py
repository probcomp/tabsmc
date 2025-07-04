#!/usr/bin/env python
"""Test GPU memory limits for SMC with varying particle counts on PUMS data."""

import jax
import jax.numpy as jnp
import numpy as np
import time
import gc
from tabsmc.smc import init_empty, smc_step, gibbs
import subprocess


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


def load_pums_data(batch_size=1000):
    """Load PUMS data and prepare batch."""
    print(f"Loading PUMS data with batch size {batch_size}...")
    
    # Load raw PUMS data
    X_train_raw = np.load('data/pums10000.npy')
    mask = np.load('data/pums_mask.npy')
    
    # Convert to one-hot
    X_train = np.where(X_train_raw == 0.0, 1.0, 0.0)
    
    # Take batch
    X_batch = jnp.array(X_train[:batch_size])
    mask = jnp.array(mask)
    
    N, D, K = X_batch.shape
    print(f"Data shape: N={N}, D={D}, K={K}")
    
    return X_batch, mask, N, D, K


def test_particle_count(n_particles, X_batch, mask, N, D, K, C=10, rejuvenation_steps=10):
    """Test SMC with given particle count."""
    print(f"\n{'='*60}")
    print(f"Testing with {n_particles} particles...")
    
    # Print initial GPU memory
    used, total = get_gpu_memory_info()
    if used is not None:
        print(f"Initial GPU memory: {used}/{total} MB ({used/total*100:.1f}%)")
    
    try:
        # Initialize
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, n_particles + 1)
        key, particle_keys = keys[0], keys[1:]
        
        alpha_pi = 1.0
        alpha_theta = 1.0
        
        # Initialize particles
        print("Initializing particles...")
        init_start = time.time()
        # Create constant arrays for vmap
        A, φ, π, θ = jax.vmap(
            lambda key: init_empty(key, C, D, K, N, alpha_pi, alpha_theta, mask)
        )(particle_keys)
        # Force computation
        A.block_until_ready()
        init_time = time.time() - init_start
        print(f"Initialization took {init_time:.2f}s")
        
        # Check memory after init
        used, total = get_gpu_memory_info()
        if used is not None:
            print(f"After init GPU memory: {used}/{total} MB ({used/total*100:.1f}%)")
        
        # Initialize weights and gammas
        log_weights = jnp.zeros(n_particles)
        log_gammas = jnp.zeros(n_particles)
        
        # Create batch indices (first batch)
        I_B = jnp.arange(N)
        
        # Run one SMC step
        print(f"Running SMC step...")
        step_start = time.time()
        
        key, subkey = jax.random.split(key)
        particles = (A, φ, π, θ)
        particles, log_weights, log_gammas = smc_step(
            subkey, particles, log_weights, log_gammas, X_batch, I_B, alpha_pi, alpha_theta, mask
        )
        
        # Run rejuvenation steps
        if rejuvenation_steps > 0:
            print(f"Running {rejuvenation_steps} rejuvenation steps...")
            A, φ, π, θ = particles
            
            for _ in range(rejuvenation_steps):
                key, subkey = jax.random.split(key)
                I_rejuv = jax.random.choice(subkey, N, shape=(N,), replace=False)
                X_rejuv = X_batch[I_rejuv]
                
                # Run Gibbs step for each particle
                key, subkey = jax.random.split(key)
                keys = jax.random.split(subkey, n_particles)
                
                def rejuvenate_particle(p_key, p_A, p_φ, p_π, p_θ):
                    A_new, φ_new, π_new, θ_new, _, _ = gibbs(
                        p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, alpha_pi, alpha_theta, mask
                    )
                    return A_new, φ_new, π_new, θ_new
                
                A, φ, π, θ = jax.vmap(rejuvenate_particle)(keys, A, φ, π, θ)
            
            particles = (A, φ, π, θ)
        
        # Force computation
        particles[0].block_until_ready()
        step_time = time.time() - step_start
        print(f"SMC step took {step_time:.2f}s")
        
        # Check final memory
        used, total = get_gpu_memory_info()
        if used is not None:
            print(f"Final GPU memory: {used}/{total} MB ({used/total*100:.1f}%)")
        
        # Calculate approximate memory per particle
        if total is not None:
            memory_per_particle = used / n_particles
            print(f"Approximate memory per particle: {memory_per_particle:.2f} MB")
            max_particles_estimate = int(total * 0.9 / memory_per_particle)  # 90% of total
            print(f"Estimated max particles (90% GPU): ~{max_particles_estimate}")
        
        print(f"✓ SUCCESS with {n_particles} particles")
        return True, init_time, step_time, used, total
        
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
            print(f"✗ OUT OF MEMORY with {n_particles} particles")
            print(f"Error: {str(e)[:200]}...")
        else:
            print(f"✗ OTHER ERROR with {n_particles} particles")
            print(f"Error: {str(e)[:200]}...")
        return False, None, None, None, None
    finally:
        # Clean up
        gc.collect()
        jax.clear_caches()


def find_memory_limit(X_batch, mask, N, D, K):
    """Find the maximum number of particles that fits in GPU memory."""
    # Test sequence - start conservative then increase
    test_sequence = [
        10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
        1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 5000,
        6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000
    ]
    
    results = []
    max_successful = 0
    
    for n_particles in test_sequence:
        print(f"Testing with {n_particles} particles...")
        success, init_time, step_time, used, total = test_particle_count(
            n_particles, X_batch, mask, N, D, K
        )
        
        results.append({
            'n_particles': n_particles,
            'success': success,
            'init_time': init_time,
            'step_time': step_time,
            'gpu_used': used,
            'gpu_total': total
        })
        
        if success:
            max_successful = n_particles
        else:
            # We hit the limit, no point testing higher
            print(f"\nMemory limit found! Maximum successful: {max_successful} particles")
            break
    
    return results, max_successful


def main():
    # Check if GPU is available
    print("JAX devices:", jax.devices())
    if not any(d.platform == 'gpu' for d in jax.devices()):
        print("WARNING: No GPU detected! Results will be for CPU.")
    
    # Load data
    batch_size = 1000
    X_batch, mask, N, D, K = load_pums_data(batch_size)
    
    # Find memory limit
    print(f"\nTesting GPU memory limits with:")
    print(f"- Batch size: {batch_size}")
    print(f"- Rejuvenation steps: 10")
    print(f"- Clusters: 10")
    print(f"- Data dimensions: D={D}, K={K}")
    
    results, max_particles = find_memory_limit(X_batch, mask, N, D, K)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Maximum successful particle count: {max_particles}")
    
    print("\nDetailed results:")
    print(f"{'Particles':<10} {'Success':<8} {'Init(s)':<8} {'Step(s)':<8} {'GPU Used':<10} {'GPU %':<6}")
    print("-"*60)
    
    for r in results:
        if r['success']:
            gpu_pct = f"{r['gpu_used']/r['gpu_total']*100:.1f}%" if r['gpu_total'] else "N/A"
            print(f"{r['n_particles']:<10} {'✓':<8} {r['init_time']:<8.2f} {r['step_time']:<8.2f} "
                  f"{r['gpu_used']:<10} {gpu_pct:<6}")
        else:
            print(f"{r['n_particles']:<10} {'✗':<8} {'OOM':<8} {'OOM':<8} {'OOM':<10} {'OOM':<6}")
    
    # Save results
    import pickle
    with open('data/gpu_memory_test_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'max_particles': max_particles,
            'config': {
                'batch_size': batch_size,
                'rejuvenation_steps': 10,
                'clusters': 10,
                'D': D,
                'K': K
            }
        }, f)
    print(f"\nResults saved to data/gpu_memory_test_results.pkl")


if __name__ == "__main__":
    main()