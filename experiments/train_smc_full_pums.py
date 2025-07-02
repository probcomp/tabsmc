#!/usr/bin/env python
"""Train SMC on full PUMS dataset with memory-aware particle sizing."""

import jax
import jax.numpy as jnp
import numpy as np
import time
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

from tabsmc.smc import init_empty, smc_step, gibbs


def load_pums_data_from_huggingface(dataset_path="data/lpm/PUMS"):
    """Load full PUMS dataset from HuggingFace using original load_data function."""
    from tabsmc.io import load_data
    import numpy as np
    
    print(f"Loading PUMS data from HuggingFace: {dataset_path}")
    print("This will download and preprocess the full dataset...")
    
    # Use original load_data function (returns log-space data)
    train_data_log, test_data_log, col_names, mask = load_data(dataset_path)
    
    # Convert from log-space to proper one-hot encoding
    # In log-space: 0 = log(1) = active category, -inf = log(0) = inactive category
    train_data = (train_data_log == 0.0).astype(np.float32)
    test_data = (test_data_log == 0.0).astype(np.float32)
    
    # Convert mask to boolean
    mask = (mask > 0).astype(bool)
    
    # Keep data as numpy arrays (CPU memory)
    X_train_cpu = train_data  # numpy array
    X_test_cpu = test_data    # numpy array
    mask_gpu = jnp.array(mask)  # mask is small, can keep on GPU
    
    N_train, D, K = X_train_cpu.shape
    N_test = X_test_cpu.shape[0]
    
    # Calculate memory usage
    train_size_gb = X_train_cpu.nbytes / (1024**3)
    test_size_gb = X_test_cpu.nbytes / (1024**3)
    
    print(f"Dataset loaded (kept in CPU memory):")
    print(f"  Training: N={N_train}, D={D}, K={K} ({train_size_gb:.2f} GB)")
    print(f"  Test: N={N_test} ({test_size_gb:.2f} GB)")
    print(f"  Features: {len(col_names)}")
    print(f"  Mask: {mask_gpu.shape} (on GPU)")
    
    return X_train_cpu, X_test_cpu, mask_gpu, col_names


# Removed likelihood computation functions - these will be in a separate evaluation script


def run_smc_full_pums(
    dataset_path="data/lpm/PUMS",
    n_particles=1000,
    n_clusters=10,
    batch_size=200,
    rejuvenation_steps=10,
    alpha_pi=1.0,
    alpha_theta=1.0,
    seed=42,
    output_dir="results",
    save_every=10,
    n_time_steps=None,
):
    """
    Run SMC on full PUMS dataset downloaded from HuggingFace.
    
    Args:
        dataset_path: HuggingFace dataset path (e.g., "data/lpm/PUMS")
        n_particles: Number of particles (recommend ≤2000 based on memory tests)
        n_clusters: Number of mixture components
        batch_size: Batch size per time step
        rejuvenation_steps: Number of rejuvenation steps per time step
        alpha_pi: Dirichlet prior for mixing weights
        alpha_theta: Dirichlet prior for emission parameters
        seed: Random seed
        output_dir: Directory to save results
        save_every: Save checkpoint every N time steps
    """
    # Load data from HuggingFace (keep in CPU memory)
    X_train_cpu, X_test_cpu, mask, col_names = load_pums_data_from_huggingface(dataset_path)
    N_train, D, K = X_train_cpu.shape
    C = n_clusters
    P = n_particles
    if n_time_steps is None:
        n_time_steps = N_train // batch_size
    
    print(f"\nExperiment configuration:")
    print(f"  Particles: {P}")
    print(f"  Clusters: {C}")
    print(f"  Time steps: {n_time_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Rejuvenation steps: {rejuvenation_steps}")
    print(f"  Total data points: {N_train}")
    print(f"  Expected memory usage: ~{P * 4.5:.0f} MB")
    
    # Validate memory usage
    if P > 2000:
        print("WARNING: Particle count > 2000 may cause OOM based on memory tests!")
    
    # Initialize random keys
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, P + 1)
    key, particle_keys = keys[0], keys[1:]
    
    # Initialize particles
    print("\nInitializing particles...")
    start_time = time.time()
    
    A, φ, π, θ = jax.vmap(
        lambda k: init_empty(k, C, D, K, N_train, alpha_pi, alpha_theta, mask)
    )(particle_keys)
    
    # Initialize SMC state
    log_weights = jnp.zeros(P)
    log_gammas = jnp.zeros(P)
    
    init_time = time.time() - start_time
    print(f"Initialization completed in {init_time:.2f}s")
    
    # Training history
    times = []
    
    # Training loop
    print(f"\nStarting SMC training...")
    training_start = time.time()
    particle_checkpoints = []
    
    for t in tqdm(range(n_time_steps), desc="SMC Steps"):
        step_start = time.time()
        
        # Create batch for this time step (sample from CPU data, move to GPU)
        start_idx = (t * batch_size) % N_train
        end_idx = start_idx + batch_size
        
        # Handle wraparound
        I_B = jnp.arange(start_idx, end_idx)
        X_B_cpu = X_train_cpu[start_idx:end_idx]  # numpy slice
       
        # Move batch to GPU
        X_B = jnp.array(X_B_cpu)
        
        # SMC step
        key, subkey = jax.random.split(key)
        particles = (A, φ, π, θ)
        particles, log_weights, log_gammas, batch_log_liks = smc_step(
            subkey, particles, log_weights, log_gammas, X_B, I_B, alpha_pi, alpha_theta, mask
        )
        A, φ, π, θ = particles
        
        # Rejuvenation steps
        if rejuvenation_steps > 0:
            for _ in range(rejuvenation_steps):
                key, subkey = jax.random.split(key)
                I_rejuv = jax.random.choice(subkey, N_train, shape=(batch_size,), replace=False)
                # Sample rejuvenation batch from CPU data, move to GPU
                X_rejuv_cpu = X_train_cpu[np.array(I_rejuv)]  # numpy indexing
                X_rejuv = jnp.array(X_rejuv_cpu)  # move to GPU
                
                # Run Gibbs step for each particle
                key, subkey = jax.random.split(key)
                keys = jax.random.split(subkey, P)
                
                A, φ, π, θ, _, _, _ = jax.vmap(gibbs, in_axes=(0, None, None, 0, 0, 0, 0, None, None, None))(
                    keys, X_rejuv, I_rejuv, A, φ, π, θ, alpha_pi, alpha_theta, mask)
        
        particles = (A, φ, π, θ)
        step_time = time.time() - step_start
        times.append(step_time)

        if t % save_every == 0:
            particle_checkpoints.append(particles)
        
    total_time = time.time() - training_start
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    
    # Final results
    results = {
        'final_particles': particles,
        'particle_checkpoints': particle_checkpoints,
        'final_log_weights': log_weights,
        'times': times,
        'total_time': total_time,
        'init_time': init_time,
        'config': {
            'dataset_path': dataset_path,
            'n_particles': n_particles,
            'n_clusters': n_clusters,
            'n_time_steps': n_time_steps,
            'batch_size': batch_size,
            'rejuvenation_steps': rejuvenation_steps,
            'alpha_pi': alpha_pi,
            'alpha_theta': alpha_theta,
            'seed': seed,
            'data_shape': (N_train, D, K),
            'col_names': col_names,
            'mask': mask,
        }
    }
    
    # Save final results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Extract dataset name from path
    dataset_name = dataset_path.split("/")[-1].lower()
    final_file = output_path / f"smc_{dataset_name}_final_P{P}_C{C}_T{n_time_steps}_B{batch_size}.pkl"
    
    with open(final_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Final results saved to {final_file}")
    
    # Print summary
    print(f"\nFinal Results:")
    print(f"  Average step time: {np.mean(times):.2f}s")
    print(f"  Total training time: {total_time/60:.2f} minutes")
    print(f"  Final log weights range: [{jnp.min(log_weights):.3f}, {jnp.max(log_weights):.3f}]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train SMC on full PUMS dataset from HuggingFace")
    parser.add_argument("--dataset", type=str, default="data/lpm/PUMD",
                       help="HuggingFace dataset path (default: data/lpm/PUMD)")
    parser.add_argument("--particles", "-P", type=int, default=5,
                       help="Number of particles (default: 5, max recommended: 2000)")
    parser.add_argument("--clusters", "-C", type=int, default=500,
                       help="Number of clusters (default: 500)")
    parser.add_argument("--batch-size", "-B", type=int, default=1000,
                       help="Batch size per time step (default: 1000)")
    parser.add_argument("--rejuvenation", "-R", type=int, default=10,
                       help="Rejuvenation steps per time step (default: 10)")
    parser.add_argument("--alpha-pi", type=float, default=1.0,
                       help="Dirichlet prior for mixing weights (default: 1.0)")
    parser.add_argument("--alpha-theta", type=float, default=1.0,
                       help="Dirichlet prior for emission parameters (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory (default: results)")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N steps (default: 10)")
    parser.add_argument("--n-time-steps", type=int, default=None,
                       help="Number of time steps (default: None)")
    args = parser.parse_args()
    
    # Run experiment
    run_smc_full_pums(
        dataset_path=args.dataset,
        n_particles=args.particles,
        n_clusters=args.clusters,
        batch_size=args.batch_size,
        rejuvenation_steps=args.rejuvenation,
        alpha_pi=args.alpha_pi,
        alpha_theta=args.alpha_theta,
        seed=args.seed,
        output_dir=args.output_dir,
        save_every=args.save_every,
        n_time_steps=args.n_time_steps
    )


if __name__ == "__main__":
    main()