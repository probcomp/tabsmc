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
    """Load full PUMS dataset from HuggingFace using io.py."""
    from tabsmc.io import load_data
    
    print(f"Loading PUMS data from HuggingFace: {dataset_path}")
    print("This will download and preprocess the full dataset...")
    
    # Load data using the io module (returns numpy arrays)
    train_data, test_data, col_names, mask = load_data(dataset_path)
    
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


@jax.jit
def compute_test_loglik_vectorized(X_test, π, θ):
    """Compute test log-likelihood for a single particle."""
    # Use where to avoid 0 * (-inf) = NaN
    observed_logprobs = jnp.where(
        X_test[:, None, :, :] == 1,  # (N, 1, D, K)
        θ[None, :, :, :],            # (1, C, D, K)
        0.0
    )
    log_px_given_c = jnp.sum(observed_logprobs, axis=(2, 3))  # (N, C)
    log_px = jax.scipy.special.logsumexp(π[None, :] + log_px_given_c, axis=1)
    return jnp.sum(log_px)


def compute_weighted_test_loglik(X_test, particles, log_weights):
    """Compute weighted average test log-likelihood."""
    _, _, π, θ = particles
    
    # Compute log-likelihood for each particle
    log_liks = jax.vmap(compute_test_loglik_vectorized, in_axes=(None, 0, 0))(X_test, π, θ)
    
    # Weighted average
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
    return jax.scipy.special.logsumexp(log_liks + log_weights_normalized)


def run_smc_full_pums(
    dataset_path="data/lpm/PUMS",
    n_particles=1000,
    n_clusters=10,
    n_time_steps=50,
    batch_size=200,
    rejuvenation_steps=10,
    alpha_pi=1.0,
    alpha_theta=1.0,
    seed=42,
    output_dir="results",
    save_every=10
):
    """
    Run SMC on full PUMS dataset downloaded from HuggingFace.
    
    Args:
        dataset_path: HuggingFace dataset path (e.g., "data/lpm/PUMS")
        n_particles: Number of particles (recommend ≤2000 based on memory tests)
        n_clusters: Number of mixture components
        n_time_steps: Number of SMC time steps
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
    train_log_liks = []
    test_log_liks = []
    times = []
    checkpoint_times = []
    
    # Training loop
    print(f"\nStarting SMC training...")
    training_start = time.time()
    
    for t in tqdm(range(n_time_steps), desc="SMC Steps"):
        step_start = time.time()
        
        # Create batch for this time step (sample from CPU data, move to GPU)
        start_idx = (t * batch_size) % N_train
        end_idx = start_idx + batch_size
        
        # Handle wraparound
        if end_idx <= N_train:
            I_B = jnp.arange(start_idx, end_idx)
            X_B_cpu = X_train_cpu[start_idx:end_idx]  # numpy slice
        else:
            # Wraparound case
            I_B = jnp.concatenate([
                jnp.arange(start_idx, N_train),
                jnp.arange(0, end_idx - N_train)
            ])
            X_B_cpu = np.concatenate([
                X_train_cpu[start_idx:N_train],
                X_train_cpu[0:end_idx - N_train]
            ])
        
        # Move batch to GPU
        X_B = jnp.array(X_B_cpu)
        
        # SMC step
        key, subkey = jax.random.split(key)
        particles = (A, φ, π, θ)
        particles, log_weights, log_gammas = smc_step(
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
                
                def rejuvenate_particle(p_key, p_A, p_φ, p_π, p_θ):
                    A_new, φ_new, π_new, θ_new, _, _ = gibbs(
                        p_key, X_rejuv, I_rejuv, p_A, p_φ, p_π, p_θ, alpha_pi, alpha_theta, mask
                    )
                    return A_new, φ_new, π_new, θ_new
                
                A, φ, π, θ = jax.vmap(rejuvenate_particle)(keys, A, φ, π, θ)
        
        particles = (A, φ, π, θ)
        step_time = time.time() - step_start
        times.append(step_time)
        
        # Compute log-likelihoods (use smaller subsets to avoid GPU OOM)
        # For training: use a subset for evaluation
        train_eval_size = min(10000, N_train)  # Use 10k samples max for evaluation
        train_indices = np.random.choice(N_train, train_eval_size, replace=False)
        X_train_eval = jnp.array(X_train_cpu[train_indices])
        train_ll = compute_weighted_test_loglik(X_train_eval, particles, log_weights)
        
        # For test: use a subset if test set is large
        test_eval_size = min(5000, X_test_cpu.shape[0])
        test_indices = np.random.choice(X_test_cpu.shape[0], test_eval_size, replace=False)
        X_test_eval = jnp.array(X_test_cpu[test_indices])
        test_ll = compute_weighted_test_loglik(X_test_eval, particles, log_weights)
        
        # Check for NaN/inf values
        if jnp.isnan(train_ll) or jnp.isinf(train_ll):
            print(f"Warning: train_ll is {train_ll} at step {t+1}")
            print(f"  log_weights range: [{jnp.min(log_weights):.3f}, {jnp.max(log_weights):.3f}]")
        
        if jnp.isnan(test_ll) or jnp.isinf(test_ll):
            print(f"Warning: test_ll is {test_ll} at step {t+1}")
        
        train_log_liks.append(float(train_ll / train_eval_size))  # Per data point
        test_log_liks.append(float(test_ll / test_eval_size))
        
        # Print progress
        if (t + 1) % 5 == 0:
            avg_time = np.mean(times[-5:])
            remaining_time = (n_time_steps - t - 1) * avg_time
            print(f"Step {t+1}/{n_time_steps}: "
                  f"Train LL/dp={train_log_liks[-1]:.4f}, "
                  f"Test LL/dp={test_log_liks[-1]:.4f}, "
                  f"Time={step_time:.2f}s, "
                  f"ETA={remaining_time/60:.1f}min")
        
        # Save checkpoint
        if (t + 1) % save_every == 0:
            checkpoint_time = time.time()
            checkpoint = {
                'step': t + 1,
                'particles': particles,
                'log_weights': log_weights,
                'train_log_liks': train_log_liks,
                'test_log_liks': test_log_liks,
                'times': times,
                'config': {
                    'n_particles': n_particles,
                    'n_clusters': n_clusters,
                    'n_time_steps': n_time_steps,
                    'batch_size': batch_size,
                    'rejuvenation_steps': rejuvenation_steps,
                    'alpha_pi': alpha_pi,
                    'alpha_theta': alpha_theta,
                    'seed': seed,
                }
            }
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            checkpoint_file = output_path / f"smc_pums_checkpoint_step_{t+1}.pkl"
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            checkpoint_times.append(time.time() - checkpoint_time)
            print(f"  Checkpoint saved to {checkpoint_file} ({checkpoint_times[-1]:.2f}s)")
    
    total_time = time.time() - training_start
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    
    # Final results
    results = {
        'final_particles': particles,
        'final_log_weights': log_weights,
        'train_log_liks': train_log_liks,
        'test_log_liks': test_log_liks,
        'times': times,
        'total_time': total_time,
        'init_time': init_time,
        'checkpoint_times': checkpoint_times,
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
        }
    }
    
    # Save final results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_file = output_path / f"smc_pums_final_P{P}_C{C}_T{n_time_steps}_B{batch_size}.pkl"
    
    with open(final_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Final results saved to {final_file}")
    
    # Print summary
    print(f"\nFinal Results:")
    print(f"  Final train LL/dp: {train_log_liks[-1]:.4f}")
    print(f"  Final test LL/dp: {test_log_liks[-1]:.4f}")
    print(f"  Average step time: {np.mean(times):.2f}s")
    print(f"  Total training time: {total_time/60:.2f} minutes")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train SMC on full PUMS dataset from HuggingFace")
    parser.add_argument("--dataset", type=str, default="data/lpm/PUMS",
                       help="HuggingFace dataset path (default: data/lpm/PUMS)")
    parser.add_argument("--particles", "-P", type=int, default=1000,
                       help="Number of particles (default: 1000, max recommended: 2000)")
    parser.add_argument("--clusters", "-C", type=int, default=10,
                       help="Number of clusters (default: 10)")
    parser.add_argument("--time-steps", "-T", type=int, default=50,
                       help="Number of time steps (default: 50)")
    parser.add_argument("--batch-size", "-B", type=int, default=200,
                       help="Batch size per time step (default: 200)")
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.particles > 2000:
        print("WARNING: Particle count > 2000 may cause GPU OOM!")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Run experiment
    run_smc_full_pums(
        dataset_path=args.dataset,
        n_particles=args.particles,
        n_clusters=args.clusters,
        n_time_steps=args.time_steps,
        batch_size=args.batch_size,
        rejuvenation_steps=args.rejuvenation,
        alpha_pi=args.alpha_pi,
        alpha_theta=args.alpha_theta,
        seed=args.seed,
        output_dir=args.output_dir,
        save_every=args.save_every
    )


if __name__ == "__main__":
    main()