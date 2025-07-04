#!/usr/bin/env python3
"""
Generate synthetic data from SMC models for timesteps 0, 10, and 100.
Generate 10k samples for each timestep and dataset.
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import polars as pl

def sample_from_particles(particles, n_samples, key):
    """Sample synthetic data from SMC particles.
    
    Args:
        particles: Tuple (A, φ, π, θ) from SMC
        n_samples: Number of synthetic samples to generate
        key: JAX random key
        
    Returns:
        Synthetic data of shape (n_samples, D, K)
    """
    A, phi, pi, theta = particles
    
    # Handle different shapes based on the data
    if len(A.shape) == 2:
        n_particles, n_data = A.shape
    elif len(A.shape) == 3:
        n_particles, n_data, n_timesteps = A.shape
        # Use the last timestep for A
        A = A[:, :, -1]
    else:
        raise ValueError(f"Unexpected A shape: {A.shape}")
    
    if len(theta.shape) == 4:
        _, n_clusters, D, K = theta.shape
    else:
        raise ValueError(f"Unexpected theta shape: {theta.shape}")
    
    # Convert to JAX arrays if needed
    pi = jnp.array(pi)
    theta = jnp.array(theta)
    
    # IMPORTANT: SMC saves parameters in log-space, need to convert to probabilities
    # Check if parameters are in log-space (negative values)
    if jnp.any(pi < 0) or jnp.any(theta < 0):
        print("  Converting log-space parameters to probabilities...")
        # Convert from log-space to probability space
        pi = jax.nn.softmax(pi, axis=-1)  # Convert log mixture weights to probabilities
        theta = jax.nn.softmax(theta, axis=-1)  # Convert log emission params to probabilities
    
    # Sample which particle to use for each synthetic sample
    key1, key2 = jax.random.split(key)
    particle_indices = jax.random.choice(key1, n_particles, shape=(n_samples,))
    
    # Generate keys for each sample
    sample_keys = jax.random.split(key2, n_samples)
    
    # Use a simpler approach without nested vmap
    synthetic_data = []
    
    for i in range(n_samples):
        p_idx = int(particle_indices[i])
        sample_key = sample_keys[i]
        
        # Get mixture weights for this particle
        pi_p = pi[p_idx]  # Shape: (n_clusters,)
        
        # Sample cluster assignment
        key_c, key_f = jax.random.split(sample_key)
        cluster_idx = jax.random.choice(key_c, n_clusters, p=pi_p)
        
        # Get emission parameters for this cluster
        theta_c = theta[p_idx, cluster_idx]  # Shape: (D, K)
        
        # Sample features from categorical distributions
        keys_d = jax.random.split(key_f, D)
        
        # Sample each dimension
        sample_features = []
        for d in range(D):
            cat_idx = jax.random.choice(keys_d[d], K, p=theta_c[d])
            one_hot = jax.nn.one_hot(cat_idx, K)
            sample_features.append(one_hot)
        
        sample = jnp.stack(sample_features)
        synthetic_data.append(sample)
    
    synthetic_data = jnp.stack(synthetic_data)
    return synthetic_data

def find_smc_files():
    """Find all SMC P1 files."""
    results_dir = Path("results")
    smc_files = list(results_dir.glob("smc_*_P1_*.pkl"))
    return sorted(smc_files)

def extract_dataset_name(file_path):
    """Extract dataset name from file path."""
    # e.g., smc_pums_final_P1_C100_T800_B1000.pkl -> pums
    filename = file_path.name
    parts = filename.split('_')
    return parts[1]  # pums, ces, etc.

def load_smc_data(file_path):
    """Load SMC data from pickle file."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def convert_to_categorical(samples, col_names, mask):
    """Convert one-hot samples back to categorical format."""
    n_samples, n_features, max_categories = samples.shape
    
    categorical_data = np.zeros((n_samples, n_features), dtype=np.int32)
    
    for i in range(n_features):
        # Get valid categories for this feature
        valid_cats = mask[i]
        n_valid = int(np.sum(valid_cats))
        
        # Get the category with max probability for each sample
        feature_samples = samples[:, i, :n_valid]
        categorical_data[:, i] = np.argmax(feature_samples, axis=1)
    
    return categorical_data

def process_smc_file(file_path, target_timesteps=[0, 10, 100], n_samples=10000):
    """Process one SMC file and generate synthetic data for target timesteps."""
    print(f"\nProcessing {file_path.name}...")
    
    dataset_name = extract_dataset_name(file_path)
    smc_data = load_smc_data(file_path)
    
    # Get configuration
    config = smc_data['config']
    col_names = config['col_names']
    mask = np.array(smc_data['config']['mask'])
    n_timesteps = config['n_time_steps']
    save_every = 10  # Default value used in SMC training
    
    print(f"  Dataset: {dataset_name}")
    print(f"  Features: {len(col_names)}")
    print(f"  Total timesteps: {n_timesteps}")
    
    # Get available checkpoints
    has_checkpoints = 'particle_checkpoints' in smc_data and len(smc_data['particle_checkpoints']) > 0
    
    if not has_checkpoints:
        print(f"    No checkpoints available, using final particles only")
        return {}
    
    checkpoints = smc_data['particle_checkpoints']
    print(f"  Available checkpoints: {len(checkpoints)}")
    
    results = {}
    key = jax.random.PRNGKey(42)
    
    for target_step in target_timesteps:
        print(f"  Generating samples for timestep {target_step}...")
        
        particle_state = None
        actual_timestep = target_step
        
        if target_step == 0:
            # Use first checkpoint
            if len(checkpoints) > 0 and checkpoints[0] is not None:
                particle_state = checkpoints[0]
                actual_timestep = 0
            else:
                print(f"    No checkpoint available for timestep 0, skipping...")
                continue
                
        elif target_step <= n_timesteps:
            # Calculate the checkpoint index
            target_idx = target_step // save_every
            
            if target_idx < len(checkpoints) and checkpoints[target_idx] is not None:
                particle_state = checkpoints[target_idx]
                actual_timestep = target_idx * save_every
            else:
                # Find closest available checkpoint
                available_indices = [i for i, cp in enumerate(checkpoints) if cp is not None]
                if available_indices:
                    closest_idx = min(available_indices, key=lambda x: abs(x * save_every - target_step))
                    particle_state = checkpoints[closest_idx]
                    actual_timestep = closest_idx * save_every
                    print(f"    Using closest checkpoint at timestep {actual_timestep}")
                else:
                    print(f"    No suitable checkpoint found for timestep {target_step}, skipping...")
                    continue
        else:
            # Use final particle for timesteps beyond the training
            particle_state = smc_data['final_particles']
            actual_timestep = n_timesteps
            print(f"    Using final particle (timestep {actual_timestep})")
        
        if particle_state is None:
            print(f"    No particle state available for timestep {target_step}, skipping...")
            continue
        
        try:
            # Generate samples
            key, subkey = jax.random.split(key)
            samples = sample_from_particles(particle_state, n_samples, subkey)
            
            # Convert to categorical format
            categorical_data = convert_to_categorical(np.array(samples), col_names, mask)
            
            # Create DataFrame
            df = pl.DataFrame(categorical_data, schema=col_names)
            
            # Save samples
            output_dir = Path("smc_synthetic_samples")
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"smc_{dataset_name}_step_{target_step}_samples.parquet"
            df.write_parquet(output_file)
            
            results[target_step] = {
                'file': output_file,
                'shape': categorical_data.shape,
                'samples': df,
                'actual_timestep': actual_timestep
            }
            
            print(f"    Generated {categorical_data.shape[0]} samples (timestep {actual_timestep}), saved to {output_file}")
            
        except Exception as e:
            print(f"    Error generating samples for timestep {target_step}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def main():
    """Main function to process all SMC files."""
    print("Generating synthetic data from SMC models...")
    print("Target timesteps: 0, 10, 100")
    print("Samples per timestep: 10,000")
    
    # Find all SMC files
    smc_files = find_smc_files()
    print(f"\nFound {len(smc_files)} SMC files:")
    for f in smc_files:
        print(f"  {f.name}")
    
    all_results = {}
    
    # Process each file
    for smc_file in smc_files:
        dataset_name = extract_dataset_name(smc_file)
        try:
            results = process_smc_file(smc_file, target_timesteps=[0, 10, 100], n_samples=10000)
            all_results[dataset_name] = results
        except Exception as e:
            print(f"\nError processing {smc_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    
    total_files = 0
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        for timestep, result in results.items():
            print(f"  Timestep {timestep}: {result['shape']} -> {result['file']}")
            total_files += 1
    
    print(f"\nTotal files generated: {total_files}")
    print("All synthetic data saved in 'smc_synthetic_samples/' directory")

if __name__ == "__main__":
    main()