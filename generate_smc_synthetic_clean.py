#!/usr/bin/env python3
"""
Generate synthetic data from SMC models using the clean tabsmc.query module.
Generate 10k samples for timesteps 0, 10, and 100 from all SMC checkpoint files.
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import polars as pl
from tabsmc.query import sample_batch

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

def get_checkpoint_at_timestep(smc_data, target_timestep):
    """Get the SMC checkpoint at or near the target timestep."""
    config = smc_data['config']
    n_timesteps = config['n_time_steps']
    save_every = 10  # Default value used in SMC training
    
    # Get available checkpoints
    has_checkpoints = 'particle_checkpoints' in smc_data and len(smc_data['particle_checkpoints']) > 0
    
    if not has_checkpoints:
        print(f"    No checkpoints available, using final particles only")
        if target_timestep <= n_timesteps:
            return smc_data['final_particles'], n_timesteps
        else:
            return None, None
    
    checkpoints = smc_data['particle_checkpoints']
    
    if target_timestep == 0:
        # Use first checkpoint
        if len(checkpoints) > 0 and checkpoints[0] is not None:
            return checkpoints[0], 0
        else:
            return None, None
            
    elif target_timestep <= n_timesteps:
        # Calculate the checkpoint index
        target_idx = target_timestep // save_every
        
        if target_idx < len(checkpoints) and checkpoints[target_idx] is not None:
            return checkpoints[target_idx], target_idx * save_every
        else:
            # Find closest available checkpoint
            available_indices = [i for i, cp in enumerate(checkpoints) if cp is not None]
            if available_indices:
                closest_idx = min(available_indices, key=lambda x: abs(x * save_every - target_timestep))
                actual_timestep = closest_idx * save_every
                print(f"    Using closest checkpoint at timestep {actual_timestep}")
                return checkpoints[closest_idx], actual_timestep
            else:
                return None, None
    else:
        # Use final particle for timesteps beyond the training
        return smc_data['final_particles'], n_timesteps

def sample_from_smc_checkpoint(checkpoint, n_samples, key):
    """Sample from an SMC checkpoint using tabsmc.query."""
    A, phi, pi, theta = checkpoint
    
    # Get dimensions
    n_particles, n_clusters = pi.shape
    _, _, n_features, n_categories = theta.shape
    
    print(f"    Checkpoint shape: {n_particles} particles, {n_clusters} clusters, {n_features} features, {n_categories} max categories")
    
    # Sample a random particle to use
    particle_key, sample_key = jax.random.split(key)
    particle_idx = jax.random.choice(particle_key, n_particles)
    
    # Extract parameters for the selected particle
    particle_pi = pi[particle_idx]  # (n_clusters,)
    particle_theta = theta[particle_idx]  # (n_clusters, n_features, n_categories)
    
    # Sample using the clean query interface
    samples = sample_batch(particle_theta, particle_pi, sample_key, n_samples)
    
    return samples

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
    
    print(f"  Dataset: {dataset_name}")
    print(f"  Features: {len(col_names)}")
    print(f"  Total timesteps: {n_timesteps}")
    
    results = {}
    key = jax.random.PRNGKey(42)
    
    for target_step in target_timesteps:
        print(f"  Generating samples for timestep {target_step}...")
        
        # Get checkpoint for this timestep
        checkpoint, actual_timestep = get_checkpoint_at_timestep(smc_data, target_step)
        
        if checkpoint is None:
            print(f"    No checkpoint available for timestep {target_step}, skipping...")
            continue
        
        try:
            # Generate samples using clean query interface
            key, subkey = jax.random.split(key)
            samples = sample_from_smc_checkpoint(checkpoint, n_samples, subkey)
            
            print(f"    Generated {samples.shape[0]} samples (actual timestep {actual_timestep})")
            
            # Convert to DataFrame with proper column names
            # Only use the number of features we actually have
            n_actual_features = len(col_names)
            samples_subset = samples[:, :n_actual_features]
            
            # Convert JAX array to numpy for Polars compatibility
            samples_numpy = np.array(samples_subset)
            
            df = pl.DataFrame(samples_numpy, schema=col_names)
            
            # Save samples
            output_dir = Path("smc_synthetic_samples_clean")
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"smc_{dataset_name}_step_{target_step}_samples.parquet"
            df.write_parquet(output_file)
            
            results[target_step] = {
                'file': output_file,
                'shape': samples_subset.shape,
                'samples': df,
                'actual_timestep': actual_timestep
            }
            
            print(f"    Saved to {output_file}")
            
        except Exception as e:
            print(f"    Error generating samples for timestep {target_step}: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def main():
    """Main function to process all SMC files."""
    print("Generating synthetic data from SMC models using tabsmc.query...")
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
    print("All synthetic data saved in 'smc_synthetic_samples_clean/' directory")

if __name__ == "__main__":
    main()