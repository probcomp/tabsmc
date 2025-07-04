#!/usr/bin/env python3
"""Compute JS distances vs time for all methods across all datasets."""

import numpy as np
import polars as pl
from pathlib import Path
from tabsmc.io import load_data, load_huggingface, make_schema
from tabsmc.distances import js_distance_categorical, onehot_to_categorical
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Dataset configurations
DATASET_CONFIGS = {
    'covertype': {
        'path': 'data/CTGAN/covertype',
        'timing_path': '../model-building-evaluation/results/CTGAN/covertype/timed-samples.parquet',
        'display_name': 'CoverType'
    },
    'kddcup': {
        'path': 'data/CTGAN/kddcup',
        'timing_path': '../model-building-evaluation/results/CTGAN/kddcup/timed-samples.parquet',
        'display_name': 'KDD Cup'
    },
    'sydt': {
        'path': 'data/CTGAN/sydt',
        'timing_path': '../model-building-evaluation/results/CTGAN/sydt/timed-samples.parquet',
        'display_name': 'Santander'
    },
    'ces': {
        'path': 'data/lpm/CES',
        'timing_path': '../model-building-evaluation/results/lpm/CES/timed-samples.parquet',
        'display_name': 'CES'
    },
    'pumd': {
        'path': 'data/lpm/PUMD',
        'timing_path': '../model-building-evaluation/results/lpm/PUMD/timed-samples.parquet',
        'display_name': 'PUMS (Domain Adapted)'
    },
    'pums': {
        'path': 'data/lpm/PUMS',
        'timing_path': '../model-building-evaluation/results/lpm/PUMS/timed-samples.parquet',
        'display_name': 'PUMS'
    }
}


def discretize_with_train_quantiles(df, train_df, schema, n_bins=20):
    """Discretize dataframe using quantiles from training data."""
    df_processed = df.clone()
    numerical_cols = schema["types"]["numerical"]
    
    for col in numerical_cols:
        if col in df.columns and col in train_df.columns:
            # Compute quantiles from training data
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = [train_df[col].quantile(q) for q in quantiles]
            bins = sorted(list(set(bins)))  # Remove duplicates
            
            # Apply to dataframe
            vals = df[col].to_numpy()
            binned = np.digitize(vals, bins)
            
            # Replace in dataframe
            df_processed = df_processed.with_columns(
                pl.Series(name=col, values=binned.astype(str))
            )
    
    return df_processed


def compute_js_for_timing_data(timing_df, test_df, train_df, schema, model_name, dataset_name, n_samples=3000):
    """Compute JS distances for all time points of a given model."""
    print(f"\n--- Processing {model_name} for {dataset_name} ---")
    
    model_data = timing_df.filter(pl.col('model') == model_name)
    time_points = model_data['time'].unique().sort()
    
    results = []
    
    for time_val in time_points:
        print(f"  Computing JS at time {time_val:.2f}s...")
        
        # Get synthetic data for this time point
        time_data = model_data.filter(pl.col('time') == time_val)
        synthetic_df = time_data.drop(['model', 'time', 'parameter', 'replicate'])
        
        if len(synthetic_df) < n_samples:
            actual_samples = len(synthetic_df)
        else:
            actual_samples = n_samples
            synthetic_df = synthetic_df.head(n_samples)
        
        try:
            # Discretize using train data quantiles
            synthetic_processed = discretize_with_train_quantiles(synthetic_df, train_df, schema)
            test_processed = discretize_with_train_quantiles(test_df.head(actual_samples), train_df, schema)
            
            # Convert to arrays and handle nulls
            synthetic_array = synthetic_processed.to_numpy().astype(str)
            test_array = test_processed.to_numpy().astype(str)
            
            # Handle None values
            synthetic_array = np.where(synthetic_array == 'None', 'missing', synthetic_array)
            test_array = np.where(test_array == 'None', 'missing', test_array)
            
            # Compute JS distance
            js_distance = js_distance_categorical(synthetic_array, test_array)
            
            results.append({
                'dataset': dataset_name,
                'method': model_name,
                'time': float(time_val),
                'js_distance': float(js_distance),
                'n_samples': actual_samples
            })
            
            print(f"    JS distance: {js_distance:.6f} (n={actual_samples})")
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    return results


def compute_smc_js_distances(dataset_name, dataset_path, n_samples=3000):
    """Compute JS distances for SMC synthetic data."""
    print(f"\n--- Processing SMC for {dataset_name} ---")
    
    # Load test data for comparison
    try:
        _, test_data_log, col_names, mask = load_data(dataset_path)
        test_data_onehot = (test_data_log == 0.0).astype(np.float32)
    except Exception as e:
        print(f"  Error loading test data: {e}")
        return []
    
    results = []
    
    # Find SMC synthetic files for this dataset
    synthetic_dir = Path("synthetic_data")
    smc_patterns = [
        f"smc_{dataset_name}_*_synthetic.pkl",
        f"{dataset_name}_*_synthetic.npz",
        f"synthetic_{dataset_name}_*.npz"
    ]
    
    synthetic_files = []
    for pattern in smc_patterns:
        synthetic_files.extend(synthetic_dir.glob(pattern))
    
    # Also check for specific known files
    if dataset_name == 'ces':
        known_files = [
            ("synthetic_data/ces_overfitted_synthetic.npz", "SMC_overfitted", 300.0),
            ("synthetic_data/synthetic_ces_near_converged_step36.npz", "SMC_converged", 600.0),
        ]
        for file_path, method_name, time_est in known_files:
            if Path(file_path).exists():
                synthetic_files.append(Path(file_path))
    
    for synthetic_file in synthetic_files:
        print(f"  Processing {synthetic_file.name}...")
        
        try:
            if synthetic_file.suffix == '.npz':
                # NPZ format
                smc_data = np.load(synthetic_file)
                X_synthetic = smc_data['X']
                
                # Estimate time based on filename or use default
                if 'overfitted' in synthetic_file.name:
                    time_est = 300.0
                    method_name = 'SMC_overfitted'
                elif 'converged' in synthetic_file.name:
                    time_est = 600.0
                    method_name = 'SMC_converged'
                else:
                    time_est = 400.0
                    method_name = 'SMC'
                
                # Convert to categorical
                synthetic_cat = onehot_to_categorical(X_synthetic[:n_samples], mask)
                test_cat = onehot_to_categorical(test_data_onehot[:n_samples], mask)
                
                # Convert to strings
                synthetic_str = synthetic_cat.astype(str)
                test_str = test_cat.astype(str)
                
                # Compute JS distance
                js_distance = js_distance_categorical(synthetic_str, test_str)
                
                results.append({
                    'dataset': dataset_name,
                    'method': method_name,
                    'time': time_est,
                    'js_distance': float(js_distance),
                    'n_samples': n_samples
                })
                
                print(f"    JS distance: {js_distance:.6f} at {time_est}s")
                
            elif synthetic_file.suffix == '.pkl':
                # Pickle format with timesteps
                with open(synthetic_file, 'rb') as f:
                    synthetic_data = pickle.load(f)
                
                if 'timesteps' in synthetic_data:
                    # Extract config from filename
                    parts = synthetic_file.stem.split('_')
                    config_str = ''
                    for part in parts:
                        if part.startswith('P') and part[1:].isdigit():
                            n_particles = int(part[1:])
                        elif part.startswith('C') and part[1:].isdigit():
                            n_clusters = int(part[1:])
                            config_str = f'SMC_P{n_particles}_C{n_clusters}'
                    
                    if not config_str:
                        config_str = 'SMC'
                    
                    # Process selected timesteps
                    timesteps_to_check = [0, len(synthetic_data['timesteps']) // 2, len(synthetic_data['timesteps']) - 1]
                    
                    for timestep_idx in timesteps_to_check:
                        if timestep_idx in synthetic_data['timesteps']:
                            timestep_data = synthetic_data['timesteps'][timestep_idx]
                            X_synthetic = timestep_data['data']
                            
                            # Estimate time
                            time_est = (timestep_idx + 1) * 2.0  # Rough estimate
                            
                            # Convert and compute JS
                            synthetic_cat = onehot_to_categorical(X_synthetic[:n_samples], mask)
                            test_cat = onehot_to_categorical(test_data_onehot[:n_samples], mask)
                            
                            synthetic_str = synthetic_cat.astype(str)
                            test_str = test_cat.astype(str)
                            
                            js_distance = js_distance_categorical(synthetic_str, test_str)
                            
                            results.append({
                                'dataset': dataset_name,
                                'method': f'{config_str}_step{timestep_idx}',
                                'time': time_est,
                                'js_distance': float(js_distance),
                                'n_samples': n_samples
                            })
                            
                            print(f"    JS distance at step {timestep_idx}: {js_distance:.6f}")
                
        except Exception as e:
            print(f"    Error processing {synthetic_file.name}: {e}")
            continue
    
    return results


def main():
    """Main function to compute JS distances for all datasets."""
    print("=== Computing JS Distances for All Datasets ===")
    
    all_results = []
    
    # Process each dataset
    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Processing {config['display_name']} ({dataset_name})")
        print(f"{'='*60}")
        
        dataset_path = config['path']
        timing_path = config['timing_path']
        
        # Try to load timing data for ARF, TVAE, Diffusion
        if Path(timing_path).exists():
            try:
                # Load data
                print(f"\nLoading data from {dataset_path}...")
                
                # Try different loading methods based on dataset type
                if 'lpm' in dataset_path:
                    train_df, test_df = load_huggingface(dataset_path)
                else:
                    # For CTGAN datasets, try loading directly
                    train_file = Path(dataset_path) / "train.csv"
                    test_file = Path(dataset_path) / "test.csv"
                    
                    if train_file.exists() and test_file.exists():
                        train_df = pl.read_csv(train_file)
                        test_df = pl.read_csv(test_file)
                    else:
                        print(f"  Warning: Could not find train/test files for {dataset_name}")
                        continue
                
                # Load timing data
                timing_df = pl.read_parquet(timing_path)
                
                # Create schema
                all_df = pl.concat([train_df, test_df], how="vertical")
                schema = make_schema(all_df)
                
                # Filter out nulls in test data
                test_df_clean = test_df.drop_nulls()
                
                print(f"  Train data: {train_df.shape}")
                print(f"  Test data: {test_df_clean.shape}")
                
                # Process timing-based methods
                models_in_timing = timing_df['model'].unique()
                print(f"  Models in timing data: {models_in_timing}")
                
                for model_name in models_in_timing:
                    model_results = compute_js_for_timing_data(
                        timing_df, test_df_clean, train_df, schema, model_name, dataset_name
                    )
                    all_results.extend(model_results)
                    
            except Exception as e:
                print(f"  Error processing timing data: {e}")
        else:
            print(f"  Warning: Timing data not found at {timing_path}")
        
        # Process SMC methods
        smc_results = compute_smc_js_distances(dataset_name, dataset_path)
        all_results.extend(smc_results)
    
    # Load existing results
    existing_results = []
    if Path('js_distance_vs_time_results.json').exists():
        with open('js_distance_vs_time_results.json', 'r') as f:
            existing_results = json.load(f)
        print(f"\n\nLoaded {len(existing_results)} existing results")
    
    # Merge results (keeping existing CES-only results for backward compatibility)
    # Add dataset field to existing results if not present
    for result in existing_results:
        if 'dataset' not in result:
            result['dataset'] = 'ces'  # Assume existing results are for CES
    
    # Combine all results
    combined_results = existing_results + all_results
    
    # Save extended results
    output_file = 'js_distance_vs_time_results_extended.json'
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n\nSaved {len(combined_results)} total results to {output_file}")
    
    # Print summary by dataset
    print("\n--- Summary by Dataset ---")
    datasets = set(r.get('dataset', 'unknown') for r in combined_results)
    
    for dataset in sorted(datasets):
        dataset_results = [r for r in combined_results if r.get('dataset', 'unknown') == dataset]
        if dataset_results:
            methods = set(r['method'] for r in dataset_results)
            print(f"\n{dataset}: {len(dataset_results)} results across {len(methods)} methods")
            
            # Best result per method
            for method in sorted(methods):
                method_results = [r for r in dataset_results if r['method'] == method]
                if method_results:
                    best_js = min(r['js_distance'] for r in method_results)
                    print(f"  {method}: Best JS = {best_js:.6f}")
    
    return combined_results


if __name__ == "__main__":
    results = main()