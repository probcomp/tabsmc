#!/usr/bin/env python3
"""
Compute JS distances for clean SMC synthetic samples against original datasets.
Compare each clean SMC synthetic sample against the original dataset.
"""

import polars as pl
import numpy as np
from pathlib import Path
from tabsmc.distances import js_distance_integer_matrices
import pickle
from tqdm import tqdm
import sys

def compute_clean_smc_js_distances():
    """
    Compute JS distances for clean SMC synthetic samples against original datasets.
    """
    # Find all clean SMC synthetic sample files
    samples_dir = Path("smc_synthetic_samples_clean")
    sample_files = sorted(list(samples_dir.glob("smc_*_step_*_samples.parquet")))
    
    print(f"Found {len(sample_files)} clean SMC synthetic sample files")
    
    # Group files by dataset to load original data only once per dataset
    dataset_files = {}
    for sample_file in sample_files:
        filename = sample_file.stem
        parts = filename.split('_')
        dataset = parts[1]  # e.g., "ces", "covertype", etc.
        timestep = int(parts[3])  # e.g., 0, 10, 100
        
        if dataset not in dataset_files:
            dataset_files[dataset] = []
        dataset_files[dataset].append((sample_file, timestep))
    
    print(f"Processing {len(dataset_files)} unique datasets: {list(dataset_files.keys())}")
    
    # Results list to store all JS distances
    results = []
    
    # Process each unique dataset
    for dataset, files_and_timesteps in dataset_files.items():
        print(f"\nProcessing dataset: {dataset}")
        
        try:
            # Determine dataset path and schema file based on dataset name
            # Map dataset names to their original paths
            dataset_mapping = {
                'ces': ('lpm', 'CES'),
                'pumd': ('lpm', 'PUMD'), 
                'pums': ('lpm', 'PUMS'),
                'covertype': ('CTGAN', 'covertype'),
                'kddcup': ('CTGAN', 'kddcup'),
                'sydt': ('CTGAN', 'sydt')
            }
            
            if dataset not in dataset_mapping:
                print(f"  Warning: Unknown dataset {dataset}, skipping...")
                continue
                
            method, dataset_name = dataset_mapping[dataset]
            
            print(f"  Method: {method}, Original dataset: {dataset_name}")
            
            # Load and process original data using load_data function (ONLY ONCE PER DATASET)
            from tabsmc.io import load_data
            
            if method == "CTGAN":
                original_dataset_path = f"data/CTGAN/{dataset_name}"
            else:  # lpm
                original_dataset_path = f"data/lpm/{dataset_name}"
            
            print(f"  Loading original data from {original_dataset_path}...")
            train_data, test_data, col_names, mask = load_data(original_dataset_path)
            
            print(f"  Test data shape: {test_data.shape}")
            
            # Use first 10k samples from test data to avoid memory issues
            n_test_samples = min(10000, test_data.shape[0])
            test_sample = test_data[:n_test_samples]
            print(f"  Using {n_test_samples} test samples")
            
            # Convert test data from one-hot back to categorical integers (ONLY ONCE PER DATASET)
            def convert_onehot_to_categorical(data, mask):
                """Convert one-hot encoded data back to categorical integers."""
                n_samples, n_features, max_cats = data.shape
                categorical_data = np.zeros((n_samples, n_features), dtype=np.int32)
                
                print(f"    Converting {n_samples} samples with {n_features} features...")
                
                for i in range(n_features):
                    # Get valid categories for this feature
                    valid_cats = mask[i].astype(bool)
                    n_valid = int(np.sum(valid_cats))
                    
                    # Get the category with max value for each sample (handling -inf)
                    feature_data = data[:, i, :n_valid]
                    # Replace -inf with very negative number for argmax
                    feature_data_clean = np.where(feature_data == -np.inf, -1e10, feature_data)
                    categorical_data[:, i] = np.argmax(feature_data_clean, axis=1)
                
                return categorical_data
            
            print(f"  Converting test data from one-hot to categorical...")
            original_categorical = convert_onehot_to_categorical(test_sample, mask)
            
            # Now process all timesteps for this dataset
            for sample_file, timestep in files_and_timesteps:
                print(f"\n    Processing timestep {timestep} for {dataset}...")
                
                # Load synthetic samples
                df_synthetic = pl.read_parquet(sample_file)
                print(f"    Synthetic data shape: {df_synthetic.shape}")
                
                # Convert to numpy matrix
                synthetic_matrix = df_synthetic.to_numpy()
                
                print(f"    Computing JS distance between {synthetic_matrix.shape} synthetic and {original_categorical.shape} original...")
                
                # Compute JS distance
                js_distance = js_distance_integer_matrices(synthetic_matrix, original_categorical)
                
                # Store result in same format as previous results
                result = {
                    'dataset': f"SMC_Clean/{dataset}",  # Mark as clean version
                    'method': 'SMC_Clean',
                    'dataset_name': dataset,
                    'model': 'smc_clean',  # Model type
                    'time': timestep,  # Use timestep as time
                    'parameter': 'P1',  # All files are P1
                    'replicate': 1,  # Always 1 as requested
                    'n_synthetic_samples': synthetic_matrix.shape[0],
                    'n_original_samples': original_categorical.shape[0],
                    'n_features': synthetic_matrix.shape[1],
                    'js_distance': js_distance
                }
                results.append(result)
                
                print(f"    JS distance: {js_distance:.6f}")
                
        except Exception as e:
            print(f"  Error processing dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert results to DataFrame and save
    print(f"\nCompleted! Total results: {len(results)}")
    
    if results:
        results_df = pl.DataFrame(results)
        
        # Save results
        output_file = "js_distances_smc_clean_results.parquet"
        results_df.write_parquet(output_file)
        print(f"Results saved to {output_file}")
        
        # Also save as CSV for easy viewing
        csv_file = "js_distances_smc_clean_results.csv"
        results_df.write_csv(csv_file)
        print(f"Results also saved to {csv_file}")
        
        # Show summary statistics
        print(f"\nSummary:")
        print(f"Total combinations: {len(results_df)}")
        print(f"Datasets: {sorted(results_df['dataset_name'].unique().to_list())}")
        print(f"Timesteps: {sorted(results_df['time'].unique().to_list())}")
        print(f"JS distance range: {results_df['js_distance'].min():.6f} to {results_df['js_distance'].max():.6f}")
        print(f"Mean JS distance: {results_df['js_distance'].mean():.6f}")
        print(f"Median JS distance: {results_df['js_distance'].median():.6f}")
        
        # Show per-dataset summary
        print(f"\nPer-dataset summary:")
        dataset_summary = results_df.group_by('dataset_name').agg([
            pl.col('js_distance').count().alias('n_timesteps'),
            pl.col('js_distance').mean().alias('mean_js'),
            pl.col('js_distance').median().alias('median_js'),
            pl.col('js_distance').min().alias('min_js'),
            pl.col('js_distance').max().alias('max_js')
        ]).sort('dataset_name')
        
        for row in dataset_summary.iter_rows(named=True):
            print(f"  {row['dataset_name']}: {row['n_timesteps']} timesteps, "
                  f"mean JS = {row['mean_js']:.4f}, median = {row['median_js']:.4f}")
        
        # Show per-timestep summary
        print(f"\nPer-timestep summary:")
        timestep_summary = results_df.group_by('time').agg([
            pl.col('js_distance').count().alias('n_datasets'),
            pl.col('js_distance').mean().alias('mean_js'),
            pl.col('js_distance').median().alias('median_js')
        ]).sort('time')
        
        for row in timestep_summary.iter_rows(named=True):
            print(f"  Timestep {row['time']}: {row['n_datasets']} datasets, "
                  f"mean JS = {row['mean_js']:.4f}, median = {row['median_js']:.4f}")
        
        # Compare with old results if available
        old_results_file = "js_distances_smc_results.parquet"
        if Path(old_results_file).exists():
            print(f"\n{'='*50}")
            print("COMPARISON WITH OLD SMC RESULTS")
            print(f"{'='*50}")
            
            old_results_df = pl.read_parquet(old_results_file)
            old_mean = old_results_df['js_distance'].mean()
            old_median = old_results_df['js_distance'].median()
            
            new_mean = results_df['js_distance'].mean()
            new_median = results_df['js_distance'].median()
            
            print(f"Old SMC results:")
            print(f"  Mean JS distance: {old_mean:.6f}")
            print(f"  Median JS distance: {old_median:.6f}")
            
            print(f"\nNew Clean SMC results:")
            print(f"  Mean JS distance: {new_mean:.6f}")
            print(f"  Median JS distance: {new_median:.6f}")
            
            print(f"\nImprovement:")
            mean_improvement = ((old_mean - new_mean) / old_mean) * 100
            median_improvement = ((old_median - new_median) / old_median) * 100
            print(f"  Mean JS improvement: {mean_improvement:.1f}%")
            print(f"  Median JS improvement: {median_improvement:.1f}%")
        
        return results_df
    else:
        print("No results to save!")
        return None


def main():
    """Main function with error handling."""
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        print("Starting Clean SMC JS distance computation...")
        print("This will compute JS distances for all clean SMC synthetic samples")
        print("against their corresponding original datasets.\n")
        
        # Run the computation
        results_df = compute_clean_smc_js_distances()
        
        if results_df is not None:
            print("\nFirst 10 results:")
            print(results_df.head(10))
            
            print(f"\nResults saved! You can view them in:")
            print(f"- js_distances_smc_clean_results.csv (for viewing)")
            print(f"- js_distances_smc_clean_results.parquet (for analysis)")
        else:
            print("No results generated!")
            
    except KeyboardInterrupt:
        print("\n\nComputation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError in main computation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()