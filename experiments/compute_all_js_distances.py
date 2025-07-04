#!/usr/bin/env python3
"""
Compute JS distances for every dataset and every model x time x replicate combination.
Compare each synthetic sample against the original dataset.
"""

import polars as pl
import numpy as np
from pathlib import Path
from tabsmc.distances import js_distance_integer_matrices
import pickle
from tqdm import tqdm
import sys

def compute_js_distances_all_datasets():
    """
    Compute JS distances for every dataset and every model x time x replicate combination.
    Compare each synthetic sample against the original dataset.
    """
    # Base paths
    results_base = Path("../model-building-evaluation/results")
    
    # Find all processed files
    processed_files = sorted(list(results_base.glob("**/timed-samples-processed.parquet")))
    
    print(f"Found {len(processed_files)} processed files")
    
    # Results list to store all JS distances
    results = []
    
    for processed_path in processed_files:
        # Extract dataset info
        parts = processed_path.parts
        method = parts[-3]  # e.g., "CTGAN" or "lpm"
        dataset = parts[-2]  # e.g., "covertype", "CES"
        dataset_name = f"{method}/{dataset}"
        
        print(f"\nProcessing {dataset_name}...")
        
        try:
            # Load processed synthetic data
            df_synthetic = pl.read_parquet(processed_path)
            print(f"  Synthetic data shape: {df_synthetic.shape}")
            
            # Load original data and process it the same way
            if method == "CTGAN":
                schema_file = f"schema_data_CTGAN_{dataset}.pkl"
            elif method == "lpm":
                schema_file = f"schema_data_lpm_{dataset}.pkl"
            else:
                print(f"  Warning: Unknown method {method}, skipping...")
                continue
                
            # Load and process original data
            from tabsmc.io import load_huggingface, discretize_with_schema, encode_with_schema
            
            if method == "CTGAN":
                original_dataset_path = f"data/CTGAN/{dataset}"
            else:  # lpm
                original_dataset_path = f"data/lpm/{dataset}"
            
            print(f"  Loading original data from {original_dataset_path}...")
            train_df, test_df = load_huggingface(original_dataset_path)
            df_original_full = test_df
            
            # Load schema and process original data
            with open(schema_file, "rb") as f:
                schema = pickle.load(f)
            
            # Process original data the same way as synthetic
            schema_columns = schema["types"]["categorical"] + schema["types"]["numerical"]
            df_original_schema = df_original_full.select(schema_columns)
            df_original_discretized = discretize_with_schema(df_original_schema, schema)
            df_original_encoded = encode_with_schema(df_original_discretized, schema)
            
            print(f"  Original data shape: {df_original_encoded.shape}")
            
            # Get data columns (exclude metadata)
            meta_columns = ['model', 'time', 'parameter', 'replicate']
            data_columns = [col for col in df_synthetic.columns if col not in meta_columns]
            
            # Convert original data to numpy
            original_matrix = df_original_encoded.select(data_columns).to_numpy()
            
            # Get unique combinations of model, time, parameter, replicate
            combinations = df_synthetic.select(meta_columns).unique().sort(['model', 'time', 'parameter', 'replicate'])
            print(f"  Found {len(combinations)} unique combinations")
            
            # Process each combination
            for i, row in enumerate(tqdm(combinations.iter_rows(named=True), 
                                       desc=f"  Computing JS for {dataset_name}", 
                                       total=len(combinations))):
                
                # Filter synthetic data for this combination
                synthetic_subset = df_synthetic.filter(
                    (pl.col('model') == row['model']) &
                    (pl.col('time') == row['time']) &
                    (pl.col('parameter') == row['parameter']) &
                    (pl.col('replicate') == row['replicate'])
                )
                
                if len(synthetic_subset) == 0:
                    print(f"    Warning: No data for combination {row}")
                    continue
                
                # Convert to numpy (only data columns)
                synthetic_matrix = synthetic_subset.select(data_columns).to_numpy()
                
                # Sample from original data if it's much larger
                original_sample = original_matrix[:10000]
               
                # Compute JS distance
                js_distance = js_distance_integer_matrices(synthetic_matrix, original_sample)
                
                # Store result
                result = {
                    'dataset': dataset_name,
                    'method': method,
                    'dataset_name': dataset,
                    'model': row['model'],
                    'time': row['time'],
                    'parameter': row['parameter'],
                    'replicate': row['replicate'],
                    'n_original_samples': original_sample.shape[0],
                    'n_features': synthetic_matrix.shape[1],
                    'js_distance': js_distance
                }
                results.append(result)
                
                # Print progress every 50 combinations
                if (i + 1) % 50 == 0:
                    print(f"    Processed {i+1}/{len(combinations)} combinations, latest JS: {js_distance:.6f}")
                
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert results to DataFrame and save
    print(f"\nCompleted! Total results: {len(results)}")
    
    if results:
        results_df = pl.DataFrame(results)
        
        # Save results
        output_file = "js_distances_all_results.parquet"
        results_df.write_parquet(output_file)
        print(f"Results saved to {output_file}")
        
        # Also save as CSV for easy viewing
        csv_file = "js_distances_all_results.csv"
        results_df.write_csv(csv_file)
        print(f"Results also saved to {csv_file}")
        
        # Show summary statistics
        print(f"\nSummary:")
        print(f"Total combinations: {len(results_df)}")
        print(f"Datasets: {sorted(results_df['dataset'].unique().to_list())}")
        print(f"Models: {sorted(results_df['model'].unique().to_list())}")
        print(f"JS distance range: {results_df['js_distance'].min():.6f} to {results_df['js_distance'].max():.6f}")
        print(f"Mean JS distance: {results_df['js_distance'].mean():.6f}")
        print(f"Median JS distance: {results_df['js_distance'].median():.6f}")
        
        # Show per-dataset summary
        print(f"\nPer-dataset summary:")
        dataset_summary = results_df.group_by('dataset').agg([
            pl.col('js_distance').count().alias('n_combinations'),
            pl.col('js_distance').mean().alias('mean_js'),
            pl.col('js_distance').median().alias('median_js'),
            pl.col('js_distance').min().alias('min_js'),
            pl.col('js_distance').max().alias('max_js')
        ]).sort('dataset')
        
        for row in dataset_summary.iter_rows(named=True):
            print(f"  {row['dataset']}: {row['n_combinations']} combinations, "
                  f"mean JS = {row['mean_js']:.4f}, median = {row['median_js']:.4f}")
        
        # Show per-model summary
        print(f"\nPer-model summary:")
        model_summary = results_df.group_by('model').agg([
            pl.col('js_distance').count().alias('n_combinations'),
            pl.col('js_distance').mean().alias('mean_js'),
            pl.col('js_distance').median().alias('median_js')
        ]).sort('model')
        
        for row in model_summary.iter_rows(named=True):
            print(f"  {row['model']}: {row['n_combinations']} combinations, "
                  f"mean JS = {row['mean_js']:.4f}, median = {row['median_js']:.4f}")
        
        return results_df
    else:
        print("No results to save!")
        return None


def main():
    """Main function with error handling."""
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        print("Starting comprehensive JS distance computation...")
        print("This will compute JS distances for ALL model x time x replicate combinations")
        print("across ALL datasets. This may take a while...\n")
        
        # Run the computation
        results_df = compute_js_distances_all_datasets()
        
        if results_df is not None:
            print("\nFirst 10 results:")
            print(results_df.head(10))
            
            print(f"\nResults saved! You can view them in:")
            print(f"- js_distances_all_results.csv (for viewing)")
            print(f"- js_distances_all_results.parquet (for analysis)")
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