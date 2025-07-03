#!/usr/bin/env python3
"""Plot JS distances vs time for all methods: SMC, ARF, TVAE, TAB-DDPM."""

import numpy as np
import polars as pl
from pathlib import Path
from tabsmc.io import load_data, load_huggingface, make_schema
from tabsmc.distances import js_distance_categorical, onehot_to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


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


def compute_js_for_timing_data(timing_df, test_df, train_df, schema, model_name, n_samples=3000):
    """Compute JS distances for all time points of a given model."""
    print(f"\n--- Processing {model_name} ---")
    
    model_data = timing_df.filter(pl.col('model') == model_name)
    time_points = model_data['time'].unique().sort()
    
    results = []
    
    for time_val in time_points:
        print(f"Computing JS for {model_name} at time {time_val:.2f}s...")
        
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
                'method': model_name,
                'time': float(time_val),
                'js_distance': float(js_distance),
                'n_samples': actual_samples
            })
            
            print(f"  JS distance: {js_distance:.6f} (n={actual_samples})")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results


def compute_smc_js_distances():
    """Compute JS distances for SMC at different stages."""
    print("\n--- Processing SMC ---")
    
    # Load test data for comparison
    _, test_data_log, col_names, mask = load_data("data/lpm/CES")
    test_data_onehot = (test_data_log == 0.0).astype(np.float32)
    
    smc_results = []
    n_samples = 3000
    
    # SMC data points with estimated times
    smc_files = [
        ("synthetic_data/ces_overfitted_synthetic.npz", "SMC_overfitted", 300.0),
        ("synthetic_data/synthetic_ces_near_converged_step36.npz", "SMC_converged", 600.0),
    ]
    
    for file_path, name, time_est in smc_files:
        if Path(file_path).exists():
            print(f"Computing JS for {name} at time {time_est:.1f}s...")
            
            try:
                # Load SMC data
                smc_data = np.load(file_path)
                X_synthetic = smc_data['X']
                
                # Load mask (try from file, fallback to loading from data)
                if 'mask' in smc_data:
                    mask_smc = smc_data['mask']
                else:
                    _, _, _, mask_smc = load_data("data/lpm/CES")
                
                # Convert to categorical
                synthetic_cat = onehot_to_categorical(X_synthetic[:n_samples], mask_smc)
                test_cat = onehot_to_categorical(test_data_onehot[:n_samples], mask_smc)
                
                # Convert to strings
                synthetic_str = synthetic_cat.astype(str)
                test_str = test_cat.astype(str)
                
                # Compute JS distance
                js_distance = js_distance_categorical(synthetic_str, test_str)
                
                smc_results.append({
                    'method': name,
                    'time': time_est,
                    'js_distance': float(js_distance),
                    'n_samples': n_samples
                })
                
                print(f"  JS distance: {js_distance:.6f}")
                
            except Exception as e:
                print(f"  Error with {name}: {e}")
    
    return smc_results


def main():
    """Main function to compute JS distances vs time for all methods."""
    print("=== JS Distance vs Time Analysis ===")
    
    # Load data
    print("\nLoading data...")
    train_df, test_df = load_huggingface("data/lpm/CES")
    timing_df = pl.read_parquet('../model-building-evaluation/results/lpm/CES/timed-samples.parquet')
    
    # Create schema for discretization
    all_df = pl.concat([train_df, test_df], how="vertical")
    schema = make_schema(all_df)
    
    # Filter out nulls in test data
    test_df_clean = test_df.drop_nulls()
    
    print(f"Original test data: {test_df.shape}")
    print(f"Clean test data: {test_df_clean.shape}")
    
    # Collect all results
    all_results = []
    
    # Process timing-based methods
    models_in_timing = timing_df['model'].unique()
    print(f"\nModels in timing data: {models_in_timing}")
    
    for model_name in models_in_timing:
        model_results = compute_js_for_timing_data(
            timing_df, test_df_clean, train_df, schema, model_name
        )
        all_results.extend(model_results)
    
    # Process SMC methods
    smc_results = compute_smc_js_distances()
    all_results.extend(smc_results)
    
    # Create plot
    print(f"\n--- Creating Plot ---")
    if not all_results:
        print("No results to plot!")
        return
    
    # Convert to DataFrame for easier plotting
    results_df = pl.DataFrame(all_results)
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Define colors for each method
    method_colors = {
        'ARF': '#1f77b4',
        'Diffusion': '#ff7f0e', 
        'TVAE': '#2ca02c',
        'SMC_overfitted': '#d62728',
        'SMC_converged': '#9467bd'
    }
    
    # Plot each method
    methods = results_df['method'].unique()
    for method in methods:
        method_data = results_df.filter(pl.col('method') == method)
        times = method_data['time'].to_numpy()
        js_distances = method_data['js_distance'].to_numpy()
        
        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        js_distances = js_distances[sort_idx]
        
        color = method_colors.get(method, '#333333')
        plt.plot(times, js_distances, 'o-', label=method, color=color, 
                 linewidth=2, markersize=8)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Jensen-Shannon Distance', fontsize=12)
    plt.title('Distributional Fidelity (JS Distance) vs Training Time', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set log scale for x-axis if there's a wide range
    times_all = results_df['time'].to_numpy()
    if np.max(times_all) / np.min(times_all) > 100:
        plt.xscale('log')
    
    # Set reasonable y-axis limits
    js_all = results_df['js_distance'].to_numpy()
    plt.ylim(0, np.max(js_all) * 1.1)
    
    plt.tight_layout()
    plt.savefig('js_distance_vs_time_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    with open('js_distance_vs_time_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n--- Summary ---")
    print(f"Total evaluations: {len(all_results)}")
    
    for method in methods:
        method_results = [r for r in all_results if r['method'] == method]
        if method_results:
            best_js = min(r['js_distance'] for r in method_results)
            best_time = [r['time'] for r in method_results if r['js_distance'] == best_js][0]
            print(f"{method}: Best JS = {best_js:.6f} at {best_time:.1f}s")
    
    print(f"\nPlot saved as 'js_distance_vs_time_plot.png'")
    print(f"Results saved as 'js_distance_vs_time_results.json'")
    
    return all_results


if __name__ == "__main__":
    results = main()