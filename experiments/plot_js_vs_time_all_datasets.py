#!/usr/bin/env python3
"""Plot JS distances vs time for SMC across all datasets using new synthetic data."""

import numpy as np
import pickle
from pathlib import Path
from tabsmc.io import load_data
from tabsmc.distances import js_distance_categorical, onehot_to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Dataset configurations
DATASET_CONFIGS = {
    'pums': {'path': 'data/lpm/PUMS', 'display_name': 'PUMS'},
    'pumd': {'path': 'data/lpm/PUMD', 'display_name': 'PUMS (Domain Adapted)'},
    'ces': {'path': 'data/lpm/CES', 'display_name': 'CES'},
    'covertype': {'path': 'data/lpm/COVERTYPE', 'display_name': 'CoverType'},
    'kddcup': {'path': 'data/lpm/KDDCUP', 'display_name': 'KDD Cup'},
    'sydt': {'path': 'data/lpm/SYDT', 'display_name': 'Santander'}
}


def extract_dataset_name(filename):
    """Extract dataset name from filename."""
    # Handle patterns like smc_pums_final_P5_C500_T800_B1000_synthetic.pkl
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[1]  # e.g., 'pums', 'ces', etc.
    return None


def extract_smc_config(filename):
    """Extract SMC configuration from filename."""
    # Pattern: smc_{dataset}_final_P{P}_C{C}_T{T}_B{B}_synthetic.pkl
    config = {}
    parts = filename.split('_')
    
    for part in parts:
        if part.startswith('P') and part[1:].isdigit():
            config['n_particles'] = int(part[1:])
        elif part.startswith('C') and part[1:].isdigit():
            config['n_clusters'] = int(part[1:])
        elif part.startswith('T') and part[1:].isdigit():
            config['n_timesteps'] = int(part[1:])
        elif part.startswith('B') and part[1:].isdigit():
            config['batch_size'] = int(part[1:])
    
    return config


def compute_js_for_smc_file(synthetic_file, dataset_name, n_samples=3000):
    """Compute JS distances for different timesteps in an SMC synthetic file."""
    print(f"\nProcessing {synthetic_file.name}...")
    
    # Load synthetic data
    with open(synthetic_file, 'rb') as f:
        synthetic_data = pickle.load(f)
    
    # Load test data for this dataset
    if dataset_name not in DATASET_CONFIGS:
        print(f"  Unknown dataset: {dataset_name}")
        return []
    
    dataset_path = DATASET_CONFIGS[dataset_name]['path']
    try:
        _, test_data_log, col_names, mask = load_data(dataset_path)
        test_data_onehot = (test_data_log == 0.0).astype(np.float32)
    except Exception as e:
        print(f"  Error loading test data for {dataset_name}: {e}")
        print(f"  Skipping this dataset...")
        return []
    
    results = []
    config = extract_smc_config(synthetic_file.name)
    
    # Get timing information from the original SMC results if available
    original_filename = synthetic_file.name.replace("_synthetic.pkl", ".pkl")
    
    # Search for the original file in various locations
    possible_paths = [
        synthetic_file.parent.parent / "results" / original_filename,
        Path("results") / original_filename,
        Path("results/test") / original_filename,
        Path("results/test_int") / original_filename,
        Path("results/hf_test") / original_filename,
        Path("results/hf_full") / original_filename,
    ]
    
    times_dict = {}
    original_file = None
    
    for path in possible_paths:
        if path.exists():
            original_file = path
            break
    
    if original_file:
        try:
            with open(original_file, 'rb') as f:
                original_data = pickle.load(f)
                if 'times' in original_data:
                    # Cumulative times for each step
                    cumulative_times = np.cumsum(original_data['times'])
                    times_dict = {i: cumulative_times[i] for i in range(len(cumulative_times))}
                    if 'init_time' in original_data:
                        # Add initialization time
                        times_dict = {k: v + original_data['init_time'] for k, v in times_dict.items()}
                print(f"  Loaded timing data from {original_file}")
        except Exception as e:
            print(f"  Warning: Could not load timing data: {e}")
    else:
        print(f"  Warning: Could not find original file for timing data")
    
    # Process each timestep
    for timestep, timestep_data in synthetic_data['timesteps'].items():
        print(f"  Computing JS for timestep {timestep}...")
        
        try:
            # Get synthetic samples
            X_synthetic = timestep_data['data']
            
            # Limit samples
            actual_samples = min(n_samples, X_synthetic.shape[0], test_data_onehot.shape[0])
            X_synthetic = X_synthetic[:actual_samples]
            X_test = test_data_onehot[:actual_samples]
            
            # Convert to categorical
            synthetic_cat = onehot_to_categorical(X_synthetic, mask)
            test_cat = onehot_to_categorical(X_test, mask)
            
            # Convert to strings for JS computation
            synthetic_str = synthetic_cat.astype(str)
            test_str = test_cat.astype(str)
            
            # Compute JS distance
            js_distance = js_distance_categorical(synthetic_str, test_str)
            
            # Estimate time if not available
            if timestep in times_dict:
                time_val = float(times_dict[timestep])
            else:
                # Rough estimate based on timestep
                time_per_step = 2.0  # seconds, adjust based on your hardware
                time_val = timestep * time_per_step
            
            result = {
                'dataset': dataset_name,
                'display_name': DATASET_CONFIGS[dataset_name]['display_name'],
                'method': f'SMC_P{config.get("n_particles", "?")}_C{config.get("n_clusters", "?")}',
                'timestep': int(timestep),
                'time': time_val,
                'js_distance': float(js_distance),
                'n_samples': actual_samples,
                'config': config
            }
            
            results.append(result)
            print(f"    JS distance: {js_distance:.6f} (time: {time_val:.1f}s)")
            
        except Exception as e:
            print(f"    Error at timestep {timestep}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def plot_results(all_results, output_file='js_distance_vs_time_plot.png'):
    """Create plots for JS distance vs time."""
    if not all_results:
        print("No results to plot!")
        return
    
    # Convert to dict for easier access
    results_by_dataset = {}
    for result in all_results:
        dataset = result['dataset']
        if dataset not in results_by_dataset:
            results_by_dataset[dataset] = []
        results_by_dataset[dataset].append(result)
    
    # Create subplots for each dataset
    n_datasets = len(results_by_dataset)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    sns.set_style("whitegrid")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, (dataset, results) in enumerate(results_by_dataset.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Group by method
        method_results = {}
        for r in results:
            method = r['method']
            if method not in method_results:
                method_results[method] = {'times': [], 'js': []}
            method_results[method]['times'].append(r['time'])
            method_results[method]['js'].append(r['js_distance'])
        
        # Plot each method
        for i, (method, data) in enumerate(method_results.items()):
            times = np.array(data['times'])
            js_vals = np.array(data['js'])
            
            # Sort by time
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            js_vals = js_vals[sort_idx]
            
            ax.plot(times, js_vals, 'o-', label=method, color=colors[i % len(colors)],
                   linewidth=2, markersize=8)
        
        ax.set_xlabel('Time (seconds)', fontsize=11)
        ax.set_ylabel('Jensen-Shannon Distance', fontsize=11)
        ax.set_title(f'{DATASET_CONFIGS[dataset]["display_name"]}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Use log scale if needed
        if len(times) > 0 and np.max(times) / (np.min(times) + 1e-6) > 100:
            ax.set_xscale('log')
    
    # Hide empty subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('SMC Performance: JS Distance vs Training Time Across Datasets', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_file}'")
    
    # Also create a combined plot
    plt.figure(figsize=(12, 8))
    
    # Plot all datasets on one plot
    for dataset, results in results_by_dataset.items():
        # Group by method
        method_results = {}
        for r in results:
            method = r['method']
            if method not in method_results:
                method_results[method] = {'times': [], 'js': []}
            method_results[method]['times'].append(r['time'])
            method_results[method]['js'].append(r['js_distance'])
        
        # Plot best performing configuration
        best_method = None
        best_final_js = float('inf')
        
        for method, data in method_results.items():
            times = np.array(data['times'])
            js_vals = np.array(data['js'])
            
            # Get final JS value
            final_idx = np.argmax(times)
            final_js = js_vals[final_idx]
            
            if final_js < best_final_js:
                best_final_js = final_js
                best_method = method
        
        if best_method:
            times = np.array(method_results[best_method]['times'])
            js_vals = np.array(method_results[best_method]['js'])
            
            # Sort by time
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            js_vals = js_vals[sort_idx]
            
            plt.plot(times, js_vals, 'o-', label=DATASET_CONFIGS[dataset]["display_name"],
                    linewidth=2, markersize=8)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Jensen-Shannon Distance', fontsize=12)
    plt.title('SMC Performance Across Datasets (Best Configuration)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    combined_file = output_file.replace('.png', '_combined.png')
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved as '{combined_file}'")


def main():
    """Main function to compute JS distances for all SMC synthetic data."""
    print("=== SMC JS Distance vs Time Analysis ===")
    
    # Find all synthetic files
    synthetic_dir = Path("synthetic_data")
    synthetic_files = list(synthetic_dir.glob("smc_*_synthetic.pkl"))
    
    print(f"\nFound {len(synthetic_files)} synthetic data files")
    
    all_results = []
    
    # Process each file
    for synthetic_file in synthetic_files:
        dataset_name = extract_dataset_name(synthetic_file.name)
        if dataset_name:
            results = compute_js_for_smc_file(synthetic_file, dataset_name)
            all_results.extend(results)
    
    # Create plots
    if all_results:
        plot_results(all_results)
        
        # Save results
        with open('js_distance_vs_time_smc_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved as 'js_distance_vs_time_smc_results.json'")
        
        # Print summary
        print("\n--- Summary by Dataset ---")
        datasets = set(r['dataset'] for r in all_results)
        for dataset in sorted(datasets):
            dataset_results = [r for r in all_results if r['dataset'] == dataset]
            if dataset_results:
                best_js = min(r['js_distance'] for r in dataset_results)
                best_result = [r for r in dataset_results if r['js_distance'] == best_js][0]
                print(f"\n{DATASET_CONFIGS[dataset]['display_name']}:")
                print(f"  Best JS: {best_js:.6f}")
                print(f"  Method: {best_result['method']}")
                print(f"  Time: {best_result['time']:.1f}s")
                print(f"  Timestep: {best_result['timestep']}")
    
    return all_results


if __name__ == "__main__":
    results = main()