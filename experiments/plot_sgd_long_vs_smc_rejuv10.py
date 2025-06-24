"""Create a focused wall-clock time comparison plot: SGD (300 steps) vs SMC with 10 rejuvenation steps."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_optimal_likelihood import compute_optimal_loglik_per_datapoint
from generate_synthetic_data import create_test_parameters


def load_results(filename):
    """Load results from pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def compute_step_averages(results, T_total=None):
    """
    Compute average time and likelihood for each step across runs.
    """
    # Find max steps if not specified
    if T_total is None:
        T_total = max(len(result['times']) for result in results)
    
    # Group results by step
    step_data = {i: {'times': [], 'logliks': []} for i in range(T_total)}
    
    for result in results:
        times = result['times']
        logliks = result['logliks']
        
        for i in range(min(len(times), T_total)):
            step_data[i]['times'].append(times[i])
            step_data[i]['logliks'].append(logliks[i])
    
    # Compute averages for each step
    avg_times = []
    avg_logliks = []
    
    for i in range(T_total):
        if step_data[i]['times']:
            avg_times.append(np.mean(step_data[i]['times']))
            avg_logliks.append(np.mean(step_data[i]['logliks']))
    
    return np.array(avg_times), np.array(avg_logliks)


def main():
    """Create focused comparison plot."""
    # Parameters
    C, D, K = 2, 5, 3
    
    # Compute optimal log-likelihood
    true_π, true_θ = create_test_parameters(C, D, K)
    optimal_loglik = float(compute_optimal_loglik_per_datapoint(true_π, true_θ))
    
    # Load results for SGD (long) and SMC with 10 rejuvenation steps
    print("Loading results...")
    results_files = {
        'sgd_long': 'data/sgd_baseline_long_results.pkl',
        'rejuv_10': 'data/smc_rejuv_10_results.pkl'
    }
    
    all_results = {}
    for method, filename in results_files.items():
        if os.path.exists(filename):
            all_results[method] = load_results(filename)
            print(f"  Loaded {method}: {len(all_results[method])} runs")
        else:
            print(f"  WARNING: {filename} not found")
            if method == 'sgd_long':
                print(f"  Please run: uv run python experiments/run_sgd_baseline_long.py")
            else:
                print(f"  Please run: uv run python experiments/run_smc_rejuv_10.py")
    
    if len(all_results) < 2:
        print("\nBoth result files are needed for comparison!")
        return
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Method configurations
    method_configs = {
        'sgd_long': {
            'color': 'orange',
            'label': 'SGD Baseline (Adam, 300 steps)',
            'marker': 'X',
            'linewidth': 3.5,
            'markersize': 10
        },
        'rejuv_10': {
            'color': 'green',
            'label': 'SMC with 10 Rejuvenation Steps (P=20, 30 steps)',
            'marker': '^',
            'linewidth': 3.5,
            'markersize': 10
        }
    }
    
    # Plot both methods
    for method, results in all_results.items():
        config = method_configs[method]
        
        # Compute average time and likelihood for each step
        avg_times, avg_logliks = compute_step_averages(results)
        
        # Plot individual runs with low alpha
        for result in results:
            plt.plot(result['times'], result['logliks'], 
                    color=config['color'], alpha=0.2, linewidth=1)
        
        # Plot mean line
        plt.plot(avg_times, avg_logliks, 
                color=config['color'], label=config['label'], 
                linewidth=config['linewidth'], alpha=0.9)
        
        # Add markers at regular time intervals
        if method == 'sgd_long':
            # More markers for longer run
            time_intervals = [10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
        else:
            time_intervals = [5, 10, 15, 20, 25, 30, 40, 50, 60]
            
        for t_mark in time_intervals:
            if len(avg_times) > 0 and t_mark <= avg_times[-1]:
                idx = np.argmin(np.abs(avg_times - t_mark))
                plt.plot(avg_times[idx], avg_logliks[idx], 
                        config['marker'], color=config['color'], 
                        markersize=config['markersize'], markeredgewidth=1.5,
                        markeredgecolor='white')
    
    # Plot optimal log-likelihood
    plt.axhline(y=optimal_loglik, color='black', linestyle='--', linewidth=2.5,
                label=f'Optimal (True Distribution = {optimal_loglik:.3f})', alpha=0.8)
    
    # Formatting
    plt.xlabel('Wall-Clock Time (seconds)', fontsize=16)
    plt.ylabel('Test Log-Likelihood per Datapoint', fontsize=16)
    plt.title('SGD (300 steps) vs SMC with 10 Rejuvenation Steps: Wall-Clock Time', fontsize=18)
    plt.legend(fontsize=14, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    all_logliks = []
    all_times = []
    for results in all_results.values():
        for result in results:
            all_logliks.extend(result['logliks'])
            all_times.extend(result['times'])
    
    if all_logliks and all_times:
        y_min = min(all_logliks)
        y_max = max(optimal_loglik + 0.01, max(all_logliks))
        plt.ylim(y_min - 0.05, y_max + 0.05)
        
        # Set x-axis limit to show full comparison
        plt.xlim(0, max(all_times) * 1.05)
    
    # Increase tick label size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'figures/sgd_long_vs_smc_rejuv10_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Optimal test log-likelihood per datapoint: {optimal_loglik:.4f}")
    
    for method, results in all_results.items():
        config = method_configs[method]
        avg_times, avg_logliks = compute_step_averages(results)
        
        if len(avg_times) > 0:
            final_time = avg_times[-1]
            final_loglik = avg_logliks[-1]
            gap_from_optimal = optimal_loglik - final_loglik
            
            print(f"\n{config['label']}:")
            print(f"  Final: {final_loglik:.4f} at {final_time:.2f}s")
            print(f"  Gap from optimal: {gap_from_optimal:.4f}")
            
            # Find time to reach specific performance thresholds
            for threshold_gap in [0.1, 0.05, 0.02, 0.01]:
                threshold_loglik = optimal_loglik - threshold_gap
                if any(avg_logliks >= threshold_loglik):
                    time_to_threshold = avg_times[np.argmax(avg_logliks >= threshold_loglik)]
                    print(f"  Time to reach {threshold_gap:.2f} gap: {time_to_threshold:.2f}s")
            
            # Print number of steps
            if method == 'sgd_long':
                print(f"  Total gradient steps: {len(avg_times)}")
            elif method == 'rejuv_10':
                print(f"  Total SMC steps: {len(avg_times)}")
                print(f"  Total rejuvenation steps: {len(avg_times) * 10}")
            
            if len(avg_times) > 1:
                total_improvement = avg_logliks[-1] - avg_logliks[0]
                efficiency = total_improvement / final_time if final_time > 0 else 0
                print(f"  Efficiency: {efficiency:.6f} loglik/second")


if __name__ == "__main__":
    main()