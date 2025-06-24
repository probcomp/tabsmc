"""Collect results from separate runs and create a Gibbs steps plot with proper averaging."""

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


def compute_step_averages(results, T_total):
    """
    Compute average Gibbs steps and likelihood for each step across runs.
    
    This addresses the issue where different runs may have different patterns
    for the same step, leading to noisy plots.
    
    Note: T_total should account for the fact that timestep 0 has been removed.
    """
    # Group results by step (remembering that timestep 0 has been removed)
    step_data = {i: {'gibbs_steps': [], 'logliks': []} for i in range(T_total)}
    
    for result in results:
        gibbs_steps = result['gibbs_steps']
        logliks = result['logliks']
        
        for i in range(min(len(gibbs_steps), T_total)):
            step_data[i]['gibbs_steps'].append(gibbs_steps[i])
            step_data[i]['logliks'].append(logliks[i])
    
    # Compute averages for each step
    avg_gibbs_steps = []
    avg_logliks = []
    
    for i in range(T_total):
        if step_data[i]['gibbs_steps']:
            avg_gibbs_steps.append(np.mean(step_data[i]['gibbs_steps']))
            avg_logliks.append(np.mean(step_data[i]['logliks']))
    
    return np.array(avg_gibbs_steps), np.array(avg_logliks)


def main():
    """Load all results and create the Gibbs steps plot."""
    # Parameters
    T = 30
    C, D, K = 2, 5, 3
    
    # Compute optimal log-likelihood
    true_π, true_θ = create_test_parameters(C, D, K)
    optimal_loglik = float(compute_optimal_loglik_per_datapoint(true_π, true_θ))
    
    # Load results from all methods
    print("Loading results...")
    results_files = {
        'no_rejuvenation': 'data/smc_no_rejuvenation_gibbs_results.pkl',
        'rejuv_1': 'data/smc_rejuv_1_gibbs_results.pkl',
        'rejuv_10': 'data/smc_rejuv_10_gibbs_results.pkl',
        'smc_then_rejuv': 'data/smc_then_rejuv_gibbs_results.pkl',
        # 'sgd_baseline': 'data/sgd_baseline_gibbs_results.pkl'
    }
    
    all_results = {}
    for method, filename in results_files.items():
        if os.path.exists(filename):
            all_results[method] = load_results(filename)
            print(f"  Loaded {method}: {len(all_results[method])} runs")
        else:
            print(f"  WARNING: {filename} not found")
    
    if not all_results:
        print("No results found! Please run the individual Gibbs steps scripts first:")
        for script in ['run_smc_no_rejuvenation_gibbs.py', 'run_smc_rejuv_1_gibbs.py', 
                       'run_smc_rejuv_10_gibbs.py', 'run_smc_then_rejuv_gibbs.py', 'run_sgd_baseline_gibbs.py']:
            print(f"  python {script}")
        return
    
    # Create plot
    plt.figure(figsize=(16, 8))
    
    # Method configurations (T-1 because timestep 0 is removed)
    method_configs = {
        'no_rejuvenation': {
            'color': 'blue',
            'label': '[SMC step] x 30 (P=20)',
            'marker': 'o',
            'T_total': T - 1  # timestep 0 removed
        },
        'rejuv_1': {
            'color': 'red',
            'label': '[SMC step + Rejuvenation step] x 30 (P=20)',
            'marker': 's',
            'T_total': T - 1  # timestep 0 removed
        },
        'rejuv_10': {
            'color': 'green',
            'label': '[SMC step + [Rejuvenation step] x 10] x 30 (P=20)',
            'marker': '^',
            'T_total': T - 1  # timestep 0 removed
        },
        'smc_then_rejuv': {
            'color': 'purple',
            'label': f'[SMC step] x 30 + [Rejuvenation step] x 30 (P=20)',
            'marker': 'D',
            'T_total': 2 * T - 1  # timestep 0 removed (only from first phase)
        },
        # 'sgd_baseline': {
        #     'color': 'orange',
        #     'label': 'SGD Baseline (Adam)',
        #     'marker': 'X',
        #     'T_total': T - 1  # timestep 0 removed
        # }
    }
    
    # Compute step-wise averages and plot
    for method, results in all_results.items():
        config = method_configs[method]
        
        # Compute average Gibbs steps and likelihood for each step
        avg_gibbs_steps, avg_logliks = compute_step_averages(results, config['T_total'])
        
        # Plot individual runs with low alpha
        for result in results:
            plt.plot(result['gibbs_steps'], result['logliks'], 
                    color=config['color'], alpha=0.3, linewidth=1.5)
        
        # Plot mean line
        plt.plot(avg_gibbs_steps, avg_logliks, 
                color=config['color'], label=config['label'], 
                linewidth=3, alpha=0.9)
        
        # Add markers at regular Gibbs step intervals
        gibbs_intervals = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300]
        for g_mark in gibbs_intervals:
            if len(avg_gibbs_steps) > 0 and g_mark <= avg_gibbs_steps[-1]:
                idx = np.argmin(np.abs(avg_gibbs_steps - g_mark))
                plt.plot(avg_gibbs_steps[idx], avg_logliks[idx], 
                        config['marker'], color=config['color'], markersize=8)
    
    # Add phase separator for SMC-then-rejuv method
    if 'smc_then_rejuv' in all_results:
        # Find average Gibbs steps at end of SMC phase (step T-2, since we removed timestep 0)
        avg_gibbs_steps_smc_rejuv, _ = compute_step_averages(
            all_results['smc_then_rejuv'], 2 * T - 1
        )
        # The SMC phase ends at step T-2 (originally T-1, but we removed step 0)
        if len(avg_gibbs_steps_smc_rejuv) > T - 2:
            smc_phase_end_gibbs = avg_gibbs_steps_smc_rejuv[T - 2]
            plt.axvline(x=smc_phase_end_gibbs, color='black', linestyle=':', 
                       linewidth=2, alpha=0.7, label='SMC → Rejuvenation-Only')
    
    # Plot optimal log-likelihood
    plt.axhline(y=optimal_loglik, color='black', linestyle='--', linewidth=2.5,
                label=f'Optimal (True Distribution = {optimal_loglik:.3f})', alpha=0.8)
    
    # Formatting
    plt.xlabel('Number of Gibbs Steps', fontsize=14)
    plt.ylabel('Test Log-Likelihood per Datapoint', fontsize=14)
    plt.title('Multi-Particle SMC Comparison: Gibbs Steps (10 runs, timestep 0 removed)', fontsize=16)
    plt.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    all_logliks = []
    for results in all_results.values():
        for result in results:
            all_logliks.extend(result['logliks'])
    
    if all_logliks:
        y_min = min(all_logliks)
        y_max = max(optimal_loglik + 0.01, max(all_logliks))
        plt.ylim(y_min - 0.05, y_max + 0.05)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'figures/multi_particle_smc_gibbs_steps_10runs_no_t0.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    
    # Print summary statistics
    print("\nPerformance Summary (10 runs after discarding first run and timestep 0):")
    print(f"  Optimal test log-likelihood per datapoint: {optimal_loglik:.4f}")
    
    for method, results in all_results.items():
        config = method_configs[method]
        avg_gibbs_steps, avg_logliks = compute_step_averages(results, config['T_total'])
        
        if len(avg_gibbs_steps) > 0:
            final_gibbs_steps = avg_gibbs_steps[-1]
            final_loglik = avg_logliks[-1]
            gap_from_optimal = optimal_loglik - final_loglik
            
            print(f"\n  {config['label']}:")
            print(f"    Final: {final_loglik:.4f} at {final_gibbs_steps:.0f} Gibbs steps")
            print(f"    Gap from optimal: {gap_from_optimal:.4f}")
            
            if len(avg_gibbs_steps) > 1:
                total_improvement = avg_logliks[-1] - avg_logliks[0]
                efficiency = total_improvement / final_gibbs_steps if final_gibbs_steps > 0 else 0
                print(f"    Efficiency: {efficiency:.6f} per Gibbs step")


if __name__ == "__main__":
    main()