"""Plot results from synthetic data experiments."""

import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_results(filename):
    """Load results from pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def plot_synthetic_results():
    """Create plot showing SMC performance on synthetic data."""
    # Load results
    results = load_results('data/smc_rejuv_10_synthetic_results.pkl')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot each run
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, result in enumerate(results):
        times = result['times']
        logliks = result['logliks']
        seed = result['seed']
        
        plt.plot(times, logliks, 
                color=colors[i % len(colors)], 
                alpha=0.7, 
                linewidth=1.5,
                label=f'Seed {seed}')
    
    # Calculate and plot mean
    max_time = max(result['times'][-1] for result in results)
    time_grid = np.linspace(0, max_time, 100)
    
    # Interpolate each run to common time grid
    interp_logliks = []
    for result in results:
        interp_loglik = np.interp(time_grid, result['times'], result['logliks'])
        interp_logliks.append(interp_loglik)
    
    mean_logliks = np.mean(interp_logliks, axis=0)
    std_logliks = np.std(interp_logliks, axis=0)
    
    plt.plot(time_grid, mean_logliks, 
             color='black', 
             linewidth=3, 
             label='Mean')
    
    plt.fill_between(time_grid, 
                     mean_logliks - std_logliks, 
                     mean_logliks + std_logliks,
                     color='gray', 
                     alpha=0.3,
                     label='±1 std')
    
    plt.xlabel('Wall Clock Time (seconds)')
    plt.ylabel('Test Log-Likelihood per Data Point')
    plt.title('SMC with 10 Rejuvenation Steps on Synthetic Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('figures/smc_synthetic_wallclock.png', dpi=300, bbox_inches='tight')
    print("Plot saved to figures/smc_synthetic_wallclock.png")
    
    # Print summary statistics
    final_logliks = [result['logliks'][-1] for result in results]
    final_times = [result['times'][-1] for result in results]
    
    print(f"\nSummary Statistics:")
    print(f"Final log-likelihood: {np.mean(final_logliks):.4f} ± {np.std(final_logliks):.4f}")
    print(f"Final time: {np.mean(final_times):.2f} ± {np.std(final_times):.2f} seconds")
    print(f"Number of runs: {len(results)}")

if __name__ == "__main__":
    plot_synthetic_results()