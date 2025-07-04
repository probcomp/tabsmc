#!/usr/bin/env python
"""Plot and analyze SMC training results on PUMS dataset."""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def load_results(results_file):
    """Load results from pickle file."""
    with open(results_file, 'rb') as f:
        return pickle.load(f)


def plot_training_curves(results, output_file=None):
    """Plot training and test log-likelihood curves."""
    config = results['config']
    train_lls = results['train_log_liks']
    test_lls = results['test_log_liks']
    times = results['times']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Log-likelihood vs time steps
    steps = range(1, len(train_lls) + 1)
    axes[0, 0].plot(steps, train_lls, label='Train', linewidth=2, alpha=0.8)
    axes[0, 0].plot(steps, test_lls, label='Test', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Log-likelihood per data point')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Wall-clock time vs performance
    cumulative_time = np.cumsum(times) / 60  # Convert to minutes
    axes[0, 1].plot(cumulative_time, train_lls, label='Train', linewidth=2, alpha=0.8)
    axes[0, 1].plot(cumulative_time, test_lls, label='Test', linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Wall-clock time (minutes)')
    axes[0, 1].set_ylabel('Log-likelihood per data point')
    axes[0, 1].set_title('Performance vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Step times
    axes[1, 0].plot(steps, times, linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Step Time (seconds)')
    axes[1, 0].set_title('Computation Time per Step')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add horizontal line for average
    avg_time = np.mean(times)
    axes[1, 0].axhline(avg_time, color='red', linestyle='--', alpha=0.7,
                      label=f'Average: {avg_time:.2f}s')
    axes[1, 0].legend()
    
    # Plot 4: Learning curves (smoothed)
    window = max(1, len(train_lls) // 10)  # 10% smoothing window
    if len(train_lls) > window:
        train_smooth = np.convolve(train_lls, np.ones(window)/window, mode='valid')
        test_smooth = np.convolve(test_lls, np.ones(window)/window, mode='valid')
        smooth_steps = range(window, len(train_lls) + 1)
        
        axes[1, 1].plot(smooth_steps, train_smooth, label='Train (smoothed)', linewidth=2)
        axes[1, 1].plot(smooth_steps, test_smooth, label='Test (smoothed)', linewidth=2)
    else:
        axes[1, 1].plot(steps, train_lls, label='Train', linewidth=2)
        axes[1, 1].plot(steps, test_lls, label='Test', linewidth=2)
    
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Log-likelihood per data point')
    axes[1, 1].set_title('Smoothed Learning Curves')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add overall title with configuration
    fig.suptitle(
        f"SMC PUMS Training Results\n"
        f"P={config['n_particles']}, C={config['n_clusters']}, "
        f"T={config['n_time_steps']}, B={config['batch_size']}, "
        f"R={config['rejuvenation_steps']}, Seed={config['seed']}",
        fontsize=14
    )
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def compare_multiple_runs(results_files, output_file=None):
    """Compare multiple SMC runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_files)))
    
    for i, results_file in enumerate(results_files):
        results = load_results(results_file)
        config = results['config']
        train_lls = results['train_log_liks']
        test_lls = results['test_log_liks']
        times = results['times']
        
        label = f"P={config['n_particles']}, C={config['n_clusters']}"
        steps = range(1, len(train_lls) + 1)
        cumulative_time = np.cumsum(times) / 60
        
        # Training curves
        axes[0, 0].plot(steps, train_lls, label=label, color=colors[i], linewidth=2)
        axes[0, 1].plot(steps, test_lls, label=label, color=colors[i], linewidth=2)
        
        # Time-based curves
        axes[1, 0].plot(cumulative_time, train_lls, label=label, color=colors[i], linewidth=2)
        axes[1, 1].plot(cumulative_time, test_lls, label=label, color=colors[i], linewidth=2)
    
    # Configure plots
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Train LL per data point')
    axes[0, 0].set_title('Training Log-likelihood')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Test LL per data point')
    axes[0, 1].set_title('Test Log-likelihood')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Wall-clock time (minutes)')
    axes[1, 0].set_ylabel('Train LL per data point')
    axes[1, 0].set_title('Training Performance vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Wall-clock time (minutes)')
    axes[1, 1].set_ylabel('Test LL per data point')
    axes[1, 1].set_title('Test Performance vs Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('SMC PUMS Comparison', fontsize=14)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {output_file}")
    else:
        plt.show()


def print_summary(results):
    """Print summary statistics."""
    config = results['config']
    train_lls = results['train_log_liks']
    test_lls = results['test_log_liks']
    times = results['times']
    total_time = results['total_time']
    
    print("=" * 60)
    print("SMC PUMS TRAINING SUMMARY")
    print("=" * 60)
    
    print(f"Configuration:")
    print(f"  Particles: {config['n_particles']}")
    print(f"  Clusters: {config['n_clusters']}")
    print(f"  Time steps: {config['n_time_steps']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Rejuvenation steps: {config['rejuvenation_steps']}")
    print(f"  Seed: {config['seed']}")
    
    print(f"\nResults:")
    print(f"  Initial train LL/dp: {train_lls[0]:.4f}")
    print(f"  Final train LL/dp: {train_lls[-1]:.4f}")
    print(f"  Train improvement: {train_lls[-1] - train_lls[0]:.4f}")
    
    print(f"  Initial test LL/dp: {test_lls[0]:.4f}")
    print(f"  Final test LL/dp: {test_lls[-1]:.4f}")
    print(f"  Test improvement: {test_lls[-1] - test_lls[0]:.4f}")
    
    print(f"\nTiming:")
    print(f"  Total training time: {total_time/60:.2f} minutes")
    print(f"  Average step time: {np.mean(times):.2f} seconds")
    print(f"  Std step time: {np.std(times):.2f} seconds")
    print(f"  Min step time: {np.min(times):.2f} seconds")
    print(f"  Max step time: {np.max(times):.2f} seconds")
    
    # Estimate throughput
    total_data_processed = config['n_time_steps'] * config['batch_size'] * (1 + config['rejuvenation_steps'])
    throughput = total_data_processed / total_time
    print(f"  Data throughput: {throughput:.1f} data points/second")


def main():
    parser = argparse.ArgumentParser(description="Plot SMC PUMS training results")
    parser.add_argument("command", choices=["single", "compare"],
                       help="Plotting mode")
    parser.add_argument("files", nargs="+",
                       help="Result file(s) to plot")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file path")
    parser.add_argument("--summary", action="store_true",
                       help="Print summary statistics")
    
    args = parser.parse_args()
    
    if args.command == "single":
        if len(args.files) != 1:
            print("Error: Single mode requires exactly one file")
            return
        
        results = load_results(args.files[0])
        
        if args.summary:
            print_summary(results)
        
        plot_training_curves(results, args.output)
    
    elif args.command == "compare":
        if len(args.files) < 2:
            print("Error: Compare mode requires at least 2 files")
            return
        
        compare_multiple_runs(args.files, args.output)


if __name__ == "__main__":
    main()