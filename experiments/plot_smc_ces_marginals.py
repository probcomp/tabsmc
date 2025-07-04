#!/usr/bin/env python3
"""
Plot 1D marginal comparisons between SMC CES timestep 0 and first 10k train datapoints.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tabsmc.io import load_data

def load_smc_data():
    """Load SMC CES timestep 0 synthetic data."""
    smc_file = Path("smc_synthetic_samples_clean/smc_ces_step_0_samples.parquet")
    if not smc_file.exists():
        raise FileNotFoundError(f"SMC file not found: {smc_file}")
    
    df_smc = pl.read_parquet(smc_file)
    print(f"SMC data shape: {df_smc.shape}")
    return df_smc.to_numpy()

def load_real_data():
    """Load first 10k train datapoints from CES dataset."""
    dataset_path = "data/lpm/CES"
    train_data, test_data, col_names, mask = load_data(dataset_path)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Using first 10k samples from train data")
    
    # Use first 10k samples from train data
    n_train_samples = min(10000, train_data.shape[0])
    train_sample = train_data[:n_train_samples]
    
    # Convert train data from one-hot back to categorical integers
    def convert_onehot_to_categorical(data, mask):
        """Convert one-hot encoded data back to categorical integers."""
        n_samples, n_features, max_cats = data.shape
        categorical_data = np.zeros((n_samples, n_features), dtype=np.int32)
        
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
    
    real_categorical = convert_onehot_to_categorical(train_sample, mask)
    return real_categorical, col_names

def compute_marginals(data):
    """Compute 1D marginals for each feature."""
    n_samples, n_features = data.shape
    marginals = []
    
    for i in range(n_features):
        feature_data = data[:, i]
        unique_vals, counts = np.unique(feature_data, return_counts=True)
        probabilities = counts / n_samples
        marginals.append((unique_vals, probabilities))
    
    return marginals

def plot_marginals_comparison(real_marginals, smc_marginals, col_names, max_features=16):
    """Plot side-by-side comparison of marginals."""
    n_features = min(len(real_marginals), len(smc_marginals), max_features)
    
    # Create subplot grid (4x4 for 16 features)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle('SMC CES Timestep 0 vs Train Data - 1D Marginals Comparison', fontsize=16)
    
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        
        # Get marginals for this feature
        real_vals, real_probs = real_marginals[i]
        smc_vals, smc_probs = smc_marginals[i]
        
        # Get all unique categories
        all_categories = np.unique(np.concatenate([real_vals, smc_vals]))
        
        # Create probability arrays for all categories
        real_prob_dict = dict(zip(real_vals, real_probs))
        smc_prob_dict = dict(zip(smc_vals, smc_probs))
        
        real_probs_full = [real_prob_dict.get(cat, 0) for cat in all_categories]
        smc_probs_full = [smc_prob_dict.get(cat, 0) for cat in all_categories]
        
        # Create bar plot
        x = np.arange(len(all_categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, real_probs_full, width, label='Data', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, smc_probs_full, width, label='Model', color='red', alpha=0.7)
        
        # Customize plot
        feature_name = col_names[i] if i < len(col_names) else f"Feature {i}"
        ax.set_title(feature_name, fontsize=10)
        ax.set_xlabel('Category', fontsize=8)
        ax.set_ylabel('Probability', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(cat) for cat in all_categories], fontsize=8)
        
        # Add legend to first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # Rotate x-axis labels if too many categories
        if len(all_categories) > 5:
            ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to create marginals comparison plot."""
    print("Loading SMC CES timestep 0 data...")
    smc_data = load_smc_data()
    
    print("Loading real CES train data...")
    real_data, col_names = load_real_data()
    
    print("Computing marginals...")
    real_marginals = compute_marginals(real_data)
    smc_marginals = compute_marginals(smc_data)
    
    print("Creating comparison plot...")
    fig = plot_marginals_comparison(real_marginals, smc_marginals, col_names)
    
    # Save plot
    output_file = "smc_ces_marginals_comparison.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show summary statistics
    print(f"\nSummary:")
    print(f"Real data shape: {real_data.shape}")
    print(f"SMC data shape: {smc_data.shape}")
    print(f"Number of features plotted: {min(len(real_marginals), len(smc_marginals), 16)}")
    
    # Compute some basic comparison metrics
    total_variation_distances = []
    for i in range(min(len(real_marginals), len(smc_marginals))):
        real_vals, real_probs = real_marginals[i]
        smc_vals, smc_probs = smc_marginals[i]
        
        # Get all categories and align probabilities
        all_categories = np.unique(np.concatenate([real_vals, smc_vals]))
        real_prob_dict = dict(zip(real_vals, real_probs))
        smc_prob_dict = dict(zip(smc_vals, smc_probs))
        
        real_probs_aligned = np.array([real_prob_dict.get(cat, 0) for cat in all_categories])
        smc_probs_aligned = np.array([smc_prob_dict.get(cat, 0) for cat in all_categories])
        
        # Compute total variation distance
        tv_distance = 0.5 * np.sum(np.abs(real_probs_aligned - smc_probs_aligned))
        total_variation_distances.append(tv_distance)
    
    print(f"\nMarginal quality (Total Variation Distance):")
    print(f"Mean TV distance: {np.mean(total_variation_distances):.4f}")
    print(f"Median TV distance: {np.median(total_variation_distances):.4f}")
    print(f"Min TV distance: {np.min(total_variation_distances):.4f}")
    print(f"Max TV distance: {np.max(total_variation_distances):.4f}")

if __name__ == "__main__":
    main()