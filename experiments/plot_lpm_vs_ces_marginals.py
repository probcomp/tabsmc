#!/usr/bin/env python3
"""Plot 1D marginals comparing LPM samples to CES test data."""

import jax
import jax.numpy as jnp
from tabsmc.io import load_data
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_marginals(X, mask=None):
    """Compute 1D marginal distributions for each feature."""
    N, D, K = X.shape
    marginals = []
    
    for d in range(D):
        if mask is not None:
            # Only consider valid categories
            valid_cats = int(jnp.sum(mask[d]))
            X_d = X[:, d, :valid_cats]
        else:
            X_d = X[:, d, :]
        
        # Convert from log-space if needed
        if jnp.any(X_d < -10):  # Detect log-space format
            X_d_prob = jnp.exp(X_d)
            X_d_prob = jnp.where(jnp.isfinite(X_d_prob), X_d_prob, 0.0)
        else:
            X_d_prob = X_d
        
        # Only average over valid (non-missing) rows
        # A row is valid if it has exactly one 1 (for one-hot data)
        row_sums = jnp.sum(X_d_prob, axis=1)
        valid_rows = jnp.abs(row_sums - 1.0) < 1e-6  # Allow small numerical errors
        
        if jnp.sum(valid_rows) > 0:
            # Compute marginal only over valid rows
            marginal = jnp.mean(X_d_prob[valid_rows], axis=0)
        else:
            # Fallback if no valid rows found
            marginal = jnp.mean(X_d_prob, axis=0)
        
        marginals.append(marginal)
    
    return marginals


def plot_lmp_vs_ces_marginals(lmp_marginals, ces_marginals, mask, col_names, save_path, features_to_plot=20):
    """Plot LPM vs CES marginal distributions."""
    
    # Create subplots
    n_cols = 5
    n_rows = (features_to_plot + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4 * n_rows))
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for feat_idx in range(min(features_to_plot, len(lmp_marginals))):
        ax = axes[feat_idx]
        
        # Get valid categories for this feature
        if mask is not None:
            valid_cats = int(jnp.sum(mask[feat_idx]))
            lmp_marg = lmp_marginals[feat_idx][:valid_cats]
            ces_marg = ces_marginals[feat_idx][:valid_cats]
        else:
            lmp_marg = lmp_marginals[feat_idx]
            ces_marg = ces_marginals[feat_idx]
            valid_cats = len(lmp_marg)
        
        # Plot both distributions
        x = np.arange(valid_cats)
        width = 0.35
        
        ax.bar(x - width/2, ces_marg, width, alpha=0.7, color='blue', 
               label='CES Test Data', edgecolor='darkblue', linewidth=0.5)
        ax.bar(x + width/2, lmp_marg, width, alpha=0.7, color='red', 
               label='LPM Samples', edgecolor='darkred', linewidth=0.5)
        
        # Feature name or index
        if col_names and feat_idx < len(col_names):
            feature_name = col_names[feat_idx]
            # Truncate long names
            if len(feature_name) > 20:
                feature_name = feature_name[:20] + "..."
        else:
            feature_name = f'Feature {feat_idx}'
        
        ax.set_title(f'{feature_name}', fontsize=10)
        ax.set_xlabel('Category', fontsize=8)
        ax.set_ylabel('Probability', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0
        max_val = max(np.max(ces_marg), np.max(lmp_marg))
        ax.set_ylim(0, max_val * 1.1)
        
        # Set x-axis limits and ticks
        ax.set_xlim(-0.5, valid_cats - 0.5)
        ax.set_xticks(x)
        
        # Rotate x-axis labels if there are many categories
        if valid_cats > 5:
            ax.tick_params(axis='x', rotation=45, labelsize=6)
        else:
            ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    # Hide unused subplots
    for i in range(features_to_plot, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('1D Marginal Distributions: LPM Samples vs CES Test Data', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Marginal comparison plot saved to {save_path}")


def main():
    """Main function to plot LPM vs CES marginals."""
    print("Loading CES test data...")
    # Load CES data
    _, test_data_log, col_names, mask = load_data("data/lpm/CES")
    
    # Convert from log-space to proper one-hot encoding
    test_data = (test_data_log == 0.0).astype(np.float32)
    
    print("Loading processed LPM data...")
    # Load processed LPM data
    data = np.load("/home/joaoloula/tabsmc/lpm_samples_onehot.npz")
    lmp_data = data["data"]
    
    # Convert mask to boolean and JAX array
    mask_bool = jnp.array(mask.astype(bool))
    
    print(f"CES test data shape: {test_data.shape}")
    print(f"LPM data shape: {lmp_data.shape}")
    
    # Use subset for faster computation if needed
    n_ces_samples = min(5000, test_data.shape[0])
    n_lmp_samples = min(5000, lmp_data.shape[0])
    
    ces_subset = test_data[:n_ces_samples]
    lmp_subset = lmp_data[:n_lmp_samples]
    
    print(f"Using {n_ces_samples} CES samples and {n_lmp_samples} LPM samples")
    
    # Compute marginals
    print("Computing CES test marginals...")
    ces_marginals = compute_marginals(ces_subset, mask_bool)
    
    print("Computing LPM marginals...")
    lmp_marginals = compute_marginals(lmp_subset, mask_bool)
    
    # Create output directory
    os.makedirs('figures', exist_ok=True)
    
    # Plot comparison
    n_features_to_plot = 20
    print(f"Plotting first {n_features_to_plot} features...")
    plot_lmp_vs_ces_marginals(
        lmp_marginals, ces_marginals, mask_bool, col_names,
        'figures/lmp_vs_ces_marginals.png',
        features_to_plot=n_features_to_plot
    )
    
    # Compute and print some summary statistics
    print("\nSummary Statistics:")
    print(f"  Total features: {len(ces_marginals)}")
    print(f"  Features plotted: {min(n_features_to_plot, len(ces_marginals))}")
    
    # Compute mean absolute differences for each feature
    mad_per_feature = []
    for i in range(len(ces_marginals)):
        if mask is not None:
            valid_cats = int(jnp.sum(mask_bool[i]))
            ces_marg = ces_marginals[i][:valid_cats]
            lmp_marg = lmp_marginals[i][:valid_cats]
        else:
            ces_marg = ces_marginals[i]
            lmp_marg = lmp_marginals[i]
        
        mad = np.mean(np.abs(ces_marg - lmp_marg))
        mad_per_feature.append(mad)
    
    overall_mad = np.mean(mad_per_feature)
    print(f"  Mean Absolute Difference (overall): {overall_mad:.6f}")
    print(f"  MAD std dev: {np.std(mad_per_feature):.6f}")
    print(f"  MAD range: [{np.min(mad_per_feature):.6f}, {np.max(mad_per_feature):.6f}]")
    
    # Find features with largest and smallest differences
    sorted_indices = np.argsort(mad_per_feature)
    print(f"\nFeatures with smallest MAD:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        feat_name = col_names[idx] if col_names and idx < len(col_names) else f"Feature {idx}"
        print(f"  {feat_name}: {mad_per_feature[idx]:.6f}")
    
    print(f"\nFeatures with largest MAD:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[-(i+1)]
        feat_name = col_names[idx] if col_names and idx < len(col_names) else f"Feature {idx}"
        print(f"  {feat_name}: {mad_per_feature[idx]:.6f}")


if __name__ == "__main__":
    main()