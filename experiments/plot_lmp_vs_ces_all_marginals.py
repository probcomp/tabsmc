#!/usr/bin/env python3
"""Plot 1D marginals for ALL features comparing LPM samples to CES test data."""

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


def plot_all_marginals(lmp_marginals, ces_marginals, mask, col_names, save_path):
    """Plot ALL marginal distributions with smaller subplots."""
    
    n_features = len(lmp_marginals)
    
    # Create subplots with smaller size for more features
    n_cols = 8  # More columns for compactness
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(32, 3 * n_rows))
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for feat_idx in range(n_features):
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
               label='CES' if feat_idx == 0 else '', edgecolor='darkblue', linewidth=0.3)
        ax.bar(x + width/2, lmp_marg, width, alpha=0.7, color='red', 
               label='LPM' if feat_idx == 0 else '', edgecolor='darkred', linewidth=0.3)
        
        # Feature name or index
        if col_names and feat_idx < len(col_names):
            feature_name = col_names[feat_idx]
            # Truncate long names
            if len(feature_name) > 15:
                feature_name = feature_name[:15] + "..."
        else:
            feature_name = f'F{feat_idx}'
        
        ax.set_title(f'{feature_name}', fontsize=7)
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, alpha=0.2)
        
        # Set y-axis to start from 0
        max_val = max(np.max(ces_marg), np.max(lmp_marg))
        ax.set_ylim(0, max_val * 1.1)
        
        # Set x-axis limits
        ax.set_xlim(-0.5, valid_cats - 0.5)
        
        # Remove x-axis labels for compactness except bottom row
        if feat_idx < n_features - n_cols:
            ax.set_xticklabels([])
        
        # Only show y-axis labels on leftmost column
        if feat_idx % n_cols != 0:
            ax.set_yticklabels([])
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    # Add legend at the top
    if n_features > 0:
        axes[0].legend(loc='upper right', fontsize=8)
    
    plt.suptitle('1D Marginal Distributions: LPM Samples vs CES Test Data (All Features)', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # Lower DPI for large plot
    print(f"All marginals plot saved to {save_path}")


def main():
    """Main function to plot all LPM vs CES marginals."""
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
    
    # Use subset for faster computation
    n_ces_samples = min(3000, test_data.shape[0])
    n_lmp_samples = min(3000, lmp_data.shape[0])
    
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
    
    # Plot all features
    print(f"Plotting all {len(ces_marginals)} features...")
    plot_all_marginals(
        lmp_marginals, ces_marginals, mask_bool, col_names,
        'figures/lmp_vs_ces_all_marginals.png'
    )
    
    print(f"Completed plotting all {len(ces_marginals)} features!")


if __name__ == "__main__":
    main()