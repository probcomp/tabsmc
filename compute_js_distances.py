#!/usr/bin/env python3
"""Compute JS distances between LMP samples and CES test set."""

import numpy as np
import polars as pl
import jax.numpy as jnp
from tabsmc.io import load_data
from tabsmc.distances import js
import jax
from scipy.spatial.distance import jensenshannon


def js_distance_categorical(X1, X2, batch_size=1000):
    """
    Compute JS distance between two datasets of categorical variables.
    
    Args:
        X1: First dataset (n_samples1, n_features)
        X2: Second dataset (n_samples2, n_features)
        batch_size: Batch size for computation
    
    Returns:
        JS distance (scalar)
    """
    n_features = X1.shape[1]
    js_distances = []
    
    for i in range(n_features):
        # Get unique categories for this feature
        cats1 = X1[:, i]
        cats2 = X2[:, i]
        
        # Get all unique categories
        all_cats = np.unique(np.concatenate([cats1, cats2]))
        
        # Compute empirical distributions
        p1 = np.array([np.mean(cats1 == cat) for cat in all_cats])
        p2 = np.array([np.mean(cats2 == cat) for cat in all_cats])
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p1 = p1 + epsilon
        p2 = p2 + epsilon
        
        # Normalize
        p1 = p1 / np.sum(p1)
        p2 = p2 / np.sum(p2)
        
        # Compute JS distance
        js_dist = jensenshannon(p1, p2)
        js_distances.append(js_dist)
    
    return np.mean(js_distances)


def compute_js_onehot():
    """Compute JS distance using one-hot encoded data."""
    print("Loading CES test data...")
    _, test_data_log, col_names, mask = load_data("data/lpm/CES")
    test_data = (test_data_log == 0.0).astype(np.float32)
    
    print("Loading processed LMP data...")
    data = np.load("/home/joaoloula/tabsmc/lpm_samples_onehot.npz")
    lmp_data = data["data"]
    
    print(f"CES test shape: {test_data.shape}")
    print(f"LMP data shape: {lmp_data.shape}")
    
    # Use smaller samples for faster computation
    n_samples = 2000
    test_subset = test_data[:n_samples]
    lmp_subset = lmp_data[:n_samples]
    
    # Reshape to 2D
    test_flat = test_subset.reshape(n_samples, -1)
    lmp_flat = lmp_subset.reshape(n_samples, -1)
    
    # Convert to boolean
    test_bool = test_flat.astype(bool)
    lmp_bool = lmp_flat.astype(bool)
    
    # Compute JS distance using existing function
    js_jit = jax.jit(js, static_argnames=("batch_size",))
    test_jax = jnp.array(test_bool)
    lmp_jax = jnp.array(lmp_bool)
    
    print(f"\nComputing JS distance with one-hot encoding...")
    distances = js_jit(lmp_jax, test_jax, batch_size=500)
    js_distance = float(jnp.mean(distances))
    
    return js_distance


def compute_js_categorical():
    """Compute JS distance using raw categorical data."""
    print("Loading CES test data (raw)...")
    from tabsmc.io import load_huggingface
    _, test_df_orig = load_huggingface("data/lpm/CES")
    
    print("Loading LMP samples (raw)...")
    lmp_samples_path = "/home/joaoloula/model-building-evaluation/results/lpm/CES/arf/samples-max-depth-200.parquet"
    lmp_df = pl.read_parquet(lmp_samples_path)
    
    # Drop replicate_id if it exists
    if "replicate_id" in lmp_df.columns:
        lmp_df = lmp_df.drop("replicate_id")
    
    print(f"CES test shape: {test_df_orig.shape}")
    print(f"LMP samples shape: {lmp_df.shape}")
    
    # Convert to numpy for easier processing
    # Handle mixed data types by converting to strings
    test_array = test_df_orig.to_numpy(use_pyarrow=False)
    lmp_array = lmp_df.to_numpy(use_pyarrow=False)
    
    # Convert all to strings for consistent comparison
    test_str = test_array.astype(str)
    lmp_str = lmp_array.astype(str)
    
    # Use smaller samples
    n_samples = 2000
    test_subset = test_str[:n_samples]
    lmp_subset = lmp_str[:n_samples]
    
    print(f"\nComputing JS distance with categorical data...")
    js_distance = js_distance_categorical(lmp_subset, test_subset)
    
    return js_distance


def main():
    """Compute JS distances using both approaches."""
    print("=== Computing JS distances ===\n")
    
    # Method 1: Using one-hot encoded data
    try:
        js_onehot = compute_js_onehot()
        print(f"JS distance (one-hot): {js_onehot:.6f}")
    except Exception as e:
        print(f"One-hot method failed: {e}")
        js_onehot = None
    
    print("\n" + "="*50 + "\n")
    
    # Method 2: Using raw categorical data
    try:
        js_categorical = compute_js_categorical()
        print(f"JS distance (categorical): {js_categorical:.6f}")
    except Exception as e:
        print(f"Categorical method failed: {e}")
        js_categorical = None
    
    print("\n=== Summary ===")
    if js_onehot is not None:
        print(f"One-hot JS distance: {js_onehot:.6f}")
    if js_categorical is not None:
        print(f"Categorical JS distance: {js_categorical:.6f}")
    
    return js_onehot, js_categorical


if __name__ == "__main__":
    js_onehot, js_categorical = main()