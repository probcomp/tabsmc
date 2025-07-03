#!/usr/bin/env python3
"""Compute JS distances properly with correct categorical conversions."""

import numpy as np
import polars as pl
from pathlib import Path
from tabsmc.io import load_data, load_huggingface, make_schema, discretize_dataframe
from tabsmc.distances import js_distance_categorical, onehot_to_categorical
import json


def compute_smc_js_distance(n_samples=5000):
    """Compute JS distance for SMC synthetic data (one-hot to categorical)."""
    print("\n=== SMC JS Distance Computation ===")
    
    # Load SMC synthetic data
    print("Loading SMC near-converged synthetic data...")
    smc_data = np.load("synthetic_data/synthetic_ces_near_converged_step36.npz")
    X_synthetic = smc_data['X']  # Shape: (10000, 88, 51)
    
    print(f"SMC data shape: {X_synthetic.shape}")
    
    # Load test data in one-hot format
    print("Loading test data in one-hot format...")
    _, test_data_log, col_names, mask = load_data("data/lpm/CES")
    test_data_onehot = (test_data_log == 0.0).astype(np.float32)
    
    print(f"Test data shape: {test_data_onehot.shape}")
    
    # Convert both to categorical
    print("Converting one-hot to categorical...")
    synthetic_cat = onehot_to_categorical(X_synthetic[:n_samples], mask)
    test_cat = onehot_to_categorical(test_data_onehot[:n_samples], mask)
    
    print(f"Synthetic categorical shape: {synthetic_cat.shape}")
    print(f"Test categorical shape: {test_cat.shape}")
    
    # Convert to strings for JS computation
    synthetic_str = synthetic_cat.astype(str)
    test_str = test_cat.astype(str)
    
    # Compute JS distance
    print("Computing JS distance...")
    js_distance = js_distance_categorical(synthetic_str, test_str)
    
    print(f"SMC JS distance: {js_distance:.6f}")
    
    return js_distance, synthetic_cat, test_cat


def compute_arf_js_distance(test_cat, n_samples=5000):
    """Compute JS distance for ARF data with proper discretization."""
    print("\n=== ARF JS Distance Computation ===")
    
    # Load raw train and test data for discretization
    print("Loading raw CES data...")
    train_df, test_df = load_huggingface("data/lpm/CES")
    
    # Load ARF samples
    print("Loading ARF synthetic data...")
    arf_path = Path("../model-building-evaluation/results/lpm/CES/arf/samples-max-depth-200.parquet")
    if not arf_path.exists():
        print(f"Error: ARF samples not found at {arf_path}")
        return None, None
    
    arf_df = pl.read_parquet(str(arf_path))
    if "replicate_id" in arf_df.columns:
        arf_df = arf_df.drop("replicate_id")
    
    print(f"ARF data shape: {arf_df.shape}")
    
    # Use same discretization process as SMC training data
    print("Discretizing data using train data ventiles...")
    
    # Combine train and test for consistent schema
    all_df = pl.concat([train_df, test_df], how="vertical")
    schema = make_schema(all_df)
    
    # Process ARF data with same discretization as training
    print("Processing ARF data...")
    
    # Get categorical and numerical columns
    categorical_cols = schema["types"]["categorical"]
    numerical_cols = schema["types"]["numerical"]
    
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")
    
    # For numerical columns, compute ventiles from training data
    n_bins = 20  # Same as in load_data
    arf_processed = arf_df.clone()
    test_processed = test_df.clone()
    
    # Discretize numerical columns using train data quantiles
    for col in numerical_cols:
        if col in arf_df.columns and col in test_df.columns:
            # Compute quantiles from training data
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = [train_df[col].quantile(q) for q in quantiles]
            bins = sorted(list(set(bins)))  # Remove duplicates
            
            # Apply to both ARF and test data
            arf_vals = arf_df[col].to_numpy()
            test_vals = test_df[col].to_numpy()
            
            # Digitize (convert to bin indices)
            arf_binned = np.digitize(arf_vals, bins)
            test_binned = np.digitize(test_vals, bins)
            
            # Replace in dataframes
            arf_processed = arf_processed.with_columns(
                pl.Series(name=col, values=arf_binned.astype(str))
            )
            test_processed = test_processed.with_columns(
                pl.Series(name=col, values=test_binned.astype(str))
            )
    
    # Take subsets
    arf_subset = arf_processed.head(n_samples)
    test_subset = test_processed.head(n_samples)
    
    # Convert to numpy arrays for JS computation
    print("Converting to categorical arrays...")
    arf_array = arf_subset.to_numpy().astype(str)
    test_array_arf = test_subset.to_numpy().astype(str)
    
    # Handle any None values
    arf_array = np.where(arf_array == 'None', 'missing', arf_array)
    test_array_arf = np.where(test_array_arf == 'None', 'missing', test_array_arf)
    
    print(f"ARF categorical shape: {arf_array.shape}")
    print(f"Test categorical shape (for ARF): {test_array_arf.shape}")
    
    # Compute JS distance
    print("Computing JS distance...")
    js_distance = js_distance_categorical(arf_array, test_array_arf)
    
    print(f"ARF JS distance: {js_distance:.6f}")
    
    return js_distance, arf_array


def main():
    """Main function to compute and compare JS distances."""
    print("=== Proper JS Distance Computation ===")
    
    n_samples = 5000
    print(f"Using {n_samples} samples for comparison")
    
    # Compute SMC JS distance
    smc_js, synthetic_cat, test_cat = compute_smc_js_distance(n_samples)
    
    # Compute ARF JS distance
    arf_js, arf_array = compute_arf_js_distance(test_cat, n_samples)
    
    # Summary
    print("\n" + "="*50)
    print("JS DISTANCE COMPARISON")
    print("="*50)
    
    if smc_js is not None:
        print(f"SMC (near-converged) JS distance: {smc_js:.6f}")
    
    if arf_js is not None:
        print(f"ARF JS distance: {arf_js:.6f}")
    
    if smc_js is not None and arf_js is not None:
        print(f"\nDifference (ARF - SMC): {arf_js - smc_js:.6f}")
        better = "SMC" if smc_js < arf_js else "ARF"
        print(f"Better method (lower JS): {better}")
    
    # Additional analysis
    print(f"\n--- Additional Analysis ---")
    if synthetic_cat is not None and test_cat is not None:
        print(f"SMC synthetic unique values per feature (first 5):")
        for i in range(min(5, synthetic_cat.shape[1])):
            n_unique_syn = len(np.unique(synthetic_cat[:, i]))
            n_unique_test = len(np.unique(test_cat[:, i]))
            print(f"  Feature {i}: Synthetic={n_unique_syn}, Test={n_unique_test}")
    
    if arf_array is not None:
        print(f"\nARF synthetic unique values per feature (first 5):")
        for i in range(min(5, arf_array.shape[1])):
            n_unique_arf = len(np.unique(arf_array[:, i]))
            print(f"  Feature {i}: ARF={n_unique_arf}")
    
    # Save results
    results = {
        "smc_js_distance": float(smc_js) if smc_js is not None else None,
        "arf_js_distance": float(arf_js) if arf_js is not None else None,
        "n_samples": n_samples,
        "method": "proper_categorical_conversion"
    }
    
    with open("js_distances_proper.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to js_distances_proper.json")
    
    return results


if __name__ == "__main__":
    results = main()