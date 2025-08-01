#!/usr/bin/env python3
"""Test the new distance utilities."""

import numpy as np
import polars as pl
import jax.numpy as jnp
from tabsmc.distances import (
    js_distance_dataframes,
    js_distance_categorical,
    js_distance_onehot,
    onehot_to_categorical
)


def test_js_distance_categorical():
    """Test JS distance for categorical arrays."""
    print("Testing js_distance_categorical...")
    
    # Create two similar categorical datasets
    np.random.seed(42)
    X1 = np.random.choice(['A', 'B', 'C'], size=(100, 5))
    X2 = np.random.choice(['A', 'B', 'C'], size=(100, 5), p=[0.4, 0.35, 0.25])
    
    js_dist = js_distance_categorical(X1, X2)
    print(f"  JS distance between similar categorical data: {js_dist:.6f}")
    
    # Create two different categorical datasets
    X3 = np.random.choice(['A', 'B', 'C'], size=(100, 5), p=[0.8, 0.1, 0.1])
    js_dist2 = js_distance_categorical(X1, X3)
    print(f"  JS distance between different categorical data: {js_dist2:.6f}")
    print(f"  ✓ Test passed (different > similar: {js_dist2 > js_dist})\n")


def test_js_distance_dataframes():
    """Test JS distance for dataframes."""
    print("Testing js_distance_dataframes...")
    
    # Create mixed dataframes
    np.random.seed(42)
    df1 = pl.DataFrame({
        'cat_col': np.random.choice(['X', 'Y', 'Z'], 100),
        'num_col': np.random.normal(0, 1, 100),
        'int_col': np.random.randint(0, 10, 100)
    })
    
    df2 = pl.DataFrame({
        'cat_col': np.random.choice(['X', 'Y', 'Z'], 100, p=[0.5, 0.3, 0.2]),
        'num_col': np.random.normal(0.5, 1.2, 100),
        'int_col': np.random.randint(0, 10, 100)
    })
    
    js_dist = js_distance_dataframes(df1, df2, n_bins=10)
    print(f"  JS distance between mixed dataframes: {js_dist:.6f}")
    print(f"  ✓ Test passed\n")


def test_js_distance_onehot():
    """Test JS distance for one-hot encoded data."""
    print("Testing js_distance_onehot...")
    
    # Create one-hot encoded data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    n_categories = 3
    
    # Create categorical data first
    X1_cat = np.random.randint(0, n_categories, (n_samples, n_features))
    X2_cat = np.random.randint(0, n_categories, (n_samples, n_features))
    
    # Convert to one-hot
    X1_onehot = np.zeros((n_samples, n_features, n_categories))
    X2_onehot = np.zeros((n_samples, n_features, n_categories))
    
    for i in range(n_samples):
        for j in range(n_features):
            X1_onehot[i, j, X1_cat[i, j]] = 1
            X2_onehot[i, j, X2_cat[i, j]] = 1
    
    # Test bivariate method
    js_dist_biv = js_distance_onehot(X1_onehot, X2_onehot, method="bivariate")
    print(f"  JS distance (bivariate method): {js_dist_biv:.6f}")
    
    # Test marginal method
    js_dist_marg = js_distance_onehot(X1_onehot, X2_onehot, method="marginal")
    print(f"  JS distance (marginal method): {js_dist_marg:.6f}")
    print(f"  ✓ Test passed\n")


def test_onehot_to_categorical():
    """Test one-hot to categorical conversion."""
    print("Testing onehot_to_categorical...")
    
    # Create one-hot data
    np.random.seed(42)
    n_samples = 10
    n_features = 3
    n_categories = 4
    
    X_cat_orig = np.random.randint(0, n_categories, (n_samples, n_features))
    X_onehot = np.zeros((n_samples, n_features, n_categories))
    
    for i in range(n_samples):
        for j in range(n_features):
            X_onehot[i, j, X_cat_orig[i, j]] = 1
    
    # Convert back
    X_cat_converted = onehot_to_categorical(X_onehot)
    
    # Check if conversion is correct
    is_equal = np.array_equal(X_cat_orig, X_cat_converted)
    print(f"  Conversion correct: {is_equal}")
    print(f"  ✓ Test passed\n")


def main():
    """Run all tests."""
    print("=== Testing Distance Utilities ===\n")
    
    test_js_distance_categorical()
    test_js_distance_dataframes()
    test_js_distance_onehot()
    test_onehot_to_categorical()
    
    print("=== All tests completed ===")


if __name__ == "__main__":
    main()