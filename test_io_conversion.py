#!/usr/bin/env python3
"""
Test suite for tabsmc/io.py conversion pipeline to ensure data integrity.
"""

import polars as pl
import numpy as np
from tabsmc.io import discretize_dataframe, discretize_with_schema, encode_with_schema
from tabsmc.smc import MISSING_VALUE


def create_test_data():
    """Create test data with known values."""
    # Create a simple test dataset with both categorical and numerical columns
    data = {
        'cat_A': ['x', 'y', 'x', 'z', 'y'],
        'num_B': [1.0, 2.5, 3.0, 3.7, 4.2], 
        'cat_C': ['a', 'b', 'a', 'a', 'c'],
        'num_D': [10.0, 20.0, 15.0, 25.0, 12.0],
        'cat_E': ['p', 'q', 'p', 'p', 'r']
    }
    
    df = pl.DataFrame(data)
    
    # Create a version with some missing values
    df_missing = df.clone()
    df_missing = df_missing.with_columns([
        pl.when(pl.int_range(len(df)) == 1).then(None).otherwise(pl.col('cat_A')).alias('cat_A'),
        pl.when(pl.int_range(len(df)) == 2).then(None).otherwise(pl.col('num_B')).alias('num_B'),
        pl.when(pl.int_range(len(df)) == 3).then(None).otherwise(pl.col('cat_C')).alias('cat_C'),
    ])
    
    return df, df_missing


def test_column_order_preservation():
    """Test that column order is preserved through the conversion pipeline."""
    print("Testing column order preservation...")
    print("="*50)
    
    df, df_missing = create_test_data()
    
    print(f"Original column order: {df.columns}")
    
    # Process full data
    schema, df_discretized, _ = discretize_dataframe(df)
    print(f"After discretization: {df_discretized.columns}")
    
    df_encoded = encode_with_schema(df_discretized, schema)
    print(f"After encoding: {df_encoded.columns}")
    
    # Check if order changed
    order_preserved = df.columns == df_encoded.columns
    print(f"\nColumn order preserved: {order_preserved}")
    
    if not order_preserved:
        print("\nColumn order mapping:")
        for i, orig_col in enumerate(df.columns):
            new_pos = df_encoded.columns.index(orig_col) if orig_col in df_encoded.columns else -1
            print(f"  {orig_col}: position {i} → {new_pos}")
    
    return order_preserved


def test_missing_data_consistency():
    """Test that missing data is processed consistently with full data."""
    print("\n\nTesting missing data consistency...")
    print("="*50)
    
    df, df_missing = create_test_data()
    
    # Process both datasets
    schema, df_full_discretized, _ = discretize_dataframe(df)
    df_missing_discretized = discretize_with_schema(df_missing, schema)
    
    df_full_encoded = encode_with_schema(df_full_discretized, schema)
    df_missing_encoded = encode_with_schema(df_missing_discretized, schema)
    
    print(f"Full data shape: {df_full_encoded.shape}")
    print(f"Missing data shape: {df_missing_encoded.shape}")
    print(f"Column order matches: {df_full_encoded.columns == df_missing_encoded.columns}")
    
    # Check non-missing values
    print(f"\nChecking non-missing values...")
    
    # Convert to numpy for comparison
    full_np = df_full_encoded.to_numpy()
    missing_np = df_missing_encoded.to_numpy()
    
    # Apply the same cleaning as in the actual pipeline
    full_clean = np.where((np.isnan(full_np)) | (full_np == -1), MISSING_VALUE, full_np)
    missing_clean = np.where((np.isnan(missing_np)) | (missing_np == -1), MISSING_VALUE, missing_np)
    
    print(f"\nRow-by-row comparison:")
    for row in range(len(df)):
        print(f"\nRow {row}:")
        print(f"  Full:    {full_clean[row]}")
        print(f"  Missing: {missing_clean[row]}")
        
        # Find positions where values differ (excluding actual missing values)
        differences = []
        for col in range(len(df_full_encoded.columns)):
            if (missing_clean[row, col] != MISSING_VALUE and 
                full_clean[row, col] != missing_clean[row, col]):
                col_name = df_full_encoded.columns[col]
                differences.append(f"col {col} ({col_name}): {full_clean[row, col]} vs {missing_clean[row, col]}")
        
        if differences:
            print(f"  DIFFERENCES: {differences}")
        else:
            print(f"  All non-missing values match ✓")


def test_value_round_trip():
    """Test that non-missing values survive the round trip correctly."""
    print("\n\nTesting value round-trip integrity...")
    print("="*50)
    
    df, df_missing = create_test_data()
    
    # Process the data
    schema, df_full_discretized, _ = discretize_dataframe(df)
    df_missing_discretized = discretize_with_schema(df_missing, schema)
    
    df_full_encoded = encode_with_schema(df_full_discretized, schema)
    df_missing_encoded = encode_with_schema(df_missing_discretized, schema)
    
    print("\nOriginal vs final encoded values:")
    
    for row in range(len(df)):
        print(f"\nRow {row}:")
        for col_name in df.columns:
            orig_val = df[col_name][row]
            missing_orig_val = df_missing[col_name][row]
            
            # Get encoded values by column name (not position)
            full_encoded_val = df_full_encoded[col_name][row]
            missing_encoded_val = df_missing_encoded[col_name][row]
            
            # Check if the missing value is actually missing
            is_missing = missing_orig_val is None or (isinstance(missing_orig_val, float) and np.isnan(missing_orig_val))
            
            print(f"  {col_name}:")
            print(f"    Original: {orig_val} → Encoded: {full_encoded_val}")
            if is_missing:
                print(f"    Missing:  {missing_orig_val} → Encoded: {missing_encoded_val} (should be -1)")
            else:
                print(f"    Missing:  {missing_orig_val} → Encoded: {missing_encoded_val}")
                if full_encoded_val != missing_encoded_val:
                    print(f"    ❌ ERROR: Non-missing value encoded differently!")
                else:
                    print(f"    ✅ OK: Non-missing value encoded consistently")


def test_schema_types():
    """Test that schema correctly identifies column types."""
    print("\n\nTesting schema type identification...")
    print("="*50)
    
    df, _ = create_test_data()
    
    schema, _, _ = discretize_dataframe(df)
    
    print(f"Categorical columns: {schema['types']['categorical']}")
    print(f"Numerical columns: {schema['types']['numerical']}")
    
    expected_categorical = ['cat_A', 'cat_C', 'cat_E']
    expected_numerical = ['num_B', 'num_D']
    
    cat_correct = set(schema['types']['categorical']) == set(expected_categorical)
    num_correct = set(schema['types']['numerical']) == set(expected_numerical)
    
    print(f"\nCategorical classification correct: {cat_correct}")
    print(f"Numerical classification correct: {num_correct}")
    
    return cat_correct and num_correct


def run_all_tests():
    """Run all tests and return results."""
    print("TABSMC IO.PY CONVERSION TESTS")
    print("="*60)
    
    results = {}
    
    # Test 1: Column order preservation
    results['column_order'] = test_column_order_preservation()
    
    # Test 2: Missing data consistency
    test_missing_data_consistency()
    
    # Test 3: Value round-trip integrity
    test_value_round_trip()
    
    # Test 4: Schema type identification
    results['schema_types'] = test_schema_types()
    
    # Summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {sum(results.values())}/{len(results)} tests passed")
    
    return results


if __name__ == "__main__":
    run_all_tests()