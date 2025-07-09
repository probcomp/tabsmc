#!/usr/bin/env python3
"""
Test to investigate the discretization consistency issue.
"""

import polars as pl
import numpy as np
from tabsmc.io import discretize_dataframe, discretize_with_schema

def test_discretization_consistency():
    """Test why identical values get different bins."""
    
    # Create simple test case with identical values but different missing patterns
    full_data = {
        'num_col': [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    
    missing_data = {
        'num_col': [1.0, None, 3.0, 4.0, 5.0]  # Same values except one missing
    }
    
    df_full = pl.DataFrame(full_data)
    df_missing = pl.DataFrame(missing_data)
    
    print("Testing discretization consistency...")
    print("="*50)
    
    # Process full data
    schema, df_full_discretized, _ = discretize_dataframe(df_full, n_bins=5)
    
    # Process missing data with same schema
    df_missing_discretized = discretize_with_schema(df_missing, schema)
    
    print("Original values:")
    print(f"  Full:    {df_full['num_col'].to_list()}")
    print(f"  Missing: {df_missing['num_col'].to_list()}")
    
    print("\nDiscretized values:")
    print(f"  Full:    {df_full_discretized['num_col'].to_list()}")
    print(f"  Missing: {df_missing_discretized['num_col'].to_list()}")
    
    print("\nQuantiles used:")
    quantiles = schema['var_metadata']['num_col']['quantiles']
    print(f"  {quantiles}")
    
    # Check each non-missing value
    print("\nChecking each non-missing value:")
    for i, (full_val, miss_val) in enumerate(zip(df_full['num_col'], df_missing['num_col'])):
        if miss_val is not None:
            full_disc = df_full_discretized['num_col'][i]
            miss_disc = df_missing_discretized['num_col'][i]
            
            print(f"  Row {i}: {full_val} vs {miss_val}")
            print(f"    Discretized to: {full_disc} vs {miss_disc}")
            if full_disc != miss_disc:
                print(f"    ERROR: Same value, different bins!")
                
                # Debug the binning process
                print(f"    Debugging binning for {miss_val}:")
                n_bins = schema['var_metadata']['num_col']['n_bins']
                
                # Apply the same logic as discretize_with_schema
                result_bin = n_bins - 1  # Default to last bin
                for j in range(n_bins - 1, 0, -1):
                    if miss_val <= quantiles[j]:
                        result_bin = j - 1
                        print(f"      {miss_val} <= {quantiles[j]} → bin {j-1}")
                        break
                    else:
                        print(f"      {miss_val} > {quantiles[j]} → continue")
                
                print(f"    Manual calculation gives bin: {result_bin}")
                print(f"    But discretize_with_schema gives: {miss_disc}")
            else:
                print(f"    OK: Same bin assignment")


if __name__ == "__main__":
    test_discretization_consistency()