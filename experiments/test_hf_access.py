#!/usr/bin/env python
"""Test HuggingFace dataset access and list available datasets."""

import os
from tabsmc.io import load_data

def test_hf_access():
    """Test HuggingFace access with different tokens."""
    print("Testing HuggingFace dataset access...")
    
    # Check current token setup
    from tabsmc.io import ACCESS_TOKEN
    print(f"Current ACCESS_TOKEN from io.py: {ACCESS_TOKEN[:10]}...")
    
    # Check environment variable
    hf_token = os.getenv("HF_TOKEN")
    print(f"HF_TOKEN environment variable: {'Set' if hf_token else 'Not set'}")
    
    # Try to access the dataset
    try:
        print("\nAttempting to load PUMS dataset...")
        train_data, test_data, col_names, mask = load_data("data/lpm/PUMS")
        print(f"✓ SUCCESS! Data loaded:")
        print(f"  Train shape: {train_data.shape}")
        print(f"  Test shape: {test_data.shape}")
        print(f"  Features: {len(col_names)}")
        print(f"  Mask shape: {mask.shape}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def list_potential_datasets():
    """List some potential dataset paths to try."""
    print("\nPotential dataset paths to try:")
    datasets = [
        "data/lpm/PUMS",
        "data/lpm/Adult", 
        "data/lpm/Covertype",
        "data/folktables/acs",
        "data/folktables/pums"
    ]
    
    for dataset in datasets:
        print(f"  - {dataset}")

if __name__ == "__main__":
    success = test_hf_access()
    if not success:
        list_potential_datasets()
        print("\nTo fix HuggingFace access:")
        print("1. Get a valid HuggingFace token from https://huggingface.co/settings/tokens")
        print("2. Set it as environment variable: export HF_TOKEN=your_token_here")
        print("3. Or update the ACCESS_TOKEN in tabsmc/io.py")