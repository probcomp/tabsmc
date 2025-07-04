#!/usr/bin/env python3
"""Evaluate synthetic data quality using CatBoost - Version 2 with proper one-hot handling."""

import numpy as np
import polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tabsmc.io import load_data, load_huggingface
from tabsmc.distances import onehot_to_categorical
import json
import warnings
warnings.filterwarnings('ignore')


def prepare_onehot_for_catboost(X_onehot, target_idx, mask=None):
    """Convert one-hot encoded data to categorical for CatBoost."""
    # Convert from one-hot to categorical indices
    X_cat = onehot_to_categorical(X_onehot, mask)
    
    # Separate features and target
    y = X_cat[:, target_idx]
    X = np.delete(X_cat, target_idx, axis=1)
    
    return X, y


def evaluate_arf_synthetic(test_df, target_col, n_train=5000, n_test=2000):
    """Evaluate ARF synthetic data quality."""
    print("\n=== Evaluating ARF Synthetic Data ===")
    
    # Load ARF samples
    arf_path = Path("../model-building-evaluation/results/lpm/CES/arf/samples-max-depth-200.parquet")
    if not arf_path.exists():
        print(f"Error: ARF samples not found at {arf_path}")
        return None
    
    arf_df = pl.read_parquet(str(arf_path))
    if "replicate_id" in arf_df.columns:
        arf_df = arf_df.drop("replicate_id")
    
    print(f"Loaded ARF data shape: {arf_df.shape}")
    
    # Filter out nulls in target column
    arf_df = arf_df.filter(pl.col(target_col).is_not_null())
    test_df = test_df.filter(pl.col(target_col).is_not_null())
    
    # Prepare training data (synthetic)
    arf_subset = arf_df.head(n_train)
    test_subset = test_df.head(n_test)
    
    # Separate features and target
    feature_cols = [col for col in arf_df.columns if col != target_col]
    
    # Fill nulls in features
    for col in feature_cols:
        if arf_subset[col].dtype == pl.Utf8:
            arf_subset = arf_subset.with_columns(pl.col(col).fill_null("missing"))
            test_subset = test_subset.with_columns(pl.col(col).fill_null("missing"))
        else:
            arf_subset = arf_subset.with_columns(pl.col(col).fill_null(-999))
            test_subset = test_subset.with_columns(pl.col(col).fill_null(-999))
    
    X_train = arf_subset.select(feature_cols).to_numpy()
    y_train = arf_subset.select(target_col).to_numpy().ravel()
    
    X_test = test_subset.select(feature_cols).to_numpy()
    y_test = test_subset.select(target_col).to_numpy().ravel()
    
    # Handle categorical features for CatBoost
    cat_features = []
    for i, col in enumerate(feature_cols):
        if arf_df[col].dtype == pl.Utf8:
            cat_features.append(i)
    
    # Encode labels
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_test])
    le.fit(all_labels)
    
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print(f"\nTraining CatBoost on {len(X_train)} ARF samples...")
    print(f"Target: {target_col}, Classes: {le.classes_}")
    
    # Train CatBoost
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        verbose=False,
        random_seed=42
    )
    
    model.fit(X_train, y_train_encoded)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    # ROC AUC
    if len(le.classes_) == 2:
        roc_auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='macro')
    
    print(f"\nARF Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    return {
        "method": "ARF",
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "n_train": n_train,
        "n_test": n_test,
        "target": target_col
    }


def evaluate_smc_synthetic_onehot(col_names, target_col, n_train=5000, n_test=2000):
    """Evaluate SMC synthetic data using one-hot format."""
    print("\n=== Evaluating SMC Synthetic Data (One-hot) ===")
    
    # Load SMC synthetic data
    smc_data = np.load("synthetic_data/synthetic_ces_near_converged_step36.npz")
    X_synthetic = smc_data['X']  # Shape: (10000, 88, 51)
    
    # Load mask from the original data loading
    _, _, _, mask = load_data("data/lpm/CES")
    
    print(f"Loaded SMC data shape: {X_synthetic.shape}")
    
    # Load test data in one-hot format
    print("Loading test data in one-hot format...")
    _, test_data_log, col_names_loaded, mask_loaded = load_data("data/lpm/CES")
    test_data_onehot = (test_data_log == 0.0).astype(np.float32)
    
    # Find target column index
    if target_col not in col_names:
        print(f"Warning: {target_col} not found in col_names, using first column")
        target_idx = 0
    else:
        target_idx = col_names.index(target_col)
    
    print(f"Target column: {target_col} (index: {target_idx})")
    
    # Convert to categorical
    print("Converting one-hot to categorical...")
    synthetic_cat = onehot_to_categorical(X_synthetic[:n_train], mask)
    test_cat = onehot_to_categorical(test_data_onehot[:n_test], mask)
    
    # Prepare data for CatBoost
    X_train, y_train = prepare_onehot_for_catboost(X_synthetic[:n_train], target_idx, mask)
    X_test, y_test = prepare_onehot_for_catboost(test_data_onehot[:n_test], target_idx, mask)
    
    # Remove samples where target is missing (if any)
    # In one-hot format, missing might be represented as all zeros
    valid_train = ~np.isnan(y_train)
    valid_test = ~np.isnan(y_test)
    
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    X_test = X_test[valid_test]
    y_test = y_test[valid_test]
    
    print(f"\nTraining CatBoost on {len(X_train)} SMC samples...")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Unique target values in train: {np.unique(y_train)}")
    print(f"Unique target values in test: {np.unique(y_test)}")
    
    # All features are categorical in this representation
    cat_features = list(range(X_train.shape[1]))
    
    # Train CatBoost
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_features,
        verbose=False,
        random_seed=42
    )
    
    # Convert to int for CatBoost
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)
    
    model.fit(X_train, y_train_int)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_int, y_pred)
    
    # ROC AUC
    n_classes = len(np.unique(y_test_int))
    if n_classes == 2:
        roc_auc = roc_auc_score(y_test_int, y_pred_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test_int, y_pred_proba, multi_class='ovr', average='macro')
    
    print(f"\nSMC Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    return {
        "method": "SMC",
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "n_train": n_train,
        "n_test": n_test,
        "target": target_col,
        "target_idx": target_idx
    }


def main():
    """Main evaluation function."""
    print("=== Synthetic Data Quality Evaluation using CatBoost (v2) ===")
    
    # Load data to get column names
    print("\nLoading CES data...")
    train_df_real, test_df = load_huggingface("data/lpm/CES")
    _, _, col_names, _ = load_data("data/lpm/CES")
    
    # Choose a target column
    # Pick a categorical column with reasonable cardinality
    potential_targets = []
    for col in test_df.columns:
        if test_df[col].dtype == pl.Utf8:
            n_unique = test_df[col].n_unique()
            if 2 <= n_unique <= 10 and col in col_names:  # Make sure it's in one-hot data
                potential_targets.append((col, n_unique))
    
    if not potential_targets:
        print("No suitable target columns found")
        return None
    
    # Sort by number of unique values and pick one
    potential_targets.sort(key=lambda x: x[1])
    target_col = potential_targets[len(potential_targets)//2][0]
    
    print(f"\nSelected target column: {target_col}")
    print(f"Number of unique values: {test_df[target_col].n_unique()}")
    
    # Evaluate both methods
    results = {}
    
    # ARF evaluation
    arf_results = evaluate_arf_synthetic(test_df, target_col)
    if arf_results:
        results['arf'] = arf_results
    
    # SMC evaluation with one-hot data
    try:
        smc_results = evaluate_smc_synthetic_onehot(col_names, target_col)
        results['smc'] = smc_results
    except Exception as e:
        print(f"\nError evaluating SMC: {e}")
        import traceback
        traceback.print_exc()
        results['smc'] = {"error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    if 'arf' in results and 'accuracy' in results['arf']:
        print(f"\nARF Synthetic Data:")
        print(f"  Accuracy: {results['arf']['accuracy']:.4f}")
        print(f"  ROC AUC: {results['arf']['roc_auc']:.4f}")
    
    if 'smc' in results and 'accuracy' in results['smc']:
        print(f"\nSMC Synthetic Data:")
        print(f"  Accuracy: {results['smc']['accuracy']:.4f}")
        print(f"  ROC AUC: {results['smc']['roc_auc']:.4f}")
        
        # Compare results
        if 'arf' in results and 'accuracy' in results['arf']:
            print(f"\nComparison:")
            print(f"  Accuracy difference (ARF - SMC): {results['arf']['accuracy'] - results['smc']['accuracy']:.4f}")
            print(f"  ROC AUC difference (ARF - SMC): {results['arf']['roc_auc'] - results['smc']['roc_auc']:.4f}")
    
    # Save results
    with open("catboost_evaluation_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to catboost_evaluation_results_v2.json")
    
    return results


if __name__ == "__main__":
    results = main()