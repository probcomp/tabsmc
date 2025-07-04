#!/usr/bin/env python3
"""Evaluate synthetic data quality using CatBoost prediction task."""

import numpy as np
import polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tabsmc.io import load_huggingface
import json
import warnings
warnings.filterwarnings('ignore')


def prepare_data_for_catboost(df, target_col, n_samples=None):
    """Prepare dataframe for CatBoost training."""
    # First filter out nulls in target column
    df = df.filter(pl.col(target_col).is_not_null())
    
    if n_samples is not None and len(df) > n_samples:
        df = df.head(n_samples)
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df.select(feature_cols)
    y = df.select(target_col)
    
    # Fill remaining nulls in features with a placeholder
    # For categorical columns, use "missing"
    # For numeric columns, use -999
    for col in feature_cols:
        if X[col].dtype == pl.Utf8:
            X = X.with_columns(pl.col(col).fill_null("missing"))
        else:
            X = X.with_columns(pl.col(col).fill_null(-999))
    
    # Convert to numpy
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()
    
    # Handle categorical features
    cat_features = []
    for i, col in enumerate(feature_cols):
        if X[col].dtype == pl.Utf8:
            cat_features.append(i)
    
    return X_np, y_np, feature_cols, cat_features


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
    
    # Prepare training data (synthetic)
    X_train, y_train, feature_cols, cat_features = prepare_data_for_catboost(
        arf_df, target_col, n_train
    )
    
    # Prepare test data (real)
    X_test, y_test, _, _ = prepare_data_for_catboost(
        test_df, target_col, n_test
    )
    
    # Encode labels if needed
    le = LabelEncoder()
    # Fit on all possible labels from both train and test
    all_labels = np.concatenate([y_train, y_test])
    # Remove any None values
    all_labels = all_labels[all_labels != None]
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
    
    # ROC AUC for binary or compute macro average for multiclass
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


def evaluate_smc_synthetic(test_df, target_col, n_train=5000, n_test=2000):
    """Evaluate SMC synthetic data quality."""
    print("\n=== Evaluating SMC Synthetic Data ===")
    
    # Load SMC synthetic data
    smc_data = np.load("synthetic_data/ces_overfitted_synthetic.npz")
    X_synthetic = smc_data['X']  # Shape: (10000, 88, 51)
    
    print(f"Loaded SMC data shape: {X_synthetic.shape}")
    
    # Convert one-hot to categorical
    synthetic_cat = np.argmax(X_synthetic[:n_train], axis=2)
    
    # Need to map back to original feature names and values
    # For now, we'll use the categorical indices directly
    # This is a simplification - ideally we'd have the reverse mapping
    
    # Find target column index (this is approximate)
    test_cols = test_df.columns
    target_idx = test_cols.index(target_col)
    
    # Prepare data
    y_train = synthetic_cat[:, target_idx]
    X_train = np.delete(synthetic_cat, target_idx, axis=1)
    
    # Prepare test data
    test_array = test_df.head(n_test).to_numpy()
    y_test = test_array[:, target_idx]
    X_test = np.delete(test_array, target_idx, axis=1)
    
    # Convert to appropriate types
    # For CatBoost, we need to handle mixed types properly
    print(f"\nTraining CatBoost on {len(X_train)} SMC samples...")
    print(f"Target: {target_col} (column index: {target_idx})")
    
    # Train CatBoost
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        verbose=False,
        random_seed=42
    )
    
    # Encode labels
    le = LabelEncoder()
    
    # Fit on combined labels to ensure all classes are known
    all_labels = np.concatenate([y_train.astype(str), y_test.astype(str)])
    le.fit(all_labels)
    
    y_train_encoded = le.transform(y_train.astype(str))
    y_test_encoded = le.transform(y_test.astype(str))
    
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
    
    print(f"\nSMC Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    
    return {
        "method": "SMC",
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "n_train": n_train,
        "n_test": n_test,
        "target": target_col
    }


def main():
    """Main evaluation function."""
    print("=== Synthetic Data Quality Evaluation using CatBoost ===")
    
    # Load test data
    print("\nLoading CES test data...")
    train_df_real, test_df = load_huggingface("data/lpm/CES")
    
    # Choose a target column to predict
    # Let's pick a categorical column with reasonable cardinality
    potential_targets = []
    for col in test_df.columns:
        if test_df[col].dtype == pl.Utf8:
            n_unique = test_df[col].n_unique()
            if 2 <= n_unique <= 20:  # Reasonable number of classes
                potential_targets.append((col, n_unique))
    
    # Sort by number of unique values and pick one in the middle
    potential_targets.sort(key=lambda x: x[1])
    target_col = potential_targets[len(potential_targets)//2][0] if potential_targets else test_df.columns[0]
    
    print(f"\nSelected target column: {target_col}")
    print(f"Number of unique values: {test_df[target_col].n_unique()}")
    
    # Evaluate both methods
    results = {}
    
    # ARF evaluation
    arf_results = evaluate_arf_synthetic(test_df, target_col)
    if arf_results:
        results['arf'] = arf_results
    
    # SMC evaluation  
    try:
        smc_results = evaluate_smc_synthetic(test_df, target_col)
        results['smc'] = smc_results
    except Exception as e:
        print(f"\nError evaluating SMC: {e}")
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
    
    # Save results
    with open("catboost_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to catboost_evaluation_results.json")
    
    return results


if __name__ == "__main__":
    results = main()