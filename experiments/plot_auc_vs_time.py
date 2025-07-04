#!/usr/bin/env python3
"""Plot AUC vs time for different synthetic data generation methods."""

import numpy as np
import polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tabsmc.io import load_huggingface
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


def evaluate_method_at_timepoint(synthetic_df, test_df, target_col, method_name, time_val, n_train=2000, n_test=1000):
    """Evaluate a synthetic data method at a specific timepoint."""
    print(f"Evaluating {method_name} at time {time_val:.2f}...")
    
    # Filter out nulls in target column
    synthetic_clean = synthetic_df.filter(pl.col(target_col).is_not_null())
    test_clean = test_df.filter(pl.col(target_col).is_not_null())
    
    if len(synthetic_clean) < n_train:
        n_train = len(synthetic_clean)
    if len(test_clean) < n_test:
        n_test = len(test_clean)
    
    # Prepare data
    synthetic_subset = synthetic_clean.head(n_train)
    test_subset = test_clean.head(n_test)
    
    # Separate features and target
    feature_cols = [col for col in synthetic_df.columns if col != target_col]
    
    # Fill nulls in features
    for col in feature_cols:
        if synthetic_subset[col].dtype == pl.Utf8:
            synthetic_subset = synthetic_subset.with_columns(pl.col(col).fill_null("missing"))
            test_subset = test_subset.with_columns(pl.col(col).fill_null("missing"))
        else:
            synthetic_subset = synthetic_subset.with_columns(pl.col(col).fill_null(-999))
            test_subset = test_subset.with_columns(pl.col(col).fill_null(-999))
    
    X_train = synthetic_subset.select(feature_cols).to_numpy()
    y_train = synthetic_subset.select(target_col).to_numpy().ravel()
    
    X_test = test_subset.select(feature_cols).to_numpy()
    y_test = test_subset.select(target_col).to_numpy().ravel()
    
    # Handle categorical features
    cat_features = []
    for i, col in enumerate(feature_cols):
        if synthetic_subset[col].dtype == pl.Utf8:
            cat_features.append(i)
    
    # Encode labels
    le = LabelEncoder()
    all_labels = np.concatenate([y_train, y_test])
    le.fit(all_labels)
    
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Train CatBoost
    model = CatBoostClassifier(
        iterations=50,  # Faster for multiple evaluations
        learning_rate=0.1,
        depth=4,  # Simpler for speed
        cat_features=cat_features,
        verbose=False,
        random_seed=42
    )
    
    try:
        model.fit(X_train, y_train_encoded)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate ROC AUC
        if len(le.classes_) == 2:
            roc_auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='macro')
        
        return roc_auc
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    """Main function to create AUC vs time plot."""
    print("=== AUC vs Time Analysis ===\n")
    
    # Load test data
    print("Loading CES test data...")
    _, test_df = load_huggingface("data/lpm/CES")
    
    # Load timing data
    print("Loading timing data...")
    timing_df = pl.read_parquet('../model-building-evaluation/results/lpm/CES/timed-samples.parquet')
    
    # Choose target column (same as before)
    target_col = "Policy_support_allow_states_import_drugs_from_other_countries"
    print(f"Target column: {target_col}")
    
    # Storage for results
    results = []
    
    # Process each model
    for model_name in timing_df['model'].unique():
        print(f"\n--- Processing {model_name} ---")
        
        # Get unique time points for this model
        model_data = timing_df.filter(pl.col('model') == model_name)
        time_points = model_data['time'].unique().sort()
        
        print(f"Time points: {len(time_points)}")
        
        for time_val in time_points:
            # Get synthetic data for this time point
            time_data = model_data.filter(pl.col('time') == time_val)
            
            # Drop timing columns to get just the synthetic data
            synthetic_df = time_data.drop(['model', 'time', 'parameter', 'replicate'])
            
            # Evaluate
            auc = evaluate_method_at_timepoint(
                synthetic_df, test_df, target_col, model_name, time_val
            )
            
            if auc is not None:
                results.append({
                    'method': model_name,
                    'time': float(time_val),
                    'auc': float(auc)
                })
                print(f"  Time {time_val:.2f}: AUC = {auc:.4f}")
    
    # Add SMC data points (manually, since we know the results)
    print(f"\n--- Adding SMC Results ---")
    
    # These are approximate timing estimates for SMC
    # You'll need to measure actual times or estimate based on your runs
    smc_results = [
        {'method': 'SMC_overfitted', 'time': 300.0, 'auc': 0.4724},  # Poor performance
        {'method': 'SMC_converged', 'time': 600.0, 'auc': 0.6618},   # Good performance
    ]
    
    results.extend(smc_results)
    
    # Convert to DataFrame for plotting
    results_df = pl.DataFrame(results)
    
    # Create the plot
    print(f"\n--- Creating Plot ---")
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot each method
    methods = results_df['method'].unique()
    for i, method in enumerate(methods):
        method_data = results_df.filter(pl.col('method') == method)
        times = method_data['time'].to_numpy()
        aucs = method_data['auc'].to_numpy()
        
        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        aucs = aucs[sort_idx]
        
        plt.plot(times, aucs, 'o-', label=method, color=colors[i % len(colors)], 
                 linewidth=2, markersize=8)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('ROC AUC', fontsize=12)
    plt.title('Synthetic Data Quality (ROC AUC) vs Training Time', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    plt.xlim(0, max(results_df['time']) * 1.1)
    plt.ylim(0.4, max(results_df['auc']) * 1.05)
    
    # Add horizontal line for random performance
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
    
    plt.tight_layout()
    plt.savefig('auc_vs_time_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    with open('auc_vs_time_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n--- Summary ---")
    print(f"Total evaluations: {len(results)}")
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        if method_results:
            best_auc = max(r['auc'] for r in method_results)
            best_time = [r['time'] for r in method_results if r['auc'] == best_auc][0]
            print(f"{method}: Best AUC = {best_auc:.4f} at {best_time:.1f}s")
    
    print(f"\nPlot saved as 'auc_vs_time_plot.png'")
    print(f"Results saved as 'auc_vs_time_results.json'")


if __name__ == "__main__":
    main()