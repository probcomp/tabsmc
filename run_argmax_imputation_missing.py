#!/usr/bin/env python3
"""
Run batched argmax conditional imputation on all datasets with missing data.
Uses trained models from results_missing directory.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Any
import gc
from scipy import stats

from tabsmc.query import argmax_conditional
from argmax_conditional_particles import argmax_conditional_batched_particles
from tabsmc.smc import MISSING_VALUE
from tabsmc.io import discretize_dataframe, encode_with_schema, discretize_with_schema
import polars as pl


# All datasets with missing data
DATASETS = {
    "ctgan": ["covertype", "kddcup", "sydt"],
    "lpm": ["CES", "PUMD", "PUMS"]
}


def load_missing_data_for_dataset(dataset_name: str, data_root: str = "../model-building-evaluation", load_true_data: bool = False) -> Tuple:
    """Load missing data for a specific dataset (same as run_smc_missing_data_clean.py)."""
    
    # Determine dataset paths based on type
    if dataset_name.lower() in ["covertype", "kddcup", "sydt"]:
        train_path = f"{data_root}/data/CTGAN/{dataset_name}/data-train-num-missing.parquet"
        train_full_path = f"{data_root}/data/CTGAN/{dataset_name}/data-train-num.parquet"
        test_path = f"{data_root}/data/CTGAN/{dataset_name}/data-test-full-num.parquet"
    elif dataset_name.upper() in ["CES", "PUMD", "PUMS"]:
        train_path = f"{data_root}/data/lpm/{dataset_name.upper()}/data-train-num-missing.parquet"
        train_full_path = f"{data_root}/data/lpm/{dataset_name.upper()}/data-train-num.parquet"
        test_path = f"{data_root}/data/lpm/{dataset_name.upper()}/data-test-full-num.parquet"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loading missing data for {dataset_name}")
    print(f"  Train (with missing): {train_path}")
    print(f"  Test (regular): {test_path}")
    
    # Check if files exist
    train_file = Path(train_path)
    train_full_file = Path(train_full_path)
    test_file = Path(test_path)
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not train_full_file.exists():
        raise FileNotFoundError(f"Training full file not found: {train_full_path}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    # Load the parquet files directly
    print("Loading training data...")
    train_df = pl.read_parquet(train_path)
    train_full_df = pl.read_parquet(train_full_path)
    print("Loading test data...")
    test_df = pl.read_parquet(test_path)
    
    print(f"Raw data loaded: train={train_df.shape}, test={test_df.shape}")
    
    # Process the data similar to load_data function but for missing data
    # First discretize the training data to get schema
    print("Processing training data...")
    train_full_schema, train_full_discretized, _ = discretize_dataframe(train_full_df)
    
    train_discretized = discretize_with_schema(train_df, train_full_schema)
    train_encoded = encode_with_schema(train_discretized, train_full_schema)
    
    # Process test data using the same schema
    print("Processing test data...")
    test_discretized = discretize_with_schema(test_df, train_full_schema)
    test_encoded = encode_with_schema(test_discretized, train_full_schema)
    
    # Convert to numpy arrays with proper missing value handling
    train_data_raw = train_encoded.to_numpy()
    test_data_raw = test_encoded.to_numpy()
    
    # Handle missing values by replacing NaN and -1 with MISSING_VALUE
    train_data_clean = np.where(
        (np.isnan(train_data_raw)) | (train_data_raw == -1), 
        MISSING_VALUE, 
        train_data_raw
    )
    test_data_clean = np.where(
        (np.isnan(test_data_raw)) | (test_data_raw == -1), 
        MISSING_VALUE, 
        test_data_raw
    )
    
    # Convert to uint32 to prevent negative indexing issues
    train_data = train_data_clean.astype(np.uint32)
    test_data = test_data_clean.astype(np.uint32)
    
    # Get column names
    col_names = train_encoded.columns
    
    # Determine K (max categories per feature)
    K_values = []
    for col in col_names:
        if col in train_full_schema["types"]["categorical"]:
            K_col = len(train_full_schema["var_metadata"][col]["levels"])
        else:  # numerical
            K_col = train_full_schema["var_metadata"][col]["n_bins"]
        K_values.append(K_col)
    
    K = max(K_values)
    
    # Create mask for valid categories per feature
    mask = np.zeros((len(col_names), K), dtype=bool)
    for i, (col, K_col) in enumerate(zip(col_names, K_values)):
        mask[i, :K_col] = True
    
    # Count missing values
    train_missing = np.sum(train_data == MISSING_VALUE)
    test_missing = np.sum(test_data == MISSING_VALUE)
    total_train_values = train_data.size
    total_test_values = test_data.size
    
    print(f"Dataset {dataset_name} processed:")
    print(f"  Training: {train_data.shape} ({train_missing:,}/{total_train_values:,} missing = {100*train_missing/total_train_values:.1f}%)")
    print(f"  Test: {test_data.shape} ({test_missing:,}/{total_test_values:,} missing = {100*test_missing/total_test_values:.1f}%)")
    print(f"  Features: {len(col_names)}, Max categories: {K}")
    print(f"  Data types: train={train_data.dtype}, test={test_data.dtype}")
    print(f"  Missing value constant: {MISSING_VALUE}")
    
    if load_true_data:
        # Also process the full training data (without missing values) for comparison
        print("Processing full training data for accuracy comparison...")
        train_full_encoded = encode_with_schema(train_full_discretized, train_full_schema)
        train_full_data_raw = train_full_encoded.to_numpy()
        
        # Handle any missing values in the "full" data
        train_full_data_clean = np.where(
            (np.isnan(train_full_data_raw)) | (train_full_data_raw == -1), 
            MISSING_VALUE, 
            train_full_data_raw
        )
        train_full_data = train_full_data_clean.astype(np.uint32)
        
        full_missing = np.sum(train_full_data == MISSING_VALUE)
        print(f"  Full training data: {train_full_data.shape} ({full_missing:,} missing values)")
        
        return (jnp.array(train_data), jnp.array(test_data), col_names, jnp.array(mask), K, 
                jnp.array(train_full_data))
    else:
        return jnp.array(train_data), jnp.array(test_data), col_names, jnp.array(mask), K


def find_best_model_for_dataset(dataset_name: str, results_dir: str = "results_missing") -> str:
    """Find the best trained model for a dataset."""
    results_path = Path(results_dir)
    
    # Look for model files for this dataset
    pattern = f"smc_{dataset_name.lower()}_missing_*.pkl"
    model_files = list(results_path.glob(pattern))
    
    if not model_files:
        raise FileNotFoundError(f"No trained models found for dataset {dataset_name} in {results_dir}")
    
    # Choose the most recent or best model (for now just take the first one)
    # In practice, you might want to load and compare models
    best_model = model_files[0]
    
    print(f"Found {len(model_files)} models for {dataset_name}, using: {best_model.name}")
    
    return str(best_model)


def load_trained_model(model_path: str, use_all_particles: bool = True) -> Tuple:
    """Load theta and pi from a trained model.
    
    Args:
        model_path: Path to the trained model
        use_all_particles: If True, return all particles; if False, return only first particle
    
    Returns:
        If use_all_particles is True: (theta_particles, pi_particles, log_weights)
        If use_all_particles is False: (theta, pi)
    """
    print(f"Loading model from {model_path}")
    
    with open(model_path, 'rb') as f:
        results = pickle.load(f)
    
    if not results.get('success', False):
        raise ValueError(f"Model training was not successful: {model_path}")
    
    # Extract parameters from particles
    particles = results['smc_results']['particles']
    log_weights = results['smc_results']['log_weights']
    
    # Based on the structure we found:
    # particles[0]: allocations (4, n_train)
    # particles[1]: theta parameters (possibly counts) (4, n_clusters, n_features, n_categories)
    # particles[2]: pi parameters (log mixture weights) (4, n_clusters)
    # particles[3]: theta in log space (log emission parameters) (4, n_clusters, n_features, n_categories)
    
    theta_particles = particles[3]  # Shape: (n_particles, n_clusters, n_features, n_categories)
    pi_particles = particles[2]     # Shape: (n_particles, n_clusters)
    
    print(f"Found {theta_particles.shape[0]} particles")
    print(f"Log weights: {log_weights}")
    
    if use_all_particles:
        print(f"Returning all {theta_particles.shape[0]} particles for proper weighted inference")
        print(f"Theta particles shape: {theta_particles.shape}")
        print(f"Pi particles shape: {pi_particles.shape}")
        print(f"Theta range: {jnp.min(theta_particles):.3f} to {jnp.max(theta_particles):.3f}")
        print(f"Pi range: {jnp.min(pi_particles):.3f} to {jnp.max(pi_particles):.3f}")
        return theta_particles, pi_particles, log_weights
    else:
        # Just return first particle (backward compatibility)
        theta = theta_particles[0]
        pi = pi_particles[0]
        print(f"Returning first particle only")
        print(f"Theta shape: {theta.shape}, Pi shape: {pi.shape}")
        print(f"Theta range: {jnp.min(theta):.3f} to {jnp.max(theta):.3f}")
        print(f"Pi range: {jnp.min(pi):.3f} to {jnp.max(pi):.3f}")
        return theta, pi


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute confidence interval for a set of values.
    
    Args:
        data: Array of values
        confidence: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        (mean, lower_bound, upper_bound, std_error)
    """
    n = len(data)
    if n < 2:
        return (float(data[0]) if n == 1 else 0.0, 0.0, 0.0, 0.0)
    
    mean = np.mean(data)
    std_error = stats.sem(data)
    
    # Compute confidence interval using t-distribution
    confidence_interval = stats.t.interval(
        confidence, 
        n - 1, 
        loc=mean, 
        scale=std_error
    )
    
    return mean, confidence_interval[0], confidence_interval[1], std_error


def compute_binomial_confidence_interval(successes, trials, confidence=0.95):
    """
    Compute confidence interval for a proportion using Wilson score interval.
    
    Args:
        successes: Number of successes
        trials: Total number of trials
        confidence: Confidence level (default: 0.95 for 95% CI)
        
    Returns:
        (proportion, lower_bound, upper_bound)
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
    
    # Wilson score interval
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials
    
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    
    margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return p_hat, lower, upper


def compute_imputation_accuracy(
    imputed_data: jnp.ndarray,
    true_data: jnp.ndarray,
    missing_data: jnp.ndarray,
    max_rows: int = None
) -> Dict[str, float]:
    """
    Compute accuracy of imputation by comparing to true values.
    
    Args:
        imputed_data: Data with imputed values
        true_data: True data without missing values
        missing_data: Original data with missing values (MISSING_VALUE for missing)
        max_rows: Maximum number of rows to evaluate (should match imputation)
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Limit rows if specified
    if max_rows is not None:
        imputed_data = imputed_data[:max_rows]
        true_data = true_data[:max_rows]
        missing_data = missing_data[:max_rows]
    
    # Create mask for values that were imputed (missing in original data)
    imputed_mask = (missing_data == MISSING_VALUE)
    
    # Create mask for values that are missing in the true data (should be excluded)
    true_missing_mask = (true_data == MISSING_VALUE)
    
    # Combined mask: only evaluate values that were imputed AND have true values
    eval_mask = imputed_mask & ~true_missing_mask
    
    # Compute accuracy only on imputed values that have ground truth
    if jnp.sum(eval_mask) == 0:
        print("Warning: No values to evaluate (all imputed values lack ground truth)")
        return {
            'accuracy': 0.0,
            'n_evaluated': 0,
            'n_imputed': int(jnp.sum(imputed_mask)),
            'n_true_missing': int(jnp.sum(true_missing_mask)),
        }
    
    # Compare imputed values to true values
    correct = (imputed_data == true_data) & eval_mask
    
    # Compute accuracy using nanmean approach
    accuracy_matrix = jnp.where(eval_mask, correct.astype(float), jnp.nan)
    accuracy = jnp.nanmean(accuracy_matrix)
    
    # Compute per-feature accuracy
    per_feature_accuracy = jnp.nanmean(accuracy_matrix, axis=0)
    
    # Count statistics
    n_evaluated = int(jnp.sum(eval_mask))
    n_correct = int(jnp.sum(correct))
    n_imputed = int(jnp.sum(imputed_mask))
    n_true_missing = int(jnp.sum(true_missing_mask))
    n_imputed_with_truth = int(jnp.sum(imputed_mask & ~true_missing_mask))
    
    results = {
        'accuracy': float(accuracy),
        'n_correct': n_correct,
        'n_evaluated': n_evaluated,
        'n_imputed': n_imputed,
        'n_imputed_with_truth': n_imputed_with_truth,
        'n_true_missing': n_true_missing,
        'per_feature_accuracy': per_feature_accuracy.tolist(),
        'accuracy_std': float(jnp.nanstd(accuracy_matrix)),
    }
    
    return results


def argmax_conditional_batched(
    data: jnp.ndarray,
    theta: jnp.ndarray,
    pi: jnp.ndarray,
    batch_size: int = 1000,
    max_rows: int = None
) -> jnp.ndarray:
    """
    Perform batched argmax conditional imputation.
    
    Args:
        data: Data with missing values (MISSING_VALUE = -1)
        theta: Log emission parameters (n_clusters, n_features, n_categories)
        pi: Log mixture weights (n_clusters,)
        batch_size: Size of batches for processing
        max_rows: Maximum number of rows to process (for testing)
        
    Returns:
        Imputed data with same shape as input (or truncated if max_rows specified)
    """
    n_rows, n_features = data.shape
    
    # Limit rows if specified
    if max_rows is not None:
        n_rows = min(n_rows, max_rows)
        data = data[:n_rows]
        print(f"Limiting processing to first {n_rows} rows")
    
    # Convert MISSING_VALUE to -1 for argmax_conditional function
    data_for_imputation = jnp.where(data == MISSING_VALUE, -1, data)
    
    # Process in batches
    imputed_batches = []
    
    for i in range(0, n_rows, batch_size):
        end_idx = min(i + batch_size, n_rows)
        batch = data_for_imputation[i:end_idx]
        
        print(f"Processing batch {i//batch_size + 1}/{(n_rows + batch_size - 1)//batch_size}: rows {i} to {end_idx-1}")
        
        # Apply argmax_conditional to each row in the batch
        batch_imputed = jax.vmap(
            lambda row: argmax_conditional(row, theta, pi)
        )(batch)
        
        imputed_batches.append(batch_imputed)
    
    # Concatenate all batches
    imputed_data = jnp.concatenate(imputed_batches, axis=0)
    
    return imputed_data


def run_argmax_imputation_for_dataset(
    dataset_name: str,
    data_root: str = "../model-building-evaluation",
    results_dir: str = "results_missing",
    batch_size: int = 1000,
    output_dir: str = "argmax_imputation_results",
    use_all_particles: bool = True
) -> Dict[str, Any]:
    """Run argmax conditional imputation for a single dataset."""
    
    print(f"\n{'='*70}")
    print(f"Running argmax imputation for {dataset_name}")
    print(f"{'='*70}")
    
    try:
        # Load missing data with true data for accuracy comparison
        data_result = load_missing_data_for_dataset(
            dataset_name, data_root, load_true_data=True
        )
        
        if len(data_result) == 6:
            train_data, test_data, col_names, mask, K, train_full_data = data_result
        else:
            # Fallback if load_true_data failed
            train_data, test_data, col_names, mask, K = data_result
            train_full_data = None
        
        # Find and load trained model
        model_path = find_best_model_for_dataset(dataset_name, results_dir)
        
        if use_all_particles:
            theta_particles, pi_particles, log_weights = load_trained_model(model_path, use_all_particles=True)
            
            # Count missing values before imputation
            train_missing_before = jnp.sum(train_data == MISSING_VALUE)
            test_missing_before = jnp.sum(test_data == MISSING_VALUE)
            
            print(f"\nMissing values before imputation:")
            print(f"  Training: {train_missing_before:,}/{train_data.size:,} ({100*train_missing_before/train_data.size:.1f}%)")
            print(f"  Test: {test_missing_before:,}/{test_data.size:,} ({100*test_missing_before/test_data.size:.1f}%)")
            
            # Perform argmax imputation using all particles
            print(f"\nPerforming argmax imputation with all particles, batch size {batch_size}...")
            start_time = time.time()
            
            print("Imputing training data...")
            train_result = argmax_conditional_batched_particles(
                train_data, theta_particles, pi_particles, log_weights, batch_size, max_rows=10000, track_timing=True
            )
            train_imputed, train_batch_times = train_result
            
            print("Imputing test data...")
            test_result = argmax_conditional_batched_particles(
                test_data, theta_particles, pi_particles, log_weights, batch_size, max_rows=10000, track_timing=True
            )
            test_imputed, test_batch_times = test_result
        else:
            # Backward compatibility: use single particle
            theta, pi = load_trained_model(model_path, use_all_particles=False)
            
            # Count missing values before imputation
            train_missing_before = jnp.sum(train_data == MISSING_VALUE)
            test_missing_before = jnp.sum(test_data == MISSING_VALUE)
            
            print(f"\nMissing values before imputation:")
            print(f"  Training: {train_missing_before:,}/{train_data.size:,} ({100*train_missing_before/train_data.size:.1f}%)")
            print(f"  Test: {test_missing_before:,}/{test_data.size:,} ({100*test_missing_before/test_data.size:.1f}%)")
            
            # Perform argmax imputation
            print(f"\nPerforming argmax imputation with single particle, batch size {batch_size}...")
            start_time = time.time()
            
            print("Imputing training data...")
            train_imputed = argmax_conditional_batched(train_data, theta, pi, batch_size, max_rows=10000)
            
            print("Imputing test data...")
            test_imputed = argmax_conditional_batched(test_data, theta, pi, batch_size, max_rows=10000)
        
        imputation_time = time.time() - start_time
        
        # Compute timing statistics and confidence intervals
        timing_stats = None
        if use_all_particles:
            # Combine all batch times
            all_batch_times = train_batch_times + test_batch_times
            all_batch_sizes = [batch_size] * len(all_batch_times)
            
            # Compute time per row for each batch
            time_per_row_list = [bt / bs for bt, bs in zip(all_batch_times, all_batch_sizes)]
            
            # Compute confidence intervals
            time_per_row_mean, time_per_row_lower, time_per_row_upper, time_per_row_stderr = compute_confidence_interval(time_per_row_list)
            
            timing_stats = {
                'time_per_row_mean': time_per_row_mean,
                'time_per_row_95ci_lower': time_per_row_lower,
                'time_per_row_95ci_upper': time_per_row_upper,
                'time_per_row_stderr': time_per_row_stderr,
                'batch_times': all_batch_times,
                'n_batches': len(all_batch_times),
                'avg_batch_size': np.mean(all_batch_sizes),
            }
            
            print(f"Timing per row: {time_per_row_mean*1000:.2f}ms (95% CI: [{time_per_row_lower*1000:.2f}, {time_per_row_upper*1000:.2f}]ms)")
        
        # Verify imputation
        train_missing_after = jnp.sum(train_imputed == MISSING_VALUE)
        test_missing_after = jnp.sum(test_imputed == MISSING_VALUE)
        
        print(f"\nImputation completed in {imputation_time:.2f} seconds")
        print(f"Missing values after imputation:")
        print(f"  Training: {train_missing_after:,}/{train_imputed.size:,}")
        print(f"  Test: {test_missing_after:,}/{test_imputed.size:,}")
        
        if train_missing_after == 0 and test_missing_after == 0:
            print("✅ All missing values successfully imputed!")
            success = True
        else:
            print("⚠️  Some missing values remain")
            success = False
        
        # Compute accuracy if we have true data
        accuracy_results = None
        if train_full_data is not None:
            print("\nComputing imputation accuracy...")
            accuracy_results = compute_imputation_accuracy(
                train_imputed, 
                train_full_data, 
                train_data,
                max_rows=10000  # Same limit as imputation
            )
            
            # Compute confidence interval for accuracy using binomial distribution
            acc_prop, acc_lower, acc_upper = compute_binomial_confidence_interval(
                accuracy_results['n_correct'], 
                accuracy_results['n_evaluated']
            )
            
            # Add confidence interval to results
            accuracy_results['accuracy_95ci_lower'] = acc_lower
            accuracy_results['accuracy_95ci_upper'] = acc_upper
            
            print(f"Imputation accuracy: {accuracy_results['accuracy']:.3f} (95% CI: [{acc_lower:.3f}, {acc_upper:.3f}])")
            print(f"  Evaluated {accuracy_results['n_evaluated']:,} imputed values")
            print(f"  ({accuracy_results['n_correct']:,} correct predictions)")
            print(f"  Note: {accuracy_results['n_imputed'] - accuracy_results['n_imputed_with_truth']:,} imputed values had no ground truth")
            
            # Print first row comparison for debugging
            print(f"\nFirst row comparison for {dataset_name}:")
            print(f"  True row:    {train_full_data[0]}")
            print(f"  Missing row: {train_data[0]}")
            print(f"  Argmax row:  {train_imputed[0]}")
            
            # Show which values were imputed in first row
            imputed_positions = jnp.where(train_data[0] == MISSING_VALUE)[0]
            if len(imputed_positions) > 0:
                print(f"  Imputed positions: {imputed_positions}")
                print(f"  True values at imputed positions: {train_full_data[0][imputed_positions]}")
                print(f"  Argmax values at imputed positions: {train_imputed[0][imputed_positions]}")
                first_row_correct = jnp.sum(train_full_data[0][imputed_positions] == train_imputed[0][imputed_positions])
                print(f"  First row accuracy: {first_row_correct}/{len(imputed_positions)} = {first_row_correct/len(imputed_positions):.3f}")
            else:
                print(f"  No missing values in first row")
        
        # Prepare results
        results = {
            'dataset_name': dataset_name,
            'success': success,
            'imputation_time': imputation_time,
            'model_path': model_path,
            'train_data_original': train_data,
            'test_data_original': test_data,
            'train_data_imputed': train_imputed,
            'test_data_imputed': test_imputed,
            'missing_stats': {
                'train_missing_before': int(train_missing_before),
                'train_missing_after': int(train_missing_after),
                'test_missing_before': int(test_missing_before),
                'test_missing_after': int(test_missing_after),
                'train_missing_pct_before': float(100 * train_missing_before / train_data.size),
                'train_missing_pct_after': float(100 * train_missing_after / train_imputed.size),
                'test_missing_pct_before': float(100 * test_missing_before / test_data.size),
                'test_missing_pct_after': float(100 * test_missing_after / test_imputed.size),
            },
            'accuracy_results': accuracy_results,
            'timing_stats': timing_stats,
            'data_info': {
                'col_names': col_names,
                'train_shape': train_data.shape,
                'test_shape': test_data.shape,
                'n_features': len(col_names),
                'max_categories': K,
                'batch_size': batch_size,
                'missing_value_constant': MISSING_VALUE,
            },
            'model_info': {
                'use_all_particles': use_all_particles,
                'theta_shape': theta_particles.shape if use_all_particles else theta.shape,
                'pi_shape': pi_particles.shape if use_all_particles else pi.shape,
                'theta_range': [float(jnp.min(theta_particles if use_all_particles else theta)), 
                               float(jnp.max(theta_particles if use_all_particles else theta))],
                'pi_range': [float(jnp.min(pi_particles if use_all_particles else pi)), 
                            float(jnp.max(pi_particles if use_all_particles else pi))],
                'n_particles': theta_particles.shape[0] if use_all_particles else 1,
                'log_weights': log_weights.tolist() if use_all_particles else None,
            }
        }
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"argmax_imputation_{dataset_name.lower()}_results.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {output_file}")
        
        # Clean up memory
        del train_data, test_data, train_imputed, test_imputed
        if use_all_particles:
            del theta_particles, pi_particles, log_weights
        else:
            del theta, pi
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"❌ {dataset_name} failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'dataset_name': dataset_name,
            'success': False,
            'error': str(e),
            'imputation_time': 0,
        }


def run_all_datasets(
    data_root: str = "../model-building-evaluation",
    results_dir: str = "results_missing",
    batch_size: int = 1000,
    output_dir: str = "argmax_imputation_results",
    datasets_to_run: List[str] = None,
    use_all_particles: bool = True
) -> Dict[str, Any]:
    """Run argmax imputation on all datasets."""
    
    # Determine which datasets to run
    if datasets_to_run is None:
        all_datasets = []
        for dataset_list in DATASETS.values():
            all_datasets.extend(dataset_list)
        datasets_to_run = all_datasets
    
    print(f"Running argmax imputation on {len(datasets_to_run)} datasets")
    print(f"Datasets: {datasets_to_run}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track results
    all_results = []
    total_start = time.time()
    
    # Run each dataset
    for dataset_name in datasets_to_run:
        dataset_results = run_argmax_imputation_for_dataset(
            dataset_name, data_root, results_dir, batch_size, output_dir, use_all_particles
        )
        all_results.append(dataset_results)
    
    # Final summary
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - ARGMAX IMPUTATION ON ALL DATASETS")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print("\nResults:")
    
    successes = 0
    for res in all_results:
        status = "✅" if res.get("success", False) else "❌"
        time_str = f"{res.get('imputation_time', 0):.2f}s"
        missing_info = ""
        accuracy_info = ""
        if 'missing_stats' in res:
            missing_before = res['missing_stats']['train_missing_pct_before']
            missing_after = res['missing_stats']['train_missing_pct_after']
            missing_info = f" ({missing_before:.1f}% → {missing_after:.1f}%)"
        if 'accuracy_results' in res and res['accuracy_results'] is not None:
            accuracy = res['accuracy_results']['accuracy']
            acc_lower = res['accuracy_results'].get('accuracy_95ci_lower', 0)
            acc_upper = res['accuracy_results'].get('accuracy_95ci_upper', 0)
            accuracy_info = f", acc: {accuracy:.3f} [{acc_lower:.3f}, {acc_upper:.3f}]"
        
        timing_info = ""
        if 'timing_stats' in res and res['timing_stats'] is not None:
            time_per_row = res['timing_stats']['time_per_row_mean'] * 1000  # Convert to ms
            timing_info = f", {time_per_row:.1f}ms/row"
        
        print(f"  {status} {res['dataset_name']}: {time_str}{missing_info}{accuracy_info}{timing_info}")
        if res.get("success", False):
            successes += 1
    
    print(f"\n✅ Completed: {successes}/{len(datasets_to_run)} datasets")
    
    # Save combined results
    summary_results = {
        'total_time': total_time,
        'total_datasets': len(datasets_to_run),
        'successful_datasets': successes,
        'batch_size': batch_size,
        'datasets_processed': datasets_to_run,
        'individual_results': all_results,
        'missing_value_constant': MISSING_VALUE,
        'timestamp': time.time()
    }
    
    summary_file = output_path / "argmax_imputation_all_results.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(summary_results, f)
    
    print(f"Combined results saved to {summary_file}")
    
    return summary_results


def main():
    parser = argparse.ArgumentParser(description="Run argmax conditional imputation on datasets with missing data")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Specific dataset to run (default: all)")
    parser.add_argument("--batch-size", "-B", type=int, default=1000,
                       help="Batch size for processing (default: 1000)")
    parser.add_argument("--data-root", type=str, default="../model-building-evaluation",
                       help="Root directory for model-building-evaluation data")
    parser.add_argument("--results-dir", type=str, default="results_missing",
                       help="Directory containing trained models (default: results_missing)")
    parser.add_argument("--output-dir", type=str, default="argmax_imputation_results",
                       help="Output directory for imputation results (default: argmax_imputation_results)")
    parser.add_argument("--use-all-particles", action="store_true", default=True,
                       help="Use all particles for weighted inference (default: True)")
    parser.add_argument("--single-particle", action="store_true",
                       help="Use only first particle (overrides --use-all-particles)")
    
    args = parser.parse_args()
    
    # Run experiment
    if args.dataset:
        # Run single dataset
        print(f"Running single dataset: {args.dataset}")
        use_all_particles = not args.single_particle
        result = run_argmax_imputation_for_dataset(
            args.dataset, args.data_root, args.results_dir, args.batch_size, args.output_dir, use_all_particles
        )
        print(f"\nSingle dataset run completed: {'✅' if result.get('success', False) else '❌'}")
    else:
        # Run all datasets
        print("Running all datasets")
        use_all_particles = not args.single_particle
        summary_results = run_all_datasets(
            args.data_root, args.results_dir, args.batch_size, args.output_dir, 
            use_all_particles=use_all_particles
        )
        print(f"\nAll datasets completed: {summary_results['successful_datasets']}/{summary_results['total_datasets']} successful")


if __name__ == "__main__":
    main()