import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool
from functools import partial
import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon
from typing import Union, Optional, Dict, Any


# @partial(jax.jit, static_argnames=("batch_size",))
def js(x: Bool[Array, "n k"], y: Bool[Array, "m k"], batch_size: int = 1000) -> Array:
    k = x.shape[1]
    col_combinations = jnp.array([
        [i, j] for i in range(k) for j in range(i + 1, k)
    ])

    x_indexed = jax.vmap(lambda c: x.take(c, axis=1))(col_combinations)
    y_indexed = jax.vmap(lambda c: y.take(c, axis=1))(col_combinations)

    # return jax.lax.map(batch_js_bivariate, (x_indexed, y_indexed), batch_size=batch_size)
    return jax.vmap(js_bivariate, in_axes=(0, 0))(x_indexed, y_indexed)

def batch_js_bivariate(args):
    x, y = args
    return js_bivariate(x, y)

def js_bivariate(x: Bool[Array, "n 2"], y: Bool[Array, "m 2"]) -> Array:
    vals = jnp.array([
        [True, True],
        [True, False],
        [False, True],
        [False, False],
    ])

    x_vals = jax.vmap(jax.vmap(jnp.array_equal, in_axes=(0, None)), in_axes=(None, 0))(x, vals)
    x_vals = jnp.mean(x_vals, axis=1)
    y_vals = jax.vmap(jax.vmap(jnp.array_equal, in_axes=(0, None)), in_axes=(None, 0))(y, vals)
    y_vals = jnp.mean(y_vals, axis=1)

    m_vals = (x_vals + y_vals) / 2

    kl_x_m = jnp.nansum(x_vals * jnp.log(x_vals / m_vals)) 
    kl_y_m = jnp.nansum(y_vals * jnp.log(y_vals / m_vals))

    return (kl_x_m + kl_y_m) / 2


def js_distance_dataframes(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    train_df: Optional[pl.DataFrame] = None,
    n_bins: int = 20,
    epsilon: float = 1e-10
) -> float:
    """
    Compute JS distance between two dataframes with mixed categorical/numerical data.
    
    For numerical columns, discretizes using quantiles from train_df (or df1 if not provided).
    For categorical columns, uses the raw values.
    
    Args:
        df1: First dataframe
        df2: Second dataframe  
        train_df: Training dataframe to compute quantiles from (optional)
        n_bins: Number of bins for discretizing numerical features
        epsilon: Small value to avoid log(0)
    
    Returns:
        Average JS distance across all features
    """
    if train_df is None:
        train_df = df1
    
    js_distances = []
    
    for col in df1.columns:
        if col not in df2.columns:
            continue
            
        # Handle numerical columns
        if train_df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            # Compute quantiles from training data
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = [train_df[col].quantile(q) for q in quantiles]
            bins = sorted(list(set(bins)))  # Remove duplicates and sort
            
            # Discretize both dataframes using training quantiles
            vals1 = np.digitize(df1[col].to_numpy(), bins)
            vals2 = np.digitize(df2[col].to_numpy(), bins)
        else:
            # For categorical columns, use raw values
            vals1 = df1[col].to_numpy()
            vals2 = df2[col].to_numpy()
        
        # Convert to strings for consistent comparison
        vals1 = vals1.astype(str)
        vals2 = vals2.astype(str)
        
        # Get all unique values
        all_vals = np.unique(np.concatenate([vals1, vals2]))
        
        # Compute empirical distributions
        p1 = np.array([np.mean(vals1 == val) for val in all_vals])
        p2 = np.array([np.mean(vals2 == val) for val in all_vals])
        
        # Add epsilon and normalize
        p1 = (p1 + epsilon) / np.sum(p1 + epsilon)
        p2 = (p2 + epsilon) / np.sum(p2 + epsilon)
        
        # Compute JS distance for this feature
        js_dist = jensenshannon(p1, p2)
        js_distances.append(js_dist)
    
    return np.mean(js_distances)


def js_distance_categorical(
    X1: np.ndarray,
    X2: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute JS distance between two datasets of categorical variables.
    
    Args:
        X1: First dataset (n_samples1, n_features), can be strings or integers
        X2: Second dataset (n_samples2, n_features), can be strings or integers
        epsilon: Small value to avoid log(0)
    
    Returns:
        Average JS distance across all features
    """
    n_features = X1.shape[1]
    js_distances = []
    
    for i in range(n_features):
        cats1 = X1[:, i]
        cats2 = X2[:, i]
        
        # Get all unique categories
        all_cats = np.unique(np.concatenate([cats1, cats2]))
        
        # Compute empirical distributions
        p1 = np.array([np.mean(cats1 == cat) for cat in all_cats])
        p2 = np.array([np.mean(cats2 == cat) for cat in all_cats])
        
        # Add epsilon and normalize
        p1 = (p1 + epsilon) / np.sum(p1 + epsilon)
        p2 = (p2 + epsilon) / np.sum(p2 + epsilon)
        
        # Compute JS distance
        js_dist = jensenshannon(p1, p2)
        js_distances.append(js_dist)
    
    return np.mean(js_distances)


def js_distance_onehot(
    X1: Union[np.ndarray, jnp.ndarray],
    X2: Union[np.ndarray, jnp.ndarray],
    batch_size: int = 1000,
    method: str = "bivariate"
) -> float:
    """
    Compute JS distance between two one-hot encoded datasets.
    
    Args:
        X1: First dataset (n_samples1, n_features, n_categories) or flattened
        X2: Second dataset (n_samples2, n_features, n_categories) or flattened
        batch_size: Batch size for JAX computation
        method: "bivariate" to use pairwise feature JS, "marginal" for per-feature JS
    
    Returns:
        JS distance (scalar)
    """
    # Ensure JAX arrays
    X1 = jnp.array(X1) if not isinstance(X1, jnp.ndarray) else X1
    X2 = jnp.array(X2) if not isinstance(X2, jnp.ndarray) else X2
    
    # If 3D, flatten to 2D
    if len(X1.shape) == 3:
        n1, d, k = X1.shape
        X1 = X1.reshape(n1, d * k)
    if len(X2.shape) == 3:
        n2, d, k = X2.shape
        X2 = X2.reshape(n2, d * k)
    
    # Convert to boolean
    X1_bool = X1.astype(bool)
    X2_bool = X2.astype(bool)
    
    if method == "bivariate":
        # Use existing bivariate JS function
        js_jit = jax.jit(js, static_argnames=("batch_size",))
        distances = js_jit(X1_bool, X2_bool, batch_size=batch_size)
        return float(jnp.mean(distances))
    else:
        # Marginal JS distance
        js_distances = []
        n_features = X1_bool.shape[1]
        
        for i in range(n_features):
            p1 = jnp.mean(X1_bool[:, i])
            p2 = jnp.mean(X2_bool[:, i])
            
            # Binary JS distance
            p1_vec = jnp.array([1 - p1, p1])
            p2_vec = jnp.array([1 - p2, p2])
            m_vec = (p1_vec + p2_vec) / 2
            
            kl_1_m = jnp.sum(p1_vec * jnp.log(p1_vec / m_vec + 1e-10))
            kl_2_m = jnp.sum(p2_vec * jnp.log(p2_vec / m_vec + 1e-10))
            js_dist = (kl_1_m + kl_2_m) / 2
            
            js_distances.append(float(js_dist))
        
        return np.mean(js_distances)


def onehot_to_categorical(
    X_onehot: Union[np.ndarray, jnp.ndarray],
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert one-hot encoded array to categorical (integer) representation.
    
    Args:
        X_onehot: One-hot array of shape (n_samples, n_features, n_categories)
        mask: Optional mask indicating valid categories per feature
    
    Returns:
        Categorical array of shape (n_samples, n_features)
    """
    X_onehot = np.array(X_onehot) if isinstance(X_onehot, jnp.ndarray) else X_onehot
    
    if len(X_onehot.shape) == 2:
        # Assume it needs to be reshaped based on mask
        if mask is None:
            raise ValueError("Mask required for 2D input")
        n_samples = X_onehot.shape[0]
        n_features = mask.shape[0]
        n_categories = mask.shape[1]
        X_onehot = X_onehot.reshape(n_samples, n_features, n_categories)
    
    # Convert to categorical by taking argmax
    X_categorical = np.argmax(X_onehot, axis=2)
    
    return X_categorical
