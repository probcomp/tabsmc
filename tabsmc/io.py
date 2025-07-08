import polars as pl
import numpy as np
from os import getenv

ACCESS_TOKEN = getenv("HF_TOKEN")


def discretize_dataframe(df: pl.DataFrame, n_bins: int = 20):
    schema = make_schema(df)
    categorical_df = df.select(schema["types"]["categorical"])
    numerical_df = df.select(schema["types"]["numerical"])
    
    # Calculate and store quantiles for numerical columns
    for col in schema["types"]["numerical"]:
        quantiles = np.linspace(0, 1, n_bins + 1)
        quantile_values = [numerical_df[col].quantile(q) for q in quantiles]
        schema["var_metadata"][col] = {"quantiles": quantile_values, "n_bins": n_bins}
    
    numerical_df = numerical_df.with_columns(
        pl.all()
        .qcut(quantiles=n_bins, labels=[str(i) for i in range(n_bins)])
        .name.keep()
    )

    categorical_idxs = np.concatenate(
        [
            idx * np.ones(len(schema["var_metadata"][col]["levels"]))
            for idx, col in enumerate(categorical_df.columns)
        ]
        + [
            len(categorical_df.columns) + idx * np.ones(n_bins)
            for idx, col in enumerate(numerical_df.columns)
        ]
    ).astype(np.int32)

    df = pl.concat((categorical_df, numerical_df), how="horizontal")
    return schema, df, categorical_idxs


# Removed to_dummies function - no longer needed with integer encoding


def dummies_to_padded_array(df: pl.DataFrame, categorical_idxs: np.ndarray):
    data = df.to_numpy().astype(np.bool_)
    n_categories = max(categorical_idxs) + 1
    max_categories = max(sum(categorical_idxs == i) for i in range(n_categories))
    col_names = []

    padded_data = np.zeros(
        (data.shape[0], n_categories, max_categories), dtype=np.bool_
    )
    cat_lenghts = []
    for i in range(n_categories):
        cat_length = sum(categorical_idxs == i)
        cat_start = np.where(categorical_idxs == i)[0][0]
        padded_data[:, i, :cat_length] = data[:, cat_start : cat_start + cat_length]
        col_names.append(df.columns[cat_start].rsplit("_", maxsplit=1)[0])
        cat_lenghts.append(cat_length)

    return padded_data, col_names, cat_lenghts


def load_huggingface(dataset_path):
    splits = {
        "train": f"{dataset_path}/data-train-num.parquet",
        "test": f"{dataset_path}/data-test-full-num.parquet",
    }
    train_df = pl.read_parquet(
        f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['train']}",
        storage_options={"token": ACCESS_TOKEN},
    )
    test_df = pl.read_parquet(
        f"hf://datasets/Large-Population-Model/model-building-evaluation/{splits['test']}",
        storage_options={"token": ACCESS_TOKEN},
    )

    return train_df, test_df


def make_schema(df: pl.DataFrame):
    schema = {"types": {"numerical": [], "categorical": []}, "var_metadata": {}}
    for c in df.columns:
        if df[c].dtype == pl.Utf8:
            schema["types"]["categorical"].append(c)
            schema["var_metadata"][c] = {
                "levels": df[c].drop_nulls().unique().sort().to_list()
            }
        elif df[c].dtype == pl.Float64:
            schema["types"]["numerical"].append(c)
        else:
            raise ValueError(c)
    return schema


def discretize_with_schema(df: pl.DataFrame, schema: dict):
    """Discretize a dataframe using quantiles from a pre-computed schema."""
    categorical_df = df.select(schema["types"]["categorical"])
    numerical_df = df.select(schema["types"]["numerical"])
    
    # Discretize numerical columns using schema quantiles
    for col in schema["types"]["numerical"]:
        if col in numerical_df.columns:
            quantiles = schema["var_metadata"][col]["quantiles"]
            n_bins = schema["var_metadata"][col]["n_bins"]
            
            # Create binning expression
            # Map values to bins based on quantiles
            bin_expr = pl.lit(n_bins - 1)  # Default to last bin
            for i in range(n_bins - 1, 0, -1):
                bin_expr = pl.when(pl.col(col) <= quantiles[i]).then(pl.lit(i - 1)).otherwise(bin_expr)
            
            numerical_df = numerical_df.with_columns(
                bin_expr.alias(col)
            )
    
    # Combine categorical and discretized numerical columns
    df = pl.concat((categorical_df, numerical_df), how="horizontal")
    return df


def encode_with_schema(df: pl.DataFrame, schema: dict):
    """Convert categorical variables to integer indices using schema."""
    result_df = df.clone()
    
    # Encode categorical variables
    for col in schema["types"]["categorical"]:
        if col in result_df.columns:
            levels = schema["var_metadata"][col]["levels"]
            # Create mapping from level to index
            level_to_idx = {level: idx for idx, level in enumerate(levels)}
            
            # Apply mapping using replace
            result_df = result_df.with_columns(
                pl.col(col).replace(level_to_idx, default=-1).alias(col)
            )
    
    # Numerical columns (already discretized) just need to be converted to int
    for col in schema["types"]["numerical"]:
        if col in result_df.columns:
            result_df = result_df.with_columns(
                pl.col(col).cast(pl.Int32).alias(col)
            )
    
    return result_df


def from_dummies(df, separator="_"):
    col_exprs = {}

    for col in df.columns:
        name, value = col.rsplit(separator, maxsplit=1)
        expr = pl.when(pl.col(col) == 1).then(pl.lit(value))
        col_exprs.setdefault(name, []).append(expr)

    return df.select(
        pl.coalesce(exprs).alias(  # keep the first non-null expression value by row
            name
        )
        for name, exprs in col_exprs.items()
    )


def load_data(dataset_path):
    """Load data in integer format with proper missing value handling."""
    from tabsmc.smc import MISSING_VALUE
    
    train_df, test_df = load_huggingface(dataset_path)
    
    # Process training and test data separately to avoid memory issues
    print("Processing training data...")
    train_schema, train_discretized, train_categorical_idxs = discretize_dataframe(train_df)
    train_encoded = encode_with_schema(train_discretized, train_schema)
    
    print("Processing test data...")
    test_discretized = discretize_with_schema(test_df, train_schema)
    test_encoded = encode_with_schema(test_discretized, train_schema)
    
    # Convert to numpy arrays with proper missing value handling
    train_data_raw = train_encoded.to_numpy()
    test_data_raw = test_encoded.to_numpy()
    
    # Handle missing values by replacing NaN with MISSING_VALUE
    train_data_clean = np.where(np.isnan(train_data_raw), MISSING_VALUE, train_data_raw)
    test_data_clean = np.where(np.isnan(test_data_raw), MISSING_VALUE, test_data_raw)
    
    # Convert to uint32 to prevent negative indexing issues
    train_data = train_data_clean.astype(np.uint32)
    test_data = test_data_clean.astype(np.uint32)
    
    # Get column names
    col_names = train_encoded.columns
    
    # Determine K (max categories per feature)
    # For each feature, find the maximum valid category index
    K_values = []
    for col in col_names:
        if col in train_schema["types"]["categorical"]:
            K_col = len(train_schema["var_metadata"][col]["levels"])
        else:  # numerical
            K_col = train_schema["var_metadata"][col]["n_bins"]
        K_values.append(K_col)
    
    K = max(K_values)
    
    # Create mask for valid categories per feature
    mask = np.zeros((len(col_names), K), dtype=bool)
    for i, (col, K_col) in enumerate(zip(col_names, K_values)):
        mask[i, :K_col] = True
    
    print(f"Loaded data shapes: train={train_data.shape}, test={test_data.shape}")
    print(f"Max categories K: {K}")
    print(f"Missing values in train: {np.sum(train_data == MISSING_VALUE):,}")
    print(f"Missing values in test: {np.sum(test_data == MISSING_VALUE):,}")
    
    return train_data, test_data, col_names, mask, K
