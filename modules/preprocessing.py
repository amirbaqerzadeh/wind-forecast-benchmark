import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Store original pd.read_csv for safe wrapper
__original_csv = pd.read_csv


def safe_read_csv(path, *args, **kwargs):
    """
    Safe wrapper around pd.read_csv to handle common issues.
    ````````````````````````
    This function provides enhanced error handling and logging for CSV reading operations.
    It can be used as a drop-in replacement for pd.read_csv.
    
    Parameters:
    - path (str): Path to the CSV file.
    - *args: Additional positional arguments passed to pd.read_csv.
    - **kwargs: Additional keyword arguments passed to pd.read_csv.
    
    Returns:
    - pd.DataFrame: The loaded DataFrame.
    
    Raises:
    - FileNotFoundError: If the CSV file doesn't exist.
    - pd.errors.EmptyDataError: If the CSV file is empty.
    - pd.errors.ParserError: If there's an error parsing the CSV.
    - Exception: For any other unexpected errors.
    """
    try:
        return __original_csv(path, *args, **kwargs)
    except FileNotFoundError:
        logger.error("CSV file not found: %s", path)
        raise
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty: %s", path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file: %s. Error: %s", path, e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred while reading CSV file: %s. Error: %s", path, e)
        raise


def load_and_basic_clean(
    path: str,
    required_columns: Optional[List[str]] = None,
    sort_by: Optional[str] = 'datetime',
    dtype_cast: Optional[Dict[str, Any]] = None,
    return_metadata: bool = False
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Load a CSV file into a DataFrame, select required columns, sort, and cast data types.
    
    Parameters:
    - path (str): Path to the CSV file.
    - required_columns (list of str, optional): List of columns to select from the DataFrame.
        If None, all columns are kept.
    - sort_by (str, optional): Column name to sort the DataFrame by. Default is 'datetime'.
        Set to None to skip sorting.
    - dtype_cast (dict, optional): Dictionary specifying data types to cast columns to.
        Example: {'price': float, 'quantity': int}
    - return_metadata (bool, optional): If True, returns metadata about the loading process.
        Default is False.
    
    Returns:
    - pd.DataFrame: The cleaned DataFrame.
    - dict (optional): Metadata about the loading process (only if return_metadata=True).
        Contains: rows_loaded, columns_original, columns_selected, columns_sorted_by, 
        columns_cast, warnings
    
    Raises:
    - KeyError: If required columns are missing from the DataFrame.
    - ValueError: If there's an error casting column data types.
    
    Example:
```
    df = load_and_basic_clean(
        'data.csv',
        required_columns=['datetime', 'price', 'volume'],
        sort_by='datetime',
        dtype_cast={'price': float, 'volume': int}
    )
```
    """
    metadata = {
        'rows_loaded': 0,
        'columns_original': 0,
        'columns_selected': 0,
        'columns_sorted_by': None,
        'columns_cast': [],
        'warnings': []
    }
    
    # Load CSV using safe wrapper
    df = safe_read_csv(path)
    metadata['rows_loaded'] = len(df)
    metadata['columns_original'] = len(df.columns)
    
    logger.info("Loaded CSV from '%s': %d rows, %d columns", path, len(df), len(df.columns))
    
    # Select required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            raise KeyError(error_msg)
        df = df[required_columns]
        metadata['columns_selected'] = len(df.columns)
        logger.info("Selected %d required columns", len(required_columns))
    else:
        metadata['columns_selected'] = metadata['columns_original']
    
    # Sort DataFrame
    if sort_by:
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by).reset_index(drop=True)
            metadata['columns_sorted_by'] = sort_by
            logger.info("Sorted DataFrame by column '%s'", sort_by)
        else:
            warning_msg = f"Sort column '{sort_by}' not found in DataFrame. Skipping sorting."
            logger.warning(warning_msg)
            metadata['warnings'].append(warning_msg)
    
    # Cast data types
    if dtype_cast:
        for col, dtype in dtype_cast.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                    metadata['columns_cast'].append(col)
                    logger.info("Cast column '%s' to %s", col, dtype)
                except Exception as e:
                    error_msg = f"Error casting column '{col}' to {dtype}: {e}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                warning_msg = f"Column '{col}' not found in DataFrame. Skipping dtype casting."
                logger.warning(warning_msg)
                metadata['warnings'].append(warning_msg)
    
    logger.info("Data loading and cleaning completed successfully")
    
    if return_metadata:
        return df, metadata
    return df, None


# Optional: If you still want to monkey-patch pd.read_csv globally
# Uncomment the line below, but be aware this affects all pandas operations in your app
# pd.read_csv = safe_read_csv
            
def add_time_features(df, datetime_col='datetime', cyclical_cols=None):
    """
    Add time-based features to the DataFrame.
    - Extract hour, month, day of week
    - Create cyclical features (sin/cos) for specific columns.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - datetime_col (str): Name of the datetime column. Default is 'datetime'.
    - cyclical_cols (list of str, optional): List of columns to encode cyclically. Default is None.
    
    Returns:
    - pd.DataFrame: DataFrame with added time features.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    if datetime_col not in df.columns:
        raise KeyError(f"Datetime column '{datetime_col}' not found in DataFrame.")
    
    if not np.issubdtype(df[datetime_col].dtype, np.datetime64):
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        except Exception as e:
            raise ValueError(f"Error converting column '{datetime_col}' to datetime: {e}")

    # Extract time features
    df['hour'] = df[datetime_col].dt.hour
    df['month'] = df[datetime_col].dt.month
    df['day_of_week'] = df[datetime_col].dt.dayofweek

    # Create cyclical features
    if cyclical_cols is None:
        cyclical_cols = ['hour']
    
    for col in cyclical_cols:
        if col not in df.columns:
            logger.warning("Cyclical column '%s' not found in DataFrame. Skipping cyclical encoding.", col)
            continue
        
        period = 24 if col == 'hour' else 12 if col == 'month' else 7 if col == 'day_of_week' else None
        
        if period is None:
            logger.warning("No defined period for column '%s'. Skipping cyclical encoding.", col)
            continue
        
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    
    return df      



def add_lag_and_rolling_features(df, target_cols, lags=None, rolling_windows=None, dropna=True):
    """
    Add lag and rolling window features with leakage prevention and optimization.
    """
    # Normalize target_cols to list
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    
    # Input validation (Keep your existing validation logic here)
    if not isinstance(target_cols, list) or len(target_cols) == 0:
        raise ValueError("target_cols must be a non-empty string or list.")
    missing_cols = [col for col in target_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Target columns not found: {missing_cols}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted.")

    # Defaults
    lags = lags or [1, 3, 6]
    rolling_windows = rolling_windows or [3, 6, 12]

    # Container for new features
    new_features = []

    for target_col in target_cols:
        # 1. Add Lag Features
        for lag in lags:
            feat_name = f'{target_col}_lag_{lag}'
            new_features.append(
                df[target_col].shift(lag).rename(feat_name)
            )
        
        # 2. Add Rolling Features (Shifted to prevent leakage)
        # We shift by 1 so the window uses t-1, t-2, t-3... 
        # independent of the current value at t.
        target_shifted = df[target_col].shift(1)
        
        for window in rolling_windows:
            new_features.append(target_shifted.rolling(window).mean().rename(f'{target_col}_roll_mean_{window}'))
            new_features.append(target_shifted.rolling(window).std().rename(f'{target_col}_roll_std_{window}'))
            new_features.append(target_shifted.rolling(window).min().rename(f'{target_col}_roll_min_{window}'))
            new_features.append(target_shifted.rolling(window).max().rename(f'{target_col}_roll_max_{window}'))
    df_out = pd.concat([df] + new_features, axis=1)
    
     # HANDLE NANS HERE
    if dropna:
        df_out = df_out.dropna()
        
    return df_out
def split_train_test(
    df,
    target_col,
    test_size=720,
    drop_cols=None      
):
    """
    Split time-series DataFrame into train/test sets using last N rows as test.
    Handles both single string targets and list of targets.
    """
    # 1. Normalize target_col to a list for consistent logic
    if isinstance(target_col, str):
        target_cols = [target_col]
    else:
        target_cols = target_col

    # Validation
    missing_targets = [t for t in target_cols if t not in df.columns]
    if missing_targets:
        raise KeyError(f"Target columns not found: {missing_targets}")
        
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted before splitting.")
        
    if not isinstance(test_size, int) or test_size <= 0:
        raise ValueError("test_size must be a positive integer.")
    
    if test_size >= len(df):
        raise ValueError("test_size is larger than the DataFrame length.")
    
    # Handle columns to drop
    if drop_cols is None:
        drop_cols = []
    # Filter only existing columns to drop
    drop_cols = [c for c in drop_cols if c in df.columns]

    # 2. Define X and y
    # If target_col was a single string, y returns a Series. 
    # If list, y returns a DataFrame.
    y = df[target_col] 
    
    # Drop columns + targets from X
    cols_to_exclude = drop_cols + target_cols
    X = df.drop(columns=cols_to_exclude)

    # 3. Split by Index
    train_end = len(df) - test_size

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[train_end:]
    y_test = y.iloc[train_end:]

    # Keep a copy of the raw test data for plotting later
    test_df = df.iloc[train_end:].copy()

    return X_train, y_train, X_test, y_test, test_df


def build_feature_list(
    df,
    lag_prefix='_L',
    include_current=None,
    include_rolling=None,
    include_time_features=None
):
    """
    Build a validated list of feature columns for ML models.

    Parameters:
    - df: DataFrame containing all engineered features.
    - lag_prefix: str — substring that identifies lag features (default "_L").
    - include_current: list of str — columns to include as 'current' features.
    - include_rolling: list of str — rolling/statistical features.
    - include_time_features: list of str — cyclic or datetime-based features.

    Returns:
    - List[str] of valid feature names
    """
    features = []
    
    # Get lag columns
    lag_cols = [c for c in df.columns if lag_prefix in c]
    features.extend(lag_cols)

    # Validate and add current features
    if include_current:
        missing = [c for c in include_current if c not in df.columns]  # Fixed: was "if c in"
        if missing:
            raise KeyError(f"Current observation columns not found: {missing}")
        features.extend(include_current)
    
    # Validate and add rolling features
    if include_rolling:
        missing = [c for c in include_rolling if c not in df.columns]
        if missing:
            raise KeyError(f"Rolling feature columns not found: {missing}")
        features.extend(include_rolling)

    # Validate and add time features
    if include_time_features:  # Fixed typo: was "incluce_time_features"
        missing = [c for c in include_time_features if c not in df.columns]
        if missing:
            raise KeyError(f"Time-feature columns not found: {missing}")
        features.extend(include_time_features)
    
    # Remove duplicates while preserving order
    features = list(dict.fromkeys(features))

    return features



def prepare_dataset(
    path,
    required_columns=None,
    dtypes=None,
    sort_by='datetime',
    datetime_col='datetime',
    time_features=True,
    cyclical_cols=None,
    lag_cols=None,
    lags=None,
    rolling_targets=None,
    rolling_windows=None,
    target_col='target_wind_speed',
    test_size=720,
    return_metadata=False
):
    """
    Master dataset preparation pipeline.
    
    Runs loading, cleaning, feature engineering, target creation,
    feature list building, and train/test split.
    
    Parameters:
    - path (str): Path to the CSV file.
    - required_columns (list of str, optional): Columns to load from CSV.
    - dtypes (dict, optional): Data types for columns.
    - sort_by (str, optional): Column to sort by. Default is 'datetime'.
    - datetime_col (str): Name of datetime column. Default is 'datetime'.
    - time_features (bool): Whether to add time-based features. Default is True.
    - cyclical_cols (list of str, optional): Columns for cyclical encoding. Default is ['hour'].
    - lag_cols (list of str, optional): Columns to create lag features for.
    - lags (list of int, optional): Lag periods. Default is [1, 3, 6].
    - rolling_targets (list of str, optional): Columns for rolling features.
    - rolling_windows (list of int, optional): Rolling window sizes. Default is [3, 6, 12].
    - target_col (str): Name for the target column. Default is 'target_wind_speed'.
    - test_size (int): Number of rows for test set. Default is 720.
    - return_metadata (bool): Whether to return metadata. Default is False.
    
    Returns:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series): Test target.
    - FEATS (list): List of feature column names.
    - df (pd.DataFrame): Full processed DataFrame.
    - test_df (pd.DataFrame): Test subset with all columns.
    - metadata (dict, optional): Pipeline metadata if return_metadata=True.
    
    Raises:
    - KeyError: If required columns are missing.
    - ValueError: If data processing fails.
    
    Example:
```
    X_train, y_train, X_test, y_test, features, df, test_df = prepare_dataset(
        path='weather_data.csv',
        required_columns=['datetime', 'wind_speed', 'pressure', 'temp'],
        lag_cols=['wind_speed'],
        lags=[1, 3, 6],
        rolling_targets=['wind_speed'],
        rolling_windows=[3, 6, 12]
    )
```
    """
    metadata = {
        'original_shape': None,
        'final_shape': None,
        'features_count': 0,
        'train_size': 0,
        'test_size': 0,
        'rows_dropped': 0
    }
    
    logger.info("Starting dataset preparation pipeline for: %s", path)
    
    # -------------------------------------------------------------
    # 1. Load & Basic Clean
    # -------------------------------------------------------------
    df, load_meta = load_and_basic_clean(
        path=path,
        required_columns=required_columns,
        sort_by=sort_by,
        dtype_cast=dtypes,  # Fixed: was 'dtyp' (typo)
        return_metadata=True
    )
    metadata['original_shape'] = (load_meta['rows_loaded'], load_meta['columns_selected'])
    logger.info("Step 1/7: Data loaded - %d rows, %d columns", *metadata['original_shape'])
    
    # -------------------------------------------------------------
    # 2. Datetime Parsing + Time Features
    # -------------------------------------------------------------
    if datetime_col not in df.columns:
        raise KeyError(f"Datetime column '{datetime_col}' not found in DataFrame.")
    
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    if time_features:
        df = add_time_features(
            df=df,
            datetime_col=datetime_col,
            cyclical_cols=cyclical_cols if cyclical_cols else ['hour']
        )
        logger.info("Step 2/7: Time features added")
    else:
        logger.info("Step 2/7: Time features skipped")
    
    # -------------------------------------------------------------
    # 3. Lag Features
    # -------------------------------------------------------------
    if lag_cols and lags:
        df = add_lag_and_rolling_features(
            df=df,
            target_cols=lag_cols,
            lags=lags,
            rolling_windows=None  # Only add lags here
        )
        logger.info("Step 3/7: Lag features added for columns: %s", lag_cols)
    else:
        logger.info("Step 3/7: Lag features skipped")
    
    # -------------------------------------------------------------
    # 4. Rolling Features
    # -------------------------------------------------------------
    if rolling_targets and rolling_windows:
        df = add_lag_and_rolling_features(
            df=df,
            target_cols=rolling_targets,
            lags=None,  # Only add rolling features here
            rolling_windows=rolling_windows
        )
        logger.info("Step 4/7: Rolling features added for columns: %s", rolling_targets)
    else:
        logger.info("Step 4/7: Rolling features skipped")
    
    # -------------------------------------------------------------
    # 5. Create Target Column (1-step ahead)
    # -------------------------------------------------------------
    if 'wind_speed' not in df.columns:
        raise KeyError("Column 'wind_speed' required for target creation.")
    
    df[target_col] = df['wind_speed'].shift(-1)
    
    rows_before_dropna = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after_dropna = len(df)
    metadata['rows_dropped'] = rows_before_dropna - rows_after_dropna
    
    logger.info("Step 5/7: Target column '%s' created, dropped %d rows with NaN", 
                target_col, metadata['rows_dropped'])
    
    # -------------------------------------------------------------
    # 6. Build Feature List
    # -------------------------------------------------------------
    # Collect rolling feature columns dynamically
    rolling_cols = [c for c in df.columns if 'roll_' in c]
    
    # Collect time feature columns if they were created
    time_feature_cols = []
    if time_features:
        time_feature_cols = [c for c in df.columns if c in ['hour', 'month', 'day_of_week'] 
                            or '_sin' in c or '_cos' in c]
    
    FEATS = build_feature_list(
        df=df,
        lag_prefix="_lag_",  # Match the naming convention from add_lag_and_rolling_features
        include_current=['pressure', 'temperature', 'u', 'v', 'wind_speed'] if 'pressure' in df.columns else None,
        include_rolling=rolling_cols if rolling_cols else None,
        include_time_features=time_feature_cols if time_feature_cols else None
    )
    metadata['features_count'] = len(FEATS)
    logger.info("Step 6/7: Built feature list with %d features", len(FEATS))
    
    # -------------------------------------------------------------
    # 7. Split train/test
    # -------------------------------------------------------------
    X_train, y_train, X_test, y_test, test_df = split_train_test(
        df=df,
        target_col=target_col,
        test_size=test_size,
        drop_cols=[datetime_col]
    )
    
    metadata['train_size'] = len(X_train)
    metadata['test_size'] = len(X_test)
    metadata['final_shape'] = df.shape
    
    logger.info("Step 7/7: Train/test split complete - Train: %d, Test: %d", 
                len(X_train), len(X_test))
    logger.info("Dataset preparation pipeline completed successfully")
    
    if return_metadata:
        return X_train, y_train, X_test, y_test, FEATS, df, test_df, metadata
    
    return X_train, y_train, X_test, y_test, FEATS, df, test_df




# At the end of preprocessing.py

# -------------------------------------------------------------
# Module Exports (for cleaner imports)
# -------------------------------------------------------------
__all__ = [
    # Loading functions
    'safe_read_csv',
    'load_and_basic_clean',
    
    # Feature engineering functions
    'add_time_features',
    'add_lag_and_rolling_features',
    'build_feature_list',
    
    # Splitting functions
    'split_train_test',
    
    # Pipeline functions
    'prepare_dataset',
]


# -------------------------------------------------------------
# Module-level configuration (optional)
# -------------------------------------------------------------
def set_log_level(level=logging.INFO):
    """
    Set the logging level for the preprocessing module.
    
    Parameters:
    - level: logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    
    Example:
```
    from preprocessing import set_log_level
    import logging
    set_log_level(logging.DEBUG)
```
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


# -------------------------------------------------------------
# Optional: Convenience function for common preprocessing
# -------------------------------------------------------------
def quick_prepare(path, target_col='target', test_size=720):
    """
    Quick dataset preparation with sensible defaults.
    
    Useful for rapid prototyping when you just want to get started quickly.
    
    Parameters:
    - path (str): Path to CSV file
    - target_col (str): Name of target column
    - test_size (int): Test set size
    
    Returns:
    - X_train, y_train, X_test, y_test, features
    """
    logger.info("Running quick_prepare with default settings")
    return prepare_dataset(
        path=path,
        target_col=target_col,
        test_size=test_size,
        time_features=True,
        lags=[1, 3, 6],
        rolling_windows=[3, 6, 12]
    )[:5]  # Return only the first 5 items (excluding df and test_df)




# -------------------------------------------------------------
# Default Configuration
# -------------------------------------------------------------
DEFAULT_CONFIG = {
    'path': None,  # Must be provided
    'required_columns': None,
    'dtypes': None,
    'sort_by': 'datetime',
    'datetime_col': 'datetime',
    'time_features': True,
    'cyclical_cols': ['hour'],
    'lag_cols': None,
    'lags': [1, 3, 6],
    'rolling_targets': None,
    'rolling_windows': [3, 6, 12],
    'target_col': 'target_wind_speed',
    'test_size': 720,
    'return_metadata': False
}


def prepare_dataset_from_config(config):
    """
    Prepare dataset using a configuration dictionary.
    
    Parameters:
    - config (dict): Configuration dictionary. Any missing keys will use DEFAULT_CONFIG values.
    
    Returns:
    - Same as prepare_dataset()
    
    Example:
```
    config = {
        'path': 'weather_data.csv',
        'lag_cols': ['wind_speed', 'pressure'],
        'lags': [1, 6, 12],
        'test_size': 1000
    }
    X_train, y_train, X_test, y_test, features, df, test_df = prepare_dataset_from_config(config)
```
    """
    # Merge user config with defaults
    full_config = DEFAULT_CONFIG.copy()
    full_config.update(config)
    
    # Validate required parameters
    if full_config['path'] is None:
        raise ValueError("'path' must be specified in config")
    
    # Call the main function
    return prepare_dataset(**full_config)


import numpy as np
import os

def save_predictions(preds, filename, folder="../../results/"):
    """
    Save a numpy array to a folder with error handling.

    Parameters
    ----------
    preds : np.ndarray
        Prediction array to save.
    filename : str
        Name of the file (e.g., 'svr_preds.npy').
    folder : str
        Path to save folder.
    """
    try:
        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)
        
        # Full path
        save_path = os.path.join(folder, filename)
        
        # Save numpy array
        np.save(save_path, preds)
        
        print(f"[SUCCESS] Saved {filename} to {folder}")
        
    except Exception as e:
        print(f"[ERROR] Could not save {filename}: {e}")





# -------------------------------------------------------------
# Version info (optional but recommended)
# -------------------------------------------------------------
__version__ = '1.0.0'
__author__ = 'Amir Baqerzadeh'


# -------------------------------------------------------------
# Module initialization message (optional)
# -------------------------------------------------------------
if __name__ != "__main__":
    logger.info("preprocessing module loaded (v%s)", __version__)


if __name__ == "__main__":
    """
    Test/demo code that runs when you execute: python preprocessing.py
    """
    import sys
    
    print(f"Preprocessing Module v{__version__}")
    print(f"Available functions: {', '.join(__all__)}")
    print("\nTo use this module, import it in your code:")
    print("  from preprocessing import prepare_dataset, load_and_basic_clean")
    print("\nFor help on any function, use:")
    print("  help(prepare_dataset)")
    
    # Optional: Add a simple test if a test file path is provided
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"\nTesting with file: {test_file}")
        try:
            df, _ = load_and_basic_clean(test_file, return_metadata=False)
            print(f"✓ Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"✗ Error loading file: {e}")
    else:
        print("\nTo test loading a file, run:")
        print("  python preprocessing.py path/to/your/data.csv")