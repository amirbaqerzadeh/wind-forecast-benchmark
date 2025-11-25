import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

__original_csv = pd.read_csv

def _safe_read_csv(path, *args, **kwargs):
    """
    safe wrapper around pd.read_csv to handle common issues.
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
    
pd.read_csv = _safe_read_csv

def load_and_basic_clean(path, required_columns=None, sort_by='datetime', dtype_cast=None):
    """""
    Load a CSV file into a DataFrame, select required columns, sort, and cast data types.
    Parameters:
    - path (str): Path to the CSV file.
    - required_columns (list of str, optional): List of columns to select from the DataFrame.
    - sort_by (str, optional): Column name to sort the DataFrame by. Default is 'datetime'.
    - dtype_cast (dict, optional): Dictionary specifying data types to cast columns to.
    Returns:
    - pd.DataFrame: The cleaned DataFrame.

    """
    df = pd.read_csv(path)
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        df = df[required_columns]

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by).reset_index(drop=True)
    else:
        logger.warning("Sort column '%s' not found in DataFrame. Skipping sorting.", sort_by)

    if dtype_cast:
        for col, dtype in dtype_cast.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    raise ValueError(f"Error casting column '{col}' to {dtype}: {e}")
            else:
                logger.warning("Column '%s' not found in DataFrame. Skipping dtype casting.", col)
    
    return df


            
def add_time_features(df, datetime_col='datetime', cyclical_cols=None):
    """"
    Add time-based features to the DataFrame.
    - Extract hour, month, day of week
    - create cyclical features (sin/cos) for specific columns.
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - datetime_col (str): Name of the datetime column. Default is 'datetime'.
    - cyclical_cols (list of str, optional): List of columns to encode cyclically. Default is None.
    
    Returns:
    - df (pd.DataFrame): DataFrame with added time features.
    """
    if datetime_col not in df.columns:
        raise KeyError(f"Datetime column '{datetime_col}' not found in DataFrame.")
    if not np.issubdtype(df[datetime_col].dtype, np.datetime64):
        try:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        except Exception as e:
            raise ValueError(f"Error converting column '{datetime_col}' to datetime: {e}")

    df['hour'] = df[datetime_col].dt.hour
    df['month'] = df[datetime_col].dt.month
    df['day_of_week'] = df[datetime_col].dt.dayofweek

    if cyclical_cols is None:
        cyclical_cols = ['hour']
    
    for col in cyclical_cols:
        if col not in df.columns:
            logger.warning("Cyclical column '%s' not found in DataFrame. Skipping cyclical encoding.", col)
            continue
        period = 24 if col == 'hour' else 12 if col == 'month' else 7 if col == 'day_of_week' else None
        if period is None:
            logger.warning("No defined period for column '%s'. Skipping cyclical encoding.", col), 
            continue
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    return df        



































