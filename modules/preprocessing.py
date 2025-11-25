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
            




































