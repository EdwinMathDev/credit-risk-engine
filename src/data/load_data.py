import os
import pandas as pd

def load_raw_data(filename: str, data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load raw dataset from the specified directory.

    Parameters
    ----------
    filename : str
        Name of the file to load (e.g., 'credit_default.csv').
    data_dir : str, optional
        Directory where raw data is stored. Default is 'data/raw'.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame with the correct target column
        'default payment_next_month' preserved.

    Raises
    ------
    FileNotFoundError
        If the file does not exist in the given directory.
    ValueError
        If the file extension is not supported.
    """

    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")

    # Support for CSV, Excel, Parquet
    if filename.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError("Unsupported file format. Use CSV, XLSX, or Parquet.")

    # Ensure target column exists with exact name
    if "default payment_next_month" not in df.columns:
        raise KeyError("Target column 'default payment_next_month' not found in dataset.")

    return df


def save_processed_data(df: pd.DataFrame, filename: str, data_dir: str = "data/processed") -> None:
    """
    Save processed dataset to the specified directory.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Name of the output file (e.g., 'credit_default_clean.csv').
    data_dir : str, optional
        Directory where processed data will be stored. Default is 'data/processed'.

    Returns
    -------
    None
    """

    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Processed data saved to {filepath}")
