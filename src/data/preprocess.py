# ============================================================
# Module for preprocessing datasets:
# - Handle missing values
# - Encode categorical variables
# - Scale numerical features
# - Transform skewed distributions
# - Balance classes (optional)
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "default payment_next_month"

# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="median")
    df[df.columns] = imputer.fit_transform(df)
    logging.info("Missing values handled with median imputation.")
    return df

# ------------------------------------------------------------
# Categorical encoding
# ------------------------------------------------------------
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) == 0:
        logging.info("No categorical columns found.")
        return df

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)
    logging.info(f"Categorical columns encoded: {list(cat_cols)}")
    return df

# ------------------------------------------------------------
# Scaling numerical features
# ------------------------------------------------------------
def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET_COL, errors="ignore")
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    logging.info("Numerical features scaled with StandardScaler.")
    return df

# ------------------------------------------------------------
# Outlier transformation (log for skewed payments)
# ------------------------------------------------------------
def transform_outliers(df: pd.DataFrame) -> pd.DataFrame:
    pay_amt_cols = [col for col in df.columns if "PAY_AMT" in col]
    for col in pay_amt_cols:
        df[col] = np.log1p(df[col])  # log(1+x) to handle zeros
    logging.info(f"Log-transform applied to skewed payment columns: {pay_amt_cols}")
    return df

# ------------------------------------------------------------
# Class balancing (SMOTE)
# ------------------------------------------------------------
def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    df_res = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=TARGET_COL)], axis=1)
    logging.info(f"Class balancing applied with SMOTE. New shape: {df_res.shape}")
    return df_res

# ------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------
def preprocess_pipeline(df: pd.DataFrame, balance: bool = True) -> pd.DataFrame:
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = transform_outliers(df)
    df = scale_numeric(df)
    if balance:
        df = balance_classes(df)
    logging.info("Preprocessing pipeline completed.")
    return df

# ------------------------------------------------------------
# Quick test block
# ------------------------------------------------------------
if __name__ == "__main__":
    from load_data import load_raw_data, save_processed_data

    try:
        df = load_raw_data("credit_card_default.csv")
        df_clean = preprocess_pipeline(df, balance=True)
        save_processed_data(df_clean, "credit_card_default_preprocessed.csv")
    except Exception as e:
        logging.error(f"Error in preprocessing pipeline: {e}")
