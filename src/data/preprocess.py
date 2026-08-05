# ============================================================
# Data CLEANING module (pipeline step 1)
# src/data/preprocess.py
#
# IMPORTANT: this module no longer scales, encodes, or balances
# the data. Those steps were moved to later stages of the pipeline
# (after feature engineering and AFTER the train/test split)
# to avoid data leakage and so that the financial ratios in
# build_features.py are calculated on real, unscaled values.
# ============================================================

import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "default payment_next_month"

# Columnas categóricas conocidas del dataset (códigos numéricos, no strings)
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa nulos con la mediana, solo en columnas numéricas."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])
    logging.info(f"Missing values imputados (mediana) en: {list(num_cols)}")
    return df


# ------------------------------------------------------------
# Limpieza de categorías inválidas
# ------------------------------------------------------------
def clean_invalid_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Este dataset (UCI default of credit card clients) tiene códigos
    documentados y códigos "basura" que aparecen en la práctica:

    EDUCATION: 1=grad school, 2=university, 3=high school, 4=others
               -> valores 0, 5, 6 NO están documentados, se agrupan en 4 (others)
    MARRIAGE:  1=married, 2=single, 3=others
               -> valor 0 NO está documentado, se agrupa en 3 (others)
    """
    before_edu = df["EDUCATION"].value_counts(dropna=False).to_dict()
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x in [1, 2, 3] else 4)

    before_mar = df["MARRIAGE"].value_counts(dropna=False).to_dict()
    df["MARRIAGE"] = df["MARRIAGE"].apply(lambda x: x if x in [1, 2] else 3)

    logging.info(f"EDUCATION antes de limpiar: {before_edu}")
    logging.info(f"EDUCATION después de limpiar: {df['EDUCATION'].value_counts().to_dict()}")
    logging.info(f"MARRIAGE antes de limpiar: {before_mar}")
    logging.info(f"MARRIAGE después de limpiar: {df['MARRIAGE'].value_counts().to_dict()}")
    return df


# ------------------------------------------------------------
# Tipado explícito de categóricas
# ------------------------------------------------------------
def cast_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte SEX/EDUCATION/MARRIAGE a dtype 'category'.
    Esto es clave para que el encoder posterior las detecte
    explícitamente en vez de depender de select_dtypes(include='object'),
    que con estas columnas (ya enteras) no detecta nada.
    """
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    logging.info(f"Columnas casteadas a category: {CATEGORICAL_COLS}")
    return df


# ------------------------------------------------------------
# Pipeline de limpieza (SIN escalar, SIN encodear, SIN balancear)
# ------------------------------------------------------------
def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = handle_missing_values(df)
    df = clean_invalid_categories(df)
    df = cast_categorical_dtypes(df)
    logging.info("Pipeline de limpieza completado (sin scaling/encoding/balancing).")
    return df


if __name__ == "__main__":
    from load_data import load_raw_data, save_processed_data

    try:
        df = load_raw_data("credit_card_default.csv")
        df_clean = clean_pipeline(df)
        # Guardamos como "clean", no "preprocessed" — este archivo
        # es el que debe consumir build_features.py
        save_processed_data(df_clean, "credit_card_default_clean.csv")
    except Exception as e:
        logging.error(f"Error en pipeline de limpieza: {e}")