# ============================================================
# Final data preparation pipeline for modeling (step 3)
# src/models/train_pipeline.py
#
# Correct order:
#   1. Load data/features/credit_card_features.csv
#      (output of build_features.py — ratios already computed,
#       categoricals still UNENCODED, nothing scaled)
#   2. Stratified train_test_split (before any fit)
#   3. Categorical encoding — fit on train only
#   4. log1p on PAY_AMT (deterministic transform, no fit required)
#   5. StandardScaler — fit on train only
#   6. SMOTE — train only
#   7. Save X_train/X_test/y_train/y_test + encoder/scaler
#      (the encoder and scaler are saved because the API will
#       need them to transform new incoming requests)
# ============================================================

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "default payment_next_month"
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE", "age_group"]

FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"


def load_features(filename: str = "credit_card_features.csv") -> pd.DataFrame:
    path = os.path.join(FEATURES_DIR, filename)
    df = pd.read_csv(path)
    logging.info(f"Features cargadas: {df.shape}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(f"Split: train={X_train.shape}, test={X_test.shape}")
    logging.info(f"Balance de clases train: {y_train.value_counts(normalize=True).to_dict()}")
    logging.info(f"Balance de clases test:  {y_test.value_counts(normalize=True).to_dict()}")
    return X_train, X_test, y_train, y_test


def encode_categoricals(X_train, X_test, cat_cols=CATEGORICAL_COLS):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[cat_cols])  # fit SOLO en train

    train_enc = pd.DataFrame(
        encoder.transform(X_train[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train.index,
    )
    test_enc = pd.DataFrame(
        encoder.transform(X_test[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_test.index,
    )

    X_train = pd.concat([X_train.drop(columns=cat_cols), train_enc], axis=1)
    X_test = pd.concat([X_test.drop(columns=cat_cols), test_enc], axis=1)
    logging.info(f"Encoding aplicado (fit en train). Nuevas columnas: {list(train_enc.columns)}")
    return X_train, X_test, encoder


def transform_skewed(X_train, X_test):
    pay_amt_cols = [c for c in X_train.columns if "PAY_AMT" in c]
    for col in pay_amt_cols:
        X_train[col] = np.log1p(X_train[col].clip(lower=0))
        X_test[col] = np.log1p(X_test[col].clip(lower=0))
    logging.info(f"log1p aplicado a columnas sesgadas: {pay_amt_cols}")
    return X_train, X_test


def scale_numeric(X_train, X_test):
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])  # fit SOLO en train
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    logging.info(f"Scaling aplicado (fit en train) a {len(num_cols)} columnas numéricas.")
    return X_train, X_test, scaler


def impute_remaining_na(X_train, X_test):
    """Los NaN/inf que pudo generar build_features.py se resuelven aquí,
    con la mediana calculada SOLO en train."""
    medians = X_train.median(numeric_only=True)
    n_train_na = X_train.isna().sum().sum()
    n_test_na = X_test.isna().sum().sum()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    if n_train_na or n_test_na:
        logging.info(f"NaNs imputados con mediana de train — train: {n_train_na}, test: {n_test_na}")
    return X_train, X_test


def balance_train_only(X_train, y_train, random_state: int = 42):
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logging.info(f"SMOTE aplicado SOLO en train. Shape antes: {X_train.shape}, después: {X_res.shape}")
    return X_res, y_res


def run_pipeline():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_features()
    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test, encoder = encode_categoricals(X_train, X_test)
    X_train, X_test = transform_skewed(X_train, X_test)
    X_train, X_test = impute_remaining_na(X_train, X_test)
    X_train, X_test, scaler = scale_numeric(X_train, X_test)

    X_train_bal, y_train_bal = balance_train_only(X_train, y_train)

    # Guardar datasets listos para modelar
    X_train_bal.assign(**{TARGET_COL: y_train_bal.values}).to_csv(
        os.path.join(FEATURES_DIR, "train_final.csv"), index=False
    )
    X_test.assign(**{TARGET_COL: y_test.values}).to_csv(
        os.path.join(FEATURES_DIR, "test_final.csv"), index=False
    )

    # Guardar artefactos — la API los necesita para transformar requests nuevos
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, "encoder.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))

    logging.info("Pipeline de preparación para modelado completado.")
    logging.info(f"Artefactos guardados en {ARTIFACTS_DIR}/ (encoder.joblib, scaler.joblib)")
    return X_train_bal, X_test, y_train_bal, y_test


if __name__ == "__main__":
    run_pipeline()