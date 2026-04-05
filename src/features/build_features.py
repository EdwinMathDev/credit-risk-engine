# ============================================================
# Feature engineering module
# src/features/build_features.py
# ============================================================

import logging
import pandas as pd
import numpy as np
from src.data.load_data import load_raw_data, save_processed_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default payment_next_month"
]

PAY_DELAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    # ── 1. Filtrar columnas base ──────────────────────────────
    logging.info("Filtrando columnas relevantes...")
    df = df[BASE_COLUMNS].copy()

    # ── 2. Ratios de utilización y pago por mes ───────────────
    logging.info("Creando features de utilización y pago por mes...")
    for i in range(1, 7):
        # Qué % del límite está siendo usado (clipeado para cubrir sobregiro moderado)
        df[f"utilization_ratio_{i}"] = (
            df[f"BILL_AMT{i}"] / df["LIMIT_BAL"]
        ).clip(0, 2)

        # Qué % de la deuda fue pagado ese mes (0 si no había deuda)
        df[f"payment_ratio_{i}"] = np.where(
            df[f"BILL_AMT{i}"] > 0,
            df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i}"],
            0
        )

    # ── 3. Tendencias mes a mes ───────────────────────────────
    logging.info("Creando features de tendencia mes a mes...")
    for i in range(1, 6):
        df[f"bill_trend_{i}"] = df[f"BILL_AMT{i+1}"] - df[f"BILL_AMT{i}"]
        df[f"pay_trend_{i}"]  = df[f"PAY_AMT{i+1}"]  - df[f"PAY_AMT{i}"]

    # ── 4. Volatilidad de pagos y facturas ────────────────────
    logging.info("Creando features de volatilidad...")
    df["payment_volatility"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].std(axis=1)
    df["bill_volatility"]    = df[[f"BILL_AMT{i}" for i in range(1, 7)]].std(axis=1)

    # ── 5. Historial de delays ────────────────────────────────
    logging.info("Creando features de historial de delays...")
    df["recent_delay"]   = (df["PAY_0"] > 0).astype(int)
    df["avg_pay_delay"]  = df[PAY_DELAY_COLS].mean(axis=1)
    df["max_pay_delay"]  = df[PAY_DELAY_COLS].max(axis=1)
    df["months_delayed"] = (df[PAY_DELAY_COLS] > 0).sum(axis=1)

    # ── 6. Capacidad de pago global ───────────────────────────
    logging.info("Creando features de capacidad de pago global...")
    df["total_bill"]    = df[[f"BILL_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["total_payment"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["global_payment_ratio"] = np.where(
        df["total_bill"] > 0,
        df["total_payment"] / df["total_bill"],
        0
    )

    # ── 7. Grupo etario ordinal ───────────────────────────────
    logging.info("Creando feature de grupo etario...")
    df["age_group"] = pd.cut(
        df["AGE"],
        bins=[20, 30, 40, 50, 60, 100],
        labels=[0, 1, 2, 3, 4],
        right=False
    ).astype("Int64")  # Int64 soporta NaN si hay edades fuera del rango

    logging.info(f"Features creadas: {df.shape[1]} columnas — {df.shape[0]} filas.")
    return df


if __name__ == "__main__":
    processed_file = "credit_card_default_preprocessed.csv"
    logging.info(f"Cargando dataset desde data/processed/{processed_file}...")
    df = load_raw_data(processed_file, data_dir="data/processed")

    df_features = create_features(df)

    save_processed_data(df_features, "credit_card_features.csv", data_dir="data/features")
    logging.info("Pipeline de feature engineering completado.")