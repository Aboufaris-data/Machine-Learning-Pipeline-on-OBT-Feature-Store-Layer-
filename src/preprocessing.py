import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.logger import get_logger

logger = get_logger()

PROCESSED_DATA_PATH = "data/processed/"
TARGET_REGRESSION = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CATEGORICAL_COLS = ["ville", "quartier"]

def preprocess_for_regression(df: pd.DataFrame):
    logger.info("[preprocessing] === Starting Preprocessing ===")
    
    # 1. Drop metadata and leakage tracking columns
    cols_to_drop = ['annonce_id', 'title', 'link', 'location', 'price_m', 'price_category']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 2. Separate Target and Features
    if TARGET_REGRESSION not in df.columns:
        raise KeyError(f"Target column '{TARGET_REGRESSION}' missing from data columns.")
    X = df.drop(columns=[TARGET_REGRESSION])
    y = df[TARGET_REGRESSION]
    
    # 3. Train/Test Split (Executed first to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 4. Handle Missing Values using Training Median values
    for col in X_train.select_dtypes(include=[np.number]).columns:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    # 5. Handle Categorical Columns (Label Encoding)
    for col in CATEGORICAL_COLS:
        if col in X_train.columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = X_test[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # 6. Save split snapshots
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "X_test_raw.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "y_test.csv"), index=False)

    logger.info("[preprocessing] Preprocessing and base splits completed.")
    return X_train, X_test, y_train, y_test