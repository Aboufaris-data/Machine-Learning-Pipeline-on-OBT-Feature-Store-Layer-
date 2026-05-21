import os
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger()
PROCESSED_PATH = "data/processed/"

def engineer_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    logger.info("[feature_engineering] === Generating Advanced Features ===")
    X_train, X_test = X_train.copy(), X_test.copy()
    
    # 1. Clean accidental tracking leakage columns
    leakage_cols = ['price_m', 'price_category', 'price_per_m2', 'price_dh', 'price']
    X_train = X_train.drop(columns=[c for c in leakage_cols if c in X_train.columns], errors='ignore')
    X_test = X_test.drop(columns=[c for c in leakage_cols if c in X_test.columns], errors='ignore')

    # 2. Log Transformation for Surface scale stability
    if "surface" in X_train.columns:
        X_train["log_surface"] = np.log1p(X_train["surface"])
        X_test["log_surface"] = np.log1p(X_test["surface"])
        
    # 3. Ratios
    if "rooms" in X_train.columns and "surface" in X_train.columns:
        X_train["surface_per_room"] = X_train["surface"] / (X_train["rooms"] + 1)
        X_test["surface_per_room"] = X_test["surface"] / (X_test["rooms"] + 1)
        
    if "baths" in X_train.columns and "rooms" in X_train.columns:
        X_train["bath_room_ratio"] = X_train["baths"] / (X_train["rooms"] + 1)
        X_test["bath_room_ratio"] = X_test["baths"] / (X_test["rooms"] + 1)

    # 4. Interaction Spaces
    if "surface" in X_train.columns and "rooms" in X_train.columns:
        X_train["surface_x_rooms"] = X_train["surface"] * X_train["rooms"]
        X_test["surface_x_rooms"] = X_test["surface"] * X_test["rooms"]

    # 5. Save engineered snapshots
    X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train_engineered.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test_engineered.csv"), index=False)
    
    logger.info(f"[feature_engineering] Engineering complete. Features shape: {X_train.shape}")
    return X_train, X_test