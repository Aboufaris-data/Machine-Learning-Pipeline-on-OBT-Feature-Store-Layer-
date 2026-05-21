import os
import pandas as pd
from src.logger import get_logger
from src.extraction import extract_obt
from src.preprocessing import preprocess_for_regression
from src.feature_eng import engineer_features
from src.train import train_regression_model

logger = get_logger()

def main():
    logger.info("="*50)
    logger.info(" RUNNING MOROCCAN REAL ESTATE ML REGRESSION")
    logger.info("="*50)

    # STEP 1: DB Extraction or Local CSV Upload Loader
    logger.info("\n--- STEP 1: DATA EXTRACTION ---")
    try:
        df_raw = extract_obt(save_csv=True)
    except Exception as e:
        logger.warning(f"Database extraction failed: {e}. Trying direct local uploads.")
        if os.path.exists("obt_raw.csv"):
            df_raw = pd.read_csv("obt_raw.csv")
        elif os.path.exists("data/raw/obt_raw.csv"):
            df_raw = pd.read_csv("data/raw/obt_raw.csv")
        else:
            logger.error("No valid input dataset found. Stopping execution pipeline.")
            return

    # STEP 2: Preprocessing and Splits
    logger.info("\n--- STEP 2: PREPROCESSING ---")
    X_train, X_test, y_train, y_test = preprocess_for_regression(df_raw)

    # STEP 3: Advanced Feature Matrix Construction
    logger.info("\n--- STEP 3: FEATURE ENGINEERING ---")
    X_train_final, X_test_final = engineer_features(X_train, X_test)

    # STEP 4: Scaling & Train Fitting
    logger.info("\n--- STEP 4: MODEL TRAINING ---")
    model, metrics = train_regression_model()

    logger.info("="*50)
    logger.info(" Pipeline execution completed successfully with new CSV schema!")
    logger.info("="*50)

if __name__ == "__main__":
    main()