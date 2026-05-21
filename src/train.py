import os
import json
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger import get_logger

logger = get_logger()
PROCESSED_DATA_PATH = "data/processed/"
MODELS_PATH = "models/"

def train_regression_model():
    logger.info("[training] === Starting Regression Model Training ===")
    
    # 1. Load data
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_train_engineered.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_test_engineered.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_test.csv")).values.ravel()
    
    # 2. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Fit Regressor
    logger.info("[training] Fitting RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # 4. Metrics Evaluator
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2_Score": round(r2, 4)
    }
    
    print("\n" + "="*40)
    print("   REGRESSION MODEL METRICS RESULTS")
    print("="*40)
    print(f"MAE      (Mean Absolute Error):   {metrics['MAE']} DH")
    print(f"RMSE     (Root Mean Sq. Error):   {metrics['RMSE']} DH")
    print(f"R² Score (Coefficient of Det.):   {metrics['R2_Score']}")
    print("="*40 + "\n")
    
    # 5. Save Artifacts
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_PATH, "regression_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_PATH, "feature_scaler.pkl"))
    
    with open(os.path.join(MODELS_PATH, "regression_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("[training] Regression pipeline items saved successfully.")
    return model, metrics