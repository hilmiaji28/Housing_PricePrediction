import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)

import pandas as pd
import numpy as np

import sys
import json
import os

from datetime import datetime


def train_model(data_path, experiment_name="housing-price-prediction"):
    """Train model with MLflow tracking"""

    # =========================================================
    # MLflow Configuration
    # =========================================================
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"Using MLflow Tracking URI: {tracking_uri}")

    # =========================================================
    # Load Data
    # =========================================================
    print(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)

    X = df[["OverallQual", "GarageCars", "GrLivArea"]]
    y = df["SalePrice"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # =========================================================
    # Start MLflow Run
    # =========================================================
    with mlflow.start_run(
        run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):

        # =====================================================
        # Hyperparameters
        # =====================================================
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
        }

        mlflow.log_params(params)

        # =====================================================
        # Train Model
        # =====================================================
        print("Training model...")

        model = RandomForestRegressor(**params)

        model.fit(X_train, y_train)

        # =====================================================
        # Prediction
        # =====================================================
        y_pred = model.predict(X_val)

        # =====================================================
        # Metrics
        # =====================================================
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
        }

        mlflow.log_metrics(metrics)

        # =====================================================
        # Log & Register Model
        # =====================================================
        print("Logging model to MLflow...")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="housing-price-model",
        )

        # =====================================================
        # Additional Metadata
        # =====================================================
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("validation_samples", len(X_val))

        # =====================================================
        # Save Metrics Locally
        # =====================================================
        BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        logs_dir = os.path.join(BASE_DIR, "logs")

        os.makedirs(logs_dir, exist_ok=True)

        metrics_path = os.path.join(
            logs_dir,
            "latest_metrics.json",
        )

        with open(metrics_path, "w") as f:
            json.dump(
                {
                    **metrics,
                    "timestamp": datetime.now().isoformat(),
                    "run_id": mlflow.active_run().info.run_id,
                },
                f,
                indent=2,
            )

        # =====================================================
        # Console Output
        # =====================================================
        print("\nModel trained successfully!")
        print("Metrics:")
        print(f"  RMSE : ${rmse:,.2f}")
        print(f"  MAE  : ${mae:,.2f}")
        print(f"  R²   : {r2:.4f}")
        print(f"  Run ID : {mlflow.active_run().info.run_id}")

        return mlflow.active_run().info.run_id


# =============================================================
# Main
# =============================================================
if __name__ == "__main__":

    data_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Data/train.csv"
    )

    train_model(data_path)