from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import mlflow
import mlflow.sklearn
import pandas as pd
import json
from datetime import datetime
import os
import uvicorn

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Housing Price Prediction API"
)

# =========================
# HOME ENDPOINT
# =========================
@app.get("/")
def home():
    return {
        "message": "Housing Price Prediction API Running"
    }

# =========================
# LOAD MODEL FROM MLFLOW
# =========================

# otomatis:
# - pakai env variable jika ada (Docker)
# - fallback ke localhost jika dijalankan lokal
tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5001"
)

mlflow.set_tracking_uri(tracking_uri)

print(f"Using MLflow Tracking URI: {tracking_uri}")

MODEL_NAME = "housing-price-model"
MODEL_VERSION = "latest"

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

print(f"Loading model: {MODEL_NAME} version {MODEL_VERSION}")

model = mlflow.sklearn.load_model(model_uri)

print("Model loaded successfully")

# =========================
# MONITORING VARIABLES
# =========================
prediction_count = 0
predictions_log = []

# buat folder logs jika belum ada
os.makedirs("logs", exist_ok=True)

# =========================
# INPUT SCHEMA
# =========================
class HouseData(BaseModel):
    OverallQual: int
    GarageCars: int
    GrLivArea: float

# =========================
# HEALTH CHECK ENDPOINT
# =========================
@app.get("/health")
async def health():

    return JSONResponse(
        {
            "status": "healthy",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "predictions_served": prediction_count,
        }
    )

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
async def predict(data: HouseData):

    global prediction_count, predictions_log

    try:

        # buat dataframe
        df = pd.DataFrame(
            [{
                "OverallQual": data.OverallQual,
                "GarageCars": data.GarageCars,
                "GrLivArea": data.GrLivArea
            }]
        )

        # lakukan prediksi
        prediction = model.predict(df)[0]

        # update monitoring
        prediction_count += 1

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": data.dict(),
            "prediction": float(prediction),
            "model_version": MODEL_VERSION,
        }

        predictions_log.append(log_entry)

        # simpan hanya 100 log terakhir
        if len(predictions_log) > 100:
            predictions_log.pop(0)

        # save prediction log
        with open("logs/predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # return response
        return JSONResponse(
            {
                "prediction": round(float(prediction), 2),
                "model_version": MODEL_VERSION,
                "timestamp": log_entry["timestamp"],
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

# =========================
# METRICS ENDPOINT
# =========================
@app.get("/metrics")
async def metrics():

    return JSONResponse(
        {
            "total_predictions": prediction_count,
            "recent_predictions": predictions_log[-10:],
        }
    )

# =========================
# ROLLBACK ENDPOINT
# =========================
@app.post("/rollback")
async def rollback(version: int):

    global model, MODEL_VERSION

    try:

        MODEL_VERSION = str(version)

        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

        model = mlflow.sklearn.load_model(model_uri)

        return JSONResponse(
            {
                "status": "success",
                "message": f"Rolled back to version {version}",
                "model_version": MODEL_VERSION,
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

# =========================
# RUN FASTAPI SERVER
# =========================
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000
    )