from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
from datetime import datetime

app = FastAPI(
    title="MLOps Monitoring API",
    description="Monitoring model drift untuk prediksi harga rumah",
    version="1.0.0" 
)


#monitoring
THRESHOLD_VARIANCE = 100000
THRESHOLD_SAMPLES = 20
THRESHOLD_MEAN_SHIFT = 0.2


#response schema
class MonitorResponse(BaseModel):
    status: str
    total_predictions: int
    mean_prediction: float
    std_prediction: float
    drift_detected: bool
    reasons: list


#root endpoint
@app.get("/")
def home():
    return {
        "message": "MLOps Monitoring API Running 🚀"
    }


#monitoring endpoint
@app.get("/monitor", response_model=MonitorResponse)
def monitor_model():

    try:
        predictions = []

        # baca prediction logs
        with open("logs/predictions.jsonl", "r") as f:
            for line in f:
                predictions.append(json.loads(line))

        # cek jumlah sample
        if len(predictions) < THRESHOLD_SAMPLES:
            return {
                "status": "Not enough samples",
                "total_predictions": len(predictions),
                "mean_prediction": 0,
                "std_prediction": 0,
                "drift_detected": False,
                "reasons": [
                    f"Need at least {THRESHOLD_SAMPLES} samples"
                ]
            }

        # ambil recent predictions
        recent = predictions[-THRESHOLD_SAMPLES:]

        pred_values = [p["prediction"] for p in recent]

        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)

        drift_detected = False
        drift_reasons = []

        #deteksi drift
        #1. variance tinggi
        if std_pred > THRESHOLD_VARIANCE:
            drift_detected = True
            drift_reasons.append(
                f"High variance detected: {std_pred:.2f}"
            )

        #2. Distribusi prediksi berubah
        if mean_pred > 500000 or mean_pred < 100000:
            drift_detected = True
            drift_reasons.append(
                f"Prediction distribution shift: {mean_pred:.2f}"
            )

        #3. Concept drift
        if len(predictions) >= THRESHOLD_SAMPLES * 2:

            earlier = predictions[
                -(THRESHOLD_SAMPLES * 2):-THRESHOLD_SAMPLES
            ]

            earlier_mean = np.mean(
                [p["prediction"] for p in earlier]
            )

            mean_shift = abs(mean_pred - earlier_mean) / earlier_mean

            if mean_shift > THRESHOLD_MEAN_SHIFT:
                drift_detected = True
                drift_reasons.append(
                    f"Concept drift detected: {mean_shift*100:.1f}% shift"
                )

        #save alert
        if drift_detected:

            alert = {
                "timestamp": datetime.now().isoformat(),
                "mean_prediction": float(mean_pred),
                "std_prediction": float(std_pred),
                "reasons": drift_reasons
            }

            with open("logs/alerts.jsonl", "a") as f:
                f.write(json.dumps(alert) + "\n")

        return {
            "status": "Monitoring completed",
            "total_predictions": len(predictions),
            "mean_prediction": float(mean_pred),
            "std_prediction": float(std_pred),
            "drift_detected": drift_detected,
            "reasons": drift_reasons
        }

    except FileNotFoundError:
        return {
            "status": "Prediction log not found",
            "total_predictions": 0,
            "mean_prediction": 0,
            "std_prediction": 0,
            "drift_detected": False,
            "reasons": ["logs/predictions.jsonl not found"]
        }

    except Exception as e:
        return {
            "status": "Error",
            "total_predictions": 0,
            "mean_prediction": 0,
            "std_prediction": 0,
            "drift_detected": False,
            "reasons": [str(e)]
        }