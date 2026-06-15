# 🏠 Housing Price Prediction MLOps Pipeline

An end-to-end Machine Learning Operations (MLOps) project for predicting house prices using a Random Forest Regressor, MLflow Model Registry, FastAPI serving, Streamlit frontend, Docker containers, monitoring, and automated retraining workflows.

---

# 📌 Project Overview

This project demonstrates a complete MLOps lifecycle, from model training and experiment tracking to deployment, monitoring, and retraining.

The model predicts house prices based on:

* Overall Quality (`OverallQual`)
* Garage Capacity (`GarageCars`)
* Living Area (`GrLivArea`)

The project integrates:

* Machine Learning Model Training
* MLflow Experiment Tracking
* MLflow Model Registry
* FastAPI Model Serving
* Streamlit User Interface
* Docker Containerization
* Prediction Monitoring
* Drift Simulation
* Automated Retraining Trigger

---

# 🏗️ System Architecture

```text
                     +------------------+
                     |   Training Data  |
                     +--------+---------+
                              |
                              v
                     +------------------+
                     |    train.py      |
                     | Random Forest ML |
                     +--------+---------+
                              |
                              v
                     +------------------+
                     |      MLflow      |
                     | Experiment Track |
                     +--------+---------+
                              |
                              v
                     +------------------+
                     | Model Registry   |
                     | Version Control  |
                     +--------+---------+
                              |
                              v
                     +------------------+
                     |   predict.py     |
                     | FastAPI Serving  |
                     +--------+---------+
                              |
                              v
                     +------------------+
                     |  Streamlit UI    |
                     +--------+---------+
                              |
                              v
                     +------------------+
                     |     End Users    |
                     +------------------+
```

---

# 📂 Project Structure

```text
Assignment/
│
├── Data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── drift_data.csv
│   ├── House_Price.py
│   └── generate_data.py
│
├── logs/
│   ├── latest_metrics.json
│   └── predictions.jsonl
│
├── mlflow_data/
│   └── mlflow.db
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── monitor.py
│   └── retrain_trigger.py
│
├── streamlit_app.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── demo.sh
└── README.md
```

---

# ⚙️ Technologies Used

* Python 3.10+
* Scikit-Learn
* Pandas
* NumPy
* MLflow
* FastAPI
* Uvicorn
* Streamlit
* Docker
* Docker Compose

---

# 🚀 Features

## Model Training

* Random Forest Regressor
* Hyperparameter logging
* Metrics logging
* Experiment tracking
* Model registration
* Model versioning

## MLflow Integration

* Experiment Tracking
* Model Registry
* Model Versioning
* Artifact Management

## FastAPI Prediction Service

* REST API
* Health Check Endpoint
* Prediction Endpoint
* Metrics Endpoint
* Model Rollback Endpoint

## Streamlit Dashboard

* Interactive prediction interface
* Real-time prediction results
* FastAPI integration

## Monitoring

* Prediction logging
* Metrics tracking
* Model performance monitoring

## Drift Simulation

* Drift data generation
* Drift detection testing
* Retraining trigger workflow

---

# 🔧 Key Components

### train.py

Responsible for:

* Training the Random Forest model
* Logging parameters and metrics
* Registering models in MLflow Registry
* Creating model versions

### predict.py

Responsible for:

* Loading the latest registered model
* Serving predictions through FastAPI
* Logging prediction history
* Providing monitoring endpoints

### monitor.py

Responsible for:

* Monitoring prediction activity
* Tracking model performance
* Reading logged prediction data

### retrain_trigger.py

Responsible for:

* Triggering model retraining
* Supporting automated model lifecycle management

### streamlit_app.py

Responsible for:

* User interface for prediction
* Sending requests to FastAPI
* Displaying prediction results

### generate_data.py

Responsible for:

* Generating synthetic drift data
* Simulating production data changes

---

# 📊 Model Performance

Latest model performance:

| Metric   | Value      |
| -------- | ---------- |
| RMSE     | $34,100.56 |
| MAE      | $22,740.33 |
| R² Score | 0.8484     |

---

# 🐳 Running with Docker

## Build Containers

```bash
docker compose build
```

## Start MLflow Server

```bash
docker compose up mlflow
```

MLflow UI:

```text
http://localhost:5001
```

---

## Train Model

```bash
docker compose --profile train up
```

Expected output:

```text
Successfully registered model 'housing-price-model'
Created version '1' of model 'housing-price-model'
```

---

## Start Prediction API

```bash
docker compose up predictor
```

Expected output:

```text
Loading model: housing-price-model version latest
Model loaded successfully
Uvicorn running on http://0.0.0.0:5000
```

FastAPI URL:

```text
http://localhost:5000
```

Swagger Documentation:

```text
http://localhost:5000/docs
```

---

# 🎨 Streamlit Dashboard

Run Streamlit:

```bash
streamlit run streamlit_app.py
```

Access:

```text
http://localhost:8501
```

Features:

* Input house characteristics
* Real-time predictions
* User-friendly interface
* FastAPI integration

---

# 🌐 API Endpoints

## Home

```http
GET /
```

## Health Check

```http
GET /health
```

## Predict House Price

```http
POST /predict
```

Example Request:

```json
{
  "OverallQual": 7,
  "GarageCars": 2,
  "GrLivArea": 1800
}
```

Example Response:

```json
{
  "prediction": 215432.12,
  "model_version": "latest",
  "timestamp": "2026-05-27T15:55:53"
}
```

## Metrics

```http
GET /metrics
```

## Rollback Model

```http
POST /rollback?version=1
```

---

# 📊 Monitoring & Drift Detection

### Prediction Logs

Stored in:

```text
logs/predictions.jsonl
```

### Training Metrics

Stored in:

```text
logs/latest_metrics.json
```

### Drift Simulation

The file:

```text
Data/drift_data.csv
```

contains modified data used to simulate data drift and evaluate model robustness.

### Retraining

The script:

```text
src/retrain_trigger.py
```

can be used to trigger model retraining when drift is detected.

---

# 🔄 End-to-End Workflow

```text
1. Load Training Data
        ↓
2. Train Random Forest Model
        ↓
3. Log Metrics to MLflow
        ↓
4. Register Model
        ↓
5. Create Model Version
        ↓
6. Serve Model via FastAPI
        ↓
7. Predict via API or Streamlit
        ↓
8. Log Predictions
        ↓
9. Monitor Model Performance
        ↓
10. Detect Drift
        ↓
11. Trigger Retraining
```

---

# 🧪 Local Development

Create virtual environment:

```bash
python -m venv venv
```

Activate environment (Windows):

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python src/train.py Data/train.csv
```

Run prediction API:

```bash
python src/predict.py
```

Run Streamlit:

```bash
streamlit run streamlit_app.py
```

---

# 🔮 Future Improvements

* Prometheus Monitoring
* Grafana Dashboard
* Automated Drift Detection
* CI/CD with GitHub Actions
* Cloud Deployment (AWS/GCP/Azure)
* Kubernetes Deployment
* Advanced Model Monitoring

---

# 📄 License

This project was developed for educational purposes and MLOps learning practice.

---

# 👨‍💻 Author

**Hilmi Aji**

Machine Learning & MLOps Enthusiast

Built with using Python, MLflow, FastAPI, Streamlit, and Docker.
