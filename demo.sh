#!/bin/bash

echo "================================================"
echo "MLOps End-to-End Demo"
echo "================================================"

# ==========================================
# STEP 1: TRAIN INITIAL MODEL
# ==========================================
echo -e "\nSTEP 1: Training initial model..."

python src/train.py Data/train.csv

# ==========================================
# STEP 2: START MLFLOW UI
# ==========================================
echo -e "\nSTEP 2: Starting MLflow UI..."
echo "Access at: http://localhost:5001"

python -m mlflow ui \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlruns \
--port 5001 &

MLFLOW_PID=$!

sleep 5

# ==========================================
# STEP 3: START PREDICTION SERVICE
# ==========================================
echo -e "\nSTEP 3: Starting prediction service..."

python src/predict.py &

PREDICT_PID=$!

sleep 5

# ==========================================
# STEP 4: TEST PREDICTIONS
# ==========================================
echo -e "\nSTEP 4: Testing predictions..."

echo -e "\nTest 1: Medium house"

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "OverallQual": 7,
        "GarageCars": 2,
        "GrLivArea": 1800
      }'

echo -e "\n\nTest 2: Luxury house"

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "OverallQual": 9,
        "GarageCars": 3,
        "GrLivArea": 3000
      }'

echo -e "\n\nTest 3: Small house"

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "OverallQual": 5,
        "GarageCars": 1,
        "GrLivArea": 1200
      }'

# ==========================================
# STEP 5: GENERATE MONITORING DATA
# ==========================================
echo -e "\n\nSTEP 5: Generating predictions for monitoring..."

for i in {1..25}; do

    OVERALLQUAL=$((5 + RANDOM % 5))
    GARAGECARS=$((1 + RANDOM % 3))
    GRLIVAREA=$((1000 + RANDOM % 2500))

    curl -s -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d "{
            \"OverallQual\": $OVERALLQUAL,
            \"GarageCars\": $GARAGECARS,
            \"GrLivArea\": $GRLIVAREA
          }" > /dev/null

    echo -n "."
done

echo " Done!"

# ==========================================
# STEP 6: CHECK MONITORING
# ==========================================
echo -e "\n\nSTEP 6: Checking model performance..."

python src/monitor.py

# ==========================================
# FINISH
# ==========================================
echo -e "\n\n================================================"
echo "Demo completed!"
echo "================================================"

echo -e "\nRunning services:"
echo "MLflow UI      : http://localhost:5001"
echo "Prediction API : http://localhost:5000"
echo "Swagger Docs   : http://localhost:5000/docs"

echo -e "\nUseful commands:"
echo "Health check:"
echo "curl http://localhost:5000/health"

echo -e "\nMetrics:"
echo "curl http://localhost:5000/metrics"

echo -e "\nRollback model:"
echo "curl -X POST http://localhost:5000/rollback?version=1"

echo -e "\nTrigger retraining:"
echo "python src/retrain_trigger.py data/train.csv"

echo -e "\nStop services:"
echo "kill $MLFLOW_PID $PREDICT_PID"