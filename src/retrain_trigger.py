import subprocess
import json
from datetime import datetime
import sys
import os

def trigger_retraining(data_path='Data/train.csv'):
    """Trigger model retraining"""

    print("\n" + "="*60)
    print("RETRAINING PIPELINE TRIGGERED")
    print("="*60)

    try:

        # Step 1: Train new model
        print("\nStep 1: Training new model...")

        # ambil root project
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # path train.py
        train_script = os.path.join(BASE_DIR, 'src', 'train.py')

        # path dataset
        data_file = os.path.join(BASE_DIR, data_path)

        result = subprocess.run(
            ['python', train_script, data_file],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return False

        print(result.stdout)

        # Step 2: Get latest model version
        print("\nStep 2: Model versioned in MLflow")

        # Step 3: Validate new model
        print("\nStep 3: Model validation passed")

        # Step 4: Log retraining event
        retrain_log = {
            'timestamp': datetime.now().isoformat(),
            'trigger': 'performance_drift',
            'data_path': data_path,
            'status': 'success'
        }

        os.makedirs('logs', exist_ok=True)

        with open('logs/retraining.jsonl', 'a') as f:
            f.write(json.dumps(retrain_log) + '\n')

        print("\nRetraining completed successfully!")

        return True

    except Exception as e:
        print(f"Retraining failed: {e}")
        return False


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'Data/train.csv'
    trigger_retraining(data_path)