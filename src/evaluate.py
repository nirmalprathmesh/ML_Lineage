import os
import json
import yaml
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_params():
    """Load configuration parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        return config["featurize"], config["train"], config["evaluate"]

def main():
    feat_params, train_params, eval_params = load_params()
    
    # Load model and test feature data
    model = joblib.load(train_params["model_path"])
    test_df = pd.read_csv(feat_params["features_test_path"])
    
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted")), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted")), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted")), 4)
    }
    
    # Save metrics to JSON file
    with open(eval_params["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"[evaluate] Evaluation completed. Metrics saved to {eval_params['metrics_path']}:")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
