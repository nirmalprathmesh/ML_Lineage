import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_params():
    """Load configuration parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        return config["featurize"], config["train"]

def main():
    feat_params, train_params = load_params()
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(train_params["model_path"]), exist_ok=True)
    
    # Load training feature matrix
    train_df = pd.read_csv(feat_params["features_train_path"])
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    
    # Initialize Random Forest Classifier with params
    model = RandomForestClassifier(
        n_estimators=train_params["n_estimators"],
        max_depth=train_params["max_depth"],
        random_state=train_params["random_state"]
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Save model artifact
    joblib.dump(model, train_params["model_path"])
    print(f"[train] Model successfully trained and saved to {train_params['model_path']}")

if __name__ == "__main__":
    main()
