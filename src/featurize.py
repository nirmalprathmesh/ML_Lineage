import os
import yaml
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_params():
    """Load configuration parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        return config["prepare"], config["featurize"]

def main():
    prep_params, feat_params = load_params()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(feat_params["features_train_path"]), exist_ok=True)
    if "scaler_path" in feat_params:
        os.makedirs(os.path.dirname(feat_params["scaler_path"]), exist_ok=True)
    
    # Load prepared datasets
    train_df = pd.read_csv(prep_params["prepared_train_path"])
    test_df = pd.read_csv(prep_params["prepared_test_path"])
    
    # Separate features and target
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    
    # Fit StandardScaler on train features only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save fitted scaler artifact
    if "scaler_path" in feat_params:
        joblib.dump(scaler, feat_params["scaler_path"])
        print(f"[featurize] Scaler saved to {feat_params['scaler_path']}")
    
    # Combine back scaled features with target
    feat_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    feat_train_df["target"] = y_train.values
    
    feat_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    feat_test_df["target"] = y_test.values
    
    # Save transformed features
    feat_train_df.to_csv(feat_params["features_train_path"], index=False)
    feat_test_df.to_csv(feat_params["features_test_path"], index=False)
    
    print(f"[featurize] Features created and saved to {feat_params['features_train_path']} and {feat_params['features_test_path']}")

if __name__ == "__main__":
    main()
