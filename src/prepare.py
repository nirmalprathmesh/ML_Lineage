import os
import yaml
import pandas as pd
from sklearn.datasets import load_wine


def load_params():
    """Load configuration parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["prepare"]

def main():
    params = load_params()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(params["raw_data_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(params["prepared_train_path"]), exist_ok=True)
    
    # 1. Load sample tabular dataset (Wine Dataset from Scikit-Learn)
    wine = load_wine(as_frame=True)
    df = wine.frame
    
    # Save raw data to disk
    df.to_csv(params["raw_data_path"], index=False)
    print(f"[prepare] Raw data saved to {params['raw_data_path']} ({len(df)} rows)")
    
    # 2. Train-test split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["target"]
    )
    
    # 3. Save prepared train and test splits
    train_df.to_csv(params["prepared_train_path"], index=False)
    test_df.to_csv(params["prepared_test_path"], index=False)
    print(f"[prepare] Prepared data saved: Train ({len(train_df)} rows), Test ({len(test_df)} rows)")

if __name__ == "__main__":
    main()
