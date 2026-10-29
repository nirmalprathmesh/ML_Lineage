"""
ml_lineage.wine
===============
Explicit schema definitions and constants for the Scikit-Learn Wine classification demo.
Preserves backward compatibility for the reference demo while isolating domain-specific
tags and pipeline definitions away from the generic core.
"""

# Artifact hash tags recorded in MLflow runs for the Wine demonstration
WINE_DVC_TAG_MAP = {
    "dvc.hash.raw_dataset":     "data/raw/wine_dataset.csv",
    "dvc.hash.prepared_train":  "data/prepared/train.csv",
    "dvc.hash.prepared_test":   "data/prepared/test.csv",
    "dvc.hash.features_train":  "data/features/train_features.csv",
    "dvc.hash.features_test":   "data/features/test_features.csv",
    "dvc.hash.scaler":          "models/scaler.pkl",
    "dvc.hash.model":           "models/model.pkl",
}

# Pipeline stage definitions for the Wine demonstration
WINE_PIPELINE_STAGES = [
    {
        "stage": "prepare",
        "description": "Raw data ingestion & stratified train/test split",
        "inputs": [],
        "outputs": [
            {"path": "data/raw/wine_dataset.csv", "tag": "dvc.hash.raw_dataset", "role": "Raw Dataset"},
            {"path": "data/prepared/train.csv", "tag": "dvc.hash.prepared_train", "role": "Prepared Train Split"},
            {"path": "data/prepared/test.csv", "tag": "dvc.hash.prepared_test", "role": "Prepared Test Split"},
        ]
    },
    {
        "stage": "featurize",
        "description": "StandardScaler fitting & feature matrix transformation",
        "inputs": ["data/prepared/train.csv", "data/prepared/test.csv"],
        "outputs": [
            {"path": "models/scaler.pkl", "tag": "dvc.hash.scaler", "role": "Fitted StandardScaler"},
            {"path": "data/features/train_features.csv", "tag": "dvc.hash.features_train", "role": "Scaled Train Features"},
            {"path": "data/features/test_features.csv", "tag": "dvc.hash.features_test", "role": "Scaled Test Features"},
        ]
    },
    {
        "stage": "train",
        "description": "RandomForestClassifier training on scaled features",
        "inputs": ["data/features/train_features.csv"],
        "outputs": [
            {"path": "models/model.pkl", "tag": "dvc.hash.model", "role": "Trained Classifier Model"},
        ]
    },
    {
        "stage": "evaluate",
        "description": "Performance evaluation on test features",
        "inputs": ["data/features/test_features.csv", "models/model.pkl"],
        "outputs": [
            {"path": "metrics.json", "tag": None, "role": "Calculated Evaluation Metrics"},
        ]
    }
]

WINE_EXPERIMENT_NAME = "ML_Lineage_Experiments"
