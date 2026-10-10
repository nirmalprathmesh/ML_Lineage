import os
import sys

# Configure MLflow environment flags before import
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import json
import yaml
import time
import platform
import subprocess
import joblib
import pandas as pd
import numpy as np
import sklearn
import dvc
import mlflow
import mlflow.sklearn

def load_params():
    """Load configuration from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def get_git_info():
    """Retrieve Git commit SHA, branch, and working tree cleanliness."""
    try:
        sha_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        commit_sha = sha_proc.stdout.strip()
    except Exception:
        commit_sha = "unavailable"

    try:
        branch_proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        branch = branch_proc.stdout.strip()
    except Exception:
        branch = "unavailable"

    try:
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        is_dirty = bool(status_proc.stdout.strip())
    except Exception:
        is_dirty = False

    return {
        "commit_sha": commit_sha,
        "branch": branch,
        "is_dirty": is_dirty
    }

def get_dvc_hashes():
    """Extract artifact MD5 hashes directly from dvc.lock."""
    hashes = {
        "raw_dataset": "unavailable",
        "prepared_train": "unavailable",
        "prepared_test": "unavailable",
        "features_train": "unavailable",
        "features_test": "unavailable",
        "scaler": "unavailable",
        "model": "unavailable"
    }

    if not os.path.exists("dvc.lock"):
        return hashes

    try:
        with open("dvc.lock", "r") as f:
            lock_data = yaml.safe_load(f)

        stages = lock_data.get("stages", {})

        # Stage: prepare
        prep_outs = {item["path"]: item.get("md5", "unavailable") for item in stages.get("prepare", {}).get("outs", [])}
        hashes["raw_dataset"] = prep_outs.get("data/raw/wine_dataset.csv", "unavailable")
        hashes["prepared_train"] = prep_outs.get("data/prepared/train.csv", "unavailable")
        hashes["prepared_test"] = prep_outs.get("data/prepared/test.csv", "unavailable")

        # Stage: featurize
        feat_outs = {item["path"]: item.get("md5", "unavailable") for item in stages.get("featurize", {}).get("outs", [])}
        hashes["features_train"] = feat_outs.get("data/features/train_features.csv", "unavailable")
        hashes["features_test"] = feat_outs.get("data/features/test_features.csv", "unavailable")
        hashes["scaler"] = feat_outs.get("models/scaler.pkl", "unavailable")

        # Stage: train
        train_outs = {item["path"]: item.get("md5", "unavailable") for item in stages.get("train", {}).get("outs", [])}
        hashes["model"] = train_outs.get("models/model.pkl", "unavailable")

    except Exception as e:
        print(f"[log_experiment] Warning: Could not parse dvc.lock: {e}")

    return hashes

def get_library_versions():
    """Retrieve environment and critical dependency versions."""
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "dvc": dvc.__version__,
        "mlflow": mlflow.__version__,
        "os_platform": platform.platform()
    }

def main():
    start_time = time.time()
    params = load_params()

    mlflow_cfg = params.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "ML_Lineage_Experiments")

    # Configure MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load evaluated metrics from metrics.json
    metrics_path = params.get("evaluate", {}).get("metrics_path", "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}. Run pipeline first.")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Gather provenance & lineage metadata
    git_info = get_git_info()
    dvc_hashes = get_dvc_hashes()
    lib_versions = get_library_versions()

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[log_experiment] Active MLflow Run ID: {run_id}")
        print(f"[log_experiment] Experiment Name: {experiment_name}")

        # 1. Log Parameters
        log_params = {
            "model_type": "RandomForestClassifier",
            "scaler_type": "StandardScaler",
            "n_estimators": params["train"]["n_estimators"],
            "max_depth": params["train"]["max_depth"],
            "random_state": params["train"]["random_state"],
            "test_size": params["prepare"]["test_size"],
            "prepare_random_state": params["prepare"]["random_state"],
            "raw_data_path": params["prepare"]["raw_data_path"],
            "prepared_train_path": params["prepare"]["prepared_train_path"],
            "prepared_test_path": params["prepare"]["prepared_test_path"],
            "features_train_path": params["featurize"]["features_train_path"],
            "features_test_path": params["featurize"]["features_test_path"],
            "scaler_path": params["featurize"]["scaler_path"],
            "model_path": params["train"]["model_path"],
            "metrics_path": metrics_path
        }
        mlflow.log_params(log_params)

        # 2. Log Metrics
        eval_metrics = {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"])
        }
        mlflow.log_metrics(eval_metrics)

        # 3. Log Provenance & Lineage Tags
        tags = {
            # Git provenance
            "git.commit_sha": git_info["commit_sha"],
            "git.branch": git_info["branch"],
            "git.is_dirty": str(git_info["is_dirty"]),
            # DVC data and model lineage hashes
            "dvc.hash.raw_dataset": dvc_hashes["raw_dataset"],
            "dvc.hash.prepared_train": dvc_hashes["prepared_train"],
            "dvc.hash.prepared_test": dvc_hashes["prepared_test"],
            "dvc.hash.features_train": dvc_hashes["features_train"],
            "dvc.hash.features_test": dvc_hashes["features_test"],
            "dvc.hash.scaler": dvc_hashes["scaler"],
            "dvc.hash.model": dvc_hashes["model"],
            # Environment & Libraries
            "env.python_version": lib_versions["python"],
            "env.os_platform": lib_versions["os_platform"],
            "env.scikit_learn": lib_versions["scikit_learn"],
            "env.pandas": lib_versions["pandas"],
            "env.numpy": lib_versions["numpy"],
            "env.dvc": lib_versions["dvc"],
            "env.mlflow": lib_versions["mlflow"]
        }
        mlflow.set_tags(tags)

        # 4. Log Artifacts
        # Model & Scaler binaries
        model_path = params["train"]["model_path"]
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, artifact_path="model")
            # Also log scikit-learn model schema
            try:
                model_obj = joblib.load(model_path)
                mlflow.sklearn.log_model(
                    sk_model=model_obj,
                    name="sklearn_model",
                    serialization_format="cloudpickle"
                )
            except Exception as e:
                print(f"[log_experiment] Warning: Could not log sklearn_model schema: {e}")

        scaler_path = params["featurize"]["scaler_path"]
        if os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path, artifact_path="scaler")

        # Metrics and params configuration files
        if os.path.exists(metrics_path):
            mlflow.log_artifact(metrics_path, artifact_path="metrics")
        if os.path.exists("params.yaml"):
            mlflow.log_artifact("params.yaml", artifact_path="config")

        # 5. Log Complete Lineage Manifest JSON
        duration = round(time.time() - start_time, 3)
        mlflow.log_metric("duration_seconds", duration)

        lineage_manifest = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "duration_seconds": duration,
            "git": git_info,
            "dvc_hashes": dvc_hashes,
            "parameters": log_params,
            "metrics": eval_metrics,
            "libraries": lib_versions
        }
        mlflow.log_dict(lineage_manifest, "lineage/lineage.json")

        print(f"[log_experiment] Run logged successfully.")
        print(f"  - Metrics: {eval_metrics}")
        print(f"  - Git Commit: {git_info['commit_sha']}")
        print(f"  - Model DVC Hash: {dvc_hashes['model']}")
        print(f"  - Scaler DVC Hash: {dvc_hashes['scaler']}")
        print(f"  - Raw Dataset DVC Hash: {dvc_hashes['raw_dataset']}")
        print(f"  - Duration: {duration}s")

if __name__ == "__main__":
    main()
