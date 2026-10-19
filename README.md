# Data Versioning and Lineage Tracking System for ML Pipelines

A robust Machine Learning operations (MLOps) pipeline integrating **Git**, **DVC**, and **MLflow** for end-to-end data versioning, pipeline automation, artifact tracking, and provenance lineage.

---

## Architecture Overview

```
                                      +-------------------------+
                                      |   Git Version Control   |
                                      |  (Code, Config, Locks)  |
                                      +------------+------------+
                                                   |
              +------------------------------------+-----------------------------------+
              |                                                                        |
              v                                                                        v
+-----------------------------+                                          +-----------------------------+
|    DVC Pipeline & Cache     |                                          |    MLflow Tracking System   |
| (data/, models/, dvc.lock)  |                                          | (Params, Metrics, Artifacts)|
+--------------+--------------+                                          +--------------+--------------+
               |                                                                        |
               +--------------------> [Lineage Bridge] <--------------------------------+
                               (Git SHA + DVC MD5 Hashes logged as MLflow tags & manifest)
```

1. **Git:** Tracks source code (`src/`), pipeline definitions (`dvc.yaml`), configurations (`params.yaml`), dependencies (`requirements.txt`), and state locks (`dvc.lock`). Large binaries, dataset files, and MLflow run databases are strictly ignored.
2. **DVC (Data Version Control):** Manages data ingestion, dataset splits, feature matrices, and trained models. Guarantees deterministic step caching and reproduction via `dvc repro`. Backed by local/remote storage (`D:\ML_DVC_Storage`).
3. **MLflow:** Tracks experiment runs, hyperparameter configurations, evaluation metrics, artifact snapshots (`model.pkl`, `scaler.pkl`, `metrics.json`, `params.yaml`), and cross-system lineage metadata.

---

## Pipeline Stages

The pipeline is defined in [`dvc.yaml`](dvc.yaml) across five modular stages:

1. **`prepare`**: Loads raw Wine dataset from Scikit-Learn, saves to `data/raw/wine_dataset.csv`, and performs stratified train/test split into `data/prepared/`.
2. **`featurize`**: Fits `StandardScaler` on training features, scales train/test sets into `data/features/`, and persists `models/scaler.pkl`.
3. **`train`**: Trains a `RandomForestClassifier` using parameters in `params.yaml` and saves `models/model.pkl`.
4. **`evaluate`**: Evaluates model on test features, computing Accuracy, Precision, Recall, and F1 score, saved to `metrics.json`.
5. **`log_experiment`**: Captures active Git commit SHA, DVC artifact hashes from `dvc.lock`, system and library versions, parameters, and metrics, logging them to MLflow with a structured `lineage/lineage.json` manifest.

---

## Setup & Reproduction

### 1. Environment Activation
Activate the project virtual environment:
```powershell
.venv\Scripts\Activate.ps1
```
*Note: If running without an activated environment, explicitly invoke DVC and scripts via `.venv\Scripts\dvc.exe` and `.venv\Scripts\python.exe`.*

### 2. Reproduce Pipeline
Run the complete pipeline using DVC:
```powershell
# If .venv is activated:
dvc repro

# Or directly:
.venv\Scripts\dvc.exe repro
```
* If dependencies or code have changed, DVC executes the required stages automatically.
* If all preprocessing and model training stages are already up to date, DVC skips them without redundant retraining while `log_experiment` creates a distinct, traceable MLflow run.

### 3. Check DVC Status
```powershell
dvc status
# Or directly:
.venv\Scripts\dvc.exe status
```

### 4. Push / Pull Data Artifacts
```powershell
# Backup tracked artifacts to the configured remote
dvc push
# Or: .venv\Scripts\dvc.exe push

# Restore tracked artifacts from remote storage
dvc pull
# Or: .venv\Scripts\dvc.exe pull
```

---

## Launching the MLflow UI

MLflow uses a local SQLite database store (`sqlite:///mlflow.db`) and local artifact store for fast querying and persistence.

To launch the web interface:

```powershell
# Using the active virtual environment
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Or directly via executable:
```powershell
.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

Inside the UI, select the experiment **`ML_Lineage_Experiments`** to inspect:
- **Parameters**: `n_estimators`, `max_depth`, `random_state`, `test_size`, paths, etc.
- **Metrics**: `accuracy`, `precision`, `recall`, `f1_score`, `duration_seconds`.
- **Tags & Lineage**:
  - `git.commit_sha`: Git commit hash for the exact code state.
  - `dvc.hash.model`: Exact MD5 hash of `models/model.pkl`.
  - `dvc.hash.scaler`: Exact MD5 hash of `models/scaler.pkl`.
  - `dvc.hash.raw_dataset`: Exact MD5 hash of raw data.
  - `dvc.hash.train_data` / `dvc.hash.test_data`: Exact MD5 hashes of dataset splits.
- **Artifacts**: Downloadable `model/model.pkl`, `scaler/scaler.pkl`, `metrics/metrics.json`, `config/params.yaml`, and `lineage/lineage.json`.

---

## Historical Run Recovery (Stage 4)

The recovery tool [`scripts/recover_run.py`](scripts/recover_run.py) inspects and restores historical experiment artifacts using recorded DVC hashes:

```powershell
# List all recorded runs
python scripts/recover_run.py --list

# Inspect run parameters, metrics, tags, and environment
python scripts/recover_run.py <run_id> --inspect

# Dry-run: preview artifacts that would be restored
python scripts/recover_run.py <run_id> --dry-run

# Verify artifact hashes without restoring files
python scripts/recover_run.py <run_id> --verify-only

# Full recovery: restore artifacts from DVC cache/remote and verify hashes
python scripts/recover_run.py <run_id>
```

---

## Lineage Report Generator (Stage 5)

The lineage reporter [`scripts/lineage_report.py`](scripts/lineage_report.py) traces end-to-end data provenance and model lineage:

`Raw dataset → prepared data → features & scaler → trained model → evaluation metrics`

```powershell
# Generate human-readable terminal lineage report
python scripts/lineage_report.py <run_id>

# Output structured JSON report to terminal
python scripts/lineage_report.py <run_id> --format json

# Export JSON report to a file
python scripts/lineage_report.py <run_id> --format json --output reports/lineage_report.json

# Export human-readable report to a file
python scripts/lineage_report.py <run_id> --output reports/lineage_report.txt

# List runs available for lineage reporting
python scripts/lineage_report.py --list
```
