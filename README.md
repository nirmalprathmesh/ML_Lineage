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

### 1. Fresh Machine Setup (Windows)

#### Step A — Clone and Enter Repository
```powershell
git clone https://github.com/nirmalprathmesh/ML_Lineage.git
cd ML_Lineage
```

#### Step B — Create Virtual Environment
```powershell
python -m venv .venv
```

#### Step C — Activate Virtual Environment
```powershell
.venv\Scripts\Activate.ps1
```
*Note: If PowerShell script execution is restricted, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` for your session.*

#### Step D — Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Step E — Verify Tools & Environment
Confirm that Python, DVC, and MLflow are installed and accessible:
```powershell
python --version
dvc --version
mlflow --version
```
*Note: You can also invoke these tools directly without shell activation using `.venv\Scripts\python.exe`, `.venv\Scripts\dvc.exe`, and `.venv\Scripts\mlflow.exe`.*

#### Step F — DVC Remote Storage Boundary
The project's configured DVC remote in `.dvc/config` points to a local directory:
```ini
[core]
    remote = local_storage
['remote "local_storage"']
    url = D:\ML_DVC_Storage
```
* **Local Storage Scope**: `D:\ML_DVC_Storage` is local disk storage on this computer, not a cloud backup.
* **Portability on Fresh Machines**: A fresh clone on another computer will not be able to pull or recover DVC-tracked data and model artifacts unless `D:\ML_DVC_Storage` is accessible at that path or another remote storage location is configured (e.g. `dvc remote modify local_storage url <path-or-uri>`).

---

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

### 3. Check DVC Status & Remote Sync
```powershell
# Check workspace pipeline status:
dvc status
# Or directly: .venv\Scripts\dvc.exe status

# Check remote synchronization status:
dvc status -c
# Or directly: .venv\Scripts\dvc.exe status -c
```

### 4. Push / Pull Data Artifacts
```powershell
# Backup tracked artifacts to the configured remote:
dvc push
# Or: .venv\Scripts\dvc.exe push

# Restore tracked artifacts from remote storage:
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

---

## Chatbot Interface (Stage 6)

The chatbot [`app.py`](app.py) provides a browser-based conversational interface to interact with the ML pipeline versioning and lineage tracking system:

```powershell
# Launch the chatbot (virtual environment must be active or use full path)
streamlit run app.py

# Or directly:
.venv\Scripts\streamlit.exe run app.py
```

Then open your browser at `http://localhost:8501`.

### Available Chat Commands

| Command | Description |
|---------|-------------|
| `list runs` | List all recorded MLflow experiment runs |
| `inspect <run_id>` | View full details of a specific run |
| `compare <run_id_a> <run_id_b>` | Compare two runs side-by-side |
| `verify <run_id>` | Verify artifact hashes against workspace |
| `lineage <run_id>` | Generate full lineage provenance report |
| `status` | Show Git, DVC, and MLflow system status |
| `help` | Show command reference |

Partial run IDs (prefix match) are supported. Natural language queries like "show me the runs" or "what's the lineage for run a0c974c4?" also work.
