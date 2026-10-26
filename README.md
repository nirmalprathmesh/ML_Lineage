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

## ML Lineage Studio Dashboard & Chat Assistant

[`app.py`](app.py) and [`dashboard_views.py`](dashboard_views.py) provide a modern, professional browser-based application for ML experiment tracking, pipeline data versioning, lineage visualization, artifact verification, historical run recovery, and conversational querying.

```powershell
# Launch the dashboard (virtual environment must be active or use full path)
streamlit run app.py

# Or directly:
.venv\Scripts\streamlit.exe run app.py
```

Then open your browser at:
```
http://localhost:8501
```

### Dashboard Sections

The application provides eight dedicated sections accessible via the sidebar navigation:

1. **Overview**: High-level KPI metrics (total runs, finished, failed, running, latest accuracy/F1, Git status, DVC remote status), Altair metric score trends across runs, and recent experiment history.
2. **Experiments**: Searchable and filterable run table with search by ID or Git commit SHA, status multiselect, accuracy slider, and detailed run inspector tabs (Parameters, Git Provenance, Environment, Artifact Hashes, and Quick Action buttons). Supports CSV export.
3. **Compare**: Side-by-side run comparison with metric delta indicators (`+0.0000`), parameter differences, artifact hash comparison, Git commit/branch diffs, and downloadable comparison CSV.
4. **Lineage**: Visual DAG trace across all 5 pipeline stages (`prepare`, `featurize`, `train`, `evaluate`, `log_experiment`). Inspect inputs, outputs, recorded MD5 hashes, verification statuses, and download structured lineage JSON.
5. **Artifacts**: Per-artifact recorded MD5 hash vs current workspace hash vs `dvc.lock` hash for all 7 tracked pipeline files. Includes live non-mutating CLI verification (`recover_run.py --verify-only`).
6. **Recovery & Workspaces**:
   - **Isolated Workspace (Recommended)**: Restores historical experiment states into a separate directory using `git worktree add <dest> <commit_sha>` without modifying active workspace files.
   - **Current-Workspace Restore**: Deliberate two-step safety workflow requiring `--dry-run` inspection and explicit user confirmation before overwriting `data/` and `models/` artifacts.
7. **Chat Assistant**: Natural language and command-based assistant with suggestion chips, strict regex intent routing, execution outputs, and safety boundaries.
8. **System Status**: Deep diagnostic checks for Git repository (branch, commit, working tree modifications), MLflow connectivity, DVC workspace pipeline (`dvc status`), and DVC remote storage synchronization (`dvc status -c`).

### Available Chat Commands

| Command | Category | Safety | Description |
|---------|----------|--------|-------------|
| `list runs` | Tracking | 🟢 Read-Only | List all recorded MLflow experiment runs |
| `inspect <run_id>` | Tracking | 🟢 Read-Only | View full details, parameters, metrics, Git tags, and DVC hashes |
| `compare <id_a> <id_b>` | Tracking | 🟢 Read-Only | Compare two runs side-by-side with metric delta indicators |
| `verify <run_id>` | Verification | 🟢 Read-Only | Verify workspace artifact hashes using `scripts/recover_run.py --verify-only` |
| `lineage <run_id>` | Lineage | 🟢 Read-Only | Generate full pipeline lineage DAG report via `scripts/lineage_report.py` |
| `preview recover <run_id>`<br>`dry-run <run_id>` | Recovery | 🟢 Read-Only | Execute `scripts/recover_run.py --dry-run` to preview artifact restoration |
| `recover <run_id>`<br>`restore <run_id>` | Recovery | 🟢 Read-Only (Until Confirmed) | Perform dry-run preview and request explicit user confirmation to restore |
| `confirm recover <run_id>`<br>`confirm restore <run_id>` | Recovery | ⚠️ **Overwrites Workspace Files** | Restore historical artifacts from DVC cache/remote into `data/` and `models/` |
| `cancel` | Recovery | 🟢 Read-Only | Cancel pending recovery operation |
| `status` | System | 🟢 Read-Only | Show Git provenance, DVC workspace pipeline, DVC remote sync, and MLflow status |
| `help` | System | 🟢 Read-Only | Show comprehensive command reference and file safety guide |

### Dry-Run Before Confirmation & File Safety Workflow

To safeguard active development and current models against unintentional data loss:
1. Entering `recover <run_id>` or `restore <run_id>` **never immediately overwrites files**.
2. The system executes a non-mutating `--dry-run` first to inspect artifact availability in the local DVC cache and configured DVC remote.
3. A prominent overwrite warning is displayed alongside the restoration plan.
4. The user must provide explicit matching confirmation by entering `confirm recover <run_id>` or clicking the **Confirm Restore** button in the Streamlit UI.
5. Confirmation state is isolated strictly per user session in `st.session_state` and is immediately consumed prior to invocation.
6. Entering `cancel` or any other recognized command immediately clears the pending confirmation state.

### Running Automated Tests and Real-Data Validation

The test suite covers backend data helpers, chatbot command routing, confirmation isolation, and real-data verification:

```powershell
# 1. Run the 24-point unit and mock test suite for chatbot recovery & isolation:
.venv\Scripts\python.exe -m unittest tests/test_chatbot_recovery.py

# 2. Run the 7-point dashboard data transformation and comparison test suite:
.venv\Scripts\python.exe -m unittest tests/test_dashboard_data.py

# 3. Run the 36-point real-data validation runner against MLflow runs & historical Git commits:
.venv\Scripts\python.exe tests/validate_real_data.py

# 4. Run real-data validation with Python UTF-8 mode:
.venv\Scripts\python.exe -X utf8 tests/validate_real_data.py
```

### Limitations & Operating Boundaries
- **Local DVC Remote**: Recovery looks for artifacts in the local DVC cache (`.dvc/cache`) and the configured local storage directory (`D:\ML_DVC_Storage`). On another machine, configure a new remote via `dvc remote modify local_storage url <path>`.
- **Intent Parsing**: The chatbot uses deterministic regex pattern matching for strict routing. Natural language phrasing containing supported keywords is supported, but arbitrary conversational queries will fall back to suggested command help.
- **Workspace Overwrite Scope**: Confirmed restoration directly overwrites the 7 tracked DVC pipeline artifacts in `data/` and `models/`. Code files, `params.yaml`, and `dvc.lock` remain governed by Git.
- **Stale Runs**: Any runs that were interrupted or failed to record complete metadata (e.g. status `RUNNING` without metrics) are honestly presented in the UI with missing fields labeled as unavailable (`—` / `N/A`), and are never silently altered or removed.
