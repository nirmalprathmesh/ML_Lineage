# ML Lineage Studio
### End-to-End MLOps Provenance, Data Versioning, Experiment Tracking, and Artifact Recovery

A comprehensive Machine Learning operations (MLOps) platform integrating **Git**, **DVC**, and **MLflow** for end-to-end data versioning, deterministic pipeline automation, cross-system artifact tracking, and verifiable provenance lineage.

---

## The Problem It Solves

Modern machine learning development faces three critical reproducibility and governance gaps:
1. **Data & Model Drift Without Auditability**: Code changes are tracked in Git, but datasets, feature matrices, and binary model artifacts evolve separately, making it difficult to answer *"Which exact dataset split and preprocessing step produced this model?"*
2. **Disconnected Experiment Tracking**: MLflow logs hyperparameters and evaluation metrics, but often lacks cryptographic links to the underlying binary files and exact pipeline state that generated them.
3. **High-Risk Historical Recovery**: Reverting to an earlier experiment run often risks accidentally overwriting active workspace files or failing because intermediate data dependencies were lost.

**ML Lineage Studio** solves these challenges by establishing an auditable **Lineage Bridge**:
- Every pipeline execution logs the active **Git commit SHA** and **DVC MD5 hashes** as MLflow tags and a structured lineage manifest (`lineage/lineage.json`).
- Workspace artifacts can be verified on-demand against recorded hashes (`scripts/recover_run.py --verify-only`).
- Historical experiments can be inspected, previewed via dry-run simulation, restored into isolated workspaces (`git worktree`), or safely recovered into the active workspace with explicit confirmation gates.

> **Note on Project Status**: ML Lineage Studio is a fully functioning MLOps engineering system and architectural reference implementation. It is designed for transparent reproducibility, local auditing, and portfolio demonstration rather than multi-tenant cloud SaaS deployment.

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

1. **Git**: Tracks source code (`src/`), pipeline orchestration (`dvc.yaml`), hyperparameters (`params.yaml`), dependencies (`requirements.txt`), and state locks (`dvc.lock`). Large binaries, datasets, and tracking databases are strictly ignored.
2. **DVC (Data Version Control)**: Manages raw data ingestion, stratified splits, feature matrices, and serialized model objects. Caches intermediate stages and enables reproducible stage execution via `dvc repro`. Configured with a dedicated storage remote.
3. **MLflow**: Tracks experiment runs, hyperparameter sweeps, evaluation metrics, artifact snapshots (`model.pkl`, `scaler.pkl`, `metrics.json`, `params.yaml`), and cross-system lineage metadata.
4. **Lineage Bridge**: Links Git code commits, DVC binary hashes, and MLflow run records into a single verifiable graph accessible via CLI tools and the interactive dashboard.

---

## Pipeline Stages

The reproducible pipeline is orchestrated in [`dvc.yaml`](dvc.yaml) across five modular stages using a Scikit-Learn Wine classification task:

1. **`prepare`**: Loads the Wine multiclass dataset, persists raw data to `data/raw/wine_dataset.csv`, and creates stratified train/test splits in `data/prepared/`.
2. **`featurize`**: Fits a `StandardScaler` on training features, scales train/test feature sets into `data/features/`, and persists the fitted transformer to `models/scaler.pkl`.
3. **`train`**: Fits a `RandomForestClassifier` according to hyperparameters defined in `params.yaml` (`n_estimators`, `max_depth`, `random_state`) and persists the model to `models/model.pkl`.
4. **`evaluate`**: Evaluates the trained classifier on test features, computing Accuracy, Precision, Recall, and F1 score, exported to `metrics.json`.
5. **`log_experiment`**: Captures the active Git commit SHA, DVC artifact hashes from `dvc.lock`, environment details, parameters, and metrics, logging them to MLflow with a structured `lineage/lineage.json` manifest.

---

## Setup & Local Reproduction

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
*Tip: You can also invoke these tools directly without activating the shell using `.venv\Scripts\python.exe`, `.venv\Scripts\dvc.exe`, and `.venv\Scripts\mlflow.exe`.*

#### Step F — DVC Remote Storage Boundary
The project's configured DVC remote in `.dvc/config` points to a local directory:
```ini
[core]
    remote = local_storage
['remote "local_storage"']
    url = D:\ML_DVC_Storage
```
* **Local Storage Scope**: `D:\ML_DVC_Storage` is local disk storage on the original host machine, not a cloud backup.
* **Portability on Another Machine**: A fresh clone on another computer will not have access to `D:\ML_DVC_Storage`. To configure a local folder or cloud storage on a new machine:
  ```powershell
  # Point the remote to a directory on your machine:
  dvc remote modify local_storage url C:\path\to\my_dvc_storage
  ```
* **Reproduction Without Pre-cached Binaries**: Running `dvc repro` executes the pipeline from scratch and regenerates all data, features, and model artifacts locally without needing external downloads.

---

### 2. Reproduce Pipeline
Run the complete pipeline using DVC:
```powershell
# With virtual environment activated:
dvc repro

# Or directly via venv executable:
.venv\Scripts\dvc.exe repro
```
* If dependencies, parameters, or code have changed, DVC executes the required stages automatically.
* If all preprocessing and model training stages are already up to date, DVC skips them without redundant computation, while `log_experiment` records a distinct, traceable MLflow run.

### 3. Check DVC Status & Remote Sync
```powershell
# Check workspace pipeline status:
dvc status

# Check remote storage synchronization status:
dvc status -c
```

### 4. Push / Pull Data Artifacts
```powershell
# Backup tracked artifacts to the configured storage remote:
dvc push

# Restore tracked artifacts from storage remote:
dvc pull
```

---

## Launching the MLflow UI

MLflow uses a local SQLite database store (`sqlite:///mlflow.db`) and local artifact store for fast querying and persistence.

To launch the web interface:

```powershell
# Using the active virtual environment:
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Or directly:
.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

Inside the UI, select the experiment **`ML_Lineage_Experiments`** to inspect:
- **Parameters**: `n_estimators`, `max_depth`, `random_state`, `test_size`, paths.
- **Metrics**: `accuracy`, `precision`, `recall`, `f1_score`, `duration_seconds`.
- **Tags & Lineage**:
  - `git.commit_sha`: Git commit hash for the exact code state.
  - `dvc.hash.model`: Exact MD5 hash of `models/model.pkl`.
  - `dvc.hash.scaler`: Exact MD5 hash of `models/scaler.pkl`.
  - `dvc.hash.raw_dataset`: Exact MD5 hash of raw data.
  - `dvc.hash.train_data` / `dvc.hash.test_data`: Exact MD5 hashes of dataset splits.
- **Artifacts**: Downloadable `model/model.pkl`, `scaler/scaler.pkl`, `metrics/metrics.json`, `config/params.yaml`, and `lineage/lineage.json`.

---

## Historical Run Recovery CLI

The command-line recovery tool [`scripts/recover_run.py`](scripts/recover_run.py) inspects and restores historical experiment artifacts using recorded DVC hashes:

```powershell
# List all recorded runs in the MLflow tracking store:
python scripts/recover_run.py --list

# Inspect run parameters, metrics, tags, and environment:
python scripts/recover_run.py <run_id> --inspect

# Dry-run: preview artifacts that would be restored (read-only):
python scripts/recover_run.py <run_id> --dry-run

# Verify current workspace artifact hashes against recorded run (read-only):
python scripts/recover_run.py <run_id> --verify-only

# Full recovery: restore artifacts from DVC cache/remote into active workspace:
python scripts/recover_run.py <run_id>
```

> ⚠️ **Warning on Full Recovery**: Running `python scripts/recover_run.py <run_id>` without flags will directly overwrite the 7 tracked DVC pipeline files in `data/` and `models/`. Always perform `--dry-run` or `--verify-only` first.

---

## Lineage Report Generator CLI

The lineage reporting tool [`scripts/lineage_report.py`](scripts/lineage_report.py) traces end-to-end data provenance and model lineage:

$$\text{Raw dataset} \longrightarrow \text{Prepared data} \longrightarrow \text{Features \& Scaler} \longrightarrow \text{Trained model} \longrightarrow \text{Evaluation metrics}$$

```powershell
# Generate human-readable terminal lineage report:
python scripts/lineage_report.py <run_id>

# Output structured JSON report to terminal:
python scripts/lineage_report.py <run_id> --format json

# Export JSON report to a file:
python scripts/lineage_report.py <run_id> --format json --output reports/lineage_report.json

# Export human-readable report to a file:
python scripts/lineage_report.py <run_id> --output reports/lineage_report.txt

# List runs available for lineage reporting:
python scripts/lineage_report.py --list
```

---

## ML Lineage Studio Dashboard & Chat Assistant

[`app.py`](app.py) and [`dashboard_views.py`](dashboard_views.py) provide a modern, professional browser-based application for ML experiment tracking, pipeline data versioning, lineage visualization, artifact verification, historical run recovery, and conversational querying.

```powershell
# Launch the dashboard:
streamlit run app.py

# Or directly via venv executable:
.venv\Scripts\streamlit.exe run app.py
```

Then open your browser at:
```
http://localhost:8501
```

### Dashboard Navigation & Sections

The application provides eight dedicated views accessible via the sidebar navigation:

1. **Overview**: Executive KPI metrics (Total Runs, Finished, Failed, Running, Latest Accuracy, Latest F1), performance trend visualization over experiment runs, and recent run history.
2. **Experiments**: Multi-filter searchable run table (search by ID or Git commit SHA, status multiselect, accuracy slider) with a detailed inspector containing Parameters, Git Provenance, Environment, and Artifact Hashes tabs. Includes CSV export.
3. **Compare Runs**: Side-by-side run comparison with metric delta indicators (`+0.0000`), hyperparameter differences, artifact hash comparison, Git commit diffs, and downloadable comparison CSV.
4. **Lineage Explorer**: Visual pipeline DAG trace across all 5 stages (`prepare`, `featurize`, `train`, `evaluate`, `log_experiment`). Inspect inputs, outputs, recorded MD5 hashes, verification statuses, and download structured lineage JSON.
5. **Artifacts**: Per-artifact recorded MD5 hash vs current workspace hash vs `dvc.lock` hash for all 7 tracked pipeline files. Includes live non-mutating CLI verification (`recover_run.py --verify-only`).
6. **Recovery Center**:
   - **Isolated Workspace (Recommended)**: Restores historical experiment states into a separate directory using `git worktree add <dest> <commit_sha>` without modifying active workspace files.
   - **Current-Workspace Restore**: Deliberate two-step safety workflow requiring `--dry-run` inspection and explicit user confirmation before overwriting `data/` and `models/` artifacts.
7. **Chat Assistant**: Natural language and command-based assistant with suggestion chips, strict regex intent routing, execution outputs, and safety boundaries. Styled with a dark composer, high-contrast readable text, and responsive action states.
8. **System Health**: Deep infrastructure diagnostics covering Git repository status (branch, commit, working tree modifications), MLflow connectivity, DVC workspace pipeline (`dvc status`), and DVC remote storage synchronization (`dvc status -c`).

---

### Available Chat Commands

| Command | Category | Safety | Description |
|:---|:---|:---|:---|
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

---

### Dry-Run Before Confirmation & Safety Workflow

To safeguard active development and current models against unintentional data loss:
1. Entering `recover <run_id>` or `restore <run_id>` **never immediately overwrites files**.
2. The system executes a non-mutating `--dry-run` first to inspect artifact availability in the local DVC cache and configured DVC remote.
3. A prominent overwrite warning is displayed alongside the restoration plan.
4. The user must provide explicit matching confirmation by entering `confirm recover <run_id>` or clicking the **Confirm Restore** button in the Streamlit UI.
5. Confirmation state is isolated strictly per user session in `st.session_state` and is immediately consumed prior to invocation.
6. Entering `cancel` or any other recognized command immediately clears the pending confirmation state.

---

## Automated Test Suite & Validation

The test suite covers backend data helpers, chatbot command routing, session confirmation isolation, and real-data verification:

```powershell
# 1. Run the entire unit and integration test suite (31 tests):
.venv\Scripts\python.exe -m unittest discover -s tests

# 2. Run the 24-point unit and mock test suite for chatbot recovery & isolation:
.venv\Scripts\python.exe -m unittest tests/test_chatbot_recovery.py

# 3. Run the 7-point dashboard data transformation and comparison test suite:
.venv\Scripts\python.exe -m unittest tests/test_dashboard_data.py

# 4. Run the 36-point real-data validation runner against MLflow runs & historical Git commits:
.venv\Scripts\python.exe tests/validate_real_data.py

# 5. Run real-data validation with Python UTF-8 mode:
.venv\Scripts\python.exe -X utf8 tests/validate_real_data.py
```

### Verified Test Results

- **Unit & Integration Suite**: 31 tests executed, **31 passed, 0 failed** in ~22s.
- **Chatbot Recovery Suite**: 24 tests executed, **24 passed, 0 failed** (covers intent parsing, session isolation, confirmation rejections, and mock subprocess routing).
- **Dashboard Data Suite**: 7 tests executed, **7 passed, 0 failed** (covers run resolution, metric delta formatting, and CSV export structures).
- **Real-Data Validation Runner**: 36 automated checks executed against live SQLite tracking data (`mlflow.db`) and historical Git commits, **36 passed, 0 failed, 0 files modified** (strictly read-only).

---

## Limitations & Operating Boundaries

- **Local Storage Remote**: DVC remote storage is configured to a local path (`D:\ML_DVC_Storage`). When cloning to another machine, update the remote target using `dvc remote modify local_storage url <new-path>`.
- **Environment & Dependency Pinned State**: DVC guarantees deterministic tracking of dataset and model binary hashes. However, exact model reproduction across different operating systems requires matching Python versions and package versions pinned in `requirements.txt`.
- **Chatbot Intent Routing**: The Chat Assistant uses deterministic regex pattern matching for strict routing. Natural language phrasing containing supported keywords is supported, while unsupported conversational queries fall back to suggested command help.
- **Workspace Overwrite Scope**: Confirmed restoration directly overwrites the 7 tracked DVC pipeline artifacts in `data/` and `models/`. Code files, `params.yaml`, and `dvc.lock` remain governed by Git.
- **Interrupted Runs**: Interrupted runs (e.g. status `RUNNING` without completed metrics) are honestly presented in the UI with missing values marked as unavailable (`—` / `N/A`), and are never silently altered or pruned.
