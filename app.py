#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py
======
ML Lineage Studio — Streamlit application.

Provides the backend data functions, chatbot command router, session isolation,
and main() entry point for the ML Lineage Studio dashboard.

Launch:
    .venv\\Scripts\\streamlit.exe run app.py
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time

import pandas as pd
import streamlit as st
import yaml

import dashboard_views

# ─────────────────────────────────────────────────────────────────────────────
# Suppress MLflow telemetry & configure environment before import
# ─────────────────────────────────────────────────────────────────────────────
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from mlflow.tracking import MlflowClient

# ─────────────────────────────────────────────────────────────────────────────
# Project Constants
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")
DVC_LOCK_PATH = os.path.join(PROJECT_ROOT, "dvc.lock")

DVC_TAG_MAP = {
    "dvc.hash.raw_dataset":     "data/raw/wine_dataset.csv",
    "dvc.hash.prepared_train":  "data/prepared/train.csv",
    "dvc.hash.prepared_test":   "data/prepared/test.csv",
    "dvc.hash.features_train":  "data/features/train_features.csv",
    "dvc.hash.features_test":   "data/features/test_features.csv",
    "dvc.hash.scaler":          "models/scaler.pkl",
    "dvc.hash.model":           "models/model.pkl",
}


# ─────────────────────────────────────────────────────────────────────────────
# MLflow / Config Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_tracking_uri() -> str:
    """Retrieve absolute tracking URI from params.yaml or fallback."""
    raw_uri = "sqlite:///mlflow.db"
    if os.path.exists(PARAMS_PATH):
        try:
            with open(PARAMS_PATH, "r", encoding="utf-8") as f:
                params = yaml.safe_load(f)
            raw_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
        except Exception:
            pass
    if raw_uri.startswith("sqlite:///") and not os.path.isabs(raw_uri[10:]):
        abs_db = os.path.join(PROJECT_ROOT, raw_uri[10:])
        return f"sqlite:///{abs_db}"
    return raw_uri


def get_mlflow_client() -> MlflowClient:
    """Create an MLflow client configured with the project's tracking URI."""
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def compute_file_md5(filepath: str) -> str:
    """Calculate MD5 hash of a local file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dvc_lock_hashes() -> dict:
    """Parse dvc.lock to extract current stage output MD5 hashes."""
    hashes = {}
    if not os.path.exists(DVC_LOCK_PATH):
        return hashes
    try:
        with open(DVC_LOCK_PATH, "r", encoding="utf-8") as f:
            lock = yaml.safe_load(f) or {}
        for stage_name, stage_info in lock.get("stages", {}).items():
            for out in stage_info.get("outs", []):
                p = out.get("path")
                md5 = out.get("md5")
                if p and md5:
                    hashes[p] = md5
    except Exception:
        pass
    return hashes


def get_python_executable() -> str:
    """Return path to verified project venv Python or sys.executable."""
    venv_py = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable


# Test-only isolated confirmation fixture hook (ONLY active when explicitly enabled by unit tests/harnesses)
_test_confirmation_override = None


class IsolatedTestSession:
    """Explicitly isolated test fixture simulating a Streamlit session outside Streamlit runtime."""

    def __init__(self, initial_state: dict = None):
        self.store = initial_state.copy() if initial_state else {}

    def __enter__(self):
        global _test_confirmation_override
        self._prev = _test_confirmation_override
        _test_confirmation_override = self.store
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _test_confirmation_override
        _test_confirmation_override = self._prev


def enable_test_session_override(store: dict = None) -> dict:
    """Enable isolated test session store for testing outside Streamlit."""
    global _test_confirmation_override
    if store is None:
        store = {}
    _test_confirmation_override = store
    return store


def disable_test_session_override() -> None:
    """Disable test session store."""
    global _test_confirmation_override
    _test_confirmation_override = None


def is_running_in_streamlit() -> bool:
    """Check if currently executing within an active Streamlit runtime context."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def get_confirmation_state() -> dict | None:
    """
    Retrieve pending recovery confirmation state.
    Strictly isolated to st.session_state when running in Streamlit.
    Outside Streamlit, only accessible if an isolated test session is explicitly active.
    """
    if _test_confirmation_override is not None:
        return _test_confirmation_override.get("pending_recovery")

    if is_running_in_streamlit():
        try:
            return st.session_state.get("pending_recovery")
        except Exception:
            return None

    return None


def set_confirmation_state(state: dict | None) -> None:
    """
    Set or clear pending recovery confirmation state.
    Strictly isolated to st.session_state in Streamlit.
    """
    if _test_confirmation_override is not None:
        if state is None:
            _test_confirmation_override.pop("pending_recovery", None)
        else:
            _test_confirmation_override["pending_recovery"] = state
        return

    if is_running_in_streamlit():
        try:
            if state is None:
                if "pending_recovery" in st.session_state:
                    del st.session_state.pending_recovery
            else:
                st.session_state.pending_recovery = state
        except Exception:
            pass


def clear_confirmation_state() -> None:
    """Clear any pending recovery confirmation state in the current session."""
    set_confirmation_state(None)


def run_recovery_subprocess(run_id: str, extra_args: list = None, timeout: int = 30) -> dict:
    """
    Safely invoke scripts/recover_run.py for a validated run_id.
    Never uses shell=True. Validates script existence and captures stdout/stderr.
    Returns:
        {
            "success": bool,
            "exit_code": int,
            "stdout": str,
            "stderr": str,
            "error": str or None
        }
    """
    if extra_args is None:
        extra_args = []

    # Validate that run_id contains only hex characters (prevents injection/path traversal)
    clean_id = str(run_id).strip().lower()
    if not re.match(r"^[a-f0-9]{4,32}$", clean_id):
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"Invalid run ID format: '{run_id}'",
            "error": "InvalidRunIDFormat",
        }

    script_path = os.path.join(PROJECT_ROOT, "scripts", "recover_run.py")
    if not os.path.exists(script_path):
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Recovery script not found at: {script_path}",
            "error": "ScriptNotFound",
        }

    python_exe = get_python_executable()
    cmd = [python_exe, script_path, clean_id] + list(extra_args)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "error": None,
        }
    except subprocess.TimeoutExpired as te:
        out = (te.stdout or "").strip() if isinstance(te.stdout, str) else ""
        return {
            "success": False,
            "exit_code": -1,
            "stdout": out,
            "stderr": f"Recovery process timed out after {timeout} seconds.",
            "error": "TimeoutExpired",
        }
    except Exception as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "error": type(e).__name__,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching Functions
# ─────────────────────────────────────────────────────────────────────────────
def fetch_all_runs() -> list:
    """Fetch all MLflow runs from the project's tracking store."""
    client = get_mlflow_client()
    exps = client.search_experiments()
    exp_ids = [e.experiment_id for e in exps]
    if not exp_ids:
        return []
    runs = client.search_runs(exp_ids, order_by=["start_time DESC"])
    return runs


def resolve_run(run_id_prefix: str):
    """Find a run by full ID or prefix. Returns the run object or None."""
    client = get_mlflow_client()
    exps = client.search_experiments()
    exp_ids = [e.experiment_id for e in exps]
    runs = client.search_runs(exp_ids)
    matches = [r for r in runs if r.info.run_id.startswith(run_id_prefix)]
    if len(matches) == 1:
        return matches[0]
    return None


def get_run_details(run) -> dict:
    """Extract a structured dict of run information."""
    client = get_mlflow_client()
    tags = run.data.tags
    params = run.data.params
    metrics = run.data.metrics
    experiment = client.get_experiment(run.info.experiment_id)

    start_ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(run.info.start_time / 1000))
    end_ts = (
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(run.info.end_time / 1000))
        if run.info.end_time else "In Progress"
    )

    return {
        "run_id": run.info.run_id,
        "experiment_name": experiment.name,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": start_ts,
        "end_time": end_ts,
        "artifact_uri": run.info.artifact_uri,
        "git": {
            "commit_sha": tags.get("git.commit_sha", "unavailable"),
            "branch": tags.get("git.branch", "unavailable"),
            "is_dirty": tags.get("git.is_dirty", "unavailable"),
        },
        "environment": {
            "python_version": tags.get("env.python_version", "unavailable"),
            "os_platform": tags.get("env.os_platform", "unavailable"),
            "scikit_learn": tags.get("env.scikit_learn", "unavailable"),
            "pandas": tags.get("env.pandas", "unavailable"),
            "numpy": tags.get("env.numpy", "unavailable"),
            "dvc": tags.get("env.dvc", "unavailable"),
            "mlflow": tags.get("env.mlflow", "unavailable"),
        },
        "parameters": params,
        "metrics": metrics,
        "dvc_hashes": {
            tag_key: tags.get(tag_key, "unavailable")
            for tag_key in DVC_TAG_MAP
        },
    }


def get_runs_dataframe(runs=None) -> pd.DataFrame:
    """Convert MLflow runs into a clean, typed pandas DataFrame for dashboard views."""
    if runs is None:
        runs = fetch_all_runs()

    if not runs:
        return pd.DataFrame(columns=[
            "run_id", "short_id", "status", "experiment", "experiment_id",
            "accuracy", "precision", "recall", "f1_score", "duration_seconds",
            "start_time", "start_epoch_ms", "git_commit", "git_commit_full",
            "git_branch", "git_dirty", "model_type", "n_estimators", "max_depth"
        ])

    client = get_mlflow_client()
    try:
        exp_map = {e.experiment_id: e.name for e in client.search_experiments()}
    except Exception:
        exp_map = {}

    records = []
    for r in runs:
        tags = r.data.tags
        metrics = r.data.metrics
        params = r.data.params
        exp_name = exp_map.get(r.info.experiment_id, r.info.experiment_id)

        start_ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(r.info.start_time / 1000))

        # Calculate duration if end_time present
        duration = metrics.get("duration_seconds")
        if duration is None and r.info.end_time and r.info.start_time:
            duration = round((r.info.end_time - r.info.start_time) / 1000.0, 3)

        records.append({
            "run_id": r.info.run_id,
            "short_id": r.info.run_id[:8],
            "status": r.info.status,
            "experiment": exp_name,
            "experiment_id": r.info.experiment_id,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "duration_seconds": duration,
            "start_time": start_ts,
            "start_epoch_ms": r.info.start_time,
            "git_commit": tags.get("git.commit_sha", "unavailable")[:8] if tags.get("git.commit_sha") else "N/A",
            "git_commit_full": tags.get("git.commit_sha", "unavailable"),
            "git_branch": tags.get("git.branch", "N/A"),
            "git_dirty": tags.get("git.is_dirty", "N/A"),
            "model_type": params.get("model_type", "N/A"),
            "n_estimators": params.get("n_estimators", "N/A"),
            "max_depth": params.get("max_depth", "N/A"),
        })

    return pd.DataFrame(records)


def fetch_lineage_report_data(run_id_prefix: str) -> dict | None:
    """Execute lineage_report.py with JSON output and return structured dict."""
    run = resolve_run(run_id_prefix)
    if not run:
        return None

    python_exe = get_python_executable()
    script_path = os.path.join(PROJECT_ROOT, "scripts", "lineage_report.py")
    if not os.path.exists(script_path):
        return None

    try:
        res = subprocess.run(
            [python_exe, script_path, run.info.run_id, "--format", "json"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode in (0, 2):
            raw = res.stdout.strip()
            idx = raw.find("{")
            if idx != -1:
                return json.loads(raw[idx:])
            return json.loads(raw)
    except Exception:
        pass
    return None


def get_run_comparison_matrix(run_id_a: str, run_id_b: str) -> dict:
    """Generate structured side-by-side comparison data for two runs."""
    run_a = resolve_run(run_id_a)
    run_b = resolve_run(run_id_b)
    if not run_a or not run_b:
        return {"error": f"One or both runs ('{run_id_a}', '{run_id_b}') could not be resolved."}

    det_a = get_run_details(run_a)
    det_b = get_run_details(run_b)

    metrics_comparison = []
    all_metrics = sorted(set(det_a["metrics"].keys()) | set(det_b["metrics"].keys()))
    for m in all_metrics:
        va = det_a["metrics"].get(m)
        vb = det_b["metrics"].get(m)
        delta = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
        metrics_comparison.append({
            "metric": m,
            "run_a": va,
            "run_b": vb,
            "delta": delta,
        })

    params_comparison = []
    all_params = sorted(set(det_a["parameters"].keys()) | set(det_b["parameters"].keys()))
    for p in all_params:
        pa = det_a["parameters"].get(p, "N/A")
        pb = det_b["parameters"].get(p, "N/A")
        params_comparison.append({
            "parameter": p,
            "run_a": pa,
            "run_b": pb,
            "changed": pa != pb,
        })

    hashes_comparison = []
    for tag_key, path in DVC_TAG_MAP.items():
        ha = det_a["dvc_hashes"].get(tag_key, "N/A")
        hb = det_b["dvc_hashes"].get(tag_key, "N/A")
        hashes_comparison.append({
            "artifact": path,
            "run_a_hash": ha,
            "run_b_hash": hb,
            "match": ha == hb and ha != "N/A",
        })

    return {
        "run_a": det_a,
        "run_b": det_b,
        "metrics": metrics_comparison,
        "parameters": params_comparison,
        "hashes": hashes_comparison,
    }


def verify_run_artifacts(run) -> list:
    """Verify DVC artifact hashes for a run against the workspace and DVC cache."""
    tags = run.data.tags
    lock_hashes = load_dvc_lock_hashes()
    results = []

    for tag_key, rel_path in DVC_TAG_MAP.items():
        expected_hash = tags.get(tag_key, "")
        abs_path = os.path.join(PROJECT_ROOT, rel_path)

        entry = {
            "artifact": rel_path,
            "expected_hash": expected_hash or "unavailable",
            "workspace_exists": os.path.exists(abs_path),
            "workspace_hash": None,
            "status": "UNVERIFIED",
            "lock_hash": lock_hashes.get(rel_path, "unavailable"),
        }

        if not expected_hash or expected_hash == "unavailable":
            entry["status"] = "NO_HASH_RECORDED"
            results.append(entry)
            continue

        if os.path.exists(abs_path):
            current_hash = compute_file_md5(abs_path)
            entry["workspace_hash"] = current_hash
            if current_hash == expected_hash:
                entry["status"] = "✅ VERIFIED"
            else:
                entry["status"] = "⚠️ HASH_MISMATCH"
        else:
            entry["status"] = "❌ MISSING"

        # Check DVC cache
        cache_file = os.path.join(PROJECT_ROOT, ".dvc", "cache", "files", "md5",
                                  expected_hash[:2], expected_hash[2:])
        if os.path.exists(cache_file):
            if entry["status"] != "✅ VERIFIED":
                entry["status"] += " (recoverable from cache)"

        results.append(entry)

    return results


def get_git_status() -> dict:
    """Get current Git status information."""
    info = {}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        info["commit"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        info["commit"] = "unavailable"

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        info["branch"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        info["branch"] = "unavailable"

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        info["clean"] = not bool(result.stdout.strip()) if result.returncode == 0 else "unknown"
    except Exception:
        info["clean"] = "unknown"

    return info


def get_dvc_status() -> str:
    """Run dvc status and return output."""
    try:
        dvc_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "dvc.exe")
        if not os.path.exists(dvc_exe):
            dvc_exe = "dvc"
        result = subprocess.run(
            [dvc_exe, "status"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30
        )
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Error running dvc status: {e}"


def get_dvc_remote_status() -> str:
    """Run dvc status -c and return output, clearly indicating remote sync status."""
    try:
        dvc_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "dvc.exe")
        if not os.path.exists(dvc_exe):
            dvc_exe = "dvc"
        result = subprocess.run(
            [dvc_exe, "status", "-c"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30
        )
        if result.returncode == 0:
            out = result.stdout.strip()
            return out if out else "Cache and remote storage are in sync."
        return f"DVC remote check returned code {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
    except Exception as e:
        return f"Could not check DVC remote: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Response Generators (Chat Logic)
# ─────────────────────────────────────────────────────────────────────────────
def handle_list_runs() -> str:
    """Generate response for listing MLflow runs."""
    runs = fetch_all_runs()
    if not runs:
        return "📭 No MLflow runs found in the tracking store."

    client = get_mlflow_client()
    exp_names = {
        e.experiment_id: e.name
        for e in client.search_experiments()
    }

    lines = [f"📋 **Found {len(runs)} MLflow run(s):**\n"]
    lines.append("| # | Run ID | Status | Experiment | Accuracy | Timestamp |")
    lines.append("|---|--------|--------|------------|----------|-----------|")

    for i, r in enumerate(runs, 1):
        accuracy = r.data.metrics.get("accuracy", None)
        acc_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
        ts = time.strftime("%Y-%m-%d %H:%M", time.gmtime(r.info.start_time / 1000))
        exp_name = exp_names.get(r.info.experiment_id, r.info.experiment_id)
        status_emoji = "✅" if r.info.status == "FINISHED" else ("❌" if r.info.status == "FAILED" else "⏳")
        short_id = r.info.run_id[:12] + "..."
        lines.append(
            f"| {i} | `{short_id}` | {status_emoji} {r.info.status} | {exp_name} | {acc_str} | {ts} |"
        )

    lines.append(f"\n💡 *Use `inspect <run_id>` to view full details of any run.*")
    return "\n".join(lines)


def handle_inspect_run(run_id_prefix: str) -> str:
    """Generate response for inspecting a specific run."""
    run = resolve_run(run_id_prefix)
    if not run:
        return f"❌ No unique run found matching `{run_id_prefix}`. Use `list runs` to see available runs."

    details = get_run_details(run)

    lines = [
        f"🔍 **Run Inspection: `{details['run_id'][:16]}...`**\n",
        "---",
        "### 📌 Run Information",
        f"- **Run ID:** `{details['run_id']}`",
        f"- **Experiment:** {details['experiment_name']} (ID: {details['experiment_id']})",
        f"- **Status:** {details['status']}",
        f"- **Started:** {details['start_time']}",
        f"- **Ended:** {details['end_time']}",
        "",
        "### 🔗 Git Provenance",
        f"- **Commit SHA:** `{details['git']['commit_sha']}`",
        f"- **Branch:** `{details['git']['branch']}`",
        f"- **Dirty Working Tree:** {details['git']['is_dirty']}",
        "",
        "### 📊 Metrics",
    ]

    for k, v in sorted(details["metrics"].items()):
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"- **{k}:** {val_str}")

    lines.extend([
        "",
        "### ⚙️ Parameters",
    ])
    for k, v in sorted(details["parameters"].items()):
        lines.append(f"- `{k}`: {v}")

    lines.extend([
        "",
        "### 🖥️ Environment",
        f"- **Python:** {details['environment']['python_version']}",
        f"- **OS:** {details['environment']['os_platform']}",
        f"- **scikit-learn:** {details['environment']['scikit_learn']}",
        f"- **pandas:** {details['environment']['pandas']}",
        f"- **numpy:** {details['environment']['numpy']}",
        f"- **DVC:** {details['environment']['dvc']}",
        f"- **MLflow:** {details['environment']['mlflow']}",
        "",
        "### 🗂️ DVC Artifact Hashes",
    ])
    for tag_key, hash_val in details["dvc_hashes"].items():
        artifact_path = DVC_TAG_MAP.get(tag_key, tag_key)
        short = hash_val[:16] + "..." if len(hash_val) > 16 else hash_val
        lines.append(f"- `{artifact_path}`: `{short}`")

    lines.append(f"\n💡 *Use `verify {details['run_id'][:8]}` to check artifact integrity.*")
    return "\n".join(lines)


def handle_verify_run(run_id_prefix: str) -> str:
    """Verify artifact integrity for a run using scripts/recover_run.py --verify-only."""
    run = resolve_run(run_id_prefix)
    if not run:
        return f"❌ No unique run found matching `{run_id_prefix}`. Use `list runs` to see available runs."

    full_id = run.info.run_id
    short_id = full_id[:8]

    proc = run_recovery_subprocess(full_id, ["--verify-only"])

    lines = [
        f"🔐 **Artifact Verification: `{full_id[:16]}...`**",
        f"*Executed via `scripts/recover_run.py {short_id} --verify-only` (Exit Code: {proc['exit_code']})*\n",
    ]

    if proc["stdout"]:
        lines.extend(["```", proc["stdout"], "```"])
    if proc["stderr"]:
        lines.extend(["\n⚠️ **Stderr Output:**", "```", proc["stderr"], "```"])

    lines.append("")
    if proc["exit_code"] == 0:
        lines.append("✅ **All checked artifacts match their recorded hashes.**")
    else:
        lines.append(f"⚠️ **Verification discrepancies detected** (Exit Code {proc['exit_code']}).")
        lines.append(f"💡 *Use `preview recover {short_id}` to preview restoration without modifying files.*")

    return "\n".join(lines)


def handle_dry_run(run_id_prefix: str) -> str:
    """Execute recovery dry-run preview using scripts/recover_run.py --dry-run."""
    run = resolve_run(run_id_prefix)
    if not run:
        return f"❌ No unique run found matching `{run_id_prefix}`. Use `list runs` to see available runs."

    full_id = run.info.run_id
    short_id = full_id[:8]

    proc = run_recovery_subprocess(full_id, ["--dry-run"])

    lines = [
        f"🛡️ **Recovery Dry-Run Preview: `{full_id[:16]}...`**",
        f"*Executed via `scripts/recover_run.py {short_id} --dry-run` (Exit Code: {proc['exit_code']})*\n",
    ]

    if proc["stdout"]:
        lines.extend(["```", proc["stdout"], "```"])
    if proc["stderr"]:
        lines.extend(["\n⚠️ **Stderr Output:**", "```", proc["stderr"], "```"])

    lines.extend([
        "",
        "ℹ️ **Safety Guarantee**: This dry-run was strictly read-only. **No workspace files were modified.**",
        f"💡 *To initiate full restoration with confirmation, type: `recover {short_id}`*",
    ])

    return "\n".join(lines)


def handle_recover_prompt(run_id_prefix: str) -> str:
    """
    Handle recover/restore request by running dry-run first and requesting confirmation.
    Does NOT restore files immediately.
    """
    run = resolve_run(run_id_prefix)
    if not run:
        return f"❌ No unique run found matching `{run_id_prefix}`. Use `list runs` to see available runs."

    full_id = run.info.run_id
    short_id = full_id[:8]

    # Step 1: Execute dry-run to obtain restoration plan
    proc = run_recovery_subprocess(full_id, ["--dry-run"])

    # Step 2: Set pending confirmation bound to this exact run ID
    set_confirmation_state({
        "run_id": full_id,
        "short_id": short_id,
        "timestamp": time.time(),
    })

    lines = [
        f"🛡️ **Recovery Preview & Confirmation Required: `{full_id[:16]}...`**\n",
        "### 📋 Restoration Dry-Run Plan",
    ]

    if proc["stdout"]:
        lines.extend(["```", proc["stdout"], "```"])
    if proc["stderr"]:
        lines.extend(["\n⚠️ **Stderr Output:**", "```", proc["stderr"], "```"])

    lines.extend([
        "",
        "---",
        "### ⚠️ WARNING: Workspace Overwrite Risk",
        "Executing full recovery will copy historical artifacts from DVC cache/remote into your workspace.",
        f"**Any existing files in `data/` and `models/` that differ from run `{short_id}` will be OVERWRITTEN.**",
        "",
        "### 🔒 Explicit Confirmation Required",
        "👉 To proceed with full restoration, enter:",
        f"   **`confirm recover {short_id}`**",
        "   *(or click the **🚨 Confirm Restore** button in the interface)*",
        "",
        "👉 To cancel without restoring, enter: **`cancel`**",
    ])

    return "\n".join(lines)


def handle_confirm_recover(run_id_prefix: str = "") -> str:
    """
    Execute full recovery ONLY after explicit confirmation matching the pending run ID.
    Clears pending confirmation afterwards.
    """
    pending = get_confirmation_state()
    if not pending:
        return (
            "❌ **No recovery operation is currently pending confirmation.**\n\n"
            "To initiate recovery, first preview and request it using:\n"
            "- `recover <run_id>` or `restore <run_id>`"
        )

    pending_full_id = pending.get("run_id", "")
    pending_short_id = pending.get("short_id", pending_full_id[:8])

    # If run_id_prefix is supplied, verify that it matches the pending run
    if run_id_prefix:
        clean_prefix = run_id_prefix.strip().lower()
        if not pending_full_id.startswith(clean_prefix):
            return (
                f"❌ **Confirmation Rejected**: Provided run ID `{run_id_prefix}` does not match "
                f"the pending recovery run `{pending_short_id}`.\n\n"
                f"To confirm the pending run, enter: `confirm recover {pending_short_id}`\n"
                f"To abort, enter: `cancel`"
            )

    # Clear pending state immediately before execution so it cannot be re-triggered
    clear_confirmation_state()

    # Step 3: Execute full recovery
    proc = run_recovery_subprocess(pending_full_id, [])

    lines = [
        f"🔄 **Executing Historical Recovery: `{pending_full_id[:16]}...`**",
        f"*Executed via `scripts/recover_run.py {pending_short_id}` (Exit Code: {proc['exit_code']})*\n",
    ]

    if proc["stdout"]:
        lines.extend(["```", proc["stdout"], "```"])
    if proc["stderr"]:
        lines.extend(["\n⚠️ **Stderr Output:**", "```", proc["stderr"], "```"])

    lines.append("")
    if proc["exit_code"] == 0:
        lines.extend([
            "✅ **Recovery Completed Successfully.**",
            f"All tracked artifacts for run `{pending_short_id}` have been restored and verified in your workspace.",
        ])
    else:
        lines.extend([
            f"❌ **Recovery Failed** (Exit Code {proc['exit_code']}).",
            "Review the output above for missing artifacts or cache discrepancies.",
        ])

    return "\n".join(lines)


def handle_cancel_recovery() -> str:
    """Cancel any pending recovery operation without file modifications."""
    pending = get_confirmation_state()
    if pending:
        short_id = pending.get("short_id", "unknown")
        clear_confirmation_state()
        return f"🛑 **Recovery Cancelled**: Pending recovery for run `{short_id}` has been cleared. **No files were modified.**"
    return "ℹ️ No recovery operation is currently pending."


def handle_lineage_report(run_id_prefix: str) -> str:
    """Run the lineage report script and return its output."""
    run = resolve_run(run_id_prefix)
    if not run:
        return f"❌ No unique run found matching `{run_id_prefix}`. Use `list runs` to see available runs."

    python_exe = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    script_path = os.path.join(PROJECT_ROOT, "scripts", "lineage_report.py")

    try:
        result = subprocess.run(
            [python_exe, script_path, run.info.run_id, "--format", "json"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30,
            encoding="utf-8", errors="replace"
        )
        if result.returncode not in (0, 2):
            return f"❌ Lineage report failed:\n```\n{result.stderr.strip()}\n```"

        raw_out = result.stdout.strip()
        brace = raw_out.find("{")
        report = json.loads(raw_out[brace:] if brace != -1 else raw_out)

        lines = [
            f"📜 **Lineage Report: `{report['run']['run_id'][:16]}...`**\n",
            f"*Generated at: {report['report_generated_at']}*\n",
            "---",
            "### 🔗 Git Provenance",
            f"- **Commit:** `{report['git_provenance']['commit_sha']}`",
            f"- **Branch:** `{report['git_provenance']['branch']}`",
            f"- **Dirty:** {report['git_provenance']['is_dirty']}",
            "",
            "### 📊 Metrics",
        ]

        for k, v in sorted(report["metrics"].items()):
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"- **{k}:** {val_str}")

        lines.extend(["", "### 🔄 Pipeline Lineage Trace"])

        for stage in report["lineage_stages"]:
            lines.append(f"\n**{stage['stage'].upper()}** — {stage['description']}")
            if stage["inputs"]:
                lines.append(f"  - Inputs: {', '.join(stage['inputs'])}")
            for out in stage["outputs"]:
                status = out.get("status", "UNKNOWN")
                emoji = "✅" if "VERIFIED" in status else ("⚠️" if "MISMATCH" in status or "HISTORICAL" in status else "❓")
                lines.append(f"  - {emoji} `{out['path']}` — {out['role']} [{status}]")

        v = report["verification_summary"]
        lines.extend([
            "",
            "### ✅ Verification Summary",
            f"- **Tracked artifacts:** {v['versioned_artifacts_tracked']}",
            f"- **Verified in workspace:** {v['verified_in_workspace']}",
            f"- **Hash mismatches:** {v['workspace_hash_mismatches']}",
            f"- **Missing:** {v['workspace_missing']}",
            f"- **Recoverable from DVC:** {v['recoverable_from_dvc']}",
            "",
        ])

        if v["is_lineage_verified"]:
            lines.append("✅ **LINEAGE FULLY VERIFIED** — All pipeline artifacts are accountable.")
        else:
            lines.append("⚠️ **LINEAGE VERIFICATION INCOMPLETE** — Some artifacts have discrepancies.")

        return "\n".join(lines)

    except json.JSONDecodeError:
        return f"❌ Failed to parse lineage report output. Raw output:\n```\n{result.stdout[:500]}\n```"
    except Exception as e:
        return f"❌ Error running lineage report: {e}"


def handle_compare_runs(run_id_a: str, run_id_b: str) -> str:
    """Compare two MLflow runs side-by-side."""
    run_a = resolve_run(run_id_a)
    run_b = resolve_run(run_id_b)

    if not run_a:
        return f"❌ Could not find run matching `{run_id_a}`."
    if not run_b:
        return f"❌ Could not find run matching `{run_id_b}`."

    details_a = get_run_details(run_a)
    details_b = get_run_details(run_b)

    lines = [
        f"⚖️ **Run Comparison**\n",
        f"| Attribute | Run A (`{details_a['run_id'][:10]}...`) | Run B (`{details_b['run_id'][:10]}...`) |",
        "|-----------|--------|--------|",
        f"| Status | {details_a['status']} | {details_b['status']} |",
        f"| Started | {details_a['start_time']} | {details_b['start_time']} |",
        f"| Git Commit | `{details_a['git']['commit_sha'][:10]}...` | `{details_b['git']['commit_sha'][:10]}...` |",
        f"| Branch | {details_a['git']['branch']} | {details_b['git']['branch']} |",
    ]

    # Compare metrics
    all_metric_keys = set(details_a["metrics"].keys()) | set(details_b["metrics"].keys())
    for k in sorted(all_metric_keys):
        val_a = details_a["metrics"].get(k, "N/A")
        val_b = details_b["metrics"].get(k, "N/A")
        str_a = f"{val_a:.4f}" if isinstance(val_a, float) else str(val_a)
        str_b = f"{val_b:.4f}" if isinstance(val_b, float) else str(val_b)
        diff_marker = ""
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            diff = val_b - val_a
            if diff > 0:
                diff_marker = f" 📈 +{diff:.4f}"
            elif diff < 0:
                diff_marker = f" 📉 {diff:.4f}"
        lines.append(f"| {k} | {str_a} | {str_b}{diff_marker} |")

    # Compare key parameters
    key_params = ["n_estimators", "max_depth", "random_state", "test_size"]
    for k in key_params:
        val_a = details_a["parameters"].get(k, "N/A")
        val_b = details_b["parameters"].get(k, "N/A")
        changed = " ⚡" if val_a != val_b else ""
        lines.append(f"| {k} | {val_a} | {val_b}{changed} |")

    return "\n".join(lines)


def handle_status() -> str:
    """Generate response for system status, distinguishing workspace and remote."""
    git_info = get_git_status()
    dvc_ws = get_dvc_status()
    dvc_rem = get_dvc_remote_status()

    runs = fetch_all_runs()
    finished = sum(1 for r in runs if r.info.status == "FINISHED")

    lines = [
        "🏠 **System Status**\n",
        "---",
        "### 📂 Git",
        f"- **Branch:** `{git_info['branch']}`",
        f"- **Commit:** `{git_info['commit'][:12]}...`" if len(git_info.get('commit', '')) > 12 else f"- **Commit:** `{git_info.get('commit', 'unavailable')}`",
        f"- **Clean Working Tree:** {'✅ Yes' if git_info['clean'] is True else '❌ No' if git_info['clean'] is False else '❓ Unknown'}",
        "",
        "### 📦 DVC Workspace Pipeline (`dvc status`)",
        f"```\n{dvc_ws if dvc_ws else 'Pipeline is up to date.'}\n```",
        "",
        "### ☁️ DVC Remote Storage Sync (`dvc status -c` - Local Storage)",
        f"```\n{dvc_rem}\n```",
        "",
        "### 🧪 MLflow Experiments",
        f"- **Total runs:** {len(runs)}",
        f"- **Finished:** {finished}",
    ]

    if runs:
        latest = runs[0]
        ts = time.strftime("%Y-%m-%d %H:%M", time.gmtime(latest.info.start_time / 1000))
        acc = latest.data.metrics.get("accuracy")
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        lines.extend([
            f"- **Latest run:** `{latest.info.run_id[:12]}...` ({ts})",
            f"- **Latest accuracy:** {acc_str}",
        ])

    return "\n".join(lines)


def handle_help() -> str:
    """Generate comprehensive command reference explaining file safety."""
    return """🤖 **ML Lineage Chatbot — Available Commands**

---

### 📋 Experiment Tracking *(Read-Only)*
| Command | Description | Safety |
|---------|-------------|--------|
| `list runs` | List all recorded MLflow experiment runs | 🟢 Read-Only |
| `inspect <run_id>` | View full parameters, metrics, Git tags, & hashes | 🟢 Read-Only |
| `compare <id_a> <id_b>` | Compare two runs side-by-side | 🟢 Read-Only |

### 🔍 Verification & Lineage *(Read-Only)*
| Command | Description | Safety |
|---------|-------------|--------|
| `verify <run_id>` | Verify workspace artifact hashes using `scripts/recover_run.py --verify-only` | 🟢 Read-Only |
| `lineage <run_id>` | Generate full pipeline lineage report using `scripts/lineage_report.py` | 🟢 Read-Only |

### 🔄 Recovery & Restoration *(Protected by Dry-Run & Confirmation)*
| Command | Description | Safety |
|---------|-------------|--------|
| `preview recover <run_id>`<br>`dry-run <run_id>` | Show restoration plan and test artifact sources without modifying files | 🟢 Read-Only |
| `recover <run_id>`<br>`restore <run_id>` | Execute dry-run preview and request explicit confirmation to restore | 🟢 Read-Only (Until Confirmed) |
| `confirm recover <run_id>`<br>`confirm restore <run_id>` | **Perform full artifact recovery**: restores files from DVC cache/remote into workspace | ⚠️ **Overwrites Workspace Files** |
| `cancel` | Cancel any pending recovery operation | 🟢 Read-Only |

### 🏠 System *(Read-Only)*
| Command | Description | Safety |
|---------|-------------|--------|
| `status` | Show Git, DVC workspace & remote, and MLflow status | 🟢 Read-Only |
| `help` | Show this command guide | 🟢 Read-Only |

---

⚠️ **File Safety Notice:**
All inspection, verification, status, and dry-run commands are strictly read-only.
Only `confirm recover <run_id>` will overwrite workspace files (`data/*`, `models/*`).
Full recovery is protected by mandatory dry-run preview and explicit confirmation.

💡 *Partial run IDs (e.g. `verify b6b15235`) are fully supported!*
"""


# ─────────────────────────────────────────────────────────────────────────────
# Intent Routing (Strict Command Matching Before Fallbacks)
# ─────────────────────────────────────────────────────────────────────────────
def extract_run_ids(text: str) -> list:
    """Extract hex strings that look like MLflow run IDs (8+ hex chars)."""
    return re.findall(r'\b([a-f0-9]{8,32})\b', text.lower())


def route_message(user_input: str) -> str:
    """Route user input to the appropriate handler with strict command matching."""
    text = user_input.strip().lower()

    # 1. Help
    if text in ("help", "/help", "?", "commands"):
        clear_confirmation_state()
        return handle_help()

    # 2. Cancel
    if text in ("cancel", "cancel recovery", "cancel restore", "abort"):
        return handle_cancel_recovery()

    # 3. Status
    if text in ("status", "system status", "system", "overview"):
        clear_confirmation_state()
        return handle_status()

    # 4. List runs
    if any(kw in text for kw in ["list runs", "list", "show runs", "all runs", "runs", "experiments"]):
        clear_confirmation_state()
        return handle_list_runs()

    # 5. Compare runs
    compare_match = re.match(r'^compare\s+([a-f0-9]+)\s+(?:and\s+|vs\s+|with\s+)?([a-f0-9]+)$', text)
    if compare_match:
        clear_confirmation_state()
        return handle_compare_runs(compare_match.group(1), compare_match.group(2))
    if text.startswith("compare"):
        clear_confirmation_state()
        parts = re.findall(r'[a-f0-9]{4,32}', text)
        if len(parts) >= 2:
            return handle_compare_runs(parts[0], parts[1])
        return "❓ Please provide two run IDs to compare. Example: `compare b6b15235 4645ac21`"

    # 6. Confirm recover / restore (MUST precede general recover/restore)
    confirm_match = re.match(r'^(?:confirm\s+recover|confirm\s+restore|confirm\s+recovery|confirm)(?:\s+([a-f0-9]+))?$', text)
    if confirm_match:
        req_id = confirm_match.group(1) or ""
        return handle_confirm_recover(req_id)

    # 7. Preview recover / dry-run (MUST precede general recover/restore)
    preview_match = re.match(r'^(?:preview\s+recover|preview\s+restore|preview|dry-run|dryrun)(?:\s+([a-f0-9]+))?$', text)
    if preview_match:
        clear_confirmation_state()
        req_id = preview_match.group(1)
        if req_id:
            return handle_dry_run(req_id)
        return "❓ Please provide a run ID to preview. Example: `preview recover b6b15235`"

    # 8. Recover / Restore (Dry-run preview + confirmation prompt)
    recover_match = re.match(r'^(?:recover|restore)(?:\s+([a-f0-9]+))?$', text)
    if recover_match:
        req_id = recover_match.group(1)
        if req_id:
            return handle_recover_prompt(req_id)
        return "❓ Please provide a run ID to recover. Example: `recover b6b15235`"

    # 9. Lineage report
    lineage_match = re.match(r'^(?:lineage|provenance|trace)(?:\s+([a-f0-9]+))?$', text)
    if lineage_match:
        clear_confirmation_state()
        req_id = lineage_match.group(1)
        if req_id:
            return handle_lineage_report(req_id)
        return "❓ Please provide a run ID. Example: `lineage b6b15235`"

    # 10. Verify run artifacts
    verify_match = re.match(r'^(?:verify|check|integrity|verify\s+run)(?:\s+([a-f0-9]+))?$', text)
    if verify_match:
        clear_confirmation_state()
        req_id = verify_match.group(1)
        if req_id:
            return handle_verify_run(req_id)
        return "❓ Please provide a run ID to verify. Example: `verify b6b15235`"

    # 11. Inspect run
    inspect_match = re.match(r'^(?:inspect|details|show\s+run|info|describe|metadata)(?:\s+([a-f0-9]+))?$', text)
    if inspect_match:
        clear_confirmation_state()
        req_id = inspect_match.group(1)
        if req_id:
            return handle_inspect_run(req_id)
        return "❓ Please provide a run ID to inspect. Example: `inspect b6b15235`"

    # 12. Strict bare run ID (user typed only hex run ID or 'run <id>')
    bare_match = re.fullmatch(r'(?:run\s+)?([a-f0-9]{8,32})', text)
    if bare_match:
        clear_confirmation_state()
        return handle_inspect_run(bare_match.group(1))

    # Clear pending recovery if user typed unrecognized text
    clear_confirmation_state()

    # Fallback for unrecognized commands
    return (
        "🤔 I didn't recognize that command. Here are supported commands you can use:\n\n"
        "- `list runs` — View all experiment runs\n"
        "- `inspect <run_id>` — View run parameters, metrics, & Git tags\n"
        "- `verify <run_id>` — Verify artifact hashes via recovery utility\n"
        "- `preview recover <run_id>` — Dry-run preview of restoration\n"
        "- `recover <run_id>` — Preview and request confirmation to restore\n"
        "- `lineage <run_id>` — Pipeline DAG lineage report\n"
        "- `compare <id_a> <id_b>` — Compare two runs side-by-side\n"
        "- `status` — Git, DVC workspace/remote, & MLflow overview\n"
        "- `help` — Complete command reference\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="ML Lineage Studio",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Compact professional CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    /* Tighten metric cards */
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-weight: 600;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        opacity: 0.7;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
    }

    /* Tighter page top padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { font-size: 0.88rem; }

    /* Buttons */
    .stButton > button {
        border-radius: 5px;
        font-weight: 500;
        font-size: 0.83rem;
        padding: 0.3rem 0.75rem;
    }

    /* Dataframe text */
    .stDataFrame { font-size: 0.82rem; }
    </style>
    """, unsafe_allow_html=True)

    # ── Navigation Items ─────────────────────────────────────────────────────
    pages = [
        "Overview",
        "Experiments",
        "Compare",
        "Lineage",
        "Artifacts",
        "Recovery & Workspaces",
        "Chat Assistant",
        "System Status",
    ]

    # Initialize current page state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Overview"

    nav_index = 0
    for i, p in enumerate(pages):
        if p == st.session_state.current_page:
            nav_index = i
            break

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("**ML Lineage Studio**")
        st.caption("Experiment tracking · Lineage · Recovery")
        st.divider()

        selected_nav = st.radio(
            "Navigation",
            pages,
            index=nav_index,
            label_visibility="collapsed",
            key="main_nav_radio",
        )
        if selected_nav != st.session_state.current_page:
            st.session_state.current_page = selected_nav
            st.rerun()

        st.divider()
        git_info = get_git_status()
        st.caption(f"Git: `{git_info.get('branch', '—')}` · `{git_info.get('commit', '—')[:7]}`")
        st.caption(f"MLflow: `mlflow.db`")
        st.caption(f"DVC: `local_storage`")

    # ── View Routing ─────────────────────────────────────────────────────────
    app_module = sys.modules[__name__]
    page = st.session_state.current_page

    if page == "Overview":
        dashboard_views.render_overview(app_module)
    elif page == "Experiments":
        dashboard_views.render_experiments(app_module)
    elif page == "Compare":
        dashboard_views.render_compare(app_module)
    elif page == "Lineage":
        dashboard_views.render_lineage(app_module)
    elif page == "Artifacts":
        dashboard_views.render_artifacts(app_module)
    elif page == "Recovery & Workspaces":
        dashboard_views.render_recovery(app_module)
    elif page == "Chat Assistant":
        dashboard_views.render_chat(app_module)
    elif page == "System Status":
        dashboard_views.render_system_status(app_module)
    else:
        dashboard_views.render_overview(app_module)


if __name__ == "__main__":
    main()
