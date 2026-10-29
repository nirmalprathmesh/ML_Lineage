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

# Attempt to import mlflow; if unavailable, set placeholders
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError:
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore

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


from ml_lineage.core.context import ProjectContext
from ml_lineage.core.hashing import compute_file_md5 as _core_compute_file_md5
from ml_lineage.core.verification import verify_run_artifacts as _core_verify_run_artifacts
from ml_lineage.adapters.git import GitAdapter
from ml_lineage.adapters.dvc import DVCAdapter
from ml_lineage.adapters.mlflow import MLflowAdapter

DEFAULT_CONTEXT = ProjectContext.from_path(PROJECT_ROOT)
_git_adapter = GitAdapter(DEFAULT_CONTEXT)
_dvc_adapter = DVCAdapter(DEFAULT_CONTEXT)
_mlflow_adapter = MLflowAdapter(DEFAULT_CONTEXT)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow / Config Helpers (Delegated to ml_lineage)
# ─────────────────────────────────────────────────────────────────────────────
def get_tracking_uri() -> str:
    """Retrieve absolute tracking URI from params.yaml or fallback."""
    return DEFAULT_CONTEXT.tracking_uri


def get_mlflow_client() -> MlflowClient:
    """Create an MLflow client configured with the project's tracking URI."""
    return _mlflow_adapter.get_client()


def compute_file_md5(filepath: str) -> str:
    """Calculate MD5 hash of a local file."""
    return _core_compute_file_md5(filepath)


def load_dvc_lock_hashes() -> dict:
    """Parse dvc.lock to extract current stage output MD5 hashes."""
    return _dvc_adapter.load_lock_hashes()


def get_python_executable() -> str:
    """Return path to verified project venv Python or sys.executable."""
    return DEFAULT_CONTEXT.python_executable


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
    return _mlflow_adapter.fetch_all_runs()


def resolve_run(run_id_prefix: str):
    """Find a run by full ID or prefix. Returns the run object or None."""
    return _mlflow_adapter.resolve_run(run_id_prefix)


def get_run_details(run) -> dict:
    """Extract a structured dict of run information."""
    return _mlflow_adapter.get_run_details(run, tag_map=DVC_TAG_MAP)


def get_runs_dataframe(runs=None) -> pd.DataFrame:
    """Convert MLflow runs into a clean, typed pandas DataFrame for dashboard views."""
    return _mlflow_adapter.get_runs_dataframe(runs=runs)


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
    return _mlflow_adapter.get_run_comparison_matrix(run_id_a, run_id_b, tag_map=DVC_TAG_MAP)


def verify_run_artifacts(run) -> list:
    """Verify DVC artifact hashes for a run against the workspace and DVC cache."""
    return _core_verify_run_artifacts(DEFAULT_CONTEXT, run, tag_map=DVC_TAG_MAP, dvc_adapter=_dvc_adapter)


def get_git_status() -> dict:
    """Get current Git status information."""
    return _git_adapter.get_status()


def get_dvc_status() -> str:
    """Run dvc status and return output."""
    return _dvc_adapter.get_status()


def get_dvc_remote_status() -> str:
    """Run dvc status -c and return output, clearly indicating remote sync status."""
    return _dvc_adapter.get_remote_status()


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
        page_icon="◇",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject the design system CSS
    dashboard_views.inject_design_system()

    # Initialize current page state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Overview"

    # ── Sidebar Navigation ──────────────────────────────────────────────────
    app_module = sys.modules[__name__]
    dashboard_views.render_sidebar(app_module)

    # ── View Routing ─────────────────────────────────────────────────────────
    page = st.session_state.current_page

    if page == "Overview":
        dashboard_views.render_overview(app_module)
    elif page == "Experiments":
        dashboard_views.render_experiments(app_module)
    elif page == "Compare Runs":
        dashboard_views.render_compare(app_module)
    elif page == "Lineage Explorer":
        dashboard_views.render_lineage(app_module)
    elif page == "Artifacts":
        dashboard_views.render_artifacts(app_module)
    elif page == "Recovery Center":
        dashboard_views.render_recovery(app_module)
    elif page == "Chat Assistant":
        dashboard_views.render_chat(app_module)
    elif page == "System Health":
        dashboard_views.render_system_status(app_module)
    else:
        dashboard_views.render_overview(app_module)


if __name__ == "__main__":
    main()
