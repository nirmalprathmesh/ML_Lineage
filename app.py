#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py
======
Stage 6: Streamlit Chatbot Interface for ML Lineage Tracking System.

A conversational web UI that allows users to interact with the existing
ML pipeline versioning and lineage tracking system through natural language
commands in a chat-based interface.

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

import streamlit as st
import yaml

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
        result = subprocess.run(
            [dvc_exe, "status"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30
        )
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Error running dvc status: {e}"


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
    """Generate response for verifying a run's artifact integrity."""
    run = resolve_run(run_id_prefix)
    if not run:
        return f"❌ No unique run found matching `{run_id_prefix}`. Use `list runs` to see available runs."

    results = verify_run_artifacts(run)

    lines = [
        f"🔐 **Artifact Verification: `{run.info.run_id[:16]}...`**\n",
        "| Artifact | Expected Hash | Status |",
        "|----------|---------------|--------|",
    ]

    verified = 0
    total_tracked = 0
    for r in results:
        if r["expected_hash"] != "unavailable":
            total_tracked += 1
        short_hash = r["expected_hash"][:12] + "..." if len(r["expected_hash"]) > 12 else r["expected_hash"]
        lines.append(f"| `{r['artifact']}` | `{short_hash}` | {r['status']} |")
        if "VERIFIED" in r["status"]:
            verified += 1

    lines.extend([
        "",
        f"**Summary:** {verified}/{total_tracked} artifacts verified in workspace.",
    ])

    if verified == total_tracked and total_tracked > 0:
        lines.append("✅ **All tracked artifacts match their recorded hashes.**")
    elif total_tracked == 0:
        lines.append("⚠️ No artifact hashes were recorded for this run.")
    else:
        lines.append("⚠️ Some artifacts have mismatches or are missing. Use `recover <run_id>` to restore.")

    return "\n".join(lines)


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

        report = json.loads(result.stdout)

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
    """Generate response for system status."""
    git_info = get_git_status()
    dvc_out = get_dvc_status()

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
        "### 📦 DVC Pipeline",
        f"```\n{dvc_out if dvc_out else 'Pipeline is up to date.'}\n```",
        "",
        "### 🧪 MLflow Experiments",
        f"- **Total runs:** {len(runs)}",
        f"- **Finished:** {finished}",
    ]

    if runs:
        latest = runs[0]
        ts = time.strftime("%Y-%m-%d %H:%M", time.gmtime(latest.info.start_time / 1000))
        acc = latest.data.metrics.get("accuracy")
        acc_str = f"{acc:.4f}" if acc else "N/A"
        lines.extend([
            f"- **Latest run:** `{latest.info.run_id[:12]}...` ({ts})",
            f"- **Latest accuracy:** {acc_str}",
        ])

    return "\n".join(lines)


def handle_help() -> str:
    """Generate the help message."""
    return """🤖 **ML Lineage Chatbot — Available Commands**

---

### 📋 Experiment Tracking
| Command | Description |
|---------|-------------|
| `list runs` | List all recorded MLflow experiment runs |
| `inspect <run_id>` | View full details of a specific run |
| `compare <run_id_a> <run_id_b>` | Compare two runs side-by-side |

### 🔐 Verification & Lineage
| Command | Description |
|---------|-------------|
| `verify <run_id>` | Verify artifact hashes for a run |
| `lineage <run_id>` | Generate full lineage provenance report |

### 🏠 System
| Command | Description |
|---------|-------------|
| `status` | Show Git, DVC, and MLflow system status |
| `help` | Show this help message |

---

💡 *You can use partial run IDs (prefix match). For example: `inspect a0c974c4` instead of the full 32-character ID.*

💡 *You can also ask questions in natural language, like "show me the runs" or "what's the lineage for run a0c974c4?"*
"""


# ─────────────────────────────────────────────────────────────────────────────
# Intent Routing (Simple Pattern Matching)
# ─────────────────────────────────────────────────────────────────────────────
def extract_run_ids(text: str) -> list:
    """Extract hex strings that look like MLflow run IDs (8+ hex chars)."""
    return re.findall(r'\b([a-f0-9]{8,32})\b', text.lower())


def route_message(user_input: str) -> str:
    """Route user input to the appropriate handler."""
    text = user_input.strip().lower()

    # Help
    if text in ("help", "/help", "?", "commands"):
        return handle_help()

    # Status
    if text in ("status", "system status", "system", "overview"):
        return handle_status()

    # List runs
    if any(kw in text for kw in ["list runs", "list", "show runs", "all runs", "runs", "experiments"]):
        return handle_list_runs()

    # Compare runs
    compare_match = re.match(r'compare\s+([a-f0-9]+)\s+(?:and\s+|vs\s+|with\s+)?([a-f0-9]+)', text)
    if compare_match:
        return handle_compare_runs(compare_match.group(1), compare_match.group(2))

    # Lineage report
    if any(kw in text for kw in ["lineage", "provenance", "trace"]):
        run_ids = extract_run_ids(text)
        if run_ids:
            return handle_lineage_report(run_ids[0])
        return "❓ Please provide a run ID. Example: `lineage a0c974c4`"

    # Verify
    if any(kw in text for kw in ["verify", "check", "integrity", "hash"]):
        run_ids = extract_run_ids(text)
        if run_ids:
            return handle_verify_run(run_ids[0])
        return "❓ Please provide a run ID. Example: `verify a0c974c4`"

    # Inspect
    if any(kw in text for kw in ["inspect", "details", "show run", "info", "describe", "metadata"]):
        run_ids = extract_run_ids(text)
        if run_ids:
            return handle_inspect_run(run_ids[0])
        return "❓ Please provide a run ID. Example: `inspect a0c974c4`"

    # Bare run ID — default to inspect
    run_ids = extract_run_ids(text)
    if run_ids and len(text.replace(run_ids[0], "").strip()) < 10:
        return handle_inspect_run(run_ids[0])

    # Fallback
    return (
        "🤔 I didn't quite understand that. Here are some things I can help with:\n\n"
        "- `list runs` — Show all experiment runs\n"
        "- `inspect <run_id>` — View run details\n"
        "- `verify <run_id>` — Check artifact integrity\n"
        "- `lineage <run_id>` — Generate lineage report\n"
        "- `compare <id_a> <id_b>` — Compare two runs\n"
        "- `status` — System overview\n"
        "- `help` — Full command reference\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="ML Lineage Chatbot",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for premium chatbot styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
        font-weight: 300;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: linear-gradient(180deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.08) 100%);
        border: 1px solid rgba(102,126,234,0.15);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-section h4 {
        color: #667eea;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    /* Quick action buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid rgba(102,126,234,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.2);
    }

    /* Chat input styling */
    .stChatInput > div {
        border-radius: 12px !important;
    }

    /* Footer */
    .footer-info {
        text-align: center;
        opacity: 0.5;
        font-size: 0.75rem;
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🧬 ML Lineage Chatbot</h1>
        <p>Interactive assistant for ML pipeline versioning and lineage tracking</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧬 ML Lineage")
        st.markdown("---")

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("#### ⚡ Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 List Runs", use_container_width=True, key="qk_list"):
                st.session_state.pending_action = "list runs"
            if st.button("🏠 Status", use_container_width=True, key="qk_status"):
                st.session_state.pending_action = "status"
        with col2:
            if st.button("❓ Help", use_container_width=True, key="qk_help"):
                st.session_state.pending_action = "help"
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("#### 📊 Quick Stats")
        try:
            runs = fetch_all_runs()
            st.metric("Total Runs", len(runs))
            finished = sum(1 for r in runs if r.info.status == "FINISHED")
            st.metric("Finished Runs", finished)
            if runs:
                latest_acc = runs[0].data.metrics.get("accuracy")
                if latest_acc:
                    st.metric("Latest Accuracy", f"{latest_acc:.4f}")
        except Exception:
            st.caption("Could not load stats.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("#### 🏗️ System Stages")
        stages = [
            ("✅", "Modular ML Pipeline"),
            ("✅", "DVC Versioning"),
            ("✅", "MLflow Tracking"),
            ("✅", "Historical Run Recovery"),
            ("✅", "Lineage Report Generator"),
            ("✅", "Chatbot Interface"),
        ]
        for icon, name in stages:
            st.markdown(f"{icon} {name}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Data Versioning & Lineage Tracking System for ML Pipelines")

    # ── Chat History ────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Welcome to the **ML Lineage Chatbot**! I can help you explore "
                    "your ML experiment history, verify artifact integrity, and trace "
                    "data lineage.\n\n"
                    "Here are some things you can try:\n"
                    "- `list runs` — View all experiment runs\n"
                    "- `inspect <run_id>` — Detailed run information\n"
                    "- `verify <run_id>` — Check artifact hashes\n"
                    "- `lineage <run_id>` — Full provenance report\n"
                    "- `status` — System overview\n"
                    "- `help` — All commands\n\n"
                    "💡 *Type a command or ask a question to get started!*"
                ),
            }
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🧬"):
            st.markdown(msg["content"])

    # Handle pending sidebar actions
    if "pending_action" in st.session_state:
        action = st.session_state.pending_action
        del st.session_state.pending_action

        st.session_state.messages.append({"role": "user", "content": action})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(action)

        with st.chat_message("assistant", avatar="🧬"):
            with st.spinner("Processing..."):
                response = route_message(action)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Chat input
    if user_input := st.chat_input("Ask about your ML experiments..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🧬"):
            with st.spinner("Thinking..."):
                response = route_message(user_input)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
