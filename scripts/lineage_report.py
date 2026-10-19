#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/lineage_report.py
=========================
Stage 5: Lineage Report Generator for ML Lineage Tracking System.

Inspects an MLflow experiment run and generates a comprehensive,
end-to-end data provenance and model lineage report tracing:
    Raw dataset -> Prepared data -> Features & Scaler -> Trained Model -> Evaluation Metrics

Supports:
- Human-readable terminal output and structured JSON export
- Verification of artifact content hashes against workspace, DVC cache, and remote
- Comparison against current dvc.lock (detects historical vs current versions)
- File output via --output <path>
- Listing and filtering available runs via --list [--experiment <NAME_OR_ID>]

Usage:
    python scripts/lineage_report.py <run_id>
    python scripts/lineage_report.py <run_id> --format json
    python scripts/lineage_report.py <run_id> --output reports/lineage_report.json --format json
    python scripts/lineage_report.py --list
    python scripts/lineage_report.py --list --experiment ML_Lineage_Experiments
"""

import argparse
import configparser
import hashlib
import json
import os
import sys
import time
import yaml

# Suppress MLflow telemetry & warnings
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from mlflow.tracking import MlflowClient

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Path Mapping
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")
DVC_LOCK_PATH = os.path.join(PROJECT_ROOT, "dvc.lock")

# Core pipeline lineage stages and artifacts
PIPELINE_STAGES = [
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

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions: Tracking URI, Hashing, DVC Storage
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


def compute_file_md5(filepath: str) -> str:
    """Calculate the MD5 hash of a local file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dvc_lock_hashes() -> dict[str, str]:
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
    except Exception as e:
        print(f"[warning] Failed to parse dvc.lock: {e}", file=sys.stderr)

    return hashes


def get_dvc_remote_base() -> str | None:
    """Read local DVC remote storage directory from .dvc/config."""
    cfg_path = os.path.join(PROJECT_ROOT, ".dvc", "config")
    if not os.path.exists(cfg_path):
        return None
    try:
        cp = configparser.ConfigParser()
        cp.read(cfg_path)
        for section in cp.sections():
            if section.startswith("'remote \"") or section.startswith('remote "'):
                url = cp[section].get("url", "")
                if url:
                    return url
    except Exception:
        pass
    return None


def check_artifact_provenance(
    rel_path: str,
    expected_hash: str | None,
    current_lock_hashes: dict[str, str],
    is_metrics_file: bool = False
) -> dict:
    """
    Examine the state of an artifact in the workspace, DVC cache, DVC remote,
    and compare against current dvc.lock.
    """
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    result = {
        "path": rel_path,
        "expected_hash": expected_hash or "unavailable",
        "lock_hash": current_lock_hashes.get(rel_path, "unavailable"),
        "workspace_exists": os.path.exists(abs_path),
        "workspace_hash": None,
        "workspace_status": "UNAVAILABLE",
        "lock_status": "UNKNOWN",
        "dvc_cache_available": False,
        "dvc_remote_available": False,
        "status": "UNVERIFIED"
    }

    # 1. Compare with current dvc.lock
    if is_metrics_file or not expected_hash or expected_hash == "unavailable":
        if rel_path in current_lock_hashes:
            result["lock_status"] = "RECORDED_IN_LOCK"
            result["lock_hash"] = current_lock_hashes[rel_path]
        else:
            result["lock_status"] = "UNTRACKED_IN_LOCK"
    elif rel_path in current_lock_hashes:
        if current_lock_hashes[rel_path] == expected_hash:
            result["lock_status"] = "MATCHES_CURRENT_LOCK"
        else:
            result["lock_status"] = "HISTORICAL_VERSION_DIFFERS_FROM_LOCK"
    else:
        result["lock_status"] = "NOT_IN_CURRENT_LOCK"

    # 2. Check local workspace
    if result["workspace_exists"]:
        try:
            curr_hash = compute_file_md5(abs_path)
            result["workspace_hash"] = curr_hash
            if is_metrics_file:
                result["workspace_status"] = "PRESENT"
            elif expected_hash and expected_hash != "unavailable":
                if curr_hash == expected_hash:
                    result["workspace_status"] = "VERIFIED"
                else:
                    result["workspace_status"] = "HASH_MISMATCH"
            else:
                result["workspace_status"] = "PRESENT_UNTRACKED"
        except Exception as e:
            result["workspace_status"] = f"READ_ERROR ({e})"
    else:
        result["workspace_status"] = "MISSING"

    # 3. Check DVC cache
    if expected_hash and expected_hash != "unavailable":
        cache_file = os.path.join(PROJECT_ROOT, ".dvc", "cache", "files", "md5", expected_hash[:2], expected_hash[2:])
        if os.path.exists(cache_file):
            result["dvc_cache_available"] = True

        # 4. Check DVC remote
        remote_base = get_dvc_remote_base()
        if remote_base:
            remote_file = os.path.join(remote_base, "files", "md5", expected_hash[:2], expected_hash[2:])
            if os.path.exists(remote_file):
                result["dvc_remote_available"] = True

    # 5. Aggregate overall status
    if is_metrics_file:
        result["status"] = "METRICS_FILE_PRESENT" if result["workspace_exists"] else "METRICS_FILE_MISSING"
    elif result["workspace_status"] == "VERIFIED":
        result["status"] = "VERIFIED_IN_WORKSPACE"
    elif result["dvc_cache_available"]:
        result["status"] = "RECOVERABLE_FROM_CACHE"
    elif result["dvc_remote_available"]:
        result["status"] = "RECOVERABLE_FROM_REMOTE"
    elif result["workspace_status"] == "HASH_MISMATCH":
        result["status"] = "MISMATCH_AND_UNAVAILABLE_IN_DVC"
    elif result["workspace_status"] == "PRESENT_UNTRACKED":
        result["status"] = "NO_HASH_RECORDED_FOR_RUN"
    elif result["workspace_status"] == "MISSING":
        result["status"] = "MISSING_FROM_WORKSPACE_AND_DVC"
    else:
        result["status"] = "UNTRACKED_OR_UNAVAILABLE"

    return result


def resolve_mlflow_run(client: MlflowClient, run_id_prefix: str):
    """Find a run by full ID or prefix across all experiments."""
    experiments = client.search_experiments()
    exp_ids = [e.experiment_id for e in experiments]
    runs = client.search_runs(exp_ids)

    matches = [r for r in runs if r.info.run_id.startswith(run_id_prefix)]
    if not matches:
        print(f"Error: No MLflow run found matching prefix '{run_id_prefix}'.", file=sys.stderr)
        print("Available runs:", file=sys.stderr)
        for r in runs:
            print(f"  {r.info.run_id}  [{r.info.status}]", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f"Error: Ambiguous run ID prefix '{run_id_prefix}'. Matches {len(matches)} runs:", file=sys.stderr)
        for r in matches:
            print(f"  {r.info.run_id}", file=sys.stderr)
        sys.exit(1)
    return matches[0]


# ─────────────────────────────────────────────────────────────────────────────
# Report Construction
# ─────────────────────────────────────────────────────────────────────────────
def build_lineage_report(client: MlflowClient, run) -> dict:
    """Build a complete, structured lineage report dictionary for a given run."""
    run_id = run.info.run_id
    tags = run.data.tags
    params = run.data.params
    metrics = run.data.metrics
    experiment = client.get_experiment(run.info.experiment_id)

    # Attempt to read lineage.json artifact if present
    lineage_manifest = None
    artifact_uri = run.info.artifact_uri
    if artifact_uri and artifact_uri.startswith("file:"):
        artifact_dir = artifact_uri.replace("file:", "").replace("/", os.sep).lstrip(os.sep)
        manifest_file = os.path.join(artifact_dir, "lineage", "lineage.json")
        if os.path.exists(manifest_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    lineage_manifest = json.load(f)
            except Exception:
                pass

    # Start and end timestamps
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(run.info.start_time / 1000))
    end_ts = (
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(run.info.end_time / 1000))
        if run.info.end_time else "In Progress"
    )

    # Git metadata
    git_info = {
        "commit_sha": tags.get("git.commit_sha", lineage_manifest.get("git", {}).get("commit_sha", "unavailable") if lineage_manifest else "unavailable"),
        "branch": tags.get("git.branch", lineage_manifest.get("git", {}).get("branch", "unavailable") if lineage_manifest else "unavailable"),
        "is_dirty": tags.get("git.is_dirty", str(lineage_manifest.get("git", {}).get("is_dirty", "unavailable")) if lineage_manifest else "unavailable")
    }

    # Environment & library metadata
    env_info = {
        "python_version": tags.get("env.python_version", lineage_manifest.get("libraries", {}).get("python", "unavailable") if lineage_manifest else "unavailable"),
        "os_platform": tags.get("env.os_platform", lineage_manifest.get("libraries", {}).get("os_platform", "unavailable") if lineage_manifest else "unavailable"),
        "scikit_learn": tags.get("env.scikit_learn", lineage_manifest.get("libraries", {}).get("scikit_learn", "unavailable") if lineage_manifest else "unavailable"),
        "pandas": tags.get("env.pandas", lineage_manifest.get("libraries", {}).get("pandas", "unavailable") if lineage_manifest else "unavailable"),
        "numpy": tags.get("env.numpy", lineage_manifest.get("libraries", {}).get("numpy", "unavailable") if lineage_manifest else "unavailable"),
        "dvc": tags.get("env.dvc", lineage_manifest.get("libraries", {}).get("dvc", "unavailable") if lineage_manifest else "unavailable"),
        "mlflow": tags.get("env.mlflow", lineage_manifest.get("libraries", {}).get("mlflow", "unavailable") if lineage_manifest else "unavailable"),
    }

    current_lock_hashes = load_dvc_lock_hashes()

    # Trace lineage stages and verify artifacts
    stages_report = []
    all_artifacts = []
    overall_all_verified = True

    for stage_def in PIPELINE_STAGES:
        stage_name = stage_def["stage"]
        stage_outs = []
        for out in stage_def["outputs"]:
            path = out["path"]
            tag_key = out["tag"]
            is_metrics_file = (tag_key is None)
            expected_hash = None
            if tag_key:
                expected_hash = tags.get(tag_key)
                if not expected_hash and lineage_manifest:
                    short_key = tag_key.replace("dvc.hash.", "")
                    expected_hash = lineage_manifest.get("dvc_hashes", {}).get(short_key)

            artifact_detail = check_artifact_provenance(
                path,
                expected_hash,
                current_lock_hashes,
                is_metrics_file=is_metrics_file
            )
            artifact_detail["role"] = out["role"]
            stage_outs.append(artifact_detail)
            all_artifacts.append(artifact_detail)

            if not is_metrics_file:
                if artifact_detail["status"] not in ("VERIFIED_IN_WORKSPACE", "RECOVERABLE_FROM_CACHE", "RECOVERABLE_FROM_REMOTE"):
                    overall_all_verified = False

        stages_report.append({
            "stage": stage_name,
            "description": stage_def["description"],
            "inputs": stage_def["inputs"],
            "outputs": stage_outs
        })

    # Model parameters
    model_params = {
        "model_type": params.get("model_type", "RandomForestClassifier"),
        "n_estimators": params.get("n_estimators", "unavailable"),
        "max_depth": params.get("max_depth", "unavailable"),
        "random_state": params.get("random_state", "unavailable"),
        "scaler_type": params.get("scaler_type", "StandardScaler"),
        "test_size": params.get("test_size", "unavailable"),
        "prepare_random_state": params.get("prepare_random_state", "unavailable"),
    }

    # Evaluation metrics
    eval_metrics = {
        "accuracy": metrics.get("accuracy", "unavailable"),
        "precision": metrics.get("precision", "unavailable"),
        "recall": metrics.get("recall", "unavailable"),
        "f1_score": metrics.get("f1_score", "unavailable"),
        "duration_seconds": metrics.get("duration_seconds", "unavailable")
    }

    # Verification summary
    versioned_artifacts = [a for a in all_artifacts if a["path"] != "metrics.json"]
    verified_in_workspace_count = sum(1 for a in versioned_artifacts if a["workspace_status"] == "VERIFIED")
    mismatched_count = sum(1 for a in versioned_artifacts if a["workspace_status"] == "HASH_MISMATCH")
    missing_workspace_count = sum(1 for a in versioned_artifacts if a["workspace_status"] == "MISSING")
    untracked_count = sum(1 for a in versioned_artifacts if a["workspace_status"] == "PRESENT_UNTRACKED")
    recoverable_count = sum(1 for a in versioned_artifacts if a["dvc_cache_available"] or a["dvc_remote_available"])
    historical_count = sum(1 for a in versioned_artifacts if a["lock_status"] == "HISTORICAL_VERSION_DIFFERS_FROM_LOCK")

    if untracked_count > 0 or verified_in_workspace_count == 0:
        overall_all_verified = False

    report = {
        "report_generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "run": {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "experiment_name": experiment.name,
            "status": run.info.status,
            "start_time": start_ts,
            "end_time": end_ts,
            "artifact_uri": run.info.artifact_uri
        },
        "git_provenance": git_info,
        "environment": env_info,
        "parameters": model_params,
        "metrics": eval_metrics,
        "lineage_stages": stages_report,
        "verification_summary": {
            "versioned_artifacts_tracked": len(versioned_artifacts),
            "verified_in_workspace": verified_in_workspace_count,
            "workspace_hash_mismatches": mismatched_count,
            "workspace_missing": missing_workspace_count,
            "untracked_or_no_hash_recorded": untracked_count,
            "recoverable_from_dvc": recoverable_count,
            "historical_versions_differing_from_current_lock": historical_count,
            "is_lineage_verified": overall_all_verified
        }
    }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Rendering: Human-Readable Text Format
# ─────────────────────────────────────────────────────────────────────────────
def render_text_report(report: dict) -> str:
    """Format the lineage report into a readable terminal presentation."""
    lines = []
    w = 78

    def banner(title: str):
        lines.append("\n" + "=" * w)
        lines.append(f"  {title.upper()}")
        lines.append("=" * w)

    def section(title: str):
        lines.append("\n" + "-" * w)
        lines.append(f"  {title}")
        lines.append("-" * w)

    banner("ML Pipeline Lineage & Provenance Report")
    r = report["run"]
    lines.append(f"  Run ID           : {r['run_id']}")
    lines.append(f"  Experiment       : {r['experiment_name']} (ID: {r['experiment_id']})")
    lines.append(f"  Status           : {r['status']}")
    lines.append(f"  Timestamp        : {r['start_time']}")
    lines.append(f"  Artifact URI     : {r['artifact_uri']}")
    lines.append(f"  Report Generated : {report['report_generated_at']}")

    # Git Provenance
    section("1. Git Code Provenance")
    g = report["git_provenance"]
    lines.append(f"  Commit SHA       : {g['commit_sha']}")
    lines.append(f"  Branch           : {g['branch']}")
    lines.append(f"  Working Tree Dirty: {g['is_dirty']}")
    if g['is_dirty'] == "True":
        lines.append("  [!] WARNING: Working tree was dirty when experiment was executed.")

    # Environment
    section("2. Execution Environment & Dependencies")
    e = report["environment"]
    lines.append(f"  Python           : {e['python_version']} ({e['os_platform']})")
    lines.append(f"  Scikit-Learn     : {e['scikit_learn']:<14} Pandas   : {e['pandas']}")
    lines.append(f"  NumPy            : {e['numpy']:<14} MLflow   : {e['mlflow']}")
    lines.append(f"  DVC              : {e['dvc']}")

    # Model Parameters & Metrics
    section("3. Parameters & Performance Metrics")
    p = report["parameters"]
    m = report["metrics"]
    lines.append(f"  Model Type       : {p['model_type']:<22} Scaler   : {p['scaler_type']}")
    lines.append(f"  Hyperparameters  : n_estimators={p['n_estimators']}, max_depth={p['max_depth']}, random_state={p['random_state']}")
    lines.append(f"  Split Parameters : test_size={p['test_size']}, prepare_random_state={p['prepare_random_state']}")
    lines.append("  ----------------------------------------------------------------------------")
    lines.append(f"  Accuracy         : {m['accuracy']:<22} Precision: {m['precision']}")
    lines.append(f"  Recall           : {m['recall']:<22} F1 Score : {m['f1_score']}")
    if m.get("duration_seconds") != "unavailable":
        lines.append(f"  Pipeline Duration: {m['duration_seconds']}s")

    # End-to-End Lineage Flow
    section("4. End-to-End Pipeline Lineage Trace")
    lines.append("  Pipeline Flow:")
    lines.append("    [Raw Dataset] -> prepare -> [Train/Test Splits]")
    lines.append("        -> featurize -> [StandardScaler & Scaled Features]")
    lines.append("        -> train     -> [RandomForest Model]")
    lines.append("        -> evaluate  -> [Evaluation Metrics]")
    lines.append("")

    for stage_info in report["lineage_stages"]:
        sname = stage_info["stage"].upper()
        sdesc = stage_info["description"]
        lines.append(f"  STAGE: {sname} ({sdesc})")
        if stage_info["inputs"]:
            lines.append(f"    Inputs  : {', '.join(stage_info['inputs'])}")
        lines.append("    Outputs :")
        for out in stage_info["outputs"]:
            role = out["role"]
            path = out["path"]
            exp_hash = out["expected_hash"]
            ws_status = out["workspace_status"]
            lock_status = out["lock_status"]
            overall_status = out["status"]

            short_hash = exp_hash[:12] + "..." if len(exp_hash) > 12 else exp_hash
            lines.append(f"      * {role} ({path}):")
            lines.append(f"          Recorded Hash : {exp_hash}")
            lines.append(f"          Workspace     : {ws_status}")
            lines.append(f"          dvc.lock Sync : {lock_status}")
            if lock_status == "HISTORICAL_VERSION_DIFFERS_FROM_LOCK":
                lines.append(f"          Current Lock  : {out['lock_hash']}")
            lines.append(f"          Provenance    : {overall_status}")
        lines.append("")

    # Verification Summary
    section("5. Lineage Verification Summary")
    v = report["verification_summary"]
    lines.append(f"  Versioned Artifacts Tracked     : {v['versioned_artifacts_tracked']}")
    lines.append(f"  Verified in Workspace           : {v['verified_in_workspace']}")
    lines.append(f"  Workspace Hash Mismatches       : {v['workspace_hash_mismatches']}")
    lines.append(f"  Missing from Workspace          : {v['workspace_missing']}")
    lines.append(f"  Untracked / No Recorded Hash    : {v['untracked_or_no_hash_recorded']}")
    lines.append(f"  Available in DVC Cache/Remote   : {v['recoverable_from_dvc']}")
    lines.append(f"  Historical vs Lock Discrepancies: {v['historical_versions_differing_from_current_lock']}")
    lines.append("")
    if v["is_lineage_verified"]:
        lines.append("  [OK] OVERALL STATUS: LINEAGE FULLY VERIFIED")
        lines.append("       All pipeline artifacts are accountable and verified against recorded provenance.")
    else:
        lines.append("  [!] OVERALL STATUS: LINEAGE VERIFICATION INCOMPLETE / UNVERIFIED")
        lines.append("       Some artifacts are untracked, missing, or have hash discrepancies.")
    lines.append("=" * w + "\n")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Listing & Commands
# ─────────────────────────────────────────────────────────────────────────────
def cmd_list_runs(args):
    """List recorded runs in MLflow tracking database."""
    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    exps = client.search_experiments()
    if args.experiment:
        exps = [e for e in exps if e.name == args.experiment or e.experiment_id == args.experiment]
        if not exps:
            print(f"Error: Experiment '{args.experiment}' not found.", file=sys.stderr)
            sys.exit(1)

    exp_ids = [e.experiment_id for e in exps]
    runs = client.search_runs(exp_ids, order_by=["start_time DESC"])

    print("\n" + "=" * 90)
    print("  Available MLflow Runs for Lineage Reporting")
    print("=" * 90)
    print(f"  {'RUN ID':<36}  {'STATUS':<10}  {'EXPERIMENT':<24}  {'ACCURACY':>8}  TIMESTAMP")
    print("  " + "-" * 88)
    exp_names = {e.experiment_id: e.name for e in client.search_experiments()}
    for r in runs:
        accuracy = r.data.metrics.get("accuracy", float("nan"))
        acc_str = f"{accuracy:.4f}" if isinstance(accuracy, float) and accuracy == accuracy else "N/A"
        ts = time.strftime("%Y-%m-%d %H:%M", time.gmtime(r.info.start_time / 1000))
        ename = exp_names.get(r.info.experiment_id, r.info.experiment_id)[:24]
        print(f"  {r.info.run_id}  {r.info.status:<10}  {ename:<24}  {acc_str:>8}  {ts}")
    print(f"\n  Total: {len(runs)} run(s)\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lineage_report.py",
        description="Stage 5: Lineage Report Generator for ML Pipelines.\n"
                    "Traces data versioning, pipeline transformations, and provenance lineage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Terminal human-readable report for a run:
  python scripts/lineage_report.py e34a5804

  # JSON report output to terminal:
  python scripts/lineage_report.py e34a5804 --format json

  # Write JSON report to file:
  python scripts/lineage_report.py e34a5804 --format json --output reports/lineage_report.json

  # Write text report to file:
  python scripts/lineage_report.py e34a5804 --output reports/lineage_report.txt

  # List available runs:
  python scripts/lineage_report.py --list
"""
    )
    p.add_argument(
        "run_id",
        nargs="?",
        help="Full or partial (prefix) MLflow run ID to analyze."
    )
    p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' (human-readable) or 'json' (structured)."
    )
    p.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        help="Optional file path to write report to (creates parent directories if needed)."
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available MLflow runs and exit."
    )
    p.add_argument(
        "--experiment",
        metavar="NAME_OR_ID",
        help="Filter --list by experiment name or ID."
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        cmd_list_runs(args)
        return

    if not args.run_id:
        parser.print_help()
        print("\nError: run_id is required unless --list is specified.", file=sys.stderr)
        sys.exit(1)

    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    run = resolve_mlflow_run(client, args.run_id)
    report = build_lineage_report(client, run)

    # Format output
    if args.format == "json":
        output_content = json.dumps(report, indent=2)
    else:
        output_content = render_text_report(report)

    # Output destination
    if args.output:
        abs_output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(abs_output), exist_ok=True)
        with open(abs_output, "w", encoding="utf-8") as f:
            f.write(output_content)
        print(f"[lineage_report] Report successfully written to: {abs_output}")
    else:
        print(output_content)

    # Verification status check
    v = report["verification_summary"]
    # If any artifact had a hash mismatch or was untracked/missing completely, exit code 2
    if v["workspace_hash_mismatches"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
