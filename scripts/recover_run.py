#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/recover_run.py
======================
Stage 4: Historical Run Recovery for ML Lineage Tracking System.

Recover or verify a historical ML experiment run recorded in MLflow,
using DVC-tracked artifact hashes for provenance verification.

Usage:
    python scripts/recover_run.py <run_id>
    python scripts/recover_run.py <run_id> --dry-run
    python scripts/recover_run.py <run_id> --verify-only
    python scripts/recover_run.py --list
    python scripts/recover_run.py --list --experiment ML_Lineage_Experiments

Example:
    python scripts/recover_run.py e34a580422034aacb402d78def5d4249
    python scripts/recover_run.py e34a5804  # short prefix also accepted
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import yaml

# Silence MLflow's tracing hint before importing mlflow
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
from mlflow.tracking import MlflowClient

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_lineage.core.context import ProjectContext
from ml_lineage.core.hashing import compute_file_md5 as _core_compute_file_md5
from ml_lineage.adapters.dvc import DVCAdapter
from ml_lineage.adapters.mlflow import MLflowAdapter
from ml_lineage.wine import WINE_DVC_TAG_MAP

DEFAULT_CONTEXT = ProjectContext.from_path(PROJECT_ROOT)
_dvc_adapter = DVCAdapter(DEFAULT_CONTEXT)
_mlflow_adapter = MLflowAdapter(DEFAULT_CONTEXT)

PARAMS_PATH = str(DEFAULT_CONTEXT.params_path)
DVC_LOCK_PATH = str(DEFAULT_CONTEXT.dvc_lock_path)
DVC_TAG_MAP = WINE_DVC_TAG_MAP

# ─────────────────────────────────────────────────────────────────────────────
# Stdout encoding: force UTF-8 on Windows to prevent cp1252 errors
# ─────────────────────────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass  # If it fails, proceed with best-effort

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours for terminal output (disabled on non-TTY)
# ─────────────────────────────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def ok(msg: str)   -> str: return _c("32", f"  [OK]  {msg}")
def warn(msg: str) -> str: return _c("33", f"  [!!]  {msg}")
def err(msg: str)  -> str: return _c("31", f"  [XX]  {msg}")
def info(msg: str) -> str: return _c("36", f"  [--]  {msg}")
def head(msg: str) -> str: return _c("1;35", f"\n{'='*60}\n  {msg}\n{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# MLflow helpers (Delegated to ml_lineage)
# ─────────────────────────────────────────────────────────────────────────────
def _get_tracking_uri() -> str:
    """Load tracking URI from params.yaml, fall back to default."""
    return DEFAULT_CONTEXT.tracking_uri


def _get_client() -> MlflowClient:
    return _mlflow_adapter.get_client()


def _resolve_run(client: MlflowClient, run_id_prefix: str):
    """
    Find a run by full ID or unambiguous prefix.
    Returns the RunInfo-like object, or raises with a helpful message.
    """
    # Search all experiments
    exps = client.search_experiments()
    all_exp_ids = [e.experiment_id for e in exps]
    runs = client.search_runs(all_exp_ids)

    matched = [r for r in runs if r.info.run_id.startswith(run_id_prefix)]
    if not matched:
        candidates = "\n".join(f"  {r.info.run_id}  [{r.info.status}]" for r in runs)
        print(err(f"No run found matching prefix '{run_id_prefix}'."))
        print(info("Available run IDs:"))
        print(candidates if candidates else "  (none)")
        sys.exit(1)
    if len(matched) > 1:
        print(err(f"Prefix '{run_id_prefix}' is ambiguous — matches {len(matched)} runs:"))
        for r in matched:
            print(f"  {r.info.run_id}")
        sys.exit(1)
    return matched[0]


# ─────────────────────────────────────────────────────────────────────────────
# MD5 verification & DVC artifact resolution (Delegated to ml_lineage)
# ─────────────────────────────────────────────────────────────────────────────
def _md5(path: str) -> str:
    return _core_compute_file_md5(path)


def _dvc_cache_path(md5_hash: str) -> str:
    """Return the path of a file inside the DVC local cache."""
    return str(_dvc_adapter.get_cache_file(md5_hash))


def _dvc_remote_path(md5_hash: str) -> str | None:
    """Return the path of a file inside the configured local DVC remote."""
    p = _dvc_adapter.get_remote_file(md5_hash)
    return str(p) if p else None


def _find_artifact_source(md5_hash: str) -> tuple[str | None, str]:
    """
    Return (source_path, source_label) for the given DVC hash,
    checking the workspace file, local DVC cache, and local DVC remote.
    """
    p, label = _dvc_adapter.find_artifact_source(md5_hash)
    return (str(p) if p else None, label)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────
def cmd_list(args):
    """List all MLflow runs, newest first."""
    client = _get_client()
    exps = client.search_experiments()

    exp_filter = args.experiment
    if exp_filter:
        exps = [e for e in exps if e.name == exp_filter or e.experiment_id == exp_filter]
        if not exps:
            print(err(f"No experiment named '{exp_filter}' found."))
            sys.exit(1)

    all_ids = [e.experiment_id for e in exps]
    runs = client.search_runs(all_ids, order_by=["start_time DESC"])

    if not runs:
        print(info("No runs found."))
        return

    print(head("MLflow Runs"))
    header = f"{'RUN ID':<36}  {'STATUS':<10}  {'EXPERIMENT':<26}  {'GIT SHA':<10}  {'ACCURACY':>8}  TIMESTAMP"
    print(f"  {header}")
    print(f"  {'─'*120}")

    exp_names = {e.experiment_id: e.name for e in client.search_experiments()}

    for r in runs:
        tags = r.data.tags
        git_sha = tags.get("git.commit_sha", "")[:8]
        accuracy = r.data.metrics.get("accuracy", float("nan"))
        acc_str = f"{accuracy:.4f}" if isinstance(accuracy, float) and accuracy == accuracy else "N/A"
        ts = time.strftime("%Y-%m-%d %H:%M", time.gmtime(r.info.start_time / 1000))
        exp_name = exp_names.get(r.info.experiment_id, r.info.experiment_id)[:26]
        status = r.info.status
        color_code = "32" if status == "FINISHED" else ("31" if status == "FAILED" else "33")
        status_str = _c(color_code, f"{status:<10}") if USE_COLOR else f"{status:<10}"
        print(f"  {r.info.run_id}  {status_str}  {exp_name:<26}  {git_sha:<10}  {acc_str:>8}  {ts}")

    print(f"\n  Total: {len(runs)} run(s)")


def cmd_inspect(run_id_prefix: str):
    """Print full details of a single run."""
    client = _get_client()
    run = _resolve_run(client, run_id_prefix)
    r = run

    print(head(f"Run Inspection: {r.info.run_id[:16]}..."))
    print(f"  Full Run ID  : {r.info.run_id}")
    print(f"  Status       : {r.info.status}")
    experiment = client.get_experiment(r.info.experiment_id)
    print(f"  Experiment   : {experiment.name}  (id={r.info.experiment_id})")
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(r.info.start_time / 1000))
    print(f"  Started At   : {ts}")
    print(f"  Artifact URI : {r.info.artifact_uri}")

    print("\n  ── Metrics ──────────────────────────────")
    for k, v in sorted(r.data.metrics.items()):
        print(f"    {k:<25} = {v}")

    print("\n  ── Parameters ───────────────────────────")
    for k, v in sorted(r.data.params.items()):
        print(f"    {k:<30} = {v}")

    print("\n  ── Git Provenance ────────────────────────")
    tags = r.data.tags
    for key in ("git.commit_sha", "git.branch", "git.is_dirty"):
        print(f"    {key:<30} = {tags.get(key, 'N/A')}")

    print("\n  ── DVC Artifact Hashes ───────────────────")
    for tag_key, artifact_path in DVC_TAG_MAP.items():
        h = tags.get(tag_key, "unavailable")
        print(f"    {artifact_path:<40} {h}")

    print("\n  ── Environment ───────────────────────────")
    for key in ("env.python_version", "env.scikit_learn", "env.pandas",
                "env.numpy", "env.dvc", "env.mlflow", "env.os_platform"):
        print(f"    {key:<30} = {tags.get(key, 'N/A')}")


def cmd_recover(run_id_prefix: str, dry_run: bool, verify_only: bool):
    """
    Main recovery logic.

    Modes:
    - verify_only: check hashes only, restore nothing
    - dry_run:     report what would be done, restore nothing
    - default:     verify + restore artifacts where needed
    """
    client = _get_client()
    run = _resolve_run(client, run_id_prefix)
    tags = run.data.tags
    run_id = run.info.run_id
    metrics = run.data.metrics
    params = run.data.params

    print(head(f"Recovery: {run_id[:16]}..."))
    mode_label = "VERIFY-ONLY" if verify_only else ("DRY-RUN" if dry_run else "RESTORE")
    print(info(f"Mode: {mode_label}"))
    print(info(f"Status: {run.info.status}"))

    if run.info.status != "FINISHED":
        print(warn(f"Run status is '{run.info.status}', not FINISHED. "
                   "Proceeding, but results may be incomplete."))

    # ── 1. Provenance metadata ──────────────────────────────────────────────
    print(head("Step 1: Provenance Metadata"))

    git_sha = tags.get("git.commit_sha", "unavailable")
    git_branch = tags.get("git.branch", "unavailable")
    git_dirty = tags.get("git.is_dirty", "unavailable")
    print(info(f"Git Commit SHA : {git_sha}"))
    print(info(f"Git Branch     : {git_branch}"))
    print(info(f"Working tree was dirty at record time: {git_dirty}"))

    if git_dirty == "True":
        print(warn("The working tree was dirty when this run was logged. "
                   "Exact code reproduction may not be possible from Git alone."))

    # ── 2. Verify/restore DVC artifacts ────────────────────────────────────
    print(head("Step 2: DVC Artifact Verification & Recovery"))

    recovery_results = {}

    for tag_key, rel_path in DVC_TAG_MAP.items():
        expected_hash = tags.get(tag_key, "")
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        label = rel_path

        if not expected_hash or expected_hash == "unavailable":
            print(warn(f"{label}: no hash recorded — skipping"))
            recovery_results[rel_path] = "no_hash"
            continue

        print(f"\n  {_c('1', rel_path)}")
        print(f"    Expected MD5 : {expected_hash}")

        # Check current workspace file
        workspace_match = False
        if os.path.exists(abs_path):
            current_hash = _md5(abs_path)
            print(f"    Current MD5  : {current_hash}")
            if current_hash == expected_hash:
                print(ok(f"Workspace file matches expected hash."))
                workspace_match = True
                recovery_results[rel_path] = "verified_in_place"
            else:
                print(warn(f"Workspace file has a different hash."))
        else:
            print(warn(f"Workspace file does not exist."))

        if workspace_match:
            continue

        # Attempt to find in DVC cache or remote
        source_path, source_label = _find_artifact_source(expected_hash)

        if source_path is None:
            print(err(f"Hash {expected_hash} not found in DVC cache or remote."))
            print(err(f"Cannot recover '{rel_path}' for this run."))
            recovery_results[rel_path] = "missing"
            continue

        print(info(f"Found in {source_label}: {source_path}"))

        # Verify the source file's own hash before copying
        source_hash = _md5(source_path)
        if source_hash != expected_hash:
            print(err(f"Source file hash mismatch! Expected {expected_hash}, got {source_hash}. "
                      "DVC cache may be corrupted."))
            recovery_results[rel_path] = "cache_corrupted"
            continue

        print(ok(f"Source hash verified: {source_hash}"))

        if verify_only:
            print(info(f"[verify-only] Would restore from {source_label}."))
            recovery_results[rel_path] = "restorable"
            continue

        if dry_run:
            print(info(f"[dry-run] Would copy {source_path} → {abs_path}"))
            recovery_results[rel_path] = "would_restore"
            continue

        # Restore
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        shutil.copy2(source_path, abs_path)
        # Verify restored file
        restored_hash = _md5(abs_path)
        if restored_hash == expected_hash:
            print(ok(f"Restored and verified: {abs_path}"))
            recovery_results[rel_path] = "restored_verified"
        else:
            print(err(f"Post-restore verification FAILED for {abs_path}! "
                      f"Expected {expected_hash}, got {restored_hash}."))
            recovery_results[rel_path] = "restore_verification_failed"

    # ── 3. Lineage manifest from MLflow artifacts ────────────────────────────
    print(head("Step 3: Lineage Manifest"))

    artifact_uri = run.info.artifact_uri
    lineage_local_path = None

    if artifact_uri and artifact_uri.startswith("file:"):
        artifact_dir = artifact_uri.replace("file:", "").replace("/", os.sep).lstrip(os.sep)
        if sys.platform == "win32" and not artifact_dir.startswith(("\\\\", "\\")):
            # Windows absolute path starts with drive letter
            pass
        lineage_json = os.path.join(artifact_dir, "lineage", "lineage.json")
        if os.path.exists(lineage_json):
            with open(lineage_json) as f:
                manifest = json.load(f)
            print(ok(f"Lineage manifest found at: {lineage_json}"))
            print(info(f"  Recorded timestamp : {manifest.get('timestamp', 'N/A')}"))
            print(info(f"  Duration (s)       : {manifest.get('duration_seconds', 'N/A')}"))
            lineage_local_path = lineage_json
        else:
            print(warn(f"Lineage manifest not found at expected path: {lineage_json}"))
    else:
        print(warn(f"Artifact URI is not a local file path: {artifact_uri}"))
        print(warn("Lineage manifest inspection skipped."))

    # ── 4. Metrics from run ─────────────────────────────────────────────────
    print(head("Step 4: Recorded Metrics"))
    for k, v in sorted(metrics.items()):
        print(f"    {k:<25} = {v}")

    # ── 5. Environment note ─────────────────────────────────────────────────
    print(head("Step 5: Environment at Time of Run"))
    for key in ("env.python_version", "env.scikit_learn", "env.pandas",
                "env.numpy", "env.dvc", "env.mlflow"):
        print(f"    {key:<30} = {tags.get(key, 'N/A')}")

    import platform, sklearn, pandas, numpy, dvc, mlflow as mlf
    current_env = {
        "env.python_version": platform.python_version(),
        "env.scikit_learn": sklearn.__version__,
        "env.pandas": pandas.__version__,
        "env.numpy": numpy.__version__,
        "env.dvc": dvc.__version__,
        "env.mlflow": mlf.__version__,
    }
    env_mismatches = []
    for key, curr_val in current_env.items():
        recorded = tags.get(key, "N/A")
        if recorded != "N/A" and curr_val != recorded:
            env_mismatches.append((key, recorded, curr_val))

    if env_mismatches:
        print(warn("Environment differences detected:"))
        for key, rec, cur in env_mismatches:
            print(warn(f"  {key}: recorded={rec}, current={cur}"))
    else:
        print(ok("Current environment matches recorded environment."))

    # ── 6. Summary ─────────────────────────────────────────────────────────
    print(head("Recovery Summary"))

    total = len(recovery_results)
    verified = sum(1 for v in recovery_results.values()
                   if v in ("verified_in_place", "restored_verified", "restorable"))
    restorable = sum(1 for v in recovery_results.values()
                     if v in ("would_restore", "restorable"))
    restored = sum(1 for v in recovery_results.values() if v == "restored_verified")
    missing = sum(1 for v in recovery_results.values() if v == "missing")
    corrupted = sum(1 for v in recovery_results.values() if v == "cache_corrupted")
    failed_verify = sum(1 for v in recovery_results.values()
                        if v == "restore_verification_failed")

    print(f"  Artifacts checked     : {total}")
    print(f"  Hash-verified         : {verified}")

    if not verify_only and not dry_run:
        print(f"  Restored & verified   : {restored}")

    if dry_run or verify_only:
        print(f"  Would restore         : {restorable}")

    if missing > 0:
        print(err(f"Unrecoverable (hash missing from cache/remote): {missing}"))
    if corrupted > 0:
        print(err(f"Cache corruption detected: {corrupted}"))
    if failed_verify > 0:
        print(err(f"Post-restore verification failures: {failed_verify}"))

    print()
    all_ok = (missing == 0 and corrupted == 0 and failed_verify == 0)
    if all_ok and not dry_run and not verify_only:
        print(ok("All artifacts successfully recovered and verified."))
        print(info(f"Run ID: {run_id}"))
        if lineage_local_path:
            print(info(f"Lineage manifest: {lineage_local_path}"))
        print()
        print(info("IMPORTANT: File hashes match the recorded versions."))
        print(info("           This does NOT guarantee identical model behaviour if"))
        print(info("           library versions differ from those recorded above."))
    elif all_ok and dry_run:
        print(ok("Dry-run complete. All artifacts are recoverable."))
    elif all_ok and verify_only:
        print(ok("Verification complete. All checked artifacts match their recorded hashes."))
    else:
        print(err("Recovery incomplete. Review errors above."))
        sys.exit(2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="recover_run.py",
        description=(
            "Historical run recovery tool for the ML Lineage Tracking System.\n\n"
            "Retrieves a recorded MLflow run's parameters, metrics, artifacts, "
            "Git SHA, and DVC provenance. Restores artifact files from the DVC "
            "cache or local remote and verifies their hashes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all recorded runs
  python scripts/recover_run.py --list

  # List runs for a specific experiment
  python scripts/recover_run.py --list --experiment ML_Lineage_Experiments

  # Inspect a run (show all metadata, no changes)
  python scripts/recover_run.py e34a580422034aacb402d78def5d4249 --inspect

  # Dry-run: show what would be restored
  python scripts/recover_run.py e34a5804 --dry-run

  # Verify hashes only (no file changes)
  python scripts/recover_run.py e34a5804 --verify-only

  # Full recovery (restore artifacts and verify hashes)
  python scripts/recover_run.py e34a580422034aacb402d78def5d4249
""",
    )

    p.add_argument(
        "run_id",
        nargs="?",
        help="Full or partial (prefix) MLflow run ID to recover.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all MLflow runs and exit.",
    )
    p.add_argument(
        "--experiment",
        metavar="NAME_OR_ID",
        help="Filter --list by experiment name or ID.",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Print full run metadata and exit without making any changes.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Show what would be done without making any file changes.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        dest="verify_only",
        help="Verify artifact hashes without restoring any files.",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Validate argument combinations
    if args.list:
        cmd_list(args)
        return

    if not args.run_id:
        parser.print_help()
        print(err("\nError: run_id is required unless --list is specified."))
        sys.exit(1)

    if args.inspect:
        cmd_inspect(args.run_id)
        return

    cmd_recover(
        run_id_prefix=args.run_id,
        dry_run=args.dry_run,
        verify_only=args.verify_only,
    )


if __name__ == "__main__":
    main()
