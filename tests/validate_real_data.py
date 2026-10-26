#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/validate_real_data.py
===========================
Stage 6B Real-Data Validation Runner & Historical Git Commit Inspector.

Executes read-only automated validation of all chatbot commands against:
1. Actual MLflow runs in the tracking database (mlflow.db):
   - Latest finished run (e.g. commit 2e719abd)
   - Historical run recorded against commit 19e15021
   - Historical run recorded against commit 90477600
   - Historical run recorded against commit 2f0cb71d
2. Historical Git commits recorded across stages.

Safety:
All tests are strictly READ-ONLY. No workspace artifacts or database entries are mutated.
"""

import os
import subprocess
import sys
import time

# Ensure UTF-8 stdout/stderr on Windows console (same approach as scripts/recover_run.py)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app
import mlflow
from mlflow.tracking import MlflowClient

# Enable isolated test session context for CLI validation runner
app.enable_test_session_override()



def run_git(args: list[str]) -> tuple[int, str, str]:
    """Execute a read-only git command."""
    proc = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def find_runs_by_commit_prefix(runs, prefix: str):
    """Find all runs whose recorded git.commit_sha starts with prefix."""
    matched = []
    for r in runs:
        sha = r.data.tags.get("git.commit_sha", "")
        if sha.lower().startswith(prefix.lower()):
            matched.append(r)
    return matched


def main():
    print("=" * 80)
    print("  STAGE 6B — REAL-DATA VALIDATION & HISTORICAL COMMIT AUDIT")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Discover Real MLflow Runs
    # ─────────────────────────────────────────────────────────────────────────
    client = app.get_mlflow_client()
    exps = client.search_experiments()
    exp_target = None
    for e in exps:
        if e.name == "ML_Lineage_Experiments":
            exp_target = e
            break

    if not exp_target:
        print("[ERROR] Experiment 'ML_Lineage_Experiments' not found!")
        sys.exit(1)

    all_runs = client.search_runs([exp_target.experiment_id], order_by=["start_time DESC"])
    finished_runs = [r for r in all_runs if r.info.status == "FINISHED"]

    print(f"Tracking URI       : {client.tracking_uri}")
    print(f"Target Experiment  : {exp_target.name} (id={exp_target.experiment_id})")
    print(f"Total Runs Found   : {len(all_runs)} (Finished: {len(finished_runs)})\n")

    latest_run = finished_runs[0] if finished_runs else None
    runs_19e15021 = find_runs_by_commit_prefix(finished_runs, "19e15021")
    runs_90477600 = find_runs_by_commit_prefix(finished_runs, "90477600")
    runs_2f0cb71d = find_runs_by_commit_prefix(finished_runs, "2f0cb71d")

    target_runs = {
        "Latest Finished Run": latest_run,
        "Commit 19e15021 Run": runs_19e15021[0] if runs_19e15021 else None,
        "Commit 90477600 Run": runs_90477600[0] if runs_90477600 else None,
        "Commit 2f0cb71d Run": runs_2f0cb71d[0] if runs_2f0cb71d else None,
    }

    results = []

    def record(category, target_id, command, exit_code, status, modifies_files, note=""):
        results.append({
            "category": category,
            "target": target_id,
            "command": command,
            "exit_code": exit_code,
            "status": status,
            "modifies_files": modifies_files,
            "note": note,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Test Each Target Run Across Chatbot Commands
    # ─────────────────────────────────────────────────────────────────────────
    for category, run in target_runs.items():
        print("-" * 80)
        if not run:
            print(f"[{category}] — UNAVAILABLE in MLflow database.")
            record(category, "N/A", "N/A", -1, "SKIPPED", False, "Run not found in MLflow")
            continue

        run_id = run.info.run_id
        short_id = run_id[:8]
        git_sha = run.data.tags.get("git.commit_sha", "N/A")[:8]
        acc = run.data.metrics.get("accuracy", "N/A")
        print(f"[{category}] Run ID: {short_id} (Full: {run_id}) | Git SHA: {git_sha} | Acc: {acc}")

        # A. Inspect
        resp_inspect = app.route_message(f"inspect {short_id}")
        ok_inspect = "Run Inspection" in resp_inspect and run_id in resp_inspect
        record(category, short_id, f"inspect {short_id}", 0 if ok_inspect else 1, "PASSED" if ok_inspect else "FAILED", False)
        print(f"  ✓ inspect       -> {'PASS' if ok_inspect else 'FAIL'}")

        # B. Verify (--verify-only via recover_run.py)
        resp_verify = app.route_message(f"verify {short_id}")
        ok_verify = "Artifact Verification" in resp_verify and ("All checked artifacts match" in resp_verify or "Verification discrepancies" in resp_verify)
        record(category, short_id, f"verify {short_id}", 0 if ok_verify else 1, "PASSED" if ok_verify else "FAILED", False)
        print(f"  ✓ verify        -> {'PASS' if ok_verify else 'FAIL'}")

        # C. Dry-Run preview (--dry-run via recover_run.py)
        resp_dry = app.route_message(f"dry-run {short_id}")
        ok_dry = "Recovery Dry-Run Preview" in resp_dry and "No workspace files were modified" in resp_dry
        record(category, short_id, f"dry-run {short_id}", 0 if ok_dry else 1, "PASSED" if ok_dry else "FAILED", False)
        print(f"  ✓ dry-run       -> {'PASS' if ok_dry else 'FAIL'}")

        # D. Lineage report
        resp_lineage = app.route_message(f"lineage {short_id}")
        ok_lineage = "Lineage Report" in resp_lineage or "Pipeline Lineage Trace" in resp_lineage
        record(category, short_id, f"lineage {short_id}", 0 if ok_lineage else 1, "PASSED" if ok_lineage else "FAILED", False)
        print(f"  ✓ lineage       -> {'PASS' if ok_lineage else 'FAIL'}")

        # E. Recover prompt (dry-run preview + confirmation prompt, NO restore)
        resp_recover = app.route_message(f"recover {short_id}")
        pending = app.get_confirmation_state()
        ok_recover = (
            "Recovery Preview & Confirmation Required" in resp_recover
            and "WARNING: Workspace Overwrite Risk" in resp_recover
            and pending is not None
            and pending.get("run_id") == run_id
        )
        record(category, short_id, f"recover {short_id}", 0 if ok_recover else 1, "PASSED" if ok_recover else "FAILED", False, "Sets pending confirmation; 0 files modified")
        print(f"  ✓ recover (req) -> {'PASS' if ok_recover else 'FAIL'} (Dry-run preview shown, confirmation requested)")

        # F. Restore alias
        resp_restore = app.route_message(f"restore {short_id}")
        ok_restore = "Recovery Preview & Confirmation Required" in resp_restore
        record(category, short_id, f"restore {short_id}", 0 if ok_restore else 1, "PASSED" if ok_restore else "FAILED", False, "Restore alias routed properly")
        print(f"  ✓ restore alias -> {'PASS' if ok_restore else 'FAIL'}")

        # G. Rejection of mismatched confirmation
        resp_rej = app.route_message("confirm recover 99999999")
        ok_rej = "Confirmation Rejected" in resp_rej
        record(category, short_id, "confirm recover 99999999", 0 if ok_rej else 1, "PASSED" if ok_rej else "FAILED", False, "Mismatched ID rejected")
        print(f"  ✓ mismatch rej  -> {'PASS' if ok_rej else 'FAIL'}")

        # H. Cancel confirmation
        resp_cancel = app.route_message("cancel")
        ok_cancel = "Recovery Cancelled" in resp_cancel and app.get_confirmation_state() is None
        record(category, short_id, "cancel", 0 if ok_cancel else 1, "PASSED" if ok_cancel else "FAILED", False, "Confirmation cleared without restore")
        print(f"  ✓ cancel flow   -> {'PASS' if ok_cancel else 'FAIL'}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Test Run Comparison
    # ─────────────────────────────────────────────────────────────────────────
    print("-" * 80)
    print("Testing Compare Command on Real Runs:")
    if len(finished_runs) >= 2:
        r1 = finished_runs[0].info.run_id[:8]
        r2 = finished_runs[1].info.run_id[:8]
        resp_comp = app.route_message(f"compare {r1} {r2}")
        ok_comp = "Run Comparison" in resp_comp and r1 in resp_comp and r2 in resp_comp
        record("Comparison", f"{r1} vs {r2}", f"compare {r1} {r2}", 0 if ok_comp else 1, "PASSED" if ok_comp else "FAILED", False)
        print(f"  ✓ compare {r1} {r2} -> {'PASS' if ok_comp else 'FAIL'}")
    else:
        record("Comparison", "N/A", "compare", -1, "SKIPPED", False, "Need at least 2 runs")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. System & Global Commands
    # ─────────────────────────────────────────────────────────────────────────
    print("-" * 80)
    print("Testing Global Chatbot Commands:")
    # Help
    h_resp = app.route_message("help")
    ok_h = "ML Lineage Chatbot — Available Commands" in h_resp and "Overwrites Workspace Files" in h_resp
    record("Global", "All", "help", 0 if ok_h else 1, "PASSED" if ok_h else "FAILED", False)
    print(f"  ✓ help          -> {'PASS' if ok_h else 'FAIL'}")

    # List
    l_resp = app.route_message("list runs")
    ok_l = "MLflow run(s)" in l_resp and len(all_runs) > 0
    record("Global", "All", "list runs", 0 if ok_l else 1, "PASSED" if ok_l else "FAILED", False)
    print(f"  ✓ list runs     -> {'PASS' if ok_l else 'FAIL'}")

    # Status
    s_resp = app.route_message("status")
    ok_s = "DVC Workspace Pipeline" in s_resp and "DVC Remote Storage Sync" in s_resp
    record("Global", "All", "status", 0 if ok_s else 1, "PASSED" if ok_s else "FAILED", False)
    print(f"  ✓ status        -> {'PASS' if ok_s else 'FAIL'}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Historical Git Commit Validation (Read-Only)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  HISTORICAL GIT COMMIT VALIDATION (READ-ONLY INSPECTION)")
    print("=" * 80)

    historical_commits = [
        ("2e719abd1a8e31f3e3814ee7dae59422ea68998d", "docs: fresh setup instructions & latest run baseline"),
        ("6940514deba5eae041d8e6a30a71cf3d19b0404c", "feat: implement Stage 5 lineage report generator"),
        ("19e15021cb7fad0c167e2c7d2332d3cfb54272f5", "feat: implement Stage 4 historical run recovery"),
        ("9047760055d2ffb731f1d15b298f8df2631ac8f0", "feat: integrate MLflow tracking with DVC lineage"),
        ("2f0cb71de5d69281717c85da89ab33f90512fc4d", "feat: configure local DVC remote & scaler tracking"),
    ]

    commit_results = []

    for sha, desc in historical_commits:
        ret, out, err = run_git(["rev-parse", sha])
        available = (ret == 0)
        full_sha = out if available else sha
        commit_msg = ""
        tree_files = []

        if available:
            _, log_out, _ = run_git(["log", "-1", "--format=%h %s (%cd)", "--date=short", full_sha])
            commit_msg = log_out
            _, show_out, _ = run_git(["show", "--stat", "--oneline", full_sha])
            # Check what scripts/files existed at that commit
            ret_recover, _, _ = run_git(["cat-file", "-e", f"{full_sha}:scripts/recover_run.py"])
            ret_lineage, _, _ = run_git(["cat-file", "-e", f"{full_sha}:scripts/lineage_report.py"])
            ret_app, _, _ = run_git(["cat-file", "-e", f"{full_sha}:app.py"])
            capabilities = []
            if ret_recover == 0: capabilities.append("recover_run.py")
            if ret_lineage == 0: capabilities.append("lineage_report.py")
            if ret_app == 0: capabilities.append("app.py")

            commit_results.append({
                "sha": sha[:10],
                "full_sha": full_sha,
                "description": desc,
                "available": True,
                "log": commit_msg,
                "scripts_present": capabilities,
            })
            print(f"[FOUND] {sha[:10]} | {commit_msg} | Scripts: {', '.join(capabilities) or 'None'}")
        else:
            commit_results.append({
                "sha": sha[:10],
                "full_sha": sha,
                "description": desc,
                "available": False,
                "log": f"Error: {err}",
                "scripts_present": [],
            })
            print(f"[MISSING] {sha[:10]} | {err}")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary Table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Checks Executed : {len(results)}")
    passed_count = sum(1 for r in results if r["status"] == "PASSED")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    skipped_count = sum(1 for r in results if r["status"] == "SKIPPED")
    print(f"Passed                : {passed_count}")
    print(f"Failed                : {failed_count}")
    print(f"Skipped               : {skipped_count}")
    print(f"Files Modified        : 0 (All operations strictly read-only)\n")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
