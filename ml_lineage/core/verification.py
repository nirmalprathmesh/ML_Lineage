"""
ml_lineage.core.verification
============================
Artifact provenance verification service.
Compares recorded artifact hashes against current workspace files,
dvc.lock state, local DVC cache, and configured remote storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_lineage.core.hashing import compute_file_md5
from ml_lineage.wine import WINE_DVC_TAG_MAP

if TYPE_CHECKING:
    from ml_lineage.core.context import ProjectContext
    from ml_lineage.adapters.dvc import DVCAdapter


def check_artifact_provenance(
    context: ProjectContext,
    rel_path: str,
    expected_hash: str | None,
    current_lock_hashes: dict[str, str],
    dvc_adapter: DVCAdapter | None = None,
    is_metrics_file: bool = False,
) -> dict[str, Any]:
    """Examine the state of an artifact in the workspace, DVC cache, and remote.

    Args:
        context: Active ProjectContext.
        rel_path: Relative path to the artifact within the project.
        expected_hash: Recorded MD5 hash from the MLflow run.
        current_lock_hashes: Current output hashes from dvc.lock.
        dvc_adapter: Optional DVCAdapter instance.
        is_metrics_file: True if file is unversioned metrics file.

    Returns:
        Structured artifact status dictionary.
    """
    abs_path = context.resolve_path(rel_path)
    workspace_exists = abs_path.is_file()

    result: dict[str, Any] = {
        "path": rel_path,
        "expected_hash": expected_hash or "unavailable",
        "lock_hash": current_lock_hashes.get(rel_path, "unavailable"),
        "workspace_exists": workspace_exists,
        "workspace_hash": None,
        "workspace_status": "UNAVAILABLE",
        "lock_status": "UNKNOWN",
        "dvc_cache_available": False,
        "dvc_remote_available": False,
        "status": "UNVERIFIED",
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
    if workspace_exists:
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

    # 3. Check DVC cache and remote if dvc_adapter available or via context
    if expected_hash and expected_hash != "unavailable":
        cache_file = context.dvc_cache_root / expected_hash[:2] / expected_hash[2:]
        if cache_file.is_file():
            result["dvc_cache_available"] = True

        remote_url = context.get_dvc_remote_url()
        if remote_url:
            remote_base = Path(remote_url)
            remote_file = remote_base / "files" / "md5" / expected_hash[:2] / expected_hash[2:]
            if remote_file.is_file():
                result["dvc_remote_available"] = True

    # 4. Aggregate overall status
    if is_metrics_file:
        result["status"] = "METRICS_FILE_PRESENT" if workspace_exists else "METRICS_FILE_MISSING"
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


def verify_run_artifacts(
    context: ProjectContext,
    run: Any,
    tag_map: dict[str, str] | None = None,
    dvc_adapter: DVCAdapter | None = None,
) -> list[dict[str, Any]]:
    """Verify all mapped artifact hashes for a run against the workspace and DVC cache.

    Compatible with app.py verify_run_artifacts interface.
    """
    if tag_map is None:
        tag_map = WINE_DVC_TAG_MAP

    from ml_lineage.adapters.dvc import DVCAdapter
    if dvc_adapter is None:
        dvc_adapter = DVCAdapter(context)

    tags = run.data.tags
    lock_hashes = dvc_adapter.load_lock_hashes()
    results = []

    for tag_key, rel_path in tag_map.items():
        expected_hash = tags.get(tag_key, "")
        abs_path = context.resolve_path(rel_path)

        entry = {
            "artifact": rel_path,
            "expected_hash": expected_hash or "unavailable",
            "workspace_exists": abs_path.is_file(),
            "workspace_hash": None,
            "status": "UNVERIFIED",
            "lock_hash": lock_hashes.get(rel_path, "unavailable"),
        }

        if not expected_hash or expected_hash == "unavailable":
            entry["status"] = "NO_HASH_RECORDED"
            results.append(entry)
            continue

        if abs_path.is_file():
            curr_hash = compute_file_md5(abs_path)
            entry["workspace_hash"] = curr_hash
            if curr_hash == expected_hash:
                entry["status"] = "✅ VERIFIED"
            else:
                entry["status"] = "⚠️ HASH_MISMATCH"
        else:
            entry["status"] = "❌ MISSING"

        # Check DVC cache
        cache_file = dvc_adapter.get_cache_file(expected_hash)
        if cache_file.is_file():
            if entry["status"] != "✅ VERIFIED":
                entry["status"] += " (recoverable from cache)"

        results.append(entry)

    return results
