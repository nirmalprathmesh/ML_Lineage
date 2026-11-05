"""
ml_lineage.core.recovery
========================
Phase 4: Safe, Project-Scoped Recovery Engine for ML Lineage Studio.

Provides pre-flight recovery planning, destination path traversal protection,
Git working-tree safety checks, collision detection, pre-overwrite backup creation,
integrity verification, transactional file restoration, and automated rollback.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_lineage.adapters.dvc import DVCAdapter
from ml_lineage.adapters.git import GitAdapter
from ml_lineage.adapters.mlflow import MLflowAdapter
from ml_lineage.core.hashing import compute_file_md5
from ml_lineage.core.lineage import find_recorded_artifact_hash
from ml_lineage.core.pipeline import parse_dvc_pipeline
from ml_lineage.wine import WINE_DVC_TAG_MAP

if TYPE_CHECKING:
    from ml_lineage.core.context import ProjectContext


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions & Shared Path Traversal Validator
# ─────────────────────────────────────────────────────────────────────────────
from ml_lineage.core.verification import (
    PathTraversalError,
    RecoveryError,
    validate_destination_path,
)


class BackupError(RecoveryError):
    """Raised when backup creation or backup verification fails."""
    pass


class RecoveryExecutionError(RecoveryError):
    """Raised when recovery file application fails."""
    def __init__(self, message: str, rollback_success: bool = True, rollback_errors: list[str] | None = None):
        super().__init__(message)
        self.rollback_success = rollback_success
        self.rollback_errors = rollback_errors or []


# ─────────────────────────────────────────────────────────────────────────────
# Path Normalization
# ─────────────────────────────────────────────────────────────────────────────
def normalize_rel_path(path_str: str) -> str:
    """Normalize a relative path string using forward slashes."""
    return os.path.normpath(path_str).replace("\\", "/").lstrip("/")


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ArtifactPlan:
    """Plan for a single artifact in the recovery operation."""

    rel_path: str
    dest_path: str
    expected_hash: str
    source_path: str | None = None
    source_type: str = "missing"  # 'workspace', 'dvc_cache', 'dvc_remote', 'missing'
    source_hash: str | None = None
    source_verified: bool = False
    dest_exists: bool = False
    dest_hash: str | None = None
    action: str = "BLOCKED"  # 'CREATE', 'REPLACE', 'SKIP_IDENTICAL', 'BLOCKED'
    conflict: bool = False
    dest_git_dirty: bool = False
    blocker_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryPlan:
    """Pre-flight recovery plan showing all planned actions, conflicts, and blockers."""

    target_project_path: str
    destination_root: str
    is_isolated_output: bool
    run_id: str
    run_status: str
    recorded_commit: str | None
    commit_exists_locally: bool
    git_is_dirty: bool
    is_git_repo: bool
    target_files_dirty: list[str] = field(default_factory=list)
    artifacts: list[ArtifactPlan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    can_recover: bool = False
    evidence_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["artifacts"] = [a.to_dict() for a in self.artifacts]
        return d


@dataclass
class RecoveryResult:
    """Outcome of a recovery execution including backup and rollback records."""

    success: bool
    run_id: str
    destination_root: str
    operations_applied: list[dict[str, Any]] = field(default_factory=list)
    backups_created: list[dict[str, Any]] = field(default_factory=list)
    rollback_occurred: bool = False
    rollback_success: bool = True
    rollback_errors: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Recovery Service
# ─────────────────────────────────────────────────────────────────────────────
class RecoveryService:
    """High-level recovery orchestrator bound to a target ProjectContext."""

    def __init__(self, context: ProjectContext, output_dir: str | Path | None = None):
        self.context = context
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.dest_root = self.output_dir if self.output_dir else self.context.root
        self.dvc_adapter = DVCAdapter(context)
        self.git_adapter = GitAdapter(context)
        self.mlflow_adapter = MLflowAdapter(context)

    def create_plan(self, run_id_or_prefix: str) -> RecoveryPlan:
        """Construct a complete pre-flight recovery plan without modifying any files.

        Validates run ID, Git state, artifact hashes, destination file conflicts,
        and ensures no path traversal or unsafe overwriting can occur.
        """
        run = self.mlflow_adapter.resolve_run(run_id_or_prefix)
        if not run:
            raise RecoveryError(f"No run found matching '{run_id_or_prefix}' or prefix is ambiguous.")

        run_id = run.info.run_id
        run_status = run.info.status
        tags = run.data.tags

        # 1. Git State Inspection
        is_git_repo = self.git_adapter.is_git_repository()
        recorded_commit = tags.get("git.commit_sha")
        commit_exists_locally = self.git_adapter.commit_exists(recorded_commit) if is_git_repo else False
        git_is_dirty = self.git_adapter.is_dirty() if is_git_repo else False

        warnings: list[str] = []
        blockers: list[str] = []

        if run_status != "FINISHED":
            warnings.append(f"Run status is '{run_status}', not 'FINISHED'. Artifacts may be partial.")

        if tags.get("git.is_dirty") == "True":
            warnings.append("Run was recorded when the working tree was dirty; exact reproduction from Git alone may fail.")

        if recorded_commit and is_git_repo and not commit_exists_locally:
            warnings.append(f"Recorded Git commit '{recorded_commit[:8]}' was not found in the local repository.")

        # 2. Gather artifacts mapped to this run
        # Read lineage manifest from MLflow artifacts if available
        lineage_manifest = None
        artifact_uri = run.info.artifact_uri
        if artifact_uri and artifact_uri.startswith("file:"):
            art_dir = artifact_uri.replace("file:", "").replace("/", os.sep).lstrip(os.sep)
            if sys.platform != "win32" and not art_dir.startswith("/"):
                art_dir = "/" + art_dir
            lineage_json = Path(art_dir) / "lineage" / "lineage.json"
            if lineage_json.is_file():
                try:
                    with lineage_json.open("r", encoding="utf-8") as f:
                        lineage_manifest = json.load(f)
                except Exception:
                    pass

        # Candidate artifact paths
        candidate_paths: list[str] = []

        # From DVC pipeline DAG outputs if dvc.yaml exists
        dag = parse_dvc_pipeline(self.context)
        if dag.has_dvc_yaml:
            for stage in dag.stages.values():
                for out in stage.outs:
                    candidate_paths.append(out.path)

        # From Wine tag map (backward compatibility)
        for _, rel in WINE_DVC_TAG_MAP.items():
            candidate_paths.append(rel)

        # From run tags directly (dvc.hash.*, artifact.hash.*, hash.*)
        for tag_key in tags.keys():
            if tag_key in WINE_DVC_TAG_MAP:
                continue
            for prefix in ("dvc.hash.", "artifact.hash.", "hash."):
                if tag_key.startswith(prefix):
                    rel = tag_key[len(prefix):]
                    if "/" in rel or "\\" in rel or "." in rel:
                        candidate_paths.append(rel)

        # From lineage manifest if present
        if lineage_manifest and isinstance(lineage_manifest.get("dvc_hashes"), dict):
            params = lineage_manifest.get("parameters", {})
            for key in lineage_manifest["dvc_hashes"].keys():
                if f"dvc.hash.{key}" in WINE_DVC_TAG_MAP:
                    candidate_paths.append(WINE_DVC_TAG_MAP[f"dvc.hash.{key}"])
                elif f"{key}_path" in params:
                    candidate_paths.append(params[f"{key}_path"])
                elif "/" in key or "\\" in key or "." in key:
                    candidate_paths.append(key)

        # Deduplicate candidates
        seen_norm: set[str] = set()
        artifact_plans: list[ArtifactPlan] = []
        target_files_dirty: list[str] = []

        is_isolated = self.output_dir is not None

        for raw_path in candidate_paths:
            norm_path = normalize_rel_path(raw_path)
            if not norm_path or norm_path in seen_norm:
                continue
            seen_norm.add(norm_path)

            expected_hash, _ = find_recorded_artifact_hash(norm_path, tags, lineage_manifest)
            if not expected_hash or expected_hash == "unavailable":
                continue

            # Validate destination path against path traversal
            try:
                dest_path = validate_destination_path(self.dest_root, norm_path)
            except PathTraversalError as e:
                plan_item = ArtifactPlan(
                    rel_path=norm_path,
                    dest_path="",
                    expected_hash=expected_hash,
                    action="BLOCKED",
                    blocker_reason=str(e),
                )
                artifact_plans.append(plan_item)
                blockers.append(str(e))
                continue

            # Check source artifact availability in DVC cache or remote
            src_path, src_label = self.dvc_adapter.find_artifact_source(expected_hash)
            source_type = "missing"
            source_verified = False
            src_hash = None
            blocker_reason = None

            if src_path and src_path.is_file():
                if "cache" in src_label.lower():
                    source_type = "dvc_cache"
                elif "remote" in src_label.lower():
                    source_type = "dvc_remote"
                else:
                    source_type = "workspace"

                try:
                    src_hash = compute_file_md5(src_path)
                    if src_hash == expected_hash:
                        source_verified = True
                    else:
                        blocker_reason = f"Corrupted source in {src_label} (expected {expected_hash}, got {src_hash})"
                        blockers.append(f"Artifact '{norm_path}': {blocker_reason}")
                except Exception as e:
                    blocker_reason = f"Failed to read source {src_path}: {e}"
                    blockers.append(f"Artifact '{norm_path}': {blocker_reason}")
            else:
                blocker_reason = f"Source artifact not found in DVC cache or remote storage"
                blockers.append(f"Artifact '{norm_path}': {blocker_reason}")

            # Check destination file state
            dest_exists = dest_path.is_file()
            dest_hash = None
            conflict = False
            dest_git_dirty = False

            if dest_exists:
                try:
                    dest_hash = compute_file_md5(dest_path)
                except Exception:
                    dest_hash = None

                if dest_hash == expected_hash:
                    action = "SKIP_IDENTICAL"
                else:
                    action = "REPLACE"
                    conflict = True

                    # Git safety: If recovering in-place into a Git repo, check if file is dirty
                    if not is_isolated and is_git_repo:
                        if self.git_adapter.is_file_dirty(norm_path):
                            dest_git_dirty = True
                            target_files_dirty.append(norm_path)
                            b_msg = (
                                f"Target file '{norm_path}' has uncommitted changes in Git. "
                                f"Commit, stash, or use --output to recover to an isolated directory."
                            )
                            blockers.append(b_msg)
                            blocker_reason = b_msg
            else:
                action = "CREATE"

            if blocker_reason and action != "SKIP_IDENTICAL":
                action = "BLOCKED"

            artifact_plans.append(
                ArtifactPlan(
                    rel_path=norm_path,
                    dest_path=str(dest_path),
                    expected_hash=expected_hash,
                    source_path=str(src_path) if src_path else None,
                    source_type=source_type,
                    source_hash=src_hash,
                    source_verified=source_verified,
                    dest_exists=dest_exists,
                    dest_hash=dest_hash,
                    action=action,
                    conflict=conflict,
                    dest_git_dirty=dest_git_dirty,
                    blocker_reason=blocker_reason,
                )
            )

        # 3. Overall Feasibility
        can_recover = len(blockers) == 0 and len(artifact_plans) > 0

        # 4. Evidence Summary
        all_sources_ok = all(a.source_verified for a in artifact_plans if a.action in ("CREATE", "REPLACE"))
        evidence_summary = {
            "metadata": "OBSERVED" if recorded_commit or tags else "UNAVAILABLE",
            "artifacts": "VERIFIED" if all_sources_ok else "UNVERIFIED",
            "git_commit": "OBSERVED_LOCAL" if commit_exists_locally else ("OBSERVED_REMOTE" if recorded_commit else "UNAVAILABLE"),
            "environment": "OBSERVED" if any(k.startswith("env.") for k in tags) else "UNAVAILABLE",
            "reproducibility": "PARTIAL (Artifact hashes verified; runtime behavior requires matching environment)",
        }

        return RecoveryPlan(
            target_project_path=str(self.context.root),
            destination_root=str(self.dest_root),
            is_isolated_output=is_isolated,
            run_id=run_id,
            run_status=run_status,
            recorded_commit=recorded_commit,
            commit_exists_locally=commit_exists_locally,
            git_is_dirty=git_is_dirty,
            is_git_repo=is_git_repo,
            target_files_dirty=target_files_dirty,
            artifacts=artifact_plans,
            warnings=warnings,
            blockers=blockers,
            can_recover=can_recover,
            evidence_summary=evidence_summary,
        )

    def execute_recovery(
        self,
        plan: RecoveryPlan,
        backup_dir_override: Path | None = None,
    ) -> RecoveryResult:
        """Execute transactional recovery using verified pre-flight backups and rollback.

        Args:
            plan: Pre-flight RecoveryPlan from create_plan.
            backup_dir_override: Optional path to store backups.

        Returns:
            RecoveryResult containing applied operations, backups, and manifest path.

        Raises:
            RecoveryError: If the plan has blockers or validation fails.
            BackupError: If backup creation fails.
            RecoveryExecutionError: If file operations fail mid-execution (triggers rollback).
        """
        if not plan.can_recover:
            raise RecoveryError(f"Cannot execute recovery; plan has {len(plan.blockers)} blocker(s): {'; '.join(plan.blockers)}")

        dest_root = Path(plan.destination_root)
        replace_items = [a for a in plan.artifacts if a.action == "REPLACE"]
        create_items = [a for a in plan.artifacts if a.action == "CREATE"]

        # 1. Create Pre-flight Backups for replaced files
        backups_created: list[dict[str, Any]] = []
        backup_dir: Path | None = None

        if replace_items:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = backup_dir_override or (dest_root / ".ml-lineage" / "backups" / f"recover_{plan.run_id[:8]}_{timestamp_str}")

            if backup_dir.exists():
                # Avoid silent overwrite of existing backup directory
                backup_dir = dest_root / ".ml-lineage" / "backups" / f"recover_{plan.run_id[:8]}_{timestamp_str}_{os.getpid()}"

            backup_dir.mkdir(parents=True, exist_ok=True)

            for item in replace_items:
                target_dest = Path(item.dest_path)
                backup_file = backup_dir / item.rel_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)

                if backup_file.exists():
                    raise BackupError(f"Backup file destination already exists: {backup_file}")

                try:
                    shutil.copy2(target_dest, backup_file)
                    backup_hash = compute_file_md5(backup_file)
                    if backup_hash != item.dest_hash:
                        raise BackupError(
                            f"Backup verification failed for {item.rel_path}! Expected {item.dest_hash}, got {backup_hash}."
                        )
                    backups_created.append({
                        "rel_path": item.rel_path,
                        "dest_path": str(target_dest),
                        "backup_path": str(backup_file),
                        "original_hash": item.dest_hash,
                        "timestamp": time.time(),
                    })
                except Exception as e:
                    # Clean up partial backup directory and abort
                    shutil.rmtree(backup_dir, ignore_errors=True)
                    raise BackupError(f"Failed to create verified backup for '{item.rel_path}': {e}") from e

        # 2. Transactional Restoration
        applied_operations: list[dict[str, Any]] = []
        rollback_errors: list[str] = []
        recovery_succeeded = False

        try:
            for item in plan.artifacts:
                if item.action == "SKIP_IDENTICAL":
                    continue

                if item.action in ("CREATE", "REPLACE"):
                    src = Path(item.source_path)
                    dst = Path(item.dest_path)
                    backup_rec = next((b for b in backups_created if b["rel_path"] == item.rel_path), None)

                    current_op = {
                        "rel_path": item.rel_path,
                        "dest_path": str(dst),
                        "action": item.action,
                        "expected_hash": item.expected_hash,
                        "backup_path": backup_rec["backup_path"] if backup_rec else None,
                        "original_hash": backup_rec["original_hash"] if backup_rec else None,
                    }
                    applied_operations.append(current_op)

                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

                    # Post-restore hash verification
                    restored_hash = compute_file_md5(dst)
                    if restored_hash != item.expected_hash:
                        raise RecoveryExecutionError(
                            f"Post-restore verification failed for '{item.rel_path}'. "
                            f"Expected {item.expected_hash}, got {restored_hash}."
                        )
                    current_op["restored_hash"] = restored_hash

            recovery_succeeded = True

        except Exception as exc:
            # 3. Rollback
            for op in reversed(applied_operations):
                try:
                    dst = Path(op["dest_path"])
                    if op["action"] == "CREATE":
                        if dst.exists():
                            dst.unlink()
                    elif op["action"] == "REPLACE":
                        b_path = Path(op["backup_path"])
                        if b_path.is_file():
                            shutil.copy2(b_path, dst)
                            restored_hash = compute_file_md5(dst)
                            if restored_hash != op["original_hash"]:
                                rollback_errors.append(
                                    f"Rollback hash mismatch on {op['rel_path']}: expected {op['original_hash']}, got {restored_hash}"
                                )
                except Exception as rbe:
                    rollback_errors.append(f"Rollback error on {op['rel_path']}: {rbe}")

            rollback_ok = len(rollback_errors) == 0

            # Record failure audit log
            self._write_audit_manifest(
                dest_root=dest_root,
                backup_dir=backup_dir,
                run_id=plan.run_id,
                status="ROLLED_BACK" if rollback_ok else "ROLLBACK_FAILED",
                applied=applied_operations,
                backups=backups_created,
                plan=plan,
                error=str(exc),
                rollback_errors=rollback_errors,
            )

            raise RecoveryExecutionError(
                message=f"Recovery failed during file restoration: {exc}",
                rollback_success=rollback_ok,
                rollback_errors=rollback_errors,
            ) from exc

        # 4. Write Success Audit Manifest
        manifest_path = self._write_audit_manifest(
            dest_root=dest_root,
            backup_dir=backup_dir,
            run_id=plan.run_id,
            status="SUCCESS",
            applied=applied_operations,
            backups=backups_created,
            plan=plan,
        )

        return RecoveryResult(
            success=True,
            run_id=plan.run_id,
            destination_root=str(dest_root),
            operations_applied=applied_operations,
            backups_created=backups_created,
            rollback_occurred=False,
            rollback_success=True,
            rollback_errors=[],
            manifest_path=str(manifest_path) if manifest_path else None,
            message=f"Successfully recovered {len(applied_operations)} artifact(s) for run {plan.run_id[:8]}.",
        )

    def _write_audit_manifest(
        self,
        dest_root: Path,
        backup_dir: Path | None,
        run_id: str,
        status: str,
        applied: list[dict[str, Any]],
        backups: list[dict[str, Any]],
        plan: RecoveryPlan,
        error: str | None = None,
        rollback_errors: list[str] | None = None,
    ) -> Path | None:
        """Write a recovery audit manifest in the project tracking dir and backup folder."""
        manifest_dir = dest_root / ".ml-lineage"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_dir / "recovery_manifest.json"

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": run_id,
            "status": status,
            "destination_root": str(dest_root),
            "target_project_path": plan.target_project_path,
            "recorded_git_commit": plan.recorded_commit,
            "operations_applied": applied,
            "backups_created": backups,
            "error": error,
            "rollback_errors": rollback_errors or [],
            "evidence_summary": plan.evidence_summary,
        }

        try:
            with manifest_file.open("w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

            if backup_dir and backup_dir.is_dir():
                backup_manifest = backup_dir / "recovery_manifest.json"
                with backup_manifest.open("w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)

            return manifest_file
        except Exception:
            return None
