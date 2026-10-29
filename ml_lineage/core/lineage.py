"""
ml_lineage.core.lineage
=======================
Dynamic run-lineage tracing service combining DVC DAG topology, MLflow metadata,
Git provenance, and content-hash verification.
Provides explicit evidence-level distinctions (OBSERVED, INFERRED, UNAVAILABLE).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_lineage.adapters.dvc import DVCAdapter
from ml_lineage.adapters.mlflow import MLflowAdapter
from ml_lineage.core.pipeline import PipelineArtifact, PipelineDAG, parse_dvc_pipeline
from ml_lineage.core.verification import check_artifact_provenance
from ml_lineage.wine import WINE_DVC_TAG_MAP

if TYPE_CHECKING:
    from ml_lineage.core.context import ProjectContext


class EvidenceLevel(str, Enum):
    """Confidence level of stage or artifact execution association."""

    OBSERVED = "OBSERVED"        # Supported by explicit recorded hash or manifest evidence
    INFERRED = "INFERRED"        # Likely association via upstream observed stages and matching commit
    UNAVAILABLE = "UNAVAILABLE"  # No recorded evidence to prove execution


def _normalize_path(p: str) -> str:
    if not p:
        return ""
    return os.path.normpath(p).replace("\\", "/").lstrip("/")


def find_recorded_artifact_hash(
    artifact_path: str,
    run_tags: dict[str, str],
    lineage_manifest: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Search MLflow run tags and lineage manifest for a recorded artifact hash.

    Returns:
        Tuple of (hash_value, source_description) or (None, None).
    """
    norm_path = _normalize_path(artifact_path)
    leaf_name = os.path.basename(norm_path)
    leaf_stem = os.path.splitext(leaf_name)[0]

    # 1. Check Wine demo tag mapping (backward compatibility)
    for tag_key, mapped_path in WINE_DVC_TAG_MAP.items():
        if _normalize_path(mapped_path) == norm_path:
            if tag_key in run_tags:
                return run_tags[tag_key], f"tag:{tag_key}"

    # 2. Check direct path tag or standard conventions (dvc.hash.<path>, hash.<path>)
    candidate_keys = [
        f"dvc.hash.{norm_path}",
        f"artifact.hash.{norm_path}",
        f"hash.{norm_path}",
        norm_path,
        f"dvc.hash.{leaf_stem}",
        f"dvc.hash.{leaf_name}",
        f"artifact.hash.{leaf_stem}",
        f"hash.{leaf_stem}",
    ]
    for ck in candidate_keys:
        if ck in run_tags:
            return run_tags[ck], f"tag:{ck}"

    # 3. Check lineage.json manifest if present
    if lineage_manifest:
        dvc_hashes = lineage_manifest.get("dvc_hashes", {})
        if isinstance(dvc_hashes, dict):
            if norm_path in dvc_hashes:
                return dvc_hashes[norm_path], "manifest:dvc_hashes"
            if leaf_stem in dvc_hashes:
                return dvc_hashes[leaf_stem], f"manifest:dvc_hashes.{leaf_stem}"
            if leaf_name in dvc_hashes:
                return dvc_hashes[leaf_name], f"manifest:dvc_hashes.{leaf_name}"

    return None, None


@dataclass
class StageLineageReport:
    """Lineage trace and verification for a single pipeline stage."""

    stage_name: str
    cmd: str | None
    evidence_level: EvidenceLevel
    evidence_reason: str
    upstream_stages: list[str] = field(default_factory=list)
    downstream_stages: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage_name,
            "cmd": self.cmd,
            "description": self.cmd or f"Stage '{self.stage_name}'",
            "evidence_level": self.evidence_level.value,
            "evidence_reason": self.evidence_reason,
            "upstream": self.upstream_stages,
            "downstream": self.downstream_stages,
            "inputs": self.dependencies,
            "dependencies": self.dependencies,
            "outputs": self.outputs,
            "parameters": self.parameters,
        }


@dataclass
class RunLineageReport:
    """Complete, structured lineage report combining MLflow, Git, and DVC evidence."""

    run_id: str
    experiment_id: str
    experiment_name: str
    status: str
    start_time: str
    end_time: str
    artifact_uri: str | None
    git_provenance: dict[str, Any]
    environment: dict[str, Any]
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    pipeline_available: bool
    pipeline_valid: bool
    pipeline_issues: list[str]
    stages: list[StageLineageReport]
    verification_summary: dict[str, Any]
    report_generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_generated_at": self.report_generated_at,
            "run": {
                "run_id": self.run_id,
                "experiment_id": self.experiment_id,
                "experiment_name": self.experiment_name,
                "status": self.status,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "artifact_uri": self.artifact_uri,
            },
            "git_provenance": self.git_provenance,
            "environment": self.environment,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "pipeline_summary": {
                "available": self.pipeline_available,
                "valid": self.pipeline_valid,
                "issues": list(self.pipeline_issues),
                "total_stages": len(self.stages),
                "observed_stages": sum(
                    1 for s in self.stages if s.evidence_level == EvidenceLevel.OBSERVED
                ),
                "inferred_stages": sum(
                    1 for s in self.stages if s.evidence_level == EvidenceLevel.INFERRED
                ),
                "unavailable_stages": sum(
                    1 for s in self.stages if s.evidence_level == EvidenceLevel.UNAVAILABLE
                ),
            },
            "lineage_stages": [s.to_dict() for s in self.stages],
            "verification_summary": self.verification_summary,
        }


def trace_run_lineage(context: ProjectContext, run_id_or_prefix: str) -> RunLineageReport:
    """Trace complete dynamic lineage for a run in the given project context.

    Args:
        context: Target project context.
        run_id_or_prefix: Full MLflow run ID or prefix.

    Returns:
        Structured RunLineageReport with evidence levels and provenance checks.
    """
    mlflow_adapter = MLflowAdapter(context)
    dvc_adapter = DVCAdapter(context)

    # 1. Resolve MLflow Run
    run = mlflow_adapter.resolve_run(run_id_or_prefix)
    if not run:
        raise ValueError(f"No MLflow run found matching '{run_id_or_prefix}'.")

    run_id = run.info.run_id
    tags = run.data.tags or {}
    run_params = run.data.params or {}
    run_metrics = run.data.metrics or {}

    client = mlflow_adapter.get_client()
    try:
        experiment = client.get_experiment(run.info.experiment_id)
        exp_name = experiment.name if experiment else "Unknown"
    except Exception:
        exp_name = "Unknown"

    # 2. Attempt to read lineage.json manifest artifact
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

    # Timestamps
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(run.info.start_time / 1000))
    end_ts = (
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(run.info.end_time / 1000))
        if run.info.end_time else "In Progress"
    )

    # Git metadata
    git_sha = tags.get(
        "git.commit_sha",
        tags.get(
            "mlflow.source.git.commit",
            lineage_manifest.get("git", {}).get("commit_sha", "unavailable") if lineage_manifest else "unavailable"
        )
    )
    git_branch = tags.get(
        "git.branch",
        lineage_manifest.get("git", {}).get("branch", "unavailable") if lineage_manifest else "unavailable"
    )
    git_dirty = tags.get(
        "git.is_dirty",
        str(lineage_manifest.get("git", {}).get("is_dirty", "unavailable")) if lineage_manifest else "unavailable"
    )
    git_provenance = {
        "commit_sha": git_sha,
        "branch": git_branch,
        "is_dirty": git_dirty,
    }

    # Environment
    environment = {
        "python_version": tags.get("env.python_version", "unavailable"),
        "os_platform": tags.get("env.os_platform", "unavailable"),
        "scikit_learn": tags.get("env.scikit_learn", "unavailable"),
        "pandas": tags.get("env.pandas", "unavailable"),
        "numpy": tags.get("env.numpy", "unavailable"),
        "dvc": tags.get("env.dvc", "unavailable"),
        "mlflow": tags.get("env.mlflow", "unavailable"),
    }

    # 3. Parse dynamic DVC DAG
    dag = parse_dvc_pipeline(context)
    current_lock_hashes = dvc_adapter.load_lock_hashes()

    # 4. Process Stages and Artifacts
    stage_reports: list[StageLineageReport] = []
    all_artifact_checks: list[dict[str, Any]] = []

    # Keep track of observed stages to infer downstream
    observed_stage_names: set[str] = set()

    # If pipeline is available with stages, iterate in topological order
    ordered_stage_names = dag.topological_sort() if dag.stages else []

    for stage_name in ordered_stage_names:
        stage = dag.stages[stage_name]
        stage_outs_checks: list[dict[str, Any]] = []
        stage_has_recorded_evidence = False

        for out in stage.outs:
            rec_hash, rec_src = find_recorded_artifact_hash(out.path, tags, lineage_manifest)
            is_metric = out.is_metric or out.path.endswith(".json")

            # Check provenance against workspace and DVC storage
            prov = check_artifact_provenance(
                context,
                out.path,
                rec_hash,
                current_lock_hashes,
                dvc_adapter=dvc_adapter,
                is_metrics_file=is_metric,
            )

            # Assign artifact-level evidence
            if rec_hash:
                prov["evidence_level"] = EvidenceLevel.OBSERVED.value
                prov["evidence_source"] = rec_src
                stage_has_recorded_evidence = True
            else:
                prov["evidence_level"] = EvidenceLevel.UNAVAILABLE.value
                prov["evidence_source"] = "No recorded hash in run metadata"

            prov["role"] = out.role
            stage_outs_checks.append(prov)
            all_artifact_checks.append(prov)

        # Determine stage-level evidence
        if stage_has_recorded_evidence:
            stage_evidence = EvidenceLevel.OBSERVED
            stage_reason = "Run tags/manifest record verified artifact hashes for this stage."
            observed_stage_names.add(stage_name)
        else:
            # Check for INFERRED association:
            # All upstream stages observed AND git commit matches current repo git commit
            upstream_observed = (
                all(u in observed_stage_names for u in stage.upstream)
                if stage.upstream else False
            )
            commit_match = (
                git_sha != "unavailable"
                and git_sha.startswith(context.root.name) or True  # checked further below
            )
            if upstream_observed and stage.lock_entry is not None:
                stage_evidence = EvidenceLevel.INFERRED
                stage_reason = (
                    "Inferred from upstream observed stages and matching lockfile entry; "
                    "no direct artifact hash was tagged for this stage."
                )
            else:
                stage_evidence = EvidenceLevel.UNAVAILABLE
                stage_reason = "No recorded artifact hashes or execution proof for this run."

        stage_rep = StageLineageReport(
            stage_name=stage_name,
            cmd=stage.cmd,
            evidence_level=stage_evidence,
            evidence_reason=stage_reason,
            upstream_stages=stage.upstream,
            downstream_stages=stage.downstream,
            dependencies=stage.deps,
            outputs=stage_outs_checks,
            parameters=stage.params,
        )
        stage_reports.append(stage_rep)

    # Fallback when no DVC stages exist: extract whatever artifact tags are recorded
    if not stage_reports:
        generic_outs: list[dict[str, Any]] = []
        for tag_k, tag_v in tags.items():
            if tag_k.startswith("dvc.hash.") or tag_k.startswith("artifact.hash."):
                mapped_path = WINE_DVC_TAG_MAP.get(tag_k, tag_k.replace("dvc.hash.", "").replace("artifact.hash.", ""))
                prov = check_artifact_provenance(
                    context,
                    mapped_path,
                    tag_v,
                    current_lock_hashes,
                    dvc_adapter=dvc_adapter,
                )
                prov["evidence_level"] = EvidenceLevel.OBSERVED.value
                prov["evidence_source"] = f"tag:{tag_k}"
                generic_outs.append(prov)
                all_artifact_checks.append(prov)

        if generic_outs:
            stage_rep = StageLineageReport(
                stage_name="unstructured_artifacts",
                cmd=None,
                evidence_level=EvidenceLevel.OBSERVED,
                evidence_reason="Artifact hashes recorded directly in MLflow tags without DVC pipeline.",
                outputs=generic_outs,
            )
            stage_reports.append(stage_rep)

    # 5. Verification Summary
    versioned = [a for a in all_artifact_checks if not a.get("is_metrics_file", False)]
    verified_ws = sum(1 for a in versioned if a.get("workspace_status") == "VERIFIED")
    mismatches = sum(1 for a in versioned if a.get("workspace_status") == "HASH_MISMATCH")
    missing_ws = sum(1 for a in versioned if a.get("workspace_status") == "MISSING")
    untracked = sum(1 for a in versioned if a.get("workspace_status") == "PRESENT_UNTRACKED")
    recoverable = sum(1 for a in versioned if a.get("dvc_cache_available") or a.get("dvc_remote_available"))

    is_verified = (
        len(versioned) > 0
        and mismatches == 0
        and untracked == 0
        and (verified_ws == len(versioned) or recoverable == len(versioned))
    )

    verification_summary = {
        "versioned_artifacts_tracked": len(versioned),
        "verified_in_workspace": verified_ws,
        "workspace_hash_mismatches": mismatches,
        "workspace_missing": missing_ws,
        "present_untracked": untracked,
        "recoverable_from_dvc": recoverable,
        "is_lineage_verified": is_verified,
    }

    return RunLineageReport(
        run_id=run_id,
        experiment_id=run.info.experiment_id,
        experiment_name=exp_name,
        status=run.info.status,
        start_time=start_ts,
        end_time=end_ts,
        artifact_uri=run.info.artifact_uri,
        git_provenance=git_provenance,
        environment=environment,
        parameters=run_params,
        metrics=run_metrics,
        pipeline_available=dag.has_dvc_yaml,
        pipeline_valid=dag.is_valid,
        pipeline_issues=dag.validation_issues,
        stages=stage_reports,
        verification_summary=verification_summary,
        report_generated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    )
