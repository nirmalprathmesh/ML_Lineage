"""
ml_lineage.adapters.mlflow
==========================
MLflow tracking adapter providing project-scoped run queries, inspection,
and comparison services.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)

# Attempt to import mlflow; if unavailable, defer error until used
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError:
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
import pandas as pd

from ml_lineage.wine import WINE_DVC_TAG_MAP

if TYPE_CHECKING:
    from ml_lineage.core.context import ProjectContext


class MLflowAdapter:
    """Provides project-scoped MLflow tracking queries and run analytics."""

    def __init__(self, context: ProjectContext):
        self.context = context
        self.tracking_uri = context.tracking_uri

    def get_client(self) -> MlflowClient:
        """Create a new MlflowClient bound to this project's tracking URI."""
        return MlflowClient(tracking_uri=self.tracking_uri)

    def search_experiments(self) -> list:
        """List all experiments in the project's tracking store."""
        client = self.get_client()
        return client.search_experiments()

    def fetch_all_runs(self, experiment_filter: str | None = None) -> list:
        """Retrieve runs from the tracking store, newest first.

        Args:
            experiment_filter: Optional experiment name or ID to filter by.
        """
        client = self.get_client()
        experiments = client.search_experiments()

        if experiment_filter:
            experiments = [
                e for e in experiments
                if e.name == experiment_filter or e.experiment_id == experiment_filter
            ]

        exp_ids = [e.experiment_id for e in experiments]
        if not exp_ids:
            return []

        return client.search_runs(exp_ids, order_by=["start_time DESC"])

    def resolve_run(self, run_id_prefix: str) -> Any | None:
        """Find an MLflow run by full ID or prefix across all experiments.

        Returns:
            The matched Run object if exactly one match exists, otherwise None.
        """
        client = self.get_client()
        experiments = client.search_experiments()
        exp_ids = [e.experiment_id for e in experiments]
        if not exp_ids:
            return None

        runs = client.search_runs(exp_ids)
        clean_prefix = run_id_prefix.strip().lower()
        matches = [r for r in runs if r.info.run_id.lower().startswith(clean_prefix)]
        if len(matches) == 1:
            return matches[0]
        return None

    def get_run_details(self, run: Any, tag_map: dict[str, str] | None = None) -> dict[str, Any]:
        """Extract a structured dict of run parameters, metrics, tags, and environment."""
        if tag_map is None:
            tag_map = WINE_DVC_TAG_MAP

        client = self.get_client()
        tags = run.data.tags
        params = run.data.params
        metrics = run.data.metrics

        try:
            experiment = client.get_experiment(run.info.experiment_id)
            exp_name = experiment.name
        except Exception:
            exp_name = run.info.experiment_id

        start_ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(run.info.start_time / 1000))
        end_ts = (
            time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(run.info.end_time / 1000))
            if run.info.end_time else "In Progress"
        )

        return {
            "run_id": run.info.run_id,
            "experiment_name": exp_name,
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
                for tag_key in tag_map
            },
        }

    def get_runs_dataframe(self, runs: list | None = None) -> pd.DataFrame:
        """Convert MLflow runs into a clean, typed pandas DataFrame."""
        if runs is None:
            runs = self.fetch_all_runs()

        cols = [
            "run_id", "short_id", "status", "experiment", "experiment_id",
            "accuracy", "precision", "recall", "f1_score", "duration_seconds",
            "start_time", "start_epoch_ms", "git_commit", "git_commit_full",
            "git_branch", "git_dirty", "model_type", "n_estimators", "max_depth"
        ]

        if not runs:
            return pd.DataFrame(columns=cols)

        client = self.get_client()
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

    def get_run_comparison_matrix(
        self,
        run_id_a: str,
        run_id_b: str,
        tag_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Generate side-by-side comparison matrix for two runs."""
        run_a = self.resolve_run(run_id_a)
        run_b = self.resolve_run(run_id_b)
        if not run_a or not run_b:
            return {"error": f"One or both runs ('{run_id_a}', '{run_id_b}') could not be resolved."}

        if tag_map is None:
            tag_map = WINE_DVC_TAG_MAP

        det_a = self.get_run_details(run_a, tag_map=tag_map)
        det_b = self.get_run_details(run_b, tag_map=tag_map)

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
        for tag_key, path in tag_map.items():
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
