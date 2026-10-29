"""
ml_lineage.adapters.dvc
=======================
DVC adapter providing project-scoped access to dvc.lock hashes,
content-addressed cache files, remote storage configuration, and pipeline status.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ml_lineage.core.pipeline import PipelineDAG, parse_dvc_pipeline

if TYPE_CHECKING:
    from ml_lineage.core.context import ProjectContext


class DVCAdapter:
    """Provides project-scoped DVC operations and artifact storage resolution."""

    def __init__(self, context: ProjectContext):
        self.context = context
        self.root = context.root

    def get_pipeline_dag(self) -> PipelineDAG:
        """Parse dvc.yaml and dvc.lock to construct the project's pipeline DAG."""
        return parse_dvc_pipeline(self.context)

    def is_dvc_initialized(self) -> bool:
        """Check whether .dvc directory exists in the project root."""
        return self.context.dvc_dir.is_dir()

    def load_lock_hashes(self) -> dict[str, str]:
        """Parse dvc.lock to extract current stage output MD5 hashes.

        Returns:
            Mapping of relative output artifact path -> MD5 hash.
        """
        hashes: dict[str, str] = {}
        if not self.context.dvc_lock_path.is_file():
            return hashes

        try:
            with self.context.dvc_lock_path.open("r", encoding="utf-8") as f:
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

    def get_cache_file(self, md5_hash: str) -> Path:
        """Resolve the expected file path inside the project's DVC cache."""
        return self.context.dvc_cache_root / md5_hash[:2] / md5_hash[2:]

    def get_remote_file(self, md5_hash: str) -> Path | None:
        """Resolve the file path inside a local DVC remote storage directory, if configured."""
        remote_url = self.context.get_dvc_remote_url()
        if not remote_url:
            return None

        # Check if remote is a local path
        remote_base = Path(remote_url)
        if remote_base.exists() or os.path.isabs(remote_url):
            return remote_base / "files" / "md5" / md5_hash[:2] / md5_hash[2:]

        return None

    def find_artifact_source(self, md5_hash: str) -> tuple[Path | None, str]:
        """Locate an artifact by MD5 hash across the local DVC cache and configured remote.

        Returns:
            Tuple of (resolved_path, source_label) where source_label is one of:
            "dvc_cache", "dvc_remote", "not_found".
        """
        # 1. Check local DVC cache
        cache_file = self.get_cache_file(md5_hash)
        if cache_file.is_file():
            return cache_file, "dvc_cache"

        # 2. Check local DVC remote
        remote_file = self.get_remote_file(md5_hash)
        if remote_file and remote_file.is_file():
            return remote_file, "dvc_remote"

        return None, "not_found"

    def get_status(self, timeout: int = 30) -> str:
        """Run `dvc status` in project directory and return output."""
        if not self.is_dvc_initialized():
            return "DVC is not initialized in this project."
        try:
            exe = self.context.dvc_executable
            res = subprocess.run(
                [exe, "status"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            return res.stdout.strip() if res.returncode == 0 else f"Error: {res.stderr.strip()}"
        except Exception as e:
            return f"Error running dvc status: {e}"

    def get_remote_status(self, timeout: int = 30) -> str:
        """Run `dvc status -c` to check cache vs remote sync status."""
        if not self.is_dvc_initialized():
            return "DVC is not initialized in this project."
        try:
            exe = self.context.dvc_executable
            res = subprocess.run(
                [exe, "status", "-c"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            if res.returncode == 0:
                out = res.stdout.strip()
                return out if out else "Cache and remote storage are in sync."
            return f"DVC remote check returned code {res.returncode}: {res.stderr.strip() or res.stdout.strip()}"
        except Exception as e:
            return f"Could not check DVC remote: {e}"
