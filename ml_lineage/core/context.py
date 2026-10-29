"""
ml_lineage.core.context
=======================
Project context representing an explicitly resolved target ML project.
Provides path-independent access to configuration, Git, DVC, and MLflow locations.
"""

from __future__ import annotations

import configparser
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


class ProjectContext:
    """Encapsulates the root path, configuration, and environment of a target ML project.

    Ensures that ML Lineage Studio operates on explicitly resolved project directories
    rather than relying on hardcoded paths or module-relative paths.
    """

    def __init__(
        self,
        root: str | Path,
        params_filename: str = "params.yaml",
        dvc_lock_filename: str = "dvc.lock",
        dvc_yaml_filename: str = "dvc.yaml",
        custom_tracking_uri: str | None = None,
    ):
        self.root: Path = Path(root).resolve()
        self.params_path: Path = self.root / params_filename
        self.dvc_lock_path: Path = self.root / dvc_lock_filename
        self.dvc_yaml_path: Path = self.root / dvc_yaml_filename
        self.dvc_dir: Path = self.root / ".dvc"
        self.dvc_config_path: Path = self.dvc_dir / "config"
        self.dvc_cache_root: Path = self.dvc_dir / "cache" / "files" / "md5"
        self._custom_tracking_uri = custom_tracking_uri
        self._cached_params: dict[str, Any] | None = None

    @classmethod
    def from_path(
        cls,
        target_path: str | Path | None = None,
        search_parents: bool = True,
    ) -> ProjectContext:
        """Resolve a ProjectContext from an explicit path or search upwards from CWD.

        Args:
            target_path: Explicit directory path. If None, uses Path.cwd().
            search_parents: If True and target is not immediately identifiable as a project,
                           traverse parents looking for markers (.git, params.yaml, dvc.yaml).

        Returns:
            ProjectContext bound to the resolved directory.
        """
        if target_path is not None:
            resolved = Path(target_path).resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Target project path does not exist: {resolved}")
            return cls(resolved)

        # Start from current working directory
        current = Path.cwd().resolve()
        markers = (".git", "params.yaml", "dvc.yaml", ".dvc", ".ml-lineage.yaml")

        if search_parents:
            probe = current
            while probe != probe.parent:
                if any((probe / marker).exists() for marker in markers):
                    return cls(probe)
                probe = probe.parent

        return cls(current)

    def resolve_path(self, rel_or_abs: str | Path) -> Path:
        """Resolve a relative or absolute path against this project's root."""
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return (self.root / p).resolve()

    @property
    def params(self) -> dict[str, Any]:
        """Load and cache project parameters from params.yaml if present."""
        if self._cached_params is None:
            if self.params_path.is_file():
                try:
                    with self.params_path.open("r", encoding="utf-8") as f:
                        self._cached_params = yaml.safe_load(f) or {}
                except Exception:
                    self._cached_params = {}
            else:
                self._cached_params = {}
        return self._cached_params

    def reload_params(self) -> dict[str, Any]:
        """Force reloading parameters from disk."""
        self._cached_params = None
        return self.params

    @property
    def tracking_uri(self) -> str:
        """Resolve absolute MLflow tracking URI for this project.

        Resolution precedence:
        1. Explicitly provided custom_tracking_uri
        2. params.yaml -> mlflow.tracking_uri (project-scoped configuration)
        3. MLFLOW_TRACKING_URI environment variable (if not configured in project)
        4. Fallback: sqlite:///<project_root>/mlflow.db
        """
        if self._custom_tracking_uri:
            return self._custom_tracking_uri

        project_raw = self.params.get("mlflow", {}).get("tracking_uri")
        if project_raw:
            if project_raw.startswith("sqlite:///"):
                db_rel = project_raw[10:]
                if not os.path.isabs(db_rel):
                    abs_db = str(self.resolve_path(db_rel))
                    return f"sqlite:///{abs_db}"
            return project_raw

        env_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if env_uri:
            return env_uri

        # Default fallback: sqlite:///<project_root>/mlflow.db
        abs_db = str(self.resolve_path("mlflow.db"))
        return f"sqlite:///{abs_db}"

    @property
    def experiment_name(self) -> str:
        """Return the configured MLflow experiment name."""
        return self.params.get("mlflow", {}).get("experiment_name", "ML_Lineage_Experiments")

    def get_dvc_remote_url(self, remote_name: str | None = None) -> str | None:
        """Read the DVC remote URL from .dvc/config.

        Args:
            remote_name: Specific remote name. If None, reads default or first remote.

        Returns:
            Configured remote URL string or None.
        """
        if not self.dvc_config_path.is_file():
            return None

        try:
            cp = configparser.ConfigParser()
            cp.read(self.dvc_config_path, encoding="utf-8")

            # If remote_name is specified, look for ['remote "name"']
            if remote_name:
                key1 = f"'remote \"{remote_name}\"'"
                key2 = f'remote "{remote_name}"'
                for k in (key1, key2):
                    if cp.has_section(k):
                        return cp[k].get("url")
                return None

            # Otherwise find the default remote or first remote
            target_remote = None
            if cp.has_section("core") and "remote" in cp["core"]:
                target_remote = cp["core"]["remote"]

            for section in cp.sections():
                if section.startswith("'remote \"") or section.startswith('remote "'):
                    # Check if matches target remote or is the first remote encountered
                    if target_remote:
                        clean_sec = section.strip("'\"").replace("remote ", "").strip('"')
                        if clean_sec == target_remote:
                            return cp[section].get("url")
                    else:
                        return cp[section].get("url")
        except Exception:
            return None
        return None

    @property
    def python_executable(self) -> str:
        """Find the virtualenv Python interpreter for this project or fallback to sys.executable."""
        candidates = [
            self.root / ".venv" / "Scripts" / "python.exe",
            self.root / ".venv" / "bin" / "python",
            self.root / "venv" / "Scripts" / "python.exe",
            self.root / "venv" / "bin" / "python",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        return sys.executable

    @property
    def dvc_executable(self) -> str:
        """Find the DVC executable in this project's venv or on PATH."""
        candidates = [
            self.root / ".venv" / "Scripts" / "dvc.exe",
            self.root / ".venv" / "bin" / "dvc",
            self.root / "venv" / "Scripts" / "dvc.exe",
            self.root / "venv" / "bin" / "dvc",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)

        which_dvc = shutil.which("dvc")
        if which_dvc:
            return which_dvc
        return "dvc"

    def to_dict(self) -> dict[str, Any]:
        """Return a structured summary of the resolved project context."""
        return {
            "root": str(self.root),
            "params_path": str(self.params_path),
            "params_exists": self.params_path.is_file(),
            "dvc_lock_path": str(self.dvc_lock_path),
            "dvc_lock_exists": self.dvc_lock_path.is_file(),
            "dvc_config_path": str(self.dvc_config_path),
            "dvc_config_exists": self.dvc_config_path.is_file(),
            "dvc_remote_url": self.get_dvc_remote_url(),
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "python_executable": self.python_executable,
            "dvc_executable": self.dvc_executable,
        }

    def __repr__(self) -> str:
        return f"ProjectContext(root={self.root})"
