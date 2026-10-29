"""
ml_lineage.core.registry
========================
Multi-project registry service managing persistent project registrations,
aliases, and configuration paths.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any


class RegistryError(Exception):
    """Base exception for project registry operations."""
    pass


class RegistryConflictError(RegistryError):
    """Raised when registering an alias or path that conflicts with an existing entry."""
    pass


class ProjectRegistry:
    """Manages persistent project registrations across multiple ML repositories.

    Default storage: ~/.ml-lineage/registry.json
    Can be overridden via ML_LINEAGE_REGISTRY_PATH or explicit constructor argument.
    """

    DEFAULT_DIR = Path.home() / ".ml-lineage"
    DEFAULT_FILE = DEFAULT_DIR / "registry.json"

    def __init__(self, registry_file: str | Path | None = None):
        if registry_file is not None:
            self.registry_file = Path(registry_file).resolve()
        else:
            env_path = os.environ.get("ML_LINEAGE_REGISTRY_PATH")
            if env_path:
                self.registry_file = Path(env_path).resolve()
            else:
                self.registry_file = self.DEFAULT_FILE

    def _ensure_parent_dir(self) -> None:
        """Create parent directory for registry file if needed."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, Any]]:
        """Load and parse the registry file.

        Returns:
            Dictionary mapping alias -> project registration metadata.

        Raises:
            RegistryError: If the registry file exists but contains invalid JSON.
        """
        if not self.registry_file.exists():
            return {}

        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise RegistryError(f"Registry root must be a JSON object, got {type(data).__name__}")
                return data
        except json.JSONDecodeError as jde:
            raise RegistryError(f"Registry file at '{self.registry_file}' contains invalid JSON: {jde}")
        except PermissionError as pe:
            raise RegistryError(f"Permission denied reading registry at '{self.registry_file}': {pe}")
        except Exception as e:
            raise RegistryError(f"Failed to read registry at '{self.registry_file}': {e}")

    def save(self, data: dict[str, dict[str, Any]]) -> None:
        """Safely persist registry data to disk using atomic replacement."""
        self._ensure_parent_dir()

        try:
            # Write to a temporary file in the same directory then rename (atomic on POSIX/Windows)
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(self.registry_file.parent),
                delete=False,
                encoding="utf-8",
            )
            with temp_file as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            # Replace target file atomically
            if os.path.exists(self.registry_file):
                os.replace(temp_file.name, self.registry_file)
            else:
                os.rename(temp_file.name, self.registry_file)
        except PermissionError as pe:
            if os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except Exception:
                    pass
            raise RegistryError(f"Permission denied writing registry to '{self.registry_file}': {pe}")
        except Exception as e:
            if os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except Exception:
                    pass
            raise RegistryError(f"Failed to save registry to '{self.registry_file}': {e}")

    def register(
        self,
        alias: str,
        project_path: str | Path,
        description: str = "",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Register a project path under a given alias.

        Args:
            alias: Unique human-readable alias (alphanumeric, underscores, hyphens).
            project_path: Path to the target project directory.
            description: Optional human-readable description.
            overwrite: If True, allow replacing an existing alias.

        Returns:
            The created project metadata dictionary.

        Raises:
            RegistryConflictError: If alias or path already registered and overwrite is False.
            RegistryError: If project_path does not exist or alias is invalid.
        """
        clean_alias = str(alias).strip().lower()
        if not re.match(r"^[a-z0-9_-]{2,64}$", clean_alias):
            raise RegistryError(
                f"Invalid project alias '{alias}'. Must be 2-64 characters using lowercase letters, numbers, hyphens, and underscores."
            )

        resolved_path = Path(project_path).resolve()
        if not resolved_path.is_dir():
            raise RegistryError(f"Target project path does not exist or is not a directory: {resolved_path}")

        data = self.load()

        # Check for alias collision
        if clean_alias in data and not overwrite:
            existing = data[clean_alias]
            if existing.get("path") != str(resolved_path):
                raise RegistryConflictError(
                    f"Alias '{clean_alias}' is already registered to '{existing.get('path')}'. "
                    f"Use --overwrite or a different alias."
                )

        # Check for path collision under a different alias
        for existing_alias, meta in data.items():
            if existing_alias != clean_alias and meta.get("path") == str(resolved_path) and not overwrite:
                raise RegistryConflictError(
                    f"Path '{resolved_path}' is already registered under alias '{existing_alias}'."
                )

        entry = {
            "alias": clean_alias,
            "path": str(resolved_path),
            "description": description,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }

        data[clean_alias] = entry
        self.save(data)
        return entry

    def unregister(self, alias: str) -> bool:
        """Remove a project from the registry by alias.

        Returns:
            True if removed, False if alias was not found.
        """
        clean_alias = str(alias).strip().lower()
        data = self.load()
        if clean_alias in data:
            del data[clean_alias]
            self.save(data)
            return True
        return False

    def get(self, alias: str) -> dict[str, Any] | None:
        """Retrieve project registration by alias."""
        clean_alias = str(alias).strip().lower()
        data = self.load()
        return data.get(clean_alias)

    def find_by_path(self, project_path: str | Path) -> str | None:
        """Find the registered alias for a project path, if any."""
        target = str(Path(project_path).resolve())
        data = self.load()
        for alias, meta in data.items():
            if meta.get("path") == target:
                return alias
        return None

    def list_projects(self) -> dict[str, dict[str, Any]]:
        """Return all registered projects."""
        return self.load()
