"""
ml_lineage.adapters.git
=======================
Git repository adapter providing read-only inspection and status queries.
Operates strictly in the working directory of the given ProjectContext.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml_lineage.core.context import ProjectContext


class GitAdapter:
    """Provides project-scoped Git operations and provenance extraction."""

    def __init__(self, context: ProjectContext):
        self.context = context
        self.root = context.root

    def run_cmd(self, args: list[str], timeout: int = 30) -> tuple[int, str, str]:
        """Execute a read-only git command strictly within this project's root.

        Never uses shell=True.
        """
        try:
            res = subprocess.run(
                ["git"] + args,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            return res.returncode, res.stdout.strip(), res.stderr.strip()
        except FileNotFoundError:
            return -1, "", "git executable not found on PATH"
        except subprocess.TimeoutExpired:
            return -1, "", f"git command timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)

    def is_git_repository(self) -> bool:
        """Check whether the project root is within a valid Git repository."""
        code, out, _ = self.run_cmd(["rev-parse", "--is-inside-work-tree"])
        return code == 0 and out.lower() == "true"

    def get_repo_toplevel(self) -> Path | None:
        """Resolve the top-level directory of the Git working tree."""
        code, out, _ = self.run_cmd(["rev-parse", "--show-toplevel"])
        if code == 0 and out:
            return Path(out).resolve()
        return None

    def get_commit_sha(self, ref: str = "HEAD") -> str:
        """Get the full commit SHA for a reference."""
        code, out, _ = self.run_cmd(["rev-parse", ref])
        return out if code == 0 else "unavailable"

    def get_branch(self) -> str:
        """Get the current Git branch name."""
        code, out, _ = self.run_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
        return out if code == 0 else "unavailable"

    def is_dirty(self) -> bool:
        """Check whether the working tree contains uncommitted or untracked changes."""
        code, out, _ = self.run_cmd(["status", "--porcelain"])
        if code == 0:
            return bool(out)
        return False

    def get_git_info(self) -> dict[str, Any]:
        """Gather commit SHA, branch, and working-tree dirty status for provenance recording."""
        return {
            "commit_sha": self.get_commit_sha(),
            "branch": self.get_branch(),
            "is_dirty": self.is_dirty(),
        }

    def get_status(self) -> dict[str, Any]:
        """Retrieve formatted Git status for dashboard/CLI display."""
        commit = self.get_commit_sha()
        branch = self.get_branch()
        code, out, _ = self.run_cmd(["status", "--porcelain"])
        clean = (not bool(out)) if code == 0 else "unknown"

        return {
            "commit": commit,
            "branch": branch,
            "clean": clean,
            "is_repo": self.is_git_repository(),
        }
