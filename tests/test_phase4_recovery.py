"""
tests/test_phase4_recovery.py
=============================
Automated test suite for Phase 4: Safe, Project-Scoped Recovery.

Covers all required Phase 4 verification items:
1. Successful recovery into an empty destination.
2. Dry-run makes zero changes.
3. Existing file conflict is detected.
4. User-modified file is not silently overwritten.
5. Dirty Git tree blocks unsafe recovery.
6. Clean Git tree permits an otherwise safe recovery.
7. Missing artifact is reported.
8. Corrupted artifact fails integrity verification.
9. Backup is created and verified.
10. Backup failure aborts recovery.
11. Interrupted recovery triggers rollback.
12. Rollback failure is reported honestly.
13. Invalid run ID is rejected.
14. Path traversal attempts are rejected.
15. Recovery cannot write outside the authorized destination.
16. Separate projects cannot interfere with each other.
17. JSON output is valid.
18. Non-interactive execution cannot bypass confirmation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
import yaml

from ml_lineage.adapters.dvc import DVCAdapter
from ml_lineage.adapters.git import GitAdapter
from ml_lineage.adapters.mlflow import MLflowAdapter
from ml_lineage.cli import main as cli_main
from ml_lineage.core.context import ProjectContext
from ml_lineage.core.hashing import compute_file_md5
from ml_lineage.core.recovery import (
    ArtifactPlan,
    BackupError,
    PathTraversalError,
    RecoveryError,
    RecoveryExecutionError,
    RecoveryPlan,
    RecoveryResult,
    RecoveryService,
    validate_destination_path,
)


def _md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


class TestPhase4Recovery(unittest.TestCase):
    """Test suite covering safe recovery planning, git safety, backups, rollback, and CLI."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="ml_test_recovery_")
        self.root = Path(self.tmp_dir).resolve()

        # Build directory structure
        self.dvc_cache = self.root / ".dvc" / "cache" / "files" / "md5"
        self.dvc_cache.mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "models").mkdir(parents=True, exist_ok=True)

        # Create dummy params.yaml and dvc.yaml
        (self.root / "params.yaml").write_text("model: wine_classifier\n", encoding="utf-8")
        dvc_yaml_data = {
            "stages": {
                "train": {
                    "cmd": "python train.py",
                    "outs": ["models/wine_model.pkl"],
                }
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(dvc_yaml_data), encoding="utf-8")

        # Put a valid sample artifact in DVC cache
        self.model_content = b"ORIGINAL_TRAINED_MODEL_V1_WEIGHTS"
        self.model_hash = _md5(self.model_content)
        cache_entry = self.dvc_cache / self.model_hash[:2] / self.model_hash[2:]
        cache_entry.parent.mkdir(parents=True, exist_ok=True)
        cache_entry.write_bytes(self.model_content)

        # Context
        self.context = ProjectContext(self.root)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _mock_run(self, run_id: str = "run_12345678", git_sha: str = "deadbeef1234", dirty: bool = False, custom_tags: dict | None = None):
        run = MagicMock()
        run.info.run_id = run_id
        run.info.status = "FINISHED"
        run.info.start_time = 1700000000000
        run.info.artifact_uri = f"file:///{self.root.as_posix()}/mlruns/1/{run_id}/artifacts"

        tags = {
            "git.commit_sha": git_sha,
            "git.branch": "master",
            "git.is_dirty": str(dirty),
            "dvc.hash.models/wine_model.pkl": self.model_hash,
            "model_path": self.model_hash,  # Wine tag map compatibility
        }
        if custom_tags:
            tags.update(custom_tags)

        run.data.tags = tags
        run.data.params = {"learning_rate": "0.01"}
        run.data.metrics = {"accuracy": 0.95}
        return run

    # 1. Successful recovery into an empty destination
    def test_01_successful_recovery_into_empty_destination(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        if dest_file.exists():
            dest_file.unlink()

        run = self._mock_run()
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")
            self.assertTrue(plan.can_recover)
            self.assertEqual(len(plan.artifacts), 1)
            self.assertEqual(plan.artifacts[0].action, "CREATE")

            result = service.execute_recovery(plan)
            self.assertTrue(result.success)
            self.assertTrue(dest_file.is_file())
            self.assertEqual(dest_file.read_bytes(), self.model_content)
            self.assertEqual(compute_file_md5(dest_file), self.model_hash)

    # 2. Dry-run makes zero changes
    def test_02_dry_run_makes_zero_changes(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        if dest_file.exists():
            dest_file.unlink()

        run = self._mock_run()
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")
            self.assertTrue(plan.can_recover)
            # Ensure destination still does not exist
            self.assertFalse(dest_file.exists())
            self.assertFalse((self.root / ".ml-lineage").exists())

    # 3. Existing file conflict is detected
    def test_03_existing_file_conflict_is_detected(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        dest_file.write_bytes(b"EXISTING_DIFFERENT_MODEL_DATA")

        run = self._mock_run()
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")
            self.assertEqual(len(plan.artifacts), 1)
            art = plan.artifacts[0]
            self.assertEqual(art.action, "REPLACE")
            self.assertTrue(art.conflict)
            self.assertTrue(art.dest_exists)
            self.assertNotEqual(art.dest_hash, self.model_hash)

    # 4. User-modified file is not silently overwritten (backup created)
    def test_04_user_modified_file_not_silently_overwritten(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        original_user_content = b"USER_MODIFIED_CUSTOM_TRAINING"
        dest_file.write_bytes(original_user_content)
        original_hash = compute_file_md5(dest_file)

        run = self._mock_run()
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")
            result = service.execute_recovery(plan)

            self.assertTrue(result.success)
            self.assertEqual(len(result.backups_created), 1)

            # Check that backup was created and has original user content
            backup_path = Path(result.backups_created[0]["backup_path"])
            self.assertTrue(backup_path.is_file())
            self.assertEqual(backup_path.read_bytes(), original_user_content)
            self.assertEqual(result.backups_created[0]["original_hash"], original_hash)

            # Check restored destination has expected recovered content
            self.assertEqual(dest_file.read_bytes(), self.model_content)

    # 5. Dirty Git tree blocks unsafe recovery
    def test_05_dirty_git_tree_blocks_unsafe_recovery(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        dest_file.write_bytes(b"DIRTY_LOCAL_EDITS")

        run = self._mock_run()
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run), \
             patch.object(service.git_adapter, "is_git_repository", return_value=True), \
             patch.object(service.git_adapter, "is_dirty", return_value=True), \
             patch.object(service.git_adapter, "is_file_dirty", return_value=True):

            plan = service.create_plan("run_1234")
            self.assertFalse(plan.can_recover)
            self.assertTrue(any("uncommitted changes in Git" in b for b in plan.blockers))

            # Attempting execution must raise RecoveryError
            with self.assertRaises(RecoveryError):
                service.execute_recovery(plan)

    # 6. Clean Git tree permits an otherwise safe recovery
    def test_06_clean_git_tree_permits_recovery(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        dest_file.write_bytes(b"OLD_STALE_FILE")

        run = self._mock_run()
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run), \
             patch.object(service.git_adapter, "is_git_repository", return_value=True), \
             patch.object(service.git_adapter, "is_dirty", return_value=False), \
             patch.object(service.git_adapter, "is_file_dirty", return_value=False), \
             patch.object(service.git_adapter, "commit_exists", return_value=True):

            plan = service.create_plan("run_1234")
            self.assertTrue(plan.can_recover)
            self.assertEqual(len(plan.blockers), 0)

            result = service.execute_recovery(plan)
            self.assertTrue(result.success)
            self.assertEqual(dest_file.read_bytes(), self.model_content)

    # 7. Missing artifact is reported
    def test_07_missing_artifact_is_reported(self):
        missing_hash = "11112222333344445555666677778888"
        run = self._mock_run(custom_tags={"dvc.hash.models/wine_model.pkl": missing_hash})
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")
            self.assertFalse(plan.can_recover)
            self.assertTrue(any("not found in DVC cache or remote" in b for b in plan.blockers))

    # 8. Corrupted artifact fails integrity verification
    def test_08_corrupted_artifact_fails_integrity(self):
        corrupted_hash = "abcdef0123456789abcdef0123456789"
        corrupted_cache = self.dvc_cache / corrupted_hash[:2] / corrupted_hash[2:]
        corrupted_cache.parent.mkdir(parents=True, exist_ok=True)
        corrupted_cache.write_bytes(b"CORRUPTED_BYTES_NOT_MATCHING_HASH")

        run = self._mock_run(custom_tags={"dvc.hash.models/wine_model.pkl": corrupted_hash})
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")
            self.assertFalse(plan.can_recover)
            self.assertTrue(any("Corrupted source" in b for b in plan.blockers))

    # 9. Backup is created and verified
    def test_09_backup_is_created_and_verified(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        dest_file.write_bytes(b"IMPORTANT_EXISTING_MODEL")
        expected_backup_hash = compute_file_md5(dest_file)

        run = self._mock_run()
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")
            result = service.execute_recovery(plan)

            self.assertEqual(len(result.backups_created), 1)
            b_info = result.backups_created[0]
            backup_file = Path(b_info["backup_path"])
            self.assertTrue(backup_file.is_file())
            self.assertEqual(compute_file_md5(backup_file), expected_backup_hash)

    # 10. Backup failure aborts recovery
    def test_10_backup_failure_aborts_recovery(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        initial_content = b"INITIAL_DATA"
        dest_file.write_bytes(initial_content)

        run = self._mock_run()
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")

            # Simulate shutil.copy2 failure during backup
            with patch("shutil.copy2", side_effect=PermissionError("Permission denied")):
                with self.assertRaises(BackupError):
                    service.execute_recovery(plan)

            # Ensure destination was untouched
            self.assertEqual(dest_file.read_bytes(), initial_content)

    # 11. Interrupted recovery triggers rollback
    def test_11_interrupted_recovery_triggers_rollback(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        initial_content = b"INITIAL_MODEL_DATA"
        dest_file.write_bytes(initial_content)

        run = self._mock_run()
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")

            # Mock compute_file_md5 during post-restore check to simulate verification failure
            orig_md5 = compute_file_md5
            def mock_md5(path):
                # When verifying dest_file after restore, pretend it failed
                if Path(path) == dest_file and dest_file.read_bytes() == self.model_content:
                    return "simulated_bad_hash"
                return orig_md5(path)

            with patch("ml_lineage.core.recovery.compute_file_md5", side_effect=mock_md5):
                with self.assertRaises(RecoveryExecutionError) as cm:
                    service.execute_recovery(plan)
                self.assertTrue(cm.exception.rollback_success)

            # Verify that rollback restored the original file
            self.assertEqual(dest_file.read_bytes(), initial_content)

    # 12. Rollback failure is reported honestly
    def test_12_rollback_failure_is_reported_honestly(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        dest_file.write_bytes(b"INITIAL_MODEL_DATA")

        run = self._mock_run()
        service = RecoveryService(self.context)

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")

            # Cause copy to fail, then cause rollback copy to also fail
            real_copy2 = shutil.copy2
            call_count = [0]
            def flaky_copy2(src, dst):
                call_count[0] += 1
                # 1 = backup (succeeds)
                # 2 = restore copy (fails)
                # 3 = rollback copy (fails)
                if call_count[0] == 2:
                    raise IOError("Disk error during file restore")
                if call_count[0] == 3:
                    raise IOError("Disk error during rollback")
                return real_copy2(src, dst)

            with patch("shutil.copy2", side_effect=flaky_copy2):
                with self.assertRaises(RecoveryExecutionError) as cm:
                    service.execute_recovery(plan)
                self.assertFalse(cm.exception.rollback_success)
                self.assertTrue(len(cm.exception.rollback_errors) > 0)

    # 13. Invalid run ID is rejected
    def test_13_invalid_run_id_is_rejected(self):
        service = RecoveryService(self.context)
        with patch.object(service.mlflow_adapter, "resolve_run", return_value=None):
            with self.assertRaises(RecoveryError):
                service.create_plan("nonexistent_run_id")

    # 14. Path traversal attempts are rejected
    def test_14_path_traversal_attempts_are_rejected(self):
        # Test paths with '..', leading slashes, and drive roots
        for bad_path in ["../../etc/passwd", "../evil.pkl", "/etc/shadow", "C:\\Windows\\system32\\calc.exe"]:
            with self.assertRaises(PathTraversalError):
                validate_destination_path(self.root, bad_path)

    # 15. Recovery cannot write outside the authorized destination
    def test_15_recovery_cannot_write_outside_authorized_destination(self):
        isolated_dir = self.root / "isolated_recovery"
        isolated_dir.mkdir(parents=True, exist_ok=True)

        service = RecoveryService(self.context, output_dir=isolated_dir)
        run = self._mock_run()

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=run):
            plan = service.create_plan("run_1234")
            self.assertTrue(plan.can_recover)
            result = service.execute_recovery(plan)
            self.assertTrue(result.success)

            # File must be written strictly inside isolated_dir
            recovered_file = isolated_dir / "models" / "wine_model.pkl"
            self.assertTrue(recovered_file.is_file())

            # Workspace original file must not exist
            workspace_file = self.root / "models" / "wine_model.pkl"
            self.assertFalse(workspace_file.exists())

    # 16. Separate projects cannot interfere with each other
    def test_16_separate_projects_cannot_interfere(self):
        # Create second independent project
        tmp2 = tempfile.mkdtemp(prefix="ml_test_proj2_")
        try:
            root2 = Path(tmp2).resolve()
            ctx2 = ProjectContext(root2)
            dvc2 = root2 / ".dvc" / "cache" / "files" / "md5"
            dvc2.mkdir(parents=True, exist_ok=True)

            content2 = b"PROJECT_2_SEPARATE_MODEL"
            hash2 = _md5(content2)
            c2_file = dvc2 / hash2[:2] / hash2[2:]
            c2_file.parent.mkdir(parents=True, exist_ok=True)
            c2_file.write_bytes(content2)

            run2 = self._mock_run(run_id="run_proj_2", custom_tags={"dvc.hash.models/wine_model.pkl": hash2})
            svc2 = RecoveryService(ctx2)

            with patch.object(svc2.mlflow_adapter, "resolve_run", return_value=run2):
                plan2 = svc2.create_plan("run_proj_2")
                res2 = svc2.execute_recovery(plan2)
                self.assertTrue(res2.success)

                # Project 2 recovered its model
                p2_model = root2 / "models" / "wine_model.pkl"
                self.assertTrue(p2_model.is_file())
                self.assertEqual(p2_model.read_bytes(), content2)

                # Project 1 workspace remains clean/unaffected
                p1_model = self.root / "models" / "wine_model.pkl"
                self.assertFalse(p1_model.exists())
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    # 17. JSON output is valid
    def test_17_cli_json_output_is_valid(self):
        runner = CliRunner()
        run = self._mock_run()

        with patch("ml_lineage.adapters.mlflow.MLflowAdapter.resolve_run", return_value=run):
            res = runner.invoke(
                cli_main,
                ["--path", str(self.root), "recover", "run_1234", "--dry-run", "--json"],
            )
            self.assertEqual(res.exit_code, 0, f"CLI error: {res.output}")
            parsed = json.loads(res.stdout)
            self.assertIn("target_project_path", parsed)
            self.assertIn("artifacts", parsed)
            self.assertIn("can_recover", parsed)
            self.assertEqual(parsed["run_id"], "run_12345678")

    # 18. Non-interactive execution cannot bypass confirmation
    def test_18_non_interactive_cannot_bypass_confirmation(self):
        runner = CliRunner()
        run = self._mock_run()

        with patch("ml_lineage.adapters.mlflow.MLflowAdapter.resolve_run", return_value=run):
            # Invoking recover without --yes in non-interactive mode must fail
            res = runner.invoke(
                cli_main,
                ["--path", str(self.root), "recover", "run_1234"],
            )
            self.assertEqual(res.exit_code, 1)
            self.assertIn("explicit confirmation flag (--yes)", res.output)

            # Dest file should NOT have been created
            dest_file = self.root / "models" / "wine_model.pkl"
            self.assertFalse(dest_file.exists())

    # 19. CLI verify subcommand
    def test_19_cli_verify_subcommand(self):
        dest_file = self.root / "models" / "wine_model.pkl"
        dest_file.write_bytes(self.model_content)

        runner = CliRunner()
        run = self._mock_run()

        with patch("ml_lineage.adapters.mlflow.MLflowAdapter.resolve_run", return_value=run):
            res = runner.invoke(
                cli_main,
                ["--path", str(self.root), "verify", "run_1234"],
            )
            self.assertEqual(res.exit_code, 0)
            self.assertIn("Artifact Verification", res.output)

            # JSON mode
            res_json = runner.invoke(
                cli_main,
                ["--path", str(self.root), "verify", "run_1234", "--json"],
            )
            self.assertEqual(res_json.exit_code, 0)
            parsed = json.loads(res_json.stdout)
            self.assertEqual(parsed["run_id"], "run_12345678")
            self.assertIn("artifacts", parsed)
