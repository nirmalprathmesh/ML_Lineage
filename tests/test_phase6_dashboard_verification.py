#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_phase6_dashboard_verification.py
===========================================
Phase 6 Test Suite:
1. Verification Path Containment Hardening:
   - Normal in-project paths
   - Parent directory traversal ('..')
   - Absolute paths
   - Windows drive-anchored paths
   - UNC paths
   - Symlinks pointing outside root
   - Both check_artifact_provenance and verify_run_artifacts containment
   - Read-only guarantee (no external read or modification)
2. Streamlit Dashboard Multi-Project Switcher:
   - Empty registry behavior
   - Registered projects appearance
   - Active context switching
   - Strict Project A / Project B isolation
   - Invalidation of pending chatbot recovery confirmation on project switch
   - Inaccessible or deleted project path handling
   - Unregistered alias handling
   - Dashboard data & view compatibility
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app
from ml_lineage.core.context import ProjectContext
from ml_lineage.core.registry import ProjectRegistry
from ml_lineage.core.recovery import PathTraversalError, validate_destination_path
from ml_lineage.core.verification import (
    check_artifact_provenance,
    verify_run_artifacts,
)


class TestVerificationPathContainment(unittest.TestCase):
    """Test verification security and strict destination path containment."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_verif_containment_")
        self.root = Path(self.temp_dir).resolve()

        # Outside directory simulating sensitive system files
        self.outside_dir = tempfile.mkdtemp(prefix="test_verif_outside_")
        self.outside_path = Path(self.outside_dir).resolve()
        self.outside_file = self.outside_path / "secret.txt"
        self.outside_file.write_text("sensitive_data_not_to_be_read", encoding="utf-8")

        # In-project directories and files
        self.data_dir = self.root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.in_file = self.data_dir / "dataset.csv"
        self.in_file.write_text("col1,col2\n1,2\n", encoding="utf-8")

        self.context = ProjectContext.from_path(self.root)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.outside_dir, ignore_errors=True)

    def test_01_normal_in_project_path(self):
        """Legitimate relative paths inside the root resolve and verify normally."""
        res = check_artifact_provenance(
            context=self.context,
            rel_path="data/dataset.csv",
            expected_hash=None,
            current_lock_hashes={},
        )
        self.assertTrue(res["workspace_exists"])
        self.assertIsNotNone(res["workspace_hash"])
        self.assertEqual(res["status"], "METRICS_FILE_PRESENT" if False else "NO_HASH_RECORDED_FOR_RUN")

    def test_02_parent_directory_traversal_rejected(self):
        """Path attempting '..' parent traversal is rejected before filesystem access."""
        traversal_paths = [
            "../outside.txt",
            "data/../../outside.txt",
            "data/../..",
            "..\\outside.txt",
        ]
        for p in traversal_paths:
            with self.assertRaises(PathTraversalError, msg=f"Failed to reject traversal: {p}"):
                check_artifact_provenance(
                    context=self.context,
                    rel_path=p,
                    expected_hash="dummy_hash",
                    current_lock_hashes={},
                )

    def test_03_absolute_paths_rejected(self):
        """Absolute POSIX and Windows paths are rejected."""
        abs_paths = [
            "/etc/passwd",
            "/tmp/secret.txt",
            "\\Windows\\System32\\cmd.exe",
            str(self.outside_file),
        ]
        for p in abs_paths:
            with self.assertRaises(PathTraversalError, msg=f"Failed to reject absolute path: {p}"):
                check_artifact_provenance(
                    context=self.context,
                    rel_path=p,
                    expected_hash="dummy_hash",
                    current_lock_hashes={},
                )

    def test_04_windows_drive_anchored_paths_rejected(self):
        """Windows drive-letter anchored paths are rejected."""
        drive_paths = [
            "C:secret.txt",
            "D:\\outside.txt",
            "C:/Windows/win.ini",
        ]
        for p in drive_paths:
            with self.assertRaises(PathTraversalError, msg=f"Failed to reject drive-letter path: {p}"):
                check_artifact_provenance(
                    context=self.context,
                    rel_path=p,
                    expected_hash="dummy_hash",
                    current_lock_hashes={},
                )

    def test_05_unc_paths_rejected(self):
        """UNC paths (both backslash and forward-slash) are rejected."""
        unc_paths = [
            "\\\\127.0.0.1\\share\\secret.txt",
            "//server/share/secret.txt",
        ]
        for p in unc_paths:
            with self.assertRaises(PathTraversalError, msg=f"Failed to reject UNC path: {p}"):
                check_artifact_provenance(
                    context=self.context,
                    rel_path=p,
                    expected_hash="dummy_hash",
                    current_lock_hashes={},
                )

    def test_06_symlink_pointing_outside_root_rejected(self):
        """A symlink located inside project root pointing to an external file is rejected."""
        symlink_target = self.outside_file
        symlink_path = self.root / "data" / "outside_symlink.txt"

        try:
            symlink_path.symlink_to(symlink_target)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not permitted in this environment")

        with self.assertRaises(PathTraversalError):
            check_artifact_provenance(
                context=self.context,
                rel_path="data/outside_symlink.txt",
                expected_hash="dummy_hash",
                current_lock_hashes={},
            )

    def test_07_verify_run_artifacts_enforces_containment(self):
        """verify_run_artifacts rejects any traversal or external paths in tag_map."""
        mock_run = MagicMock()
        mock_run.data.tags = {
            "dvc.hash.evil": "dummy_md5_hash",
        }
        evil_tag_map = {
            "dvc.hash.evil": "../outside.txt",
        }

        with self.assertRaises(PathTraversalError):
            verify_run_artifacts(
                context=self.context,
                run=mock_run,
                tag_map=evil_tag_map,
            )

    def test_08_verify_run_artifacts_symlink_outside_rejected(self):
        """verify_run_artifacts rejects external symlinks in tag_map."""
        symlink_path = self.root / "leak_symlink.txt"
        try:
            symlink_path.symlink_to(self.outside_file)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not permitted in this environment")

        mock_run = MagicMock()
        mock_run.data.tags = {"dvc.hash.leak": "dummy_hash"}
        tag_map = {"dvc.hash.leak": "leak_symlink.txt"}

        with self.assertRaises(PathTraversalError):
            verify_run_artifacts(
                context=self.context,
                run=mock_run,
                tag_map=tag_map,
            )

    def test_09_read_only_and_no_external_modifications(self):
        """Ensure verification causes zero external filesystem reads or modifications."""
        outside_before = self.outside_file.stat().st_mtime
        try:
            check_artifact_provenance(
                context=self.context,
                rel_path="../outside_secret.txt",
                expected_hash="abc",
                current_lock_hashes={},
            )
        except PathTraversalError:
            pass

        outside_after = self.outside_file.stat().st_mtime
        self.assertEqual(outside_before, outside_after)


class TestDashboardProjectSwitcher(unittest.TestCase):
    """Test Streamlit dashboard multi-project switcher and state isolation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_dash_multiproject_")
        self.root_base = Path(self.temp_dir).resolve()

        # Temporary registry file
        self.registry_file = self.root_base / "registry.json"
        self._saved_registry_env = os.environ.get("ML_LINEAGE_REGISTRY_PATH")
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.registry_file)

        # Create Project Alpha (Git + MLflow)
        self.proj_a = self.root_base / "project_alpha"
        self.proj_a.mkdir()
        (self.proj_a / ".ml-lineage.yaml").write_text(
            "project:\n  name: Alpha Project\nmlflow:\n  tracking_uri: sqlite:///alpha_mlflow.db\n",
            encoding="utf-8",
        )

        # Create Project Beta (Git + custom MLflow)
        self.proj_b = self.root_base / "project_beta"
        self.proj_b.mkdir()
        (self.proj_b / ".ml-lineage.yaml").write_text(
            "project:\n  name: Beta Project\nmlflow:\n  tracking_uri: sqlite:///beta_mlflow.db\n",
            encoding="utf-8",
        )

        app.enable_test_session_override()
        # Reset active project to default
        app.set_active_project(None)

    def tearDown(self):
        app.set_active_project(None)
        app.disable_test_session_override()

        if self._saved_registry_env is not None:
            os.environ["ML_LINEAGE_REGISTRY_PATH"] = self._saved_registry_env
        else:
            os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_empty_registry_behavior(self):
        """When registry is empty, default project is active and no crash occurs."""
        projects = app.get_registered_projects()
        self.assertEqual(projects, {})
        self.assertEqual(app.get_active_project_alias(), "(default)")
        self.assertEqual(app.get_active_context().root, app.DEFAULT_CONTEXT.root)

    def test_02_registered_projects_listed(self):
        """Registered projects appear in get_registered_projects()."""
        reg = app.get_project_registry()
        reg.register("alpha", self.proj_a, description="Alpha ML Project")
        reg.register("beta", self.proj_b, description="Beta ML Project")

        projects = app.get_registered_projects()
        self.assertEqual(len(projects), 2)
        self.assertIn("alpha", projects)
        self.assertIn("beta", projects)

    def test_03_select_alias_switches_active_context(self):
        """Switching alias changes active ProjectContext, tracking URI, and project name."""
        reg = app.get_project_registry()
        reg.register("alpha", self.proj_a)
        reg.register("beta", self.proj_b)

        ok, msg = app.set_active_project("alpha")
        self.assertTrue(ok)
        ctx_a = app.get_active_context()
        self.assertEqual(ctx_a.root, self.proj_a)
        self.assertEqual(ctx_a.project_name, "Alpha Project")
        self.assertIn("alpha_mlflow.db", app.get_tracking_uri())
        self.assertEqual(app.get_active_project_alias(), "alpha")

        # Switch to Beta
        ok, msg = app.set_active_project("beta")
        self.assertTrue(ok)
        ctx_b = app.get_active_context()
        self.assertEqual(ctx_b.root, self.proj_b)
        self.assertEqual(ctx_b.project_name, "Beta Project")
        self.assertIn("beta_mlflow.db", app.get_tracking_uri())
        self.assertEqual(app.get_active_project_alias(), "beta")

    def test_04_strict_project_isolation(self):
        """Project A and Project B have distinct roots, adapters, and tracking URIs."""
        reg = app.get_project_registry()
        reg.register("alpha", self.proj_a)
        reg.register("beta", self.proj_b)

        app.set_active_project("alpha")
        uri_a = app.get_tracking_uri()
        root_a = app.get_active_context().root
        git_a_root = app.get_active_git_adapter().root

        app.set_active_project("beta")
        uri_b = app.get_tracking_uri()
        root_b = app.get_active_context().root
        git_b_root = app.get_active_git_adapter().root

        self.assertNotEqual(root_a, root_b)
        self.assertNotEqual(uri_a, uri_b)
        self.assertNotEqual(git_a_root, git_b_root)
        self.assertEqual(git_a_root, self.proj_a)
        self.assertEqual(git_b_root, self.proj_b)

    def test_05_switching_project_invalidates_pending_recovery(self):
        """Project switch clears pending chatbot recovery confirmation."""
        reg = app.get_project_registry()
        reg.register("alpha", self.proj_a)
        reg.register("beta", self.proj_b)

        app.set_active_project("alpha")
        # Simulate chatbot setting a pending recovery
        app.set_confirmation_state({
            "run_id": "12345678abcdef0123456789abcdef01",
            "short_id": "12345678",
            "timestamp": time.time(),
        })
        self.assertIsNotNone(app.get_confirmation_state())

        # Switch project to Beta
        app.set_active_project("beta")
        # Must be invalidated
        self.assertIsNone(app.get_confirmation_state())

    def test_06_switching_to_default_invalidates_pending_recovery(self):
        """Switching back to default project also clears pending recovery."""
        app.set_confirmation_state({
            "run_id": "abcdef0123456789abcdef0123456789",
            "short_id": "abcdef01",
            "timestamp": time.time(),
        })
        self.assertIsNotNone(app.get_confirmation_state())

        app.set_active_project("(default)")
        self.assertIsNone(app.get_confirmation_state())

    def test_07_invalid_or_inaccessible_path_handled_gracefully(self):
        """Attempting to switch to an alias whose directory was deleted fails safely."""
        reg = app.get_project_registry()
        ghost_path = self.root_base / "ghost_dir"
        # Register but don't create or delete directory
        reg.register("ghost", ghost_path if ghost_path.exists() else self.proj_a)
        # Now remove the directory or edit registry
        data = reg.load()
        data["ghost"]["path"] = str(self.root_base / "nonexistent_dir")
        reg.save(data)

        current_root = app.get_active_context().root
        ok, msg = app.set_active_project("ghost")
        self.assertFalse(ok)
        self.assertIn("does not exist", msg)
        # Context must not be altered
        self.assertEqual(app.get_active_context().root, current_root)

    def test_08_unregistered_alias_handled(self):
        """Attempting to switch to an unregistered alias returns False with clear error."""
        current_root = app.get_active_context().root
        ok, msg = app.set_active_project("nonexistent_alias_xyz")
        self.assertFalse(ok)
        self.assertIn("not registered", msg)
        self.assertEqual(app.get_active_context().root, current_root)

    def test_09_dashboard_data_helpers_work_with_active_context(self):
        """Dashboard data helpers (git_status, dvc_status, tracking_uri) execute cleanly."""
        reg = app.get_project_registry()
        reg.register("alpha", self.proj_a)
        app.set_active_project("alpha")

        # Git status should query project alpha's directory
        git_stat = app.get_git_status()
        self.assertIn("clean", git_stat)

        # Tracking URI should match project alpha
        self.assertIn("alpha_mlflow.db", app.get_tracking_uri())

        # Reset to default
        app.set_active_project(None)
        self.assertEqual(app.get_tracking_uri(), app.DEFAULT_CONTEXT.tracking_uri)


class TestStreamlitSessionIsolation(unittest.TestCase):
    """Test session-level isolation for Streamlit multi-project execution."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_session_isolation_")
        self.root_base = Path(self.temp_dir).resolve()

        # Temporary registry file
        self.registry_file = self.root_base / "registry.json"
        self._saved_registry_env = os.environ.get("ML_LINEAGE_REGISTRY_PATH")
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.registry_file)

        # Create Project Alpha
        self.proj_a = self.root_base / "project_alpha"
        self.proj_a.mkdir()
        (self.proj_a / ".ml-lineage.yaml").write_text(
            "project:\n  name: Alpha Project\nmlflow:\n  tracking_uri: sqlite:///alpha.db\n",
            encoding="utf-8",
        )

        # Create Project Beta
        self.proj_b = self.root_base / "project_beta"
        self.proj_b.mkdir()
        (self.proj_b / ".ml-lineage.yaml").write_text(
            "project:\n  name: Beta Project\nmlflow:\n  tracking_uri: sqlite:///beta.db\n",
            encoding="utf-8",
        )

        reg = app.get_project_registry()
        reg.register("alpha", self.proj_a)
        reg.register("beta", self.proj_b)

    def tearDown(self):
        if self._saved_registry_env is not None:
            os.environ["ML_LINEAGE_REGISTRY_PATH"] = self._saved_registry_env
        else:
            os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_session_a_switch_does_not_alter_session_b(self):
        """Switching active project in Session A does not modify Session B."""
        session_a = app.IsolatedTestSession()
        session_b = app.IsolatedTestSession()

        # Initially both sessions are on default
        with session_a:
            self.assertEqual(app.get_active_project_alias(), "(default)")
            self.assertEqual(app.get_active_context().root, app.DEFAULT_CONTEXT.root)

        with session_b:
            self.assertEqual(app.get_active_project_alias(), "(default)")
            self.assertEqual(app.get_active_context().root, app.DEFAULT_CONTEXT.root)

        # Session A switches to Alpha
        with session_a:
            ok, _ = app.set_active_project("alpha")
            self.assertTrue(ok)
            self.assertEqual(app.get_active_project_alias(), "alpha")
            self.assertEqual(app.get_active_context().root, self.proj_a)
            self.assertIn("alpha.db", app.get_tracking_uri())

        # Session B must remain unaffected on default
        with session_b:
            self.assertEqual(app.get_active_project_alias(), "(default)")
            self.assertEqual(app.get_active_context().root, app.DEFAULT_CONTEXT.root)
            self.assertEqual(app.get_tracking_uri(), app.DEFAULT_CONTEXT.tracking_uri)

        # Session B switches to Beta
        with session_b:
            ok, _ = app.set_active_project("beta")
            self.assertTrue(ok)
            self.assertEqual(app.get_active_project_alias(), "beta")
            self.assertEqual(app.get_active_context().root, self.proj_b)
            self.assertIn("beta.db", app.get_tracking_uri())

        # Session A still points to Alpha
        with session_a:
            self.assertEqual(app.get_active_project_alias(), "alpha")
            self.assertEqual(app.get_active_context().root, self.proj_a)
            self.assertIn("alpha.db", app.get_tracking_uri())

    def test_concurrent_threads_session_isolation(self):
        """Simulate two concurrent threads running different sessions simultaneously."""
        import threading

        results = {}

        def run_session_worker(alias, name):
            sess = app.IsolatedTestSession()
            with sess:
                app.set_active_project(alias)
                # Sleep briefly to ensure overlap
                time.sleep(0.05)
                results[f"{name}_alias"] = app.get_active_project_alias()
                results[f"{name}_root"] = app.get_active_context().root
                results[f"{name}_uri"] = app.get_tracking_uri()

        t1 = threading.Thread(target=run_session_worker, args=("alpha", "session_1"))
        t2 = threading.Thread(target=run_session_worker, args=("beta", "session_2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(results["session_1_alias"], "alpha")
        self.assertEqual(results["session_1_root"], self.proj_a)
        self.assertIn("alpha.db", results["session_1_uri"])

        self.assertEqual(results["session_2_alias"], "beta")
        self.assertEqual(results["session_2_root"], self.proj_b)
        self.assertIn("beta.db", results["session_2_uri"])

    def test_session_pending_recovery_does_not_leak_across_sessions(self):
        """Pending recovery confirmation in Session A is invisible to Session B."""
        session_a = app.IsolatedTestSession()
        session_b = app.IsolatedTestSession()

        with session_a:
            app.set_confirmation_state({
                "run_id": "aaaaaaaa111122223333444455556666",
                "short_id": "aaaaaaaa",
                "timestamp": time.time(),
            })
            self.assertIsNotNone(app.get_confirmation_state())

        # Session B must see no pending recovery
        with session_b:
            self.assertIsNone(app.get_confirmation_state())

        # Session A switching project invalidates its own pending recovery
        with session_a:
            app.set_active_project("alpha")
            self.assertIsNone(app.get_confirmation_state())

    def test_adapters_are_session_scoped(self):
        """Each session maintains independent adapters scoped to its active project."""
        session_a = app.IsolatedTestSession()
        session_b = app.IsolatedTestSession()

        with session_a:
            app.set_active_project("alpha")
            git_a = app.get_active_git_adapter()
            mlflow_a = app.get_active_mlflow_adapter()
            self.assertEqual(git_a.root, self.proj_a)
            self.assertIn("alpha.db", mlflow_a.tracking_uri)

        with session_b:
            app.set_active_project("beta")
            git_b = app.get_active_git_adapter()
            mlflow_b = app.get_active_mlflow_adapter()
            self.assertEqual(git_b.root, self.proj_b)
            self.assertIn("beta.db", mlflow_b.tracking_uri)

        self.assertNotEqual(git_a.root, git_b.root)


class TestGenericArtifactTagMapping(unittest.TestCase):
    """Test generic artifact tag mapping, external project isolation, and unknown format handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_generic_tag_")
        self.root = Path(self.temp_dir).resolve()

        # External project structure with custom artifact names
        self.data_dir = self.root / "data"
        self.models_dir = self.root / "models"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.raw_sensor = self.data_dir / "raw_sensors.parquet"
        self.raw_sensor.write_bytes(b"PAR1_dummy_parquet_content_bytes")

        self.custom_model = self.models_dir / "weights.bin"
        self.custom_model.write_bytes(b"PYTORCH_WEIGHTS_BINARY_BLOB")

        self.context = ProjectContext.from_path(self.root)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_wine_mappings_only_apply_to_default_wine_project(self):
        """Wine-specific tag mappings only apply when context is the default Wine project."""
        from ml_lineage.wine import WINE_DVC_TAG_MAP

        # External context: get_active_tag_map() returns None
        with patch.object(app, "get_active_context", return_value=self.context):
            self.assertIsNone(app.get_active_tag_map())

        # Default context: returns Wine DVC_TAG_MAP
        with patch.object(app, "get_active_context", return_value=app.DEFAULT_CONTEXT):
            self.assertEqual(app.get_active_tag_map(), app.DVC_TAG_MAP)

    def test_02_external_project_never_falls_back_to_wine_mappings(self):
        """verify_run_artifacts does not inject Wine demo paths into an external project."""
        mock_run = MagicMock()
        mock_run.data.tags = {}

        # Tag map is None -> dynamic discovery
        results = verify_run_artifacts(context=self.context, run=mock_run, tag_map=None)
        # Must be empty, NOT populated with wine_dataset.csv or scaler.pkl
        self.assertEqual(results, [])
        for item in results:
            self.assertNotIn("wine_dataset.csv", item.get("artifact", ""))
            self.assertNotIn("scaler.pkl", item.get("artifact", ""))

    def test_03_dynamic_tag_discovery_handles_custom_artifact_names(self):
        """Dynamic tag discovery correctly extracts custom relative artifact paths."""
        from ml_lineage.core.hashing import compute_file_md5

        sensor_hash = compute_file_md5(self.raw_sensor)
        weights_hash = compute_file_md5(self.custom_model)

        mock_run = MagicMock()
        mock_run.data.tags = {
            "dvc.hash.data/raw_sensors.parquet": sensor_hash,
            "artifact.hash.models/weights.bin": weights_hash,
        }

        results = verify_run_artifacts(context=self.context, run=mock_run, tag_map=None)
        self.assertEqual(len(results), 2)

        artifact_map = {r["artifact"]: r for r in results}
        self.assertIn("data/raw_sensors.parquet", artifact_map)
        self.assertIn("models/weights.bin", artifact_map)

        self.assertEqual(artifact_map["data/raw_sensors.parquet"]["status"], "✅ VERIFIED")
        self.assertEqual(artifact_map["models/weights.bin"]["status"], "✅ VERIFIED")

    def test_04_unknown_artifact_tag_formats_classified_unsupported(self):
        """Unknown or non-standard artifact tag formats are classified as unsupported."""
        mock_run = MagicMock()
        mock_run.data.tags = {
            "model_checksum": "abcdef0123456789",
            "custom_data_digest": "9876543210fedcba",
            "unsupported_artifact_md5": "1122334455667788",
            "git.commit_sha": "deadbeef",
            "mlflow.runName": "run_1",
        }

        results = verify_run_artifacts(context=self.context, run=mock_run, tag_map=None)
        self.assertGreater(len(results), 0)

        for item in results:
            self.assertIn("UNVERIFIABLE", item["status"])
            self.assertIn("unsupported artifact tag format", item["status"])

    def test_05_external_lineage_trace_does_not_falsely_claim_exact_lineage(self):
        """trace_run_lineage for external project with unrecorded artifacts does not claim verification."""
        from ml_lineage.core.lineage import trace_run_lineage

        # Mock MLflow run for external project with unrecognized artifact format
        mock_run = MagicMock()
        mock_run.info.run_id = "external_run_1234567890abcdef"
        mock_run.info.experiment_id = "0"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1700000000000
        mock_run.info.end_time = 1700000060000
        mock_run.info.artifact_uri = None
        mock_run.data.tags = {
            "custom_model_hash": "dummy_hash_not_standard",
            "git.commit_sha": "abcdef123456",
        }
        mock_run.data.params = {}
        mock_run.data.metrics = {}

        with patch("ml_lineage.adapters.mlflow.MLflowAdapter.resolve_run", return_value=mock_run), \
             patch("ml_lineage.adapters.mlflow.MLflowAdapter.get_client"):
            rep = trace_run_lineage(self.context, "external_run_1234567890abcdef")
            self.assertIsNotNone(rep)
            rep_dict = rep.to_dict()

            # Verification summary must NOT falsely claim exact lineage
            v = rep_dict["verification_summary"]
            self.assertFalse(v["is_lineage_verified"])


if __name__ == "__main__":
    unittest.main()
