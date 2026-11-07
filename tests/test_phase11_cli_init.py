"""
tests.test_phase11_cli_init
===========================
Phase 11 comprehensive tests: CLI-First Project Initialization, Automatic
Discovery, Shared Registry Integration, and Multi-Project Isolation.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from ml_lineage.cli import main
from ml_lineage.core.context import ProjectContext
from ml_lineage.core.registry import ProjectRegistry, RegistryError, RegistryConflictError


def _cleanup(path: str | Path | None) -> None:
    if not path:
        return
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def _parse_json(res) -> dict:
    raw = getattr(res, "stdout", "") or res.output
    raw = raw.strip()
    idx = raw.find("{")
    if idx != -1:
        return json.loads(raw[idx:])
    return json.loads(raw)


class TestPhase11CliInit(unittest.TestCase):
    """Test Phase 11 CLI-first init, discovery, and registry synchronization."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_root = tempfile.mkdtemp()
        self.registry_file = Path(self.temp_root) / "shared_registry.json"
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.registry_file)

        # Create synthetic project directory
        self.project_a = Path(self.temp_root) / "customer_churn_model"
        self.project_a.mkdir(parents=True, exist_ok=True)
        (self.project_a / "params.yaml").write_text(
            yaml.dump({
                "mlflow": {
                    "experiment_name": "Customer_Churn_Prod",
                    "tracking_uri": "sqlite:///churn_mlflow.db",
                }
            }),
            encoding="utf-8",
        )
        (self.project_a / "data.csv").write_text("id,churn\n1,0\n2,1\n", encoding="utf-8")

    def tearDown(self):
        os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)
        _cleanup(self.temp_root)

    def test_01_init_in_cwd_auto_detects_project_root_and_registers(self):
        """Entering a project directory and executing 'ml-lineage init --yes' initializes and registers it."""
        cwd_before = os.getcwd()
        try:
            os.chdir(str(self.project_a))
            res = self.runner.invoke(main, ["init", "--yes"])
            self.assertEqual(res.exit_code, 0, f"stdout: {res.stdout}\nstderr: {res.stderr}")
            self.assertIn("Project successfully initialized", res.output)
            self.assertIn("Dashboard: Available under project switcher", res.output)

            # Check .ml-lineage.yaml was written
            config_file = self.project_a / ".ml-lineage.yaml"
            self.assertTrue(config_file.is_file())
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.assertEqual(cfg["project"]["name"], "customer_churn_model")

            # Check registry auto-registration
            reg = ProjectRegistry(self.registry_file)
            entry = reg.get("customer_churn_model")
            self.assertIsNotNone(entry)
            self.assertEqual(Path(entry["path"]).resolve(), self.project_a.resolve())
        finally:
            os.chdir(cwd_before)

    def test_02_init_custom_alias_and_sanitization(self):
        """Init with --name flags creates a sanitized valid alias."""
        res = self.runner.invoke(
            main,
            ["init", "--path", str(self.project_a), "--name", "Churn Prediction Model v2", "--yes"],
        )
        self.assertEqual(res.exit_code, 0)
        self.assertIn("churn-prediction-model-v2", res.output)

        reg = ProjectRegistry(self.registry_file)
        entry = reg.get("churn-prediction-model-v2")
        self.assertIsNotNone(entry)
        self.assertEqual(Path(entry["path"]).resolve(), self.project_a.resolve())

    def test_03_init_invalid_alias_rejected_without_modifying_target(self):
        """Invalid alias (single character, special symbols only) fails cleanly before writing files."""
        res = self.runner.invoke(
            main,
            ["init", "--path", str(self.project_a), "--name", "!", "--yes"],
        )
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Invalid project alias", res.output)
        self.assertFalse((self.project_a / ".ml-lineage.yaml").exists())

    def test_04_init_alias_collision_fails_cleanly(self):
        """Registering an existing alias to a different project path fails cleanly without mutating."""
        project_b = Path(self.temp_root) / "fraud_detection"
        project_b.mkdir(parents=True, exist_ok=True)

        # First register project_a as 'churn'
        self.runner.invoke(main, ["init", "--path", str(self.project_a), "--name", "churn", "--yes"])

        # Attempt to init project_b also as 'churn' without overwrite
        res = self.runner.invoke(main, ["init", "--path", str(project_b), "--name", "churn", "--yes"])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("already registered", res.output)
        self.assertFalse((project_b / ".ml-lineage.yaml").exists())

    def test_05_init_idempotency_preserves_existing_config(self):
        """Re-running init on an already configured project preserves config and reports dashboard readiness."""
        # First init
        self.runner.invoke(main, ["init", "--path", str(self.project_a), "--name", "churn-prod", "--yes"])
        config_file = self.project_a / ".ml-lineage.yaml"
        # Add custom field to simulate existing user config
        with open(config_file, "a", encoding="utf-8") as f:
            f.write("\ncustom_user_setting: preserve_me\n")

        # Second init without overwrite
        res = self.runner.invoke(main, ["init", "--path", str(self.project_a), "--name", "churn-prod", "--yes"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("already configured", res.output)
        self.assertIn("Dashboard", res.output)

        content = config_file.read_text(encoding="utf-8")
        self.assertIn("custom_user_setting: preserve_me", content)

    def test_06_nested_directory_upward_auto_discovery(self):
        """Commands executed from nested directories inside project resolve root correctly."""
        # Initialize project
        self.runner.invoke(main, ["init", "--path", str(self.project_a), "--name", "churn-nested", "--yes"])

        # Create nested subdirectories: src/models/deep
        nested_dir = self.project_a / "src" / "models" / "deep"
        nested_dir.mkdir(parents=True, exist_ok=True)

        cwd_before = os.getcwd()
        try:
            os.chdir(str(nested_dir))
            # Test status command without --project or --path
            res = self.runner.invoke(main, ["status", "--json"])
            self.assertEqual(res.exit_code, 0, f"stdout: {res.stdout}\nstderr: {res.stderr}")
            data = _parse_json(res)
            self.assertEqual(Path(data["project"]["root"]).resolve(), self.project_a.resolve())
            self.assertEqual(data["project"]["name"], "churn-nested")
        finally:
            os.chdir(cwd_before)

    def test_07_multi_project_isolation(self):
        """Two synthetic projects initialized via init maintain completely isolated contexts."""
        project_b = Path(self.temp_root) / "nlp_sentiment"
        project_b.mkdir(parents=True, exist_ok=True)
        (project_b / "params.yaml").write_text(
            yaml.dump({
                "mlflow": {
                    "experiment_name": "Sentiment_BERT",
                    "tracking_uri": "sqlite:///nlp_mlflow.db",
                }
            }),
            encoding="utf-8",
        )

        res_a = self.runner.invoke(main, ["init", "--path", str(self.project_a), "--name", "proj-churn", "--yes"])
        self.assertEqual(res_a.exit_code, 0)
        res_b = self.runner.invoke(main, ["init", "--path", str(project_b), "--name", "proj-sentiment", "--yes"])
        self.assertEqual(res_b.exit_code, 0)

        # Verify CLI targeting via alias
        res_status_a = self.runner.invoke(main, ["status", "--project", "proj-churn", "--json"])
        res_status_b = self.runner.invoke(main, ["status", "--project", "proj-sentiment", "--json"])

        data_a = _parse_json(res_status_a)
        data_b = _parse_json(res_status_b)

        self.assertEqual(data_a["project"]["name"], "proj-churn")
        self.assertEqual(data_b["project"]["name"], "proj-sentiment")
        self.assertIn("churn_mlflow.db", data_a["mlflow"]["tracking_uri"])
        self.assertIn("nlp_mlflow.db", data_b["mlflow"]["tracking_uri"])
        self.assertNotEqual(data_a["project"]["root"], data_b["project"]["root"])

    def test_08_dashboard_shared_registry_synchronization(self):
        """Dashboard helper functions pick up projects registered by CLI without restart."""
        import app as dashboard_app

        # Before init: registry is empty
        projects_before = dashboard_app.get_registered_projects()
        self.assertEqual(len(projects_before), 0)

        # CLI initializes project
        res = self.runner.invoke(main, ["init", "--path", str(self.project_a), "--name", "live-churn", "--yes"])
        self.assertEqual(res.exit_code, 0)

        # Dashboard helper immediately sees newly registered project
        projects_after = dashboard_app.get_registered_projects()
        self.assertIn("live-churn", projects_after)
        self.assertEqual(Path(projects_after["live-churn"]["path"]).resolve(), self.project_a.resolve())

        # Switch to project in dashboard session
        success, msg = dashboard_app.set_active_project("live-churn")
        self.assertTrue(success)
        self.assertIn("Switched to project 'live-churn'", msg)
        self.assertEqual(dashboard_app.get_active_project_alias(), "live-churn")

        # Switch back to default
        success_def, msg_def = dashboard_app.set_active_project(None)
        self.assertTrue(success_def)
        self.assertEqual(dashboard_app.get_active_project_alias(), "(default)")
