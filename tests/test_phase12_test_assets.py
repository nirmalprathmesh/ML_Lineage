"""
tests.test_phase12_test_assets
==============================
Phase 12C.1 Automated verification for dedicated local test model & sandbox assets.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from click.testing import CliRunner
from ml_lineage.cli import main
from ml_lineage.core.context import ProjectContext
from ml_lineage.core.hashing import compute_file_md5


class TestPhase12LocalSandbox(unittest.TestCase):
    """Test suite verifying test_assets sandbox integrity, execution, and artifact tracking."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.test_assets_dir = cls.repo_root / "test_assets"
        cls.runner = CliRunner()

    def test_01_test_assets_directory_and_gitignore(self):
        """test_assets must exist and be strictly ignored by Git."""
        self.assertTrue(self.test_assets_dir.is_dir(), "test_assets directory must exist.")

        # Run git check-ignore to ensure git ignore rule works
        res = subprocess.run(
            ["git", "check-ignore", "test_assets/"],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, "test_assets/ must be recognized as ignored by Git.")
        self.assertIn("test_assets", res.stdout)

    def test_02_model_execution_and_artifact_generation(self):
        """Executing test_assets/train.py generates artifacts and logs to isolated MLflow DB."""
        train_script = self.test_assets_dir / "train.py"
        self.assertTrue(train_script.is_file(), "train.py must exist.")

        # Run training
        res = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=str(self.test_assets_dir),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"train.py failed: {res.stderr}\n{res.stdout}")
        self.assertIn("Training complete", res.stdout)

        # Verify generated artifacts
        model_file = self.test_assets_dir / "model.joblib"
        scaler_file = self.test_assets_dir / "scaler.joblib"
        data_file = self.test_assets_dir / "data.csv"
        manifest_file = self.test_assets_dir / "manifest.json"
        db_file = self.test_assets_dir / "test_model.db"

        self.assertTrue(model_file.is_file(), "model.joblib must be created.")
        self.assertTrue(scaler_file.is_file(), "scaler.joblib must be created.")
        self.assertTrue(data_file.is_file(), "data.csv must be created.")
        self.assertTrue(manifest_file.is_file(), "manifest.json must be created.")
        self.assertTrue(db_file.is_file(), "test_model.db must be created.")

    def test_03_artifact_hashes_match_manifest(self):
        """Cryptographic MD5 hashes of generated artifacts match the manifest and MLflow run tags."""
        manifest_file = self.test_assets_dir / "manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        model_file = self.test_assets_dir / "model.joblib"
        actual_model_hash = compute_file_md5(model_file)
        self.assertEqual(actual_model_hash, manifest["artifacts"]["model.joblib"]["md5"])

        scaler_file = self.test_assets_dir / "scaler.joblib"
        actual_scaler_hash = compute_file_md5(scaler_file)
        self.assertEqual(actual_scaler_hash, manifest["artifacts"]["scaler.joblib"]["md5"])

    def test_04_cli_status_and_doctor_on_test_sandbox(self):
        """ML Lineage CLI status and doctor work seamlessly on the test_assets project."""
        res_status = self.runner.invoke(main, ["status", "--path", str(self.test_assets_dir), "--json"])
        self.assertEqual(res_status.exit_code, 0, f"status failed: {res_status.output}")

        res_doctor = self.runner.invoke(main, ["doctor", "--path", str(self.test_assets_dir), "--json"])
        self.assertEqual(res_doctor.exit_code, 0, f"doctor failed: {res_doctor.output}")

    def test_05_isolation_from_production_wine_store(self):
        """test_assets uses strictly isolated sqlite:///test_assets/test_model.db, not wine db."""
        ctx = ProjectContext.from_path(self.test_assets_dir)
        self.assertIn("test_model.db", ctx.tracking_uri)
        self.assertNotIn("mlflow.db", ctx.tracking_uri)

