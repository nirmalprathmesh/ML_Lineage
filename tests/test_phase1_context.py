"""
tests/test_phase1_context.py
============================
Phase 1 verification suite:
1. ProjectContext resolution (explicit path, search parents, validation).
2. Path independence and isolation across multiple independent projects.
3. Hashing utility correctness and exception handling.
4. Core package isolation (ensuring ml_lineage never imports Streamlit).
5. GitAdapter and DVCAdapter project-scoping.
6. Backward compatibility with app and script facades.
"""

import configparser
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from ml_lineage.core.context import ProjectContext
from ml_lineage.core.hashing import compute_file_md5
from ml_lineage.adapters.git import GitAdapter
from ml_lineage.adapters.dvc import DVCAdapter
from ml_lineage.adapters.mlflow import MLflowAdapter
from ml_lineage.wine import WINE_DVC_TAG_MAP, WINE_PIPELINE_STAGES


class TestProjectContext(unittest.TestCase):
    """Test explicit ProjectContext initialization and path resolution."""

    def test_01_explicit_project_root_resolution(self):
        ctx = ProjectContext.from_path(PROJECT_ROOT)
        self.assertEqual(ctx.root, Path(PROJECT_ROOT).resolve())
        self.assertEqual(ctx.params_path, Path(PROJECT_ROOT).resolve() / "params.yaml")
        self.assertEqual(ctx.dvc_lock_path, Path(PROJECT_ROOT).resolve() / "dvc.lock")
        self.assertTrue(ctx.tracking_uri.startswith("sqlite:///"))
        self.assertIn("mlflow.db", ctx.tracking_uri)

    def test_02_nonexistent_project_path_raises_error(self):
        nonexistent = Path(tempfile.gettempdir()) / "nonexistent_ml_project_xyz_12345"
        with self.assertRaises(FileNotFoundError):
            ProjectContext.from_path(nonexistent)

    def test_03_resolve_path_helper(self):
        ctx = ProjectContext.from_path(PROJECT_ROOT)
        resolved_rel = ctx.resolve_path("data/raw/wine_dataset.csv")
        self.assertEqual(resolved_rel, Path(PROJECT_ROOT).resolve() / "data" / "raw" / "wine_dataset.csv")

        # Absolute paths should remain unchanged
        abs_target = Path(tempfile.gettempdir()).resolve()
        self.assertEqual(ctx.resolve_path(abs_target), abs_target)

    def test_04_to_dict_structure(self):
        ctx = ProjectContext.from_path(PROJECT_ROOT)
        summary = ctx.to_dict()
        self.assertIn("root", summary)
        self.assertIn("tracking_uri", summary)
        self.assertIn("experiment_name", summary)
        self.assertIn("python_executable", summary)
        self.assertIn("dvc_executable", summary)


class TestMultiProjectIsolation(unittest.TestCase):
    """Test that two separate projects remain isolated without cross-contamination."""

    def setUp(self):
        self.temp_dir_a = tempfile.TemporaryDirectory()
        self.temp_dir_b = tempfile.TemporaryDirectory()
        self.path_a = Path(self.temp_dir_a.name).resolve()
        self.path_b = Path(self.temp_dir_b.name).resolve()

        # ── Setup Project A (e.g. Fraud Detection) ──
        params_a = {
            "mlflow": {
                "experiment_name": "Fraud_Detection",
                "tracking_uri": "sqlite:///fraud_mlflow.db",
            },
            "train": {"model_type": "XGBoost"},
        }
        with open(self.path_a / "params.yaml", "w", encoding="utf-8") as f:
            yaml.dump(params_a, f)

        dvc_dir_a = self.path_a / ".dvc"
        dvc_dir_a.mkdir(parents=True, exist_ok=True)
        with open(dvc_dir_a / "config", "w", encoding="utf-8") as f:
            f.write("[core]\n    remote = remote_a\n['remote \"remote_a\"']\n    url = D:\\Storage_Fraud\n")

        lock_a = {
            "stages": {
                "train": {
                    "outs": [
                        {"path": "models/fraud_model.bin", "md5": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
                    ]
                }
            }
        }
        with open(self.path_a / "dvc.lock", "w", encoding="utf-8") as f:
            yaml.dump(lock_a, f)

        # ── Setup Project B (e.g. Churn Prediction) ──
        params_b = {
            "mlflow": {
                "experiment_name": "Customer_Churn",
                "tracking_uri": "sqlite:///churn_mlflow.db",
            },
            "train": {"model_type": "LightGBM"},
        }
        with open(self.path_b / "params.yaml", "w", encoding="utf-8") as f:
            yaml.dump(params_b, f)

        dvc_dir_b = self.path_b / ".dvc"
        dvc_dir_b.mkdir(parents=True, exist_ok=True)
        with open(dvc_dir_b / "config", "w", encoding="utf-8") as f:
            f.write("[core]\n    remote = remote_b\n['remote \"remote_b\"']\n    url = E:\\Storage_Churn\n")

        lock_b = {
            "stages": {
                "train": {
                    "outs": [
                        {"path": "models/churn_model.bin", "md5": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
                    ]
                }
            }
        }
        with open(self.path_b / "dvc.lock", "w", encoding="utf-8") as f:
            yaml.dump(lock_b, f)

    def tearDown(self):
        self.temp_dir_a.cleanup()
        self.temp_dir_b.cleanup()

    def test_01_two_projects_resolve_independent_roots_and_uris(self):
        ctx_a = ProjectContext.from_path(self.path_a)
        ctx_b = ProjectContext.from_path(self.path_b)

        # Roots are isolated
        self.assertNotEqual(ctx_a.root, ctx_b.root)
        self.assertEqual(ctx_a.root, self.path_a)
        self.assertEqual(ctx_b.root, self.path_b)

        # Tracking URIs are isolated and resolved to their respective project roots
        self.assertIn("fraud_mlflow.db", ctx_a.tracking_uri)
        self.assertIn("churn_mlflow.db", ctx_b.tracking_uri)
        self.assertNotEqual(ctx_a.tracking_uri, ctx_b.tracking_uri)
        expected_db_a = str(self.path_a / "fraud_mlflow.db")
        expected_db_b = str(self.path_b / "churn_mlflow.db")
        self.assertEqual(ctx_a.tracking_uri, f"sqlite:///{expected_db_a}")
        self.assertEqual(ctx_b.tracking_uri, f"sqlite:///{expected_db_b}")

        # Experiment names are isolated
        self.assertEqual(ctx_a.experiment_name, "Fraud_Detection")
        self.assertEqual(ctx_b.experiment_name, "Customer_Churn")

    def test_02_two_projects_resolve_independent_dvc_adapters(self):
        ctx_a = ProjectContext.from_path(self.path_a)
        ctx_b = ProjectContext.from_path(self.path_b)

        dvc_a = DVCAdapter(ctx_a)
        dvc_b = DVCAdapter(ctx_b)

        # Remote URLs
        self.assertEqual(dvc_a.context.get_dvc_remote_url(), "D:\\Storage_Fraud")
        self.assertEqual(dvc_b.context.get_dvc_remote_url(), "E:\\Storage_Churn")

        # Lock hashes
        hashes_a = dvc_a.load_lock_hashes()
        hashes_b = dvc_b.load_lock_hashes()

        self.assertIn("models/fraud_model.bin", hashes_a)
        self.assertNotIn("models/churn_model.bin", hashes_a)
        self.assertEqual(hashes_a["models/fraud_model.bin"], "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        self.assertIn("models/churn_model.bin", hashes_b)
        self.assertNotIn("models/fraud_model.bin", hashes_b)
        self.assertEqual(hashes_b["models/churn_model.bin"], "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")


class TestHashingUtility(unittest.TestCase):
    """Test cryptographic MD5 verification utility."""

    def test_01_compute_file_md5_accuracy(self):
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(b"reproducible MLOps test payload 1234567890")
            temp_path = tf.name

        try:
            expected = hashlib.md5(b"reproducible MLOps test payload 1234567890").hexdigest()
            actual = compute_file_md5(temp_path)
            self.assertEqual(actual, expected)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_02_compute_file_md5_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            compute_file_md5("nonexistent_file_path_abc.bin")


class TestCoreIsolationFromStreamlit(unittest.TestCase):
    """Verify that ml_lineage core package does not import or depend on Streamlit."""

    def test_01_core_modules_do_not_import_streamlit(self):
        # Inspect module dependencies
        import ml_lineage
        import ml_lineage.core.context
        import ml_lineage.core.hashing
        import ml_lineage.core.verification
        import ml_lineage.adapters.git
        import ml_lineage.adapters.dvc
        import ml_lineage.adapters.mlflow
        import ml_lineage.wine

        # None of the ml_lineage modules should have streamlit in their __dict__
        for mod in [
            ml_lineage,
            ml_lineage.core.context,
            ml_lineage.core.hashing,
            ml_lineage.core.verification,
            ml_lineage.adapters.git,
            ml_lineage.adapters.dvc,
            ml_lineage.adapters.mlflow,
            ml_lineage.wine,
        ]:
            self.assertNotIn("streamlit", mod.__dict__, f"Module {mod.__name__} must not import streamlit!")


class TestBackwardCompatibility(unittest.TestCase):
    """Verify that app.py and legacy facades remain 100% consistent with ml_lineage."""

    def test_01_app_helpers_match_default_context(self):
        import app
        ctx = ProjectContext.from_path(PROJECT_ROOT)
        self.assertEqual(app.get_tracking_uri(), ctx.tracking_uri)
        self.assertEqual(app.get_python_executable(), ctx.python_executable)

        # Check DVC lock hashes match
        dvc_adapter = DVCAdapter(ctx)
        self.assertEqual(app.load_dvc_lock_hashes(), dvc_adapter.load_lock_hashes())

    def test_02_wine_constants_integrity(self):
        self.assertEqual(len(WINE_DVC_TAG_MAP), 7)
        self.assertEqual(len(WINE_PIPELINE_STAGES), 4)


if __name__ == "__main__":
    unittest.main()
