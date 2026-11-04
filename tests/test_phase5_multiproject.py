"""
tests/test_phase5_multiproject.py
=================================
Comprehensive end-to-end test suite for Phase 5:
1. Packaging and version verification
2. Multi-project registry commands (register, unregister, projects)
3. Configuration precedence hierarchy (CLI > Env > .ml-lineage.yaml > params.yaml > Defaults)
4. Project A: Git + MLflow without DVC (pure experiment tracking, non-accuracy metrics)
5. Project B: Git + DVC with non-Wine pipeline structure
6. History and report CLI subcommands (text, json, markdown, --output)
7. Multi-project isolation and directory independence
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from click.testing import CliRunner

import ml_lineage
from ml_lineage.adapters.mlflow import MLflowAdapter
from ml_lineage.cli import main
from ml_lineage.core.context import ProjectContext
from ml_lineage.core.lineage import trace_run_lineage
from ml_lineage.core.registry import ProjectRegistry


class TestPackagingAndVersion(unittest.TestCase):
    """Verify package version, entrypoints, and optional dependencies."""

    def test_version_bumped(self):
        self.assertEqual(ml_lineage.__version__, "0.2.0")

    def test_pyproject_toml_configuration(self):
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        self.assertTrue(pyproject_path.is_file())
        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('name = "ml-lineage"', content)
        self.assertIn('version = "0.2.0"', content)
        self.assertIn('ml-lineage = "ml_lineage.cli:main"', content)
        self.assertIn("[project.optional-dependencies]", content)
        self.assertIn("ui =", content)
        self.assertIn("dev =", content)


class TestProjectRegistryCLI(unittest.TestCase):
    """Test CLI commands: register, unregister, projects."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="lineage_reg_test_")
        self.registry_file = Path(self.temp_dir) / "registry.json"
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.registry_file)
        self.runner = CliRunner()

        # Dummy project paths
        self.proj1_path = Path(self.temp_dir) / "project_alpha"
        self.proj1_path.mkdir(parents=True, exist_ok=True)
        self.proj2_path = Path(self.temp_dir) / "project_beta"
        self.proj2_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        if hasattr(self, "_saved_mlflow_uri") and self._saved_mlflow_uri is not None:
            os.environ["MLFLOW_TRACKING_URI"] = self._saved_mlflow_uri
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_and_list_projects(self):
        # 1. Register project_alpha
        res = self.runner.invoke(
            main,
            ["register", str(self.proj1_path), "--alias", "alpha", "--description", "Alpha project", "--json"],
        )
        self.assertEqual(res.exit_code, 0, msg=res.output)
        data = json.loads(res.output)
        self.assertEqual(data["alias"], "alpha")
        self.assertEqual(Path(data["path"]).resolve(), self.proj1_path.resolve())

        # 2. Register project_beta in human format
        res2 = self.runner.invoke(
            main,
            ["register", str(self.proj2_path), "--alias", "beta", "--description", "Beta project"],
        )
        self.assertEqual(res2.exit_code, 0, msg=res2.output)
        self.assertIn("[OK] Project successfully registered:", res2.output)
        self.assertIn("beta", res2.output)

        # 3. List projects
        res_list = self.runner.invoke(main, ["projects", "--json"])
        self.assertEqual(res_list.exit_code, 0)
        projects = json.loads(res_list.output)
        self.assertIn("alpha", projects)
        self.assertIn("beta", projects)
        self.assertEqual(len(projects), 2)

    def test_register_duplicate_conflict(self):
        # Register alpha
        res1 = self.runner.invoke(main, ["register", str(self.proj1_path), "--alias", "alpha"])
        self.assertEqual(res1.exit_code, 0)

        # Register same alias with different path without --overwrite should fail
        res2 = self.runner.invoke(main, ["register", str(self.proj2_path), "--alias", "alpha"])
        self.assertEqual(res2.exit_code, 1)

        # Register same alias with --overwrite should succeed
        res3 = self.runner.invoke(main, ["register", str(self.proj2_path), "--alias", "alpha", "--overwrite"])
        self.assertEqual(res3.exit_code, 0)
        reg = ProjectRegistry(self.registry_file)
        self.assertEqual(Path(reg.get("alpha")["path"]).resolve(), self.proj2_path.resolve())

    def test_unregister_safety_and_confirmation(self):
        # Register alpha
        self.runner.invoke(main, ["register", str(self.proj1_path), "--alias", "alpha"])

        # Unregister non-existent alias should fail with code 1
        res_missing = self.runner.invoke(main, ["unregister", "nonexistent", "--yes"])
        self.assertEqual(res_missing.exit_code, 1)

        # Interactive unregister declined (input 'n') should abort with code 2
        res_declined = self.runner.invoke(main, ["unregister", "alpha"], input="n\n")
        self.assertEqual(res_declined.exit_code, 2)
        reg = ProjectRegistry(self.registry_file)
        self.assertIsNotNone(reg.get("alpha"))

        # Explicit --yes should unregister immediately
        res_yes = self.runner.invoke(main, ["unregister", "alpha", "--yes", "--json"])
        self.assertEqual(res_yes.exit_code, 0)
        data = json.loads(res_yes.output)
        self.assertEqual(data["status"], "unregistered")
        self.assertEqual(data["alias"], "alpha")
        self.assertIsNone(reg.get("alpha"))


class TestConfigurationPrecedence(unittest.TestCase):
    """Test the 5-tier configuration precedence."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="lineage_precedence_")
        self.root = Path(self.temp_dir)
        self.runner = CliRunner()

        # Clean environment
        self._saved_env = {}
        for var in ("ML_LINEAGE_PROJECT", "ML_LINEAGE_PATH", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "ML_LINEAGE_TRACKING_URI", "ML_LINEAGE_EXPERIMENT"):
            self._saved_env[var] = os.environ.pop(var, None)

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tracking_uri_precedence(self):
        # Tier 5: Default fallback
        ctx_default = ProjectContext(self.root)
        self.assertEqual(ctx_default.tracking_uri, f"sqlite:///{self.root / 'mlflow.db'}")

        # Tier 4: params.yaml
        params_file = self.root / "params.yaml"
        with open(params_file, "w", encoding="utf-8") as f:
            yaml.dump({"mlflow": {"tracking_uri": "sqlite:///params_mlflow.db"}}, f)
        ctx_params = ProjectContext(self.root)
        self.assertEqual(ctx_params.tracking_uri, f"sqlite:///{self.root / 'params_mlflow.db'}")

        # Tier 3: .ml-lineage.yaml overrides params.yaml
        config_file = self.root / ".ml-lineage.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump({"mlflow": {"tracking_uri": "sqlite:///config_mlflow.db"}}, f)
        ctx_config = ProjectContext(self.root)
        self.assertEqual(ctx_config.tracking_uri, f"sqlite:///{self.root / 'config_mlflow.db'}")

        # Tier 2: Environment variable overrides .ml-lineage.yaml and params.yaml
        os.environ["MLFLOW_TRACKING_URI"] = "http://remote-mlflow:5000"
        ctx_env = ProjectContext(self.root)
        self.assertEqual(ctx_env.tracking_uri, "http://remote-mlflow:5000")

        # Tier 1: Explicit custom argument overrides environment variable
        ctx_explicit = ProjectContext(self.root, custom_tracking_uri="http://explicit-host:8000")
        self.assertEqual(ctx_explicit.tracking_uri, "http://explicit-host:8000")

    def test_experiment_name_precedence(self):
        # Tier 4 / Default
        ctx = ProjectContext(self.root)
        self.assertEqual(ctx.experiment_name, "ML_Lineage_Experiments")

        # Tier 3: params.yaml
        with open(self.root / "params.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"mlflow": {"experiment_name": "Params_Experiment"}}, f)
        self.assertEqual(ProjectContext(self.root).experiment_name, "Params_Experiment")

        # Tier 2: .ml-lineage.yaml overrides params.yaml
        with open(self.root / ".ml-lineage.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"mlflow": {"experiment_name": "Config_Experiment"}}, f)
        self.assertEqual(ProjectContext(self.root).experiment_name, "Config_Experiment")

        # Tier 1: Environment variable overrides .ml-lineage.yaml
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "Env_Experiment"
        self.assertEqual(ProjectContext(self.root).experiment_name, "Env_Experiment")

    def test_custom_file_overrides_in_config(self):
        # Configure custom filenames in .ml-lineage.yaml
        config_file = self.root / ".ml-lineage.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump({
                "params": {"file": "custom_params.yaml"},
                "dvc": {"lock_file": "custom_dvc.lock", "pipeline_file": "custom_dvc.yaml"}
            }, f)

        ctx = ProjectContext(self.root)
        self.assertEqual(ctx.params_path, self.root / "custom_params.yaml")
        self.assertEqual(ctx.dvc_lock_path, self.root / "custom_dvc.lock")
        self.assertEqual(ctx.dvc_yaml_path, self.root / "custom_dvc.yaml")


class TestProjectA_NoDVC(unittest.TestCase):
    """Test Project A: Git + MLflow without DVC (pure experiment tracking, non-Wine layout)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="lineage_proj_a_")
        self.root = Path(self.temp_dir)
        self.runner = CliRunner()

        # Save and clear MLflow env state so it doesn't leak
        self._saved_mlflow_uri = os.environ.pop("MLFLOW_TRACKING_URI", None)

        # 1. Initialize Git repository
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.root, check=True)

        readme = self.root / "README.md"
        readme.write_text("# Project A - Pure MLflow NLP\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-m", "initial commit", "-q"], cwd=self.root, check=True)

        # 2. Setup isolated MLflow tracking store and log a run with custom metrics & artifact tags
        import mlflow
        db_path = self.root / "project_a_mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("nlp_sentiment")

        # Create dummy model file
        model_file = self.root / "sentiment_model.bin"
        model_file.write_bytes(b"dummy_weights_12345")
        import hashlib
        model_hash = hashlib.md5(b"dummy_weights_12345").hexdigest()

        with mlflow.start_run(run_name="bert_finetuned") as active_run:
            self.run_id = active_run.info.run_id
            mlflow.log_params({"model_type": "transformer", "learning_rate": 2e-5, "epochs": 3})
            mlflow.log_metrics({"loss": 0.1423, "f1_score": 0.9412})
            mlflow.set_tags({
                "git.commit_sha": "a1b2c3d4e5f6",
                "git.branch": "main",
                "git.is_dirty": "False",
                "hash.sentiment_model.bin": model_hash,
                "env.python_version": "3.11.0",
                "env.torch_version": "2.1.0",
            })

        # Save config pointing to this tracking uri
        config_file = self.root / ".ml-lineage.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump({
                "version": 1,
                "project": {"name": "nlp-project"},
                "mlflow": {
                    "tracking_uri": f"sqlite:///project_a_mlflow.db",
                    "experiment_name": "nlp_sentiment"
                }
            }, f)

    def tearDown(self):
        # Restore MLflow env state
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        if self._saved_mlflow_uri is not None:
            os.environ["MLFLOW_TRACKING_URI"] = self._saved_mlflow_uri
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_doctor_without_dvc(self):
        res = self.runner.invoke(main, ["doctor", "--path", str(self.root), "--json"])
        self.assertEqual(res.exit_code, 0, msg=res.output)
        data = json.loads(res.output)
        # DVC should be a WARN, MLflow should be OK
        dvc_check = next(c for c in data["checks"] if "DVC" in c["component"])
        self.assertEqual(dvc_check["status"], "WARN")
        mlflow_check = next(c for c in data["checks"] if "MLflow" in c["component"])
        self.assertEqual(mlflow_check["status"], "OK")

    def test_status_with_generic_metrics(self):
        res = self.runner.invoke(main, ["status", "--path", str(self.root), "--json"])
        self.assertEqual(res.exit_code, 0)
        data = json.loads(res.output)
        latest = data["mlflow"]["latest_run"]
        self.assertEqual(latest["run_id"], self.run_id)
        # Generic metric handling: f1_score should be discovered
        self.assertIn("f1_score", latest["primary_metric"]["name"])
        self.assertAlmostEqual(latest["primary_metric"]["value"], 0.9412, places=4)

    def test_runs_and_history_subcommands(self):
        # 1. runs
        res_runs = self.runner.invoke(main, ["runs", "--path", str(self.root), "--json"])
        self.assertEqual(res_runs.exit_code, 0)
        runs_data = json.loads(res_runs.output)
        self.assertEqual(len(runs_data), 1)
        self.assertEqual(runs_data[0]["run_id"], self.run_id)
        self.assertAlmostEqual(runs_data[0]["primary_metric"]["value"], 0.9412, places=4)

        # 2. history
        res_hist = self.runner.invoke(main, ["history", "--path", str(self.root), "--json"])
        self.assertEqual(res_hist.exit_code, 0)
        hist_data = json.loads(res_hist.output)
        self.assertEqual(len(hist_data), 1)
        self.assertEqual(hist_data[0]["run_id"], self.run_id)
        self.assertEqual(hist_data[0]["git_branch"], "main")
        self.assertEqual(hist_data[0]["artifact_tags_count"], 1)

    def test_report_subcommand(self):
        # Human text format
        res_text = self.runner.invoke(main, ["report", self.run_id, "--path", str(self.root)])
        self.assertEqual(res_text.exit_code, 0)
        self.assertIn("RUN PROVENANCE & LINEAGE REPORT", res_text.output)
        self.assertIn(self.run_id, res_text.output)
        self.assertIn("f1_score", res_text.output)

        # JSON format
        res_json = self.runner.invoke(main, ["report", self.run_id, "--path", str(self.root), "--format", "json"])
        self.assertEqual(res_json.exit_code, 0)
        rep = json.loads(res_json.output)
        self.assertEqual(rep["run"]["run_id"], self.run_id)
        self.assertEqual(rep["pipeline_summary"]["available"], False)
        self.assertEqual(len(rep["lineage_stages"]), 1)
        self.assertEqual(rep["lineage_stages"][0]["stage"], "unstructured_artifacts")

        # Markdown format saved to file
        out_file = self.root / "provenance_report.md"
        res_md = self.runner.invoke(
            main,
            ["report", self.run_id, "--path", str(self.root), "--format", "markdown", "--output", str(out_file)]
        )
        self.assertEqual(res_md.exit_code, 0)
        self.assertTrue(out_file.is_file())
        md_content = out_file.read_text(encoding="utf-8")
        self.assertIn("# ML Lineage Provenance Report", md_content)
        self.assertIn(self.run_id, md_content)


class TestProjectB_CustomDVC(unittest.TestCase):
    """Test Project B: Git + DVC with a custom, non-Wine multi-stage pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="lineage_proj_b_")
        self.root = Path(self.temp_dir)
        self.runner = CliRunner()

        # Save and clear MLflow env state so it doesn't leak
        self._saved_mlflow_uri = os.environ.pop("MLFLOW_TRACKING_URI", None)

        # 1. Initialize Git repository
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "TestUserB"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "testb@example.com"], cwd=self.root, check=True)

        # 2. Create custom DVC pipeline files (stages: ingest, train_model, evaluate)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "models").mkdir(parents=True, exist_ok=True)

        data_file = self.root / "data" / "raw_corpus.txt"
        data_file.write_text("sample dataset line 1\nsample dataset line 2\n", encoding="utf-8")

        model_file = self.root / "models" / "recommender.bin"
        model_file.write_bytes(b"recommender_weights_bytes_xyz")

        import hashlib
        data_hash = hashlib.md5(data_file.read_bytes()).hexdigest()
        model_hash = hashlib.md5(model_file.read_bytes()).hexdigest()

        dvc_yaml_content = {
            "stages": {
                "ingest": {
                    "cmd": "python scripts/ingest.py",
                    "outs": ["data/raw_corpus.txt"]
                },
                "train_model": {
                    "cmd": "python scripts/train.py",
                    "deps": ["data/raw_corpus.txt"],
                    "outs": ["models/recommender.bin"]
                }
            }
        }
        with open(self.root / "dvc.yaml", "w", encoding="utf-8") as f:
            yaml.dump(dvc_yaml_content, f)

        dvc_lock_content = {
            "schema": "2.0",
            "stages": {
                "ingest": {
                    "cmd": "python scripts/ingest.py",
                    "outs": [{"path": "data/raw_corpus.txt", "md5": data_hash}]
                },
                "train_model": {
                    "cmd": "python scripts/train.py",
                    "deps": [{"path": "data/raw_corpus.txt", "md5": data_hash}],
                    "outs": [{"path": "models/recommender.bin", "md5": model_hash}]
                }
            }
        }
        with open(self.root / "dvc.lock", "w", encoding="utf-8") as f:
            yaml.dump(dvc_lock_content, f)

        # 3. Setup MLflow tracking
        import mlflow
        db_path = self.root / "recommender_mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("recommendation_engine")

        with mlflow.start_run(run_name="collaborative_filtering_v1") as active_run:
            self.run_id = active_run.info.run_id
            mlflow.log_params({"latent_factors": 50, "regularization": 0.05})
            mlflow.log_metrics({"rmse": 0.8123, "mae": 0.6541})
            mlflow.set_tags({
                "git.commit_sha": "deadbeef1234",
                "git.branch": "feature/recs",
                "git.is_dirty": "False",
                "dvc.hash.data/raw_corpus.txt": data_hash,
                "dvc.hash.models/recommender.bin": model_hash,
            })

        # Save .ml-lineage.yaml
        with open(self.root / ".ml-lineage.yaml", "w", encoding="utf-8") as f:
            yaml.dump({
                "version": 1,
                "project": {"name": "recommender-system"},
                "mlflow": {
                    "tracking_uri": "sqlite:///recommender_mlflow.db",
                    "experiment_name": "recommendation_engine"
                }
            }, f)

    def tearDown(self):
        # Restore MLflow env state
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        if self._saved_mlflow_uri is not None:
            os.environ["MLFLOW_TRACKING_URI"] = self._saved_mlflow_uri
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_subcommand_custom_dag(self):
        res = self.runner.invoke(main, ["pipeline", "--path", str(self.root), "--json"])
        self.assertEqual(res.exit_code, 0, msg=res.output)
        dag = json.loads(res.output)
        self.assertEqual(len(dag["stages"]), 2)
        stages = dag["stages"]
        self.assertIn("ingest", stages)
        self.assertIn("train_model", stages)
        self.assertIn("ingest", stages["train_model"]["upstream"])

    def test_lineage_and_artifact_verification(self):
        ctx = ProjectContext.from_path(self.root)
        rep = trace_run_lineage(ctx, self.run_id)
        data = rep.to_dict()

        self.assertEqual(data["run"]["run_id"], self.run_id)
        self.assertTrue(data["pipeline_summary"]["available"])
        self.assertEqual(data["pipeline_summary"]["total_stages"], 2)
        self.assertEqual(data["pipeline_summary"]["observed_stages"], 2)

        # Verification summary: both artifacts exist in workspace and match MD5
        v = data["verification_summary"]
        self.assertTrue(v["is_lineage_verified"])
        self.assertEqual(v["versioned_artifacts_tracked"], 2)
        self.assertEqual(v["verified_in_workspace"], 2)
        self.assertEqual(v["workspace_hash_mismatches"], 0)


class TestMultiProjectIsolation(unittest.TestCase):
    """Test switching between multiple projects without state cross-contamination."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="lineage_isolation_")
        self.registry_file = Path(self.temp_dir) / "registry.json"
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.registry_file)
        # Clear MLflow global state from any prior tests
        self._saved_mlflow_uri = os.environ.pop("MLFLOW_TRACKING_URI", None)
        self.runner = CliRunner()

        # Setup Project Alpha
        self.proj_a = Path(self.temp_dir) / "alpha"
        self.proj_a.mkdir()
        (self.proj_a / ".ml-lineage.yaml").write_text(
            yaml.dump({
                "project": {"name": "alpha"},
                "mlflow": {"tracking_uri": f"sqlite:///{self.proj_a / 'alpha.db'}"}
            }),
            encoding="utf-8"
        )

        # Setup Project Beta
        self.proj_b = Path(self.temp_dir) / "beta"
        self.proj_b.mkdir()
        (self.proj_b / ".ml-lineage.yaml").write_text(
            yaml.dump({
                "project": {"name": "beta"},
                "mlflow": {"tracking_uri": f"sqlite:///{self.proj_b / 'beta.db'}"}
            }),
            encoding="utf-8"
        )

        # Register both projects
        self.runner.invoke(main, ["register", str(self.proj_a), "--alias", "alpha"])
        self.runner.invoke(main, ["register", str(self.proj_b), "--alias", "beta"])

    def tearDown(self):
        os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        if hasattr(self, "_saved_mlflow_uri") and self._saved_mlflow_uri is not None:
            os.environ["MLFLOW_TRACKING_URI"] = self._saved_mlflow_uri
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_project_switching_via_alias(self):
        # Query Alpha using --project
        res_a = self.runner.invoke(main, ["status", "--project", "alpha", "--json"])
        self.assertEqual(res_a.exit_code, 0)
        data_a = json.loads(res_a.output)
        self.assertEqual(data_a["project"]["alias"], "alpha")
        self.assertIn("alpha.db", data_a["mlflow"]["tracking_uri"])

        # Query Beta using --project
        res_b = self.runner.invoke(main, ["status", "--project", "beta", "--json"])
        self.assertEqual(res_b.exit_code, 0)
        data_b = json.loads(res_b.output)
        self.assertEqual(data_b["project"]["alias"], "beta")
        self.assertIn("beta.db", data_b["mlflow"]["tracking_uri"])

        # Ensure no cross-pollution
        self.assertNotEqual(data_a["mlflow"]["tracking_uri"], data_b["mlflow"]["tracking_uri"])


if __name__ == "__main__":
    unittest.main()
