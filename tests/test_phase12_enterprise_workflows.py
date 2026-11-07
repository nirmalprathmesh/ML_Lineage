"""
tests.test_phase12_enterprise_workflows
=======================================
Phase 12: Enterprise Workflow Reliability & Integration Hardening.
Tests real CLI-first developer workflows, multi-project isolation,
nested directory discovery, error handling, DVC/MLflow integrations,
and transactional rollback.
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

from ml_lineage.cli import main
from ml_lineage.core.context import ProjectContext
from ml_lineage.core.hashing import compute_file_md5
from ml_lineage.core.registry import ProjectRegistry


def _cleanup(path: str | Path | None) -> None:
    if path and Path(path).exists():
        shutil.rmtree(str(path), ignore_errors=True)


def _parse_json(res) -> dict:
    raw = getattr(res, "stdout", "") or res.output
    raw = raw.strip()
    idx = raw.find("{")
    if idx != -1:
        return json.loads(raw[idx:])
    return json.loads(raw)


class TestPhase12Workflows(unittest.TestCase):
    """Real engineering workflow validation across isolated synthetic projects."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_root = tempfile.mkdtemp(prefix="phase12_")
        self.registry_file = Path(self.temp_root) / "enterprise_registry.json"
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.registry_file)
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)

    def tearDown(self):
        os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
        _cleanup(self.temp_root)

    # ─────────────────────────────────────────────────────────────────────────
    # Workflow 1: New ML Project Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    def test_workflow_1_new_ml_project(self):
        """Workflow 1: Create synthetic ML repo, git init, ml-lineage init, status, doctor, projects."""
        proj = Path(self.temp_root) / "recommender_system"
        proj.mkdir(parents=True, exist_ok=True)
        (proj / "src").mkdir(exist_ok=True)
        (proj / "src" / "train.py").write_text("print('training...')", encoding="utf-8")
        (proj / "params.yaml").write_text("mlflow:\n  experiment_name: RecSys_Prod\n", encoding="utf-8")

        # Initialize Git in the project
        subprocess.run(["git", "init"], cwd=str(proj), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Test Engineer"], cwd=str(proj), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "engineer@enterprise.local"], cwd=str(proj), capture_output=True, check=True)
        subprocess.run(["git", "add", "."], cwd=str(proj), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(proj), capture_output=True, check=True)

        cwd_before = os.getcwd()
        try:
            os.chdir(str(proj))

            # 1. ml-lineage init
            res_init = self.runner.invoke(main, ["init", "--name", "recsys-prod", "--yes"])
            self.assertEqual(res_init.exit_code, 0, f"init output: {res_init.output}")
            self.assertIn("Project successfully initialized", res_init.output)

            # 2. ml-lineage status
            res_status = self.runner.invoke(main, ["status", "--json"])
            self.assertEqual(res_status.exit_code, 0, f"status output: {res_status.output}")
            data_status = _parse_json(res_status)
            self.assertEqual(data_status["project"]["name"], "recsys-prod")
            self.assertEqual(Path(data_status["project"]["root"]).resolve(), proj.resolve())

            # 3. ml-lineage doctor
            res_doc = self.runner.invoke(main, ["doctor"])
            self.assertEqual(res_doc.exit_code, 0, f"doctor output: {res_doc.output}")
            self.assertIn("[OK]    Project Directory", res_doc.output)
            self.assertIn("[OK]    Git Repository", res_doc.output)

            # 4. ml-lineage projects
            res_proj = self.runner.invoke(main, ["projects"])
            self.assertEqual(res_proj.exit_code, 0, f"projects output: {res_proj.output}")
            self.assertIn("recsys-prod", res_proj.output)
        finally:
            os.chdir(cwd_before)

    # ─────────────────────────────────────────────────────────────────────────
    # Workflow 2: Nested Development Directories
    # ─────────────────────────────────────────────────────────────────────────
    def test_workflow_2_nested_development_directories(self):
        """Workflow 2: Execute CLI commands from src/, src/models/, notebooks/ resolving project root."""
        proj = Path(self.temp_root) / "vision_defect_detector"
        proj.mkdir(parents=True, exist_ok=True)
        (proj / "src" / "models").mkdir(parents=True, exist_ok=True)
        (proj / "notebooks").mkdir(parents=True, exist_ok=True)
        (proj / "params.yaml").write_text("mlflow:\n  experiment_name: Vision_Defects\n", encoding="utf-8")

        # Initialize tracking configuration
        self.runner.invoke(main, ["init", "--path", str(proj), "--name", "vision-defect", "--yes"])

        cwd_before = os.getcwd()
        try:
            # Test 1: From src/
            os.chdir(str(proj / "src"))
            res_src = self.runner.invoke(main, ["status", "--json"])
            self.assertEqual(res_src.exit_code, 0)
            data_src = _parse_json(res_src)
            self.assertEqual(Path(data_src["project"]["root"]).resolve(), proj.resolve())

            # Test 2: From src/models/
            os.chdir(str(proj / "src" / "models"))
            res_models = self.runner.invoke(main, ["status", "--json"])
            self.assertEqual(res_models.exit_code, 0)
            data_models = _parse_json(res_models)
            self.assertEqual(Path(data_models["project"]["root"]).resolve(), proj.resolve())

            # Test 3: From notebooks/
            os.chdir(str(proj / "notebooks"))
            res_nb = self.runner.invoke(main, ["status", "--json"])
            self.assertEqual(res_nb.exit_code, 0)
            data_nb = _parse_json(res_nb)
            self.assertEqual(Path(data_nb["project"]["root"]).resolve(), proj.resolve())
        finally:
            os.chdir(cwd_before)

    # ─────────────────────────────────────────────────────────────────────────
    # Workflow 3: Multiple Projects Isolation
    # ─────────────────────────────────────────────────────────────────────────
    def test_workflow_3_three_projects_isolation(self):
        """Workflow 3: Create 3 synthetic ML projects, switch between them, verify zero cross-talk."""
        projects = {
            "credit-risk": Path(self.temp_root) / "proj_credit",
            "fraud-radar": Path(self.temp_root) / "proj_fraud",
            "supply-chain": Path(self.temp_root) / "proj_supply",
        }

        for alias, p_path in projects.items():
            p_path.mkdir(parents=True, exist_ok=True)
            db_name = f"{alias}.db"
            (p_path / "params.yaml").write_text(
                yaml.dump({"mlflow": {"experiment_name": f"Exp_{alias}", "tracking_uri": f"sqlite:///{db_name}"}}),
                encoding="utf-8",
            )
            res = self.runner.invoke(main, ["init", "--path", str(p_path), "--name", alias, "--yes"])
            self.assertEqual(res.exit_code, 0)

        # Verify targeting by alias isolates tracking stores
        for alias, p_path in projects.items():
            res = self.runner.invoke(main, ["status", "--project", alias, "--json"])
            self.assertEqual(res.exit_code, 0)
            data = _parse_json(res)
            self.assertEqual(data["project"]["name"], alias)
            self.assertEqual(Path(data["project"]["root"]).resolve(), p_path.resolve())
            self.assertIn(f"{alias}.db", data["mlflow"]["tracking_uri"])

    # ─────────────────────────────────────────────────────────────────────────
    # Workflow 4: Existing MLflow Project Artifact Verification
    # ─────────────────────────────────────────────────────────────────────────
    def test_workflow_4_mlflow_artifact_verification(self):
        """Workflow 4: MLflow experiment logging parameters, metrics, artifacts, and hash verification."""
        proj = Path(self.temp_root) / "pricing_engine"
        proj.mkdir(parents=True, exist_ok=True)
        db_path = proj / "pricing_mlflow.db"

        # Initialize project configuration
        (proj / "params.yaml").write_text(
            yaml.dump({"mlflow": {"experiment_name": "Pricing_Model", "tracking_uri": f"sqlite:///{db_path}"}}),
            encoding="utf-8",
        )
        self.runner.invoke(main, ["init", "--path", str(proj), "--name", "pricing-engine", "--yes"])

        # Create synthetic artifacts
        artifact_file = proj / "pricing_model.bin"
        artifact_file.write_text("binary-model-weights-v1", encoding="utf-8")
        expected_hash = compute_file_md5(artifact_file)

        # Log an MLflow run
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment("Pricing_Model")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_param("learning_rate", 0.05)
            mlflow.log_metric("rmse", 14.2)
            mlflow.set_tag("artifact_hash.pricing_model", expected_hash)
            mlflow.set_tag("artifact_path.pricing_model", "pricing_model.bin")

        # Inspect run via CLI
        res_runs = self.runner.invoke(main, ["runs", "--project", "pricing-engine"])
        self.assertEqual(res_runs.exit_code, 0)
        self.assertIn(run_id[:16], res_runs.output)

        # Inspect run details
        res_run = self.runner.invoke(main, ["inspect", run_id[:8], "--project", "pricing-engine", "--json"])
        self.assertEqual(res_run.exit_code, 0, f"inspect failed: {res_run.output}")
        run_data = _parse_json(res_run)
        self.assertEqual(run_data["run_id"], run_id)
        self.assertEqual(run_data["metrics"]["rmse"], 14.2)
        self.assertIn("learning_rate", run_data["parameters"])

    # ─────────────────────────────────────────────────────────────────────────
    # Workflow 5: DVC Pipeline Integration
    # ─────────────────────────────────────────────────────────────────────────
    def test_workflow_5_dvc_pipeline_detection(self):
        """Workflow 5: DVC project with dvc.yaml pipeline is parsed and tracked."""
        proj = Path(self.temp_root) / "pipeline_project"
        proj.mkdir(parents=True, exist_ok=True)

        dvc_yaml_content = {
            "stages": {
                "preprocess": {
                    "cmd": "python src/preprocess.py",
                    "deps": ["data/raw.csv"],
                    "outs": ["data/clean.csv"],
                },
                "train": {
                    "cmd": "python src/train.py",
                    "deps": ["data/clean.csv"],
                    "outs": ["models/model.pkl"],
                },
            }
        }
        (proj / "dvc.yaml").write_text(yaml.dump(dvc_yaml_content), encoding="utf-8")
        (proj / "params.yaml").write_text("mlflow:\n  experiment_name: Pipeline_Exp\n", encoding="utf-8")

        # Initialize
        res_init = self.runner.invoke(main, ["init", "--path", str(proj), "--name", "dvc-pipeline", "--yes"])
        self.assertEqual(res_init.exit_code, 0)

        # Inspect pipeline via CLI
        res_pipe = self.runner.invoke(main, ["pipeline", "--project", "dvc-pipeline", "--json"])
        self.assertEqual(res_pipe.exit_code, 0, f"pipeline output: {res_pipe.output}")
        pipe_data = _parse_json(res_pipe)
        self.assertIn("preprocess", pipe_data["stages"])
        self.assertIn("train", pipe_data["stages"])
        self.assertEqual(pipe_data["stages"]["train"]["deps"], ["data/clean.csv"])

    # ─────────────────────────────────────────────────────────────────────────
    # Workflow 6: Recovery Safety & Transactional Rollback
    # ─────────────────────────────────────────────────────────────────────────
    def test_workflow_6_recovery_safety_and_rollback(self):
        """Workflow 6: Verify artifact recovery dry-run, rollback on failure, and original preservation."""
        from unittest.mock import MagicMock, patch
        from ml_lineage.core.recovery import RecoveryService, RecoveryExecutionError

        proj = Path(self.temp_root) / "recovery_test_project"
        proj.mkdir(parents=True, exist_ok=True)
        (proj / "models").mkdir(parents=True, exist_ok=True)
        ctx = ProjectContext.from_path(proj)

        # Create original target file
        target_file = proj / "models" / "active_model.pkl"
        initial_content = b"ORIGINAL_ACTIVE_VERSION_DO_NOT_CORRUPT"
        target_file.write_bytes(initial_content)
        original_hash = compute_file_md5(target_file)

        # Create simulated cached historical source artifact in a mock DVC cache
        cache_dir = proj / ".dvc" / "cache" / "12"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_source = cache_dir / "34567890abcdef"
        new_content = b"HISTORICAL_RECOVERED_VERSION"
        cached_source.write_bytes(new_content)
        cached_hash = compute_file_md5(cached_source)

        service = RecoveryService(ctx)

        # Mock MLflow run with artifact tags
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_12345678"
        mock_run.info.status = "FINISHED"
        mock_run.info.artifact_uri = None
        mock_run.data.tags = {
            "dvc.hash.models/active_model.pkl": cached_hash,
        }
        mock_run.data.params = {}
        mock_run.data.metrics = {}

        with patch.object(service.mlflow_adapter, "resolve_run", return_value=mock_run):
            with patch.object(service.dvc_adapter, "find_artifact_source", return_value=(cached_source, "dvc_cache")):
                plan = service.create_plan("test_run_12345678")

                # 1. Pre-flight plan creation does NOT modify the active file
                self.assertTrue(plan.can_recover)
                self.assertEqual(target_file.read_bytes(), initial_content)

                # 2. Inject verification failure during recovery to test transactional rollback
                orig_md5 = compute_file_md5
                def mock_md5(path):
                    if Path(path) == target_file and target_file.read_bytes() == new_content:
                        return "simulated_corrupted_hash"
                    return orig_md5(path)

                with patch("ml_lineage.core.recovery.compute_file_md5", side_effect=mock_md5):
                    with self.assertRaises(RecoveryExecutionError) as cm:
                        service.execute_recovery(plan)
                    self.assertTrue(cm.exception.rollback_success)

                # 3. Transactional rollback verification: original file must be restored
                self.assertTrue(target_file.is_file())
                self.assertEqual(target_file.read_bytes(), initial_content)
                self.assertEqual(compute_file_md5(target_file), original_hash)


class TestPhase12CliUsabilityAndErrors(unittest.TestCase):
    """Test Phase 12D: CLI argument validation, error handling, exit codes, and JSON outputs."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_root = tempfile.mkdtemp(prefix="phase12_err_")
        self.registry_file = Path(self.temp_root) / "test_reg.json"
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.registry_file)

    def tearDown(self):
        os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)
        _cleanup(self.temp_root)

    def test_01_help_subcommands_display_cleanly(self):
        """Every supported subcommand provides formatted help text with exit code 0."""
        subcommands = ["status", "doctor", "runs", "history", "pipeline", "init", "projects", "register", "unregister"]
        for sub in subcommands:
            res = self.runner.invoke(main, [sub, "--help"])
            self.assertEqual(res.exit_code, 0, f"{sub} --help failed: {res.output}")
            self.assertIn("Usage:", res.output)

    def test_02_missing_project_alias_fails_cleanly(self):
        """Specifying a nonexistent project alias exits with code 1 and clean error message."""
        res = self.runner.invoke(main, ["status", "--project", "nonexistent-project-xyz", "--json"])
        self.assertEqual(res.exit_code, 1)
        data = _parse_json(res)
        self.assertIn("error", data)
        self.assertIn("not found in registry", data["error"])

    def test_03_invalid_run_id_rejected_cleanly(self):
        """Providing an invalid run ID (bad chars, empty) exits with code 1."""
        proj = Path(self.temp_root) / "dummy_proj"
        proj.mkdir()
        self.runner.invoke(main, ["init", "--path", str(proj), "--name", "dummy", "--yes"])

        res = self.runner.invoke(main, ["inspect", "invalid_non_hex_run_id!@#", "--project", "dummy"])
        self.assertEqual(res.exit_code, 1)

    def test_04_missing_directory_fails_cleanly(self):
        """Targeting a nonexistent directory path exits with code 1."""
        nonexistent = Path(self.temp_root) / "does_not_exist"
        res = self.runner.invoke(main, ["status", "--path", str(nonexistent)])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("does not exist", res.output)


class TestPhase12IntegrationCompatibility(unittest.TestCase):
    """Test Phase 12E: Graceful degradation across heterogeneous ML project tool combinations."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_root = tempfile.mkdtemp(prefix="phase12_compat_")
        self.registry_file = Path(self.temp_root) / "compat_reg.json"
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.registry_file)

    def tearDown(self):
        os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)
        _cleanup(self.temp_root)

    def test_01_no_tooling_ml_project(self):
        """Plain project with no Git, no DVC, no MLflow degrades gracefully without crash."""
        proj = Path(self.temp_root) / "bare_project"
        proj.mkdir()
        (proj / "train.py").write_text("# simple code", encoding="utf-8")

        res_init = self.runner.invoke(main, ["init", "--path", str(proj), "--name", "bare-proj", "--yes"])
        self.assertEqual(res_init.exit_code, 0)

        res_status = self.runner.invoke(main, ["status", "--path", str(proj), "--json"])
        self.assertEqual(res_status.exit_code, 0)
        data = _parse_json(res_status)
        self.assertEqual(data["git"]["branch"], "unavailable")
        self.assertIn("not initialized", data["dvc"]["pipeline"])

        res_doc = self.runner.invoke(main, ["doctor", "--path", str(proj)])
        self.assertEqual(res_doc.exit_code, 0)
        self.assertIn("[WARN]  Git Repository", res_doc.output)
        self.assertIn("[WARN]  DVC Setup", res_doc.output)

    def test_02_git_only_project(self):
        """Project with Git only initializes and reports git branch/commit accurately."""
        proj = Path(self.temp_root) / "git_only"
        proj.mkdir()
        subprocess.run(["git", "init"], cwd=str(proj), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Dev"], cwd=str(proj), capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email", "dev@test.local"], cwd=str(proj), capture_output=True, check=True)
        (proj / "code.py").write_text("print('hello')", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(proj), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(proj), capture_output=True, check=True)

        res_init = self.runner.invoke(main, ["init", "--path", str(proj), "--name", "git-proj", "--yes"])
        self.assertEqual(res_init.exit_code, 0)

        res_doc = self.runner.invoke(main, ["doctor", "--path", str(proj)])
        self.assertEqual(res_doc.exit_code, 0)
        self.assertIn("[OK]    Git Repository", res_doc.output)
        self.assertIn("[WARN]  DVC Setup", res_doc.output)

    def test_03_custom_params_filename(self):
        """Project with custom parameters file is handled gracefully."""
        proj = Path(self.temp_root) / "custom_cfg"
        proj.mkdir()
        (proj / ".ml-lineage.yaml").write_text(
            yaml.dump({
                "version": 1,
                "project": {"name": "custom-cfg-proj", "root": "."},
                "params": {"file": "custom_params.yaml"},
                "mlflow": {"enabled": True, "tracking_uri": "sqlite:///custom.db"},
            }),
            encoding="utf-8",
        )
        ctx = ProjectContext.from_path(proj)
        self.assertEqual(ctx.project_name, "custom-cfg-proj")
        self.assertIn("custom.db", ctx.tracking_uri)


class TestPhase12RegistryAndConcurrency(unittest.TestCase):
    """Test Phase 12F: Configuration & registry reliability, concurrency, atomic writes."""

    def setUp(self):
        self.temp_root = tempfile.mkdtemp(prefix="phase12_reg_")
        self.registry_file = Path(self.temp_root) / "concurrent_reg.json"
        self.reg = ProjectRegistry(self.registry_file)

    def tearDown(self):
        _cleanup(self.temp_root)

    def test_01_atomic_save_and_recovery(self):
        """Saving registry is atomic and does not corrupt existing data."""
        proj_a = Path(self.temp_root) / "proj_a"
        proj_a.mkdir()
        self.reg.register("proj-a", proj_a, description="Project A")

        # Load in a fresh instance
        reg2 = ProjectRegistry(self.registry_file)
        self.assertIn("proj-a", reg2.list_projects())
        self.assertEqual(Path(reg2.get("proj-a")["path"]).resolve(), proj_a.resolve())

    def test_02_concurrent_registrations(self):
        """Simultaneous registrations in multiple threads preserve registry integrity."""
        import concurrent.futures

        dirs = []
        for i in range(5):
            d = Path(self.temp_root) / f"thread_proj_{i}"
            d.mkdir()
            dirs.append((f"alias-{i}", d))

        def _reg_worker(pair):
            alias, path = pair
            reg = ProjectRegistry(self.registry_file)
            reg.register(alias, path, overwrite=True)
            return alias

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            results = list(ex.map(_reg_worker, dirs))

        final_reg = ProjectRegistry(self.registry_file)
        registered = final_reg.list_projects()
        for alias, _ in dirs:
            self.assertIn(alias, registered, f"Alias {alias} missing after concurrent writes.")

