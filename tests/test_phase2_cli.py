"""
tests/test_phase2_cli.py
========================
Phase 2 test suite:
1. Multi-project registry operations (CRUD, persistence, conflict handling, malformed JSON).
2. CLI commands: help, init, doctor, status, runs, inspect, compare, projects.
3. Interactive confirmation, cancellation, and non-interactive safety (--yes).
4. Project targeting via --path and registered --project alias.
5. Alias precedence over CWD discovery.
6. Multi-project isolation (two independent projects with distinct MLflow databases).
7. Machine-readable pure JSON stdout output verification (res.stdout is the unmixed stream).
8. Backward compatibility with the existing Wine demo.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Silence MLflow / Alembic loggers before they are imported.
for _logger_name in ("mlflow", "alembic", "mlflow.store", "mlflow.store.db", "mlflow.store.db.utils"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

from click.testing import CliRunner

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from ml_lineage.cli import main
from ml_lineage.core.registry import ProjectRegistry, RegistryError, RegistryConflictError
from ml_lineage.core.context import ProjectContext
from ml_lineage.adapters.mlflow import MLflowAdapter


def _cleanup(path: str | Path) -> None:
    """Best-effort temp directory cleanup; ignores Windows file-lock errors."""
    try:
        shutil.rmtree(str(path), ignore_errors=True)
    except Exception:
        pass


def _parse_json_stdout(res) -> dict | list:
    """Parse JSON from result.stdout (unmixed stdout stream).

    Click 8.2+ separates stdout from stderr. res.stdout contains only stdout,
    while res.output contains the terminal mix. Use res.stdout for JSON.
    """
    raw = res.stdout
    return json.loads(raw)


class TestProjectRegistry(unittest.TestCase):
    """Test suite for ml_lineage.core.registry.ProjectRegistry."""

    def setUp(self):
        self.temp_dir_path = tempfile.mkdtemp()
        self.registry_file = Path(self.temp_dir_path) / "test_registry.json"
        self.project_dir1 = Path(self.temp_dir_path) / "proj1"
        self.project_dir1.mkdir(parents=True, exist_ok=True)
        self.project_dir2 = Path(self.temp_dir_path) / "proj2"
        self.project_dir2.mkdir(parents=True, exist_ok=True)
        self.registry = ProjectRegistry(registry_file=self.registry_file)

    def tearDown(self):
        _cleanup(self.temp_dir_path)

    def test_01_empty_registry_returns_empty_dict(self):
        self.assertFalse(self.registry_file.exists())
        self.assertEqual(self.registry.load(), {})
        self.assertEqual(self.registry.list_projects(), {})

    def test_02_register_and_get_project(self):
        meta = self.registry.register("fraud-detection", self.project_dir1, description="Fraud pipeline")
        self.assertEqual(meta["alias"], "fraud-detection")
        self.assertEqual(meta["path"], str(self.project_dir1.resolve()))
        self.assertEqual(meta["description"], "Fraud pipeline")

        fetched = self.registry.get("fraud-detection")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["path"], str(self.project_dir1.resolve()))

    def test_03_registry_persistence_across_instances(self):
        self.registry.register("wine-classifier", self.project_dir1)

        reg2 = ProjectRegistry(registry_file=self.registry_file)
        fetched = reg2.get("wine-classifier")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["path"], str(self.project_dir1.resolve()))

    def test_04_alias_validation(self):
        invalid_aliases = ["a", "bad alias", "bad@alias!", "bad$name", ""]
        for bad in invalid_aliases:
            with self.assertRaises(RegistryError, msg=f"Expected error for alias '{bad}'"):
                self.registry.register(bad, self.project_dir1)

    def test_05_nonexistent_path_fails(self):
        nonexistent = Path(self.temp_dir_path) / "does_not_exist_xyz"
        with self.assertRaises(RegistryError):
            self.registry.register("valid-alias", nonexistent)

    def test_06_alias_collision_raises_conflict_error(self):
        self.registry.register("shared-alias", self.project_dir1)
        with self.assertRaises(RegistryConflictError):
            self.registry.register("shared-alias", self.project_dir2)

    def test_07_path_collision_raises_conflict_error(self):
        self.registry.register("alias-one", self.project_dir1)
        with self.assertRaises(RegistryConflictError):
            self.registry.register("alias-two", self.project_dir1)

    def test_08_overwrite_flag_allows_re_registration(self):
        self.registry.register("my-alias", self.project_dir1, description="First")
        self.registry.register("my-alias", self.project_dir2, description="Updated", overwrite=True)
        fetched = self.registry.get("my-alias")
        self.assertEqual(fetched["path"], str(self.project_dir2.resolve()))
        self.assertEqual(fetched["description"], "Updated")

    def test_09_unregister_project(self):
        self.registry.register("temp-alias", self.project_dir1)
        self.assertTrue(self.registry.unregister("temp-alias"))
        self.assertIsNone(self.registry.get("temp-alias"))
        self.assertFalse(self.registry.unregister("temp-alias"))

    def test_10_malformed_json_raises_registry_error(self):
        with open(self.registry_file, "w", encoding="utf-8") as f:
            f.write("{ invalid json corrupted ")
        with self.assertRaises(RegistryError):
            self.registry.load()

    def test_11_find_by_path(self):
        self.registry.register("churn-model", self.project_dir1)
        found = self.registry.find_by_path(self.project_dir1)
        self.assertEqual(found, "churn-model")
        self.assertIsNone(self.registry.find_by_path(self.project_dir2))


class TestCliCommands(unittest.TestCase):
    """Test suite for Click CLI subcommands using CliRunner.

    NOTE: In Click 8.2+, `result.stdout` is the pure stdout stream, while
    `result.output` is the terminal mix of stdout+stderr. We use `result.stdout`
    for JSON assertions to avoid noise from MLflow/Alembic log messages.
    """

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir_path = tempfile.mkdtemp()
        self.test_registry_file = Path(self.temp_dir_path) / "cli_registry.json"
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.test_registry_file)

    def tearDown(self):
        os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)
        _cleanup(self.temp_dir_path)

    def test_01_cli_help(self):
        res = self.runner.invoke(main, ["--help"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("ML Lineage Studio", res.output)
        for cmd in ["init", "doctor", "status", "runs", "inspect", "compare", "projects"]:
            self.assertIn(cmd, res.output)

    def test_02_subcommand_help_pages(self):
        for cmd in ["init", "doctor", "status", "runs", "inspect", "compare", "projects"]:
            res = self.runner.invoke(main, [cmd, "--help"])
            self.assertEqual(res.exit_code, 0, f"Command {cmd} --help failed:\n{res.output}")
            self.assertIn("Usage:", res.output)

    def test_03_status_command_wine_project(self):
        # Human-readable output
        res = self.runner.invoke(main, ["status", "--path", PROJECT_ROOT])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("ML Lineage — Project Status", res.output)
        self.assertIn("Git", res.output)
        self.assertIn("DVC Pipeline", res.output)
        self.assertIn("MLflow Tracking", res.output)

        # Machine-readable JSON — use res.stdout (unmixed stream in Click 8.2+)
        res_json = self.runner.invoke(main, ["status", "--path", PROJECT_ROOT, "--json"])
        self.assertEqual(res_json.exit_code, 0)
        data = _parse_json_stdout(res_json)
        self.assertIn("project", data)
        self.assertIn("git", data)
        self.assertIn("dvc", data)
        self.assertIn("mlflow", data)
        ctx = ProjectContext.from_path(PROJECT_ROOT)
        self.assertEqual(data["mlflow"]["tracking_uri"], ctx.tracking_uri)

    def test_04_runs_command_wine_project(self):
        res = self.runner.invoke(main, ["runs", "--path", PROJECT_ROOT, "--limit", "3"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("RUN ID", res.output)
        self.assertIn("STATUS", res.output)

        res_json = self.runner.invoke(main, ["runs", "--path", PROJECT_ROOT, "--limit", "2", "--json"])
        self.assertEqual(res_json.exit_code, 0)
        runs_list = _parse_json_stdout(res_json)
        self.assertIsInstance(runs_list, list)
        self.assertLessEqual(len(runs_list), 2)
        if runs_list:
            self.assertIn("run_id", runs_list[0])
            self.assertIn("status", runs_list[0])
            self.assertIn("accuracy", runs_list[0])

    def test_05_inspect_valid_run(self):
        ctx = ProjectContext.from_path(PROJECT_ROOT)
        runs = MLflowAdapter(ctx).fetch_all_runs()
        self.assertTrue(len(runs) > 0, "Need at least one run in the Wine demo tracking store")
        valid_id = runs[0].info.run_id
        short_id = valid_id[:8]

        # Human readable
        res = self.runner.invoke(main, ["inspect", short_id, "--path", PROJECT_ROOT])
        self.assertEqual(res.exit_code, 0)
        self.assertIn(valid_id, res.output)
        self.assertIn("Git Provenance", res.output)
        self.assertIn("Metrics", res.output)

        # JSON
        res_json = self.runner.invoke(main, ["inspect", short_id, "--path", PROJECT_ROOT, "--json"])
        self.assertEqual(res_json.exit_code, 0)
        details = _parse_json_stdout(res_json)
        self.assertEqual(details["run_id"], valid_id)
        self.assertIn("metrics", details)
        self.assertIn("parameters", details)
        self.assertIn("dvc_hashes", details)

    def test_06_inspect_invalid_run_exits_with_error(self):
        res = self.runner.invoke(main, ["inspect", "nonexistent_run_id_99999", "--path", PROJECT_ROOT])
        self.assertEqual(res.exit_code, 1)

        # JSON mode: should produce parseable {"error": ..., "exit_code": ...}
        res_json = self.runner.invoke(main, ["inspect", "nonexistent_run_id_99999", "--path", PROJECT_ROOT, "--json"])
        self.assertEqual(res_json.exit_code, 1)
        err = _parse_json_stdout(res_json)
        self.assertIn("error", err)

    def test_07_compare_valid_runs(self):
        ctx = ProjectContext.from_path(PROJECT_ROOT)
        runs = MLflowAdapter(ctx).fetch_all_runs()
        self.assertTrue(len(runs) >= 2, "Need at least 2 runs to compare")
        id_a = runs[0].info.run_id[:8]
        id_b = runs[1].info.run_id[:8]

        res = self.runner.invoke(main, ["compare", id_a, id_b, "--path", PROJECT_ROOT])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Run Comparison", res.output)
        self.assertIn("Metrics Comparison", res.output)

        res_json = self.runner.invoke(main, ["compare", id_a, id_b, "--path", PROJECT_ROOT, "--json"])
        self.assertEqual(res_json.exit_code, 0)
        matrix = _parse_json_stdout(res_json)
        self.assertIn("run_a", matrix)
        self.assertIn("run_b", matrix)
        self.assertIn("metrics", matrix)
        self.assertIn("parameters", matrix)
        self.assertIn("hashes", matrix)

    def test_08_compare_invalid_run_exits_with_error(self):
        res = self.runner.invoke(main, ["compare", "invalid_id_1", "invalid_id_2", "--path", PROJECT_ROOT])
        self.assertEqual(res.exit_code, 1)

    def test_09_projects_command(self):
        res_empty = self.runner.invoke(main, ["projects"])
        self.assertEqual(res_empty.exit_code, 0)
        self.assertIn("No projects registered", res_empty.output)

        reg = ProjectRegistry(registry_file=self.test_registry_file)
        reg.register("demo-wine", PROJECT_ROOT)

        res_list = self.runner.invoke(main, ["projects"])
        self.assertEqual(res_list.exit_code, 0)
        self.assertIn("demo-wine", res_list.output)

        res_json = self.runner.invoke(main, ["projects", "--json"])
        self.assertEqual(res_json.exit_code, 0)
        data = _parse_json_stdout(res_json)
        self.assertIn("demo-wine", data)
        self.assertEqual(data["demo-wine"]["path"], str(Path(PROJECT_ROOT).resolve()))

    def test_10_doctor_command_on_wine_project(self):
        res = self.runner.invoke(main, ["doctor", "--path", PROJECT_ROOT])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("ML Lineage Doctor", res.output)
        self.assertIn("Git Repository", res.output)
        self.assertIn("DVC Setup", res.output)
        self.assertIn("MLflow Store", res.output)

        res_json = self.runner.invoke(main, ["doctor", "--path", PROJECT_ROOT, "--json"])
        self.assertEqual(res_json.exit_code, 0)
        data = _parse_json_stdout(res_json)
        self.assertIn("checks", data)
        self.assertTrue(any(c["component"] == "Git Repository" for c in data["checks"]))


class TestCliInit(unittest.TestCase):
    """Test suite for ml-lineage init behavior and safety guarantees."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir_path = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir_path) / "test_ml_project"
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.test_registry_file = Path(self.temp_dir_path) / "init_registry.json"
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.test_registry_file)

    def tearDown(self):
        os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)
        _cleanup(self.temp_dir_path)

    def test_01_init_interactive_cancel_no_files_written(self):
        res = self.runner.invoke(
            main,
            ["init", "--path", str(self.project_path), "--name", "canceled-proj"],
            input="n\n",
        )
        self.assertEqual(res.exit_code, 2)
        self.assertIn("cancelled", res.output)

        config_file = self.project_path / ".ml-lineage.yaml"
        self.assertFalse(config_file.exists())

        reg = ProjectRegistry(registry_file=self.test_registry_file)
        self.assertIsNone(reg.get("canceled-proj"))

    def test_02_init_interactive_confirm_writes_config(self):
        res = self.runner.invoke(
            main,
            ["init", "--path", str(self.project_path), "--name", "interactive-proj"],
            input="y\n",
        )
        self.assertEqual(res.exit_code, 0)
        self.assertIn("successfully initialized", res.output)

        config_file = self.project_path / ".ml-lineage.yaml"
        self.assertTrue(config_file.exists())

        reg = ProjectRegistry(registry_file=self.test_registry_file)
        self.assertIsNotNone(reg.get("interactive-proj"))

    def test_03_init_non_interactive_yes(self):
        res = self.runner.invoke(
            main,
            ["init", "--path", str(self.project_path), "--name", "auto-proj", "--yes"],
        )
        self.assertEqual(res.exit_code, 0)
        self.assertIn("[OK] Project successfully initialized", res.output)

        config_file = self.project_path / ".ml-lineage.yaml"
        self.assertTrue(config_file.is_file())

        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["project"]["name"], "auto-proj")
        self.assertIn("mlflow", cfg)
        self.assertIn("git", cfg)
        self.assertIn("dvc", cfg)

    def test_04_init_already_configured_preserves_config(self):
        self.runner.invoke(
            main,
            ["init", "--path", str(self.project_path), "--name", "preserve-proj", "--yes"],
        )
        config_file = self.project_path / ".ml-lineage.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("# CUSTOM USER COMMENT\nproject:\n  name: preserve-proj\ncustom_key: preserved_value\n")

        res = self.runner.invoke(
            main,
            ["init", "--path", str(self.project_path), "--name", "preserve-proj", "--yes"],
        )
        self.assertEqual(res.exit_code, 0)
        self.assertIn("already configured", res.output)

        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("preserved_value", content)
        self.assertIn("CUSTOM USER COMMENT", content)

    def test_05_init_does_not_mutate_existing_project_files(self):
        dummy_data = self.project_path / "data.csv"
        dummy_data.write_text("a,b,c\n1,2,3", encoding="utf-8")

        self.runner.invoke(
            main,
            ["init", "--path", str(self.project_path), "--name", "safe-proj", "--yes"],
        )
        self.assertEqual(dummy_data.read_text(encoding="utf-8"), "a,b,c\n1,2,3")


class TestTargetingAndPrecedence(unittest.TestCase):
    """Test targeting project via --path, registered --project alias, and precedence."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir_path = tempfile.mkdtemp()
        self.test_registry_file = Path(self.temp_dir_path) / "target_registry.json"
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(self.test_registry_file)

        self.proj_a = Path(self.temp_dir_path) / "project_alpha"
        self.proj_a.mkdir(parents=True, exist_ok=True)
        with open(self.proj_a / "params.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"mlflow": {"experiment_name": "Exp_Alpha", "tracking_uri": "sqlite:///alpha.db"}}, f)

        self.proj_b = Path(self.temp_dir_path) / "project_beta"
        self.proj_b.mkdir(parents=True, exist_ok=True)
        with open(self.proj_b / "params.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"mlflow": {"experiment_name": "Exp_Beta", "tracking_uri": "sqlite:///beta.db"}}, f)

        self.reg = ProjectRegistry(registry_file=self.test_registry_file)
        self.reg.register("alpha", self.proj_a)
        self.reg.register("beta", self.proj_b)

    def tearDown(self):
        os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)
        _cleanup(self.temp_dir_path)

    def test_01_targeting_via_explicit_path(self):
        res = self.runner.invoke(main, ["status", "--path", str(self.proj_a), "--json"])
        self.assertEqual(res.exit_code, 0, f"stdout: {res.stdout}\nstderr: {res.stderr}")
        data = _parse_json_stdout(res)
        self.assertEqual(data["project"]["root"], str(self.proj_a.resolve()))
        self.assertIn("alpha.db", data["mlflow"]["tracking_uri"])

    def test_02_targeting_via_project_alias(self):
        res = self.runner.invoke(main, ["status", "--project", "beta", "--json"])
        self.assertEqual(res.exit_code, 0, f"stdout: {res.stdout}\nstderr: {res.stderr}")
        data = _parse_json_stdout(res)
        self.assertEqual(data["project"]["root"], str(self.proj_b.resolve()))
        self.assertEqual(data["project"]["alias"], "beta")
        self.assertIn("beta.db", data["mlflow"]["tracking_uri"])

    def test_03_alias_precedence_over_cwd(self):
        cwd_before = os.getcwd()
        try:
            os.chdir(str(self.proj_a))
            res = self.runner.invoke(main, ["status", "--project", "beta", "--json"])
            self.assertEqual(res.exit_code, 0, f"stdout: {res.stdout}\nstderr: {res.stderr}")
            data = _parse_json_stdout(res)
            # Must point to proj_b — --project overrides CWD
            self.assertEqual(data["project"]["root"], str(self.proj_b.resolve()))
            self.assertEqual(data["project"]["alias"], "beta")
        finally:
            os.chdir(cwd_before)

    def test_04_unknown_project_alias_fails_cleanly(self):
        res = self.runner.invoke(main, ["status", "--project", "unknown-alias", "--json"])
        self.assertEqual(res.exit_code, 1)
        err = _parse_json_stdout(res)
        self.assertIn("error", err)
        self.assertIn("not found in registry", err["error"])


class TestTwoIsolatedProjects(unittest.TestCase):
    """Two distinct projects must have completely isolated MLflow tracking stores."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir_path = tempfile.mkdtemp()
        os.environ["ML_LINEAGE_REGISTRY_PATH"] = str(Path(self.temp_dir_path) / "iso_registry.json")

        self.dir_x = Path(self.temp_dir_path) / "proj_x"
        self.dir_x.mkdir(parents=True, exist_ok=True)
        with open(self.dir_x / "params.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"mlflow": {"experiment_name": "Exp_X", "tracking_uri": "sqlite:///x_mlflow.db"}}, f)

        self.dir_y = Path(self.temp_dir_path) / "proj_y"
        self.dir_y.mkdir(parents=True, exist_ok=True)
        with open(self.dir_y / "params.yaml", "w", encoding="utf-8") as f:
            yaml.dump({"mlflow": {"experiment_name": "Exp_Y", "tracking_uri": "sqlite:///y_mlflow.db"}}, f)

    def tearDown(self):
        os.environ.pop("ML_LINEAGE_REGISTRY_PATH", None)
        _cleanup(self.temp_dir_path)

    def test_01_two_projects_tracking_isolation(self):
        ctx_x = ProjectContext.from_path(self.dir_x)
        ctx_y = ProjectContext.from_path(self.dir_y)

        # Verify tracking URIs are distinct and project-scoped
        self.assertIn("x_mlflow.db", ctx_x.tracking_uri)
        self.assertIn("y_mlflow.db", ctx_y.tracking_uri)
        self.assertNotEqual(ctx_x.tracking_uri, ctx_y.tracking_uri)

        # Seed Project X with a run
        client_x = MLflowAdapter(ctx_x).get_client()
        exp_id_x = client_x.create_experiment("Exp_X")
        run_x = client_x.create_run(exp_id_x)
        client_x.log_metric(run_x.info.run_id, "accuracy", 0.888)
        client_x.set_terminated(run_x.info.run_id, "FINISHED")

        # Seed Project Y with a distinct run
        client_y = MLflowAdapter(ctx_y).get_client()
        exp_id_y = client_y.create_experiment("Exp_Y")
        run_y = client_y.create_run(exp_id_y)
        client_y.log_metric(run_y.info.run_id, "accuracy", 0.999)
        client_y.set_terminated(run_y.info.run_id, "FINISHED")

        # Query Project X via CLI
        res_x = self.runner.invoke(main, ["runs", "--path", str(self.dir_x), "--json"])
        self.assertEqual(res_x.exit_code, 0, f"stderr: {res_x.stderr}")
        runs_x = _parse_json_stdout(res_x)
        self.assertEqual(len(runs_x), 1, f"Expected 1 run for X, got: {runs_x}")
        self.assertEqual(runs_x[0]["run_id"], run_x.info.run_id)
        self.assertAlmostEqual(runs_x[0]["accuracy"], 0.888, places=3)

        # Query Project Y via CLI
        res_y = self.runner.invoke(main, ["runs", "--path", str(self.dir_y), "--json"])
        self.assertEqual(res_y.exit_code, 0, f"stderr: {res_y.stderr}")
        runs_y = _parse_json_stdout(res_y)
        self.assertEqual(len(runs_y), 1, f"Expected 1 run for Y, got: {runs_y}")
        self.assertEqual(runs_y[0]["run_id"], run_y.info.run_id)
        self.assertAlmostEqual(runs_y[0]["accuracy"], 0.999, places=3)


if __name__ == "__main__":
    unittest.main()
