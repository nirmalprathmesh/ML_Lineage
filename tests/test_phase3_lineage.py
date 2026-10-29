"""
tests/test_phase3_lineage.py
============================
Comprehensive test suite for Phase 3:
1. Dynamic DVC DAG pipeline parser (linear, branching, merge, single-stage, empty, missing, malformed, cycles, conflicts, unresolved deps).
2. Dynamic run-lineage tracing (OBSERVED, INFERRED, UNAVAILABLE evidence levels, partial metadata, manifest parsing, project isolation).
3. CLI commands (`pipeline`, `lineage`) in human and --json modes with proper exit codes and stdout/stderr separation.
4. Backward compatibility with existing Wine demo, scripts/lineage_report.py, and chatbot.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
import yaml

from ml_lineage.core.context import ProjectContext
from ml_lineage.core.pipeline import (
    PipelineArtifact,
    PipelineDAG,
    PipelineStage,
    parse_dvc_pipeline,
)
from ml_lineage.core.lineage import (
    EvidenceLevel,
    find_recorded_artifact_hash,
    trace_run_lineage,
)
from ml_lineage.adapters.dvc import DVCAdapter
from ml_lineage.adapters.mlflow import MLflowAdapter
from ml_lineage.cli import main as cli_main


class TestPipelineDAGParser(unittest.TestCase):
    """Test dynamic DVC pipeline parsing across various pipeline topologies and edge cases."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="ml_test_dag_")
        self.root = Path(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_missing_dvc_yaml(self):
        ctx = ProjectContext(self.root)
        dag = parse_dvc_pipeline(ctx)
        self.assertFalse(dag.has_dvc_yaml)
        self.assertFalse(dag.is_valid)
        self.assertTrue(any("No dvc.yaml" in issue for issue in dag.validation_issues))
        self.assertEqual(dag.nodes, [])

    def test_malformed_yaml(self):
        (self.root / "dvc.yaml").write_text("stages: [invalid: yaml: :::", encoding="utf-8")
        ctx = ProjectContext(self.root)
        dag = parse_dvc_pipeline(ctx)
        self.assertTrue(dag.has_dvc_yaml)
        self.assertFalse(dag.is_valid)
        self.assertTrue(any("Malformed YAML" in issue for issue in dag.validation_issues))

    def test_empty_stages(self):
        (self.root / "dvc.yaml").write_text("stages: {}\n", encoding="utf-8")
        ctx = ProjectContext(self.root)
        dag = parse_dvc_pipeline(ctx)
        self.assertTrue(dag.has_dvc_yaml)
        self.assertTrue(dag.is_valid)
        self.assertEqual(dag.nodes, [])
        self.assertEqual(dag.edges, [])

    def test_single_stage_pipeline(self):
        content = {
            "stages": {
                "train": {
                    "cmd": "python train.py",
                    "deps": ["src/train.py"],
                    "outs": ["model.pkl"],
                }
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(content), encoding="utf-8")
        # Create src/train.py so it's not reported as unresolved dep
        (self.root / "src").mkdir()
        (self.root / "src" / "train.py").write_text("# code", encoding="utf-8")

        ctx = ProjectContext(self.root)
        dag = parse_dvc_pipeline(ctx)
        self.assertTrue(dag.is_valid)
        self.assertEqual(dag.nodes, ["train"])
        self.assertEqual(dag.edges, [])
        self.assertEqual(dag.topological_sort(), ["train"])

    def test_linear_pipeline(self):
        # A -> B -> C
        content = {
            "stages": {
                "prepare": {
                    "cmd": "python prepare.py",
                    "outs": ["data/clean.csv"],
                },
                "featurize": {
                    "cmd": "python featurize.py",
                    "deps": ["data/clean.csv"],
                    "outs": ["data/features.csv"],
                },
                "train": {
                    "cmd": "python train.py",
                    "deps": ["data/features.csv"],
                    "outs": ["models/model.pkl"],
                },
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(content), encoding="utf-8")
        ctx = ProjectContext(self.root)
        dag = parse_dvc_pipeline(ctx)

        self.assertTrue(dag.is_valid)
        self.assertFalse(dag.is_cyclic)
        self.assertEqual(dag.edges, [("featurize", "train"), ("prepare", "featurize")])
        self.assertEqual(dag.topological_sort(), ["prepare", "featurize", "train"])
        self.assertEqual(dag.get_downstream_stages("prepare"), ["featurize"])
        self.assertEqual(dag.get_upstream_stages("train"), ["featurize"])

    def test_branching_and_merge_pipeline(self):
        # Branching: A -> B, A -> C
        # Merge: B -> D, C -> D
        content = {
            "stages": {
                "split": {
                    "outs": ["train.csv", "test.csv"],
                },
                "train_model_a": {
                    "deps": ["train.csv"],
                    "outs": ["model_a.pkl"],
                },
                "train_model_b": {
                    "deps": ["train.csv"],
                    "outs": ["model_b.pkl"],
                },
                "ensemble": {
                    "deps": ["model_a.pkl", "model_b.pkl", "test.csv"],
                    "metrics": ["metrics.json"],
                },
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(content), encoding="utf-8")
        ctx = ProjectContext(self.root)
        dag = parse_dvc_pipeline(ctx)

        self.assertTrue(dag.is_valid)
        self.assertFalse(dag.is_cyclic)
        order = dag.topological_sort()
        self.assertEqual(order[0], "split")
        self.assertEqual(order[-1], "ensemble")
        self.assertIn("train_model_a", dag.get_downstream_stages("split"))
        self.assertIn("train_model_b", dag.get_downstream_stages("split"))
        self.assertEqual(sorted(dag.get_upstream_stages("ensemble")), ["split", "train_model_a", "train_model_b"])

    def test_cycle_detection(self):
        # A -> B -> C -> A
        content = {
            "stages": {
                "step_a": {
                    "deps": ["file_c.txt"],
                    "outs": ["file_a.txt"],
                },
                "step_b": {
                    "deps": ["file_a.txt"],
                    "outs": ["file_b.txt"],
                },
                "step_c": {
                    "deps": ["file_b.txt"],
                    "outs": ["file_c.txt"],
                },
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(content), encoding="utf-8")
        ctx = ProjectContext(self.root)
        dag = parse_dvc_pipeline(ctx)

        self.assertTrue(dag.is_cyclic)
        self.assertFalse(dag.is_valid)
        self.assertTrue(any("Cycle detected" in issue for issue in dag.validation_issues))
        # Topological sort must not crash or loop infinitely
        sort_res = dag.topological_sort()
        self.assertEqual(len(sort_res), 3)

    def test_conflicting_output_paths(self):
        content = {
            "stages": {
                "stage_1": {"outs": ["data/output.csv"]},
                "stage_2": {"outs": ["data/output.csv"]},
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(content), encoding="utf-8")
        ctx = ProjectContext(self.root)
        dag = parse_dvc_pipeline(ctx)
        self.assertTrue(any("Conflicting output" in issue for issue in dag.validation_issues))

    def test_unresolved_dependencies(self):
        content = {
            "stages": {
                "process": {
                    "deps": ["nonexistent_input.csv"],
                    "outs": ["output.csv"],
                }
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(content), encoding="utf-8")
        ctx = ProjectContext(self.root)
        dag = parse_dvc_pipeline(ctx)
        self.assertTrue(any("Unresolved dependency" in issue for issue in dag.validation_issues))

    def test_parameters_and_lock_hashes_extraction(self):
        dvc_yaml = {
            "stages": {
                "train": {
                    "cmd": "python train.py",
                    "params": [
                        "train.lr",
                        "train.epochs",
                    ],
                    "outs": ["model.pkl"],
                }
            }
        }
        params_yaml = {
            "train": {
                "lr": 0.001,
                "epochs": 50,
            }
        }
        dvc_lock = {
            "schema": "2.0",
            "stages": {
                "train": {
                    "cmd": "python train.py",
                    "params": {
                        "params.yaml": {
                            "train.lr": 0.001,
                            "train.epochs": 50,
                        }
                    },
                    "outs": [
                        {
                            "path": "model.pkl",
                            "hash": "md5",
                            "md5": "abc123456789def",
                            "size": 4096,
                        }
                    ]
                }
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(dvc_yaml), encoding="utf-8")
        (self.root / "params.yaml").write_text(yaml.dump(params_yaml), encoding="utf-8")
        (self.root / "dvc.lock").write_text(yaml.dump(dvc_lock), encoding="utf-8")

        ctx = ProjectContext(self.root)
        dag = parse_dvc_pipeline(ctx)
        self.assertTrue(dag.is_valid)
        stage = dag.stages["train"]
        self.assertEqual(stage.params["train.lr"], 0.001)
        self.assertEqual(stage.params["train.epochs"], 50)
        self.assertEqual(len(stage.outs), 1)
        self.assertEqual(stage.outs[0].hash_val, "abc123456789def")
        self.assertEqual(stage.outs[0].size, 4096)


class TestDynamicRunLineageService(unittest.TestCase):
    """Test dynamic run lineage tracing, evidence classification, and fallbacks."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="ml_test_lineage_")
        self.root = Path(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _setup_mock_run(self, run_id: str, tags: dict, params: dict, metrics: dict):
        mock_run = MagicMock()
        mock_run.info.run_id = run_id
        mock_run.info.experiment_id = "1"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1700000000000
        mock_run.info.end_time = 1700000010000
        mock_run.info.artifact_uri = f"file:{self.root}/mlruns/1/{run_id}/artifacts"
        mock_run.data.tags = tags
        mock_run.data.params = params
        mock_run.data.metrics = metrics
        return mock_run

    @patch("ml_lineage.core.lineage.MLflowAdapter")
    def test_observed_vs_inferred_vs_unavailable_stages(self, mock_mlflow_cls):
        # Pipeline: prepare -> train -> evaluate
        dvc_yaml = {
            "stages": {
                "prepare": {"outs": ["data/clean.csv"]},
                "train": {"deps": ["data/clean.csv"], "outs": ["models/model.pkl"]},
                "evaluate": {"deps": ["models/model.pkl"], "outs": ["metrics.json"]},
                "unrelated": {"outs": ["data/other.csv"]},
            }
        }
        dvc_lock = {
            "schema": "2.0",
            "stages": {
                "prepare": {"outs": [{"path": "data/clean.csv", "md5": "clean_md5"}]},
                "train": {"outs": [{"path": "models/model.pkl", "md5": "model_md5"}]},
                "evaluate": {"outs": [{"path": "metrics.json", "md5": "metric_md5"}]},
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(dvc_yaml), encoding="utf-8")
        (self.root / "dvc.lock").write_text(yaml.dump(dvc_lock), encoding="utf-8")

        # Mock run only tagged prepare output and train output, but not evaluate or unrelated
        tags = {
            "git.commit_sha": "abcdef123456",
            "git.branch": "main",
            "dvc.hash.data/clean.csv": "clean_md5",
            "dvc.hash.models/model.pkl": "model_md5",
        }
        mock_run = self._setup_mock_run("run_test_01", tags, {"lr": "0.01"}, {"acc": 0.95})

        mock_adapter = MagicMock()
        mock_adapter.resolve_run.return_value = mock_run
        mock_exp = MagicMock()
        mock_exp.name = "Test_Experiment"
        mock_adapter.get_client().get_experiment.return_value = mock_exp
        mock_mlflow_cls.return_value = mock_adapter

        ctx = ProjectContext(self.root)
        report = trace_run_lineage(ctx, "run_test_01")

        self.assertEqual(report.run_id, "run_test_01")
        self.assertEqual(report.git_provenance["commit_sha"], "abcdef123456")

        stage_map = {s.stage_name: s for s in report.stages}
        # prepare has direct hash tag -> OBSERVED
        self.assertEqual(stage_map["prepare"].evidence_level, EvidenceLevel.OBSERVED)
        # train has direct hash tag -> OBSERVED
        self.assertEqual(stage_map["train"].evidence_level, EvidenceLevel.OBSERVED)
        # evaluate has no hash tag, but upstream 'train' is observed and lock exists -> INFERRED
        self.assertEqual(stage_map["evaluate"].evidence_level, EvidenceLevel.INFERRED)
        # unrelated has no hash tag and no upstream observed -> UNAVAILABLE
        self.assertEqual(stage_map["unrelated"].evidence_level, EvidenceLevel.UNAVAILABLE)

        summary = report.to_dict()["pipeline_summary"]
        self.assertEqual(summary["total_stages"], 4)
        self.assertEqual(summary["observed_stages"], 2)
        self.assertEqual(summary["inferred_stages"], 1)
        self.assertEqual(summary["unavailable_stages"], 1)

    @patch("ml_lineage.core.lineage.MLflowAdapter")
    def test_run_in_project_without_dvc(self, mock_mlflow_cls):
        # Project without dvc.yaml or .dvc
        tags = {
            "dvc.hash.raw_dataset": "hash_111",
            "dvc.hash.model": "hash_222",
        }
        mock_run = self._setup_mock_run("run_nodvc", tags, {}, {})
        mock_adapter = MagicMock()
        mock_adapter.resolve_run.return_value = mock_run
        mock_mlflow_cls.return_value = mock_adapter

        ctx = ProjectContext(self.root)
        report = trace_run_lineage(ctx, "run_nodvc")

        self.assertFalse(report.pipeline_available)
        self.assertEqual(len(report.stages), 1)
        self.assertEqual(report.stages[0].stage_name, "unstructured_artifacts")
        self.assertEqual(report.stages[0].evidence_level, EvidenceLevel.OBSERVED)
        self.assertEqual(len(report.stages[0].outputs), 2)

    @patch("ml_lineage.core.lineage.MLflowAdapter")
    def test_unknown_run_raises_value_error(self, mock_mlflow_cls):
        mock_adapter = MagicMock()
        mock_adapter.resolve_run.return_value = None
        mock_mlflow_cls.return_value = mock_adapter

        ctx = ProjectContext(self.root)
        with self.assertRaises(ValueError):
            trace_run_lineage(ctx, "nonexistent_run")


class TestPipelineAndLineageCLI(unittest.TestCase):
    """Test CLI commands: ml-lineage pipeline, ml-lineage lineage."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmp_dir = tempfile.mkdtemp(prefix="ml_test_cli_p3_")
        self.root = Path(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cli_pipeline_command_human_and_json(self):
        content = {
            "stages": {
                "ingest": {"outs": ["data.csv"]},
                "train": {"deps": ["data.csv"], "outs": ["model.bin"]},
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(content), encoding="utf-8")

        # 1. Human output
        res_human = self.runner.invoke(cli_main, ["pipeline", "-p", str(self.root)])
        self.assertEqual(res_human.exit_code, 0)
        self.assertIn("DVC PIPELINE TOPOLOGY & DAG", res_human.stdout)
        self.assertIn("[Stage] ingest", res_human.stdout)
        self.assertIn("[Stage] train", res_human.stdout)
        self.assertIn("ingest  -->  train", res_human.stdout)

        # 2. JSON output
        res_json = self.runner.invoke(cli_main, ["pipeline", "-p", str(self.root), "--json"])
        self.assertEqual(res_json.exit_code, 0)
        parsed = json.loads(res_json.stdout)
        self.assertTrue(parsed["is_valid"])
        self.assertEqual(parsed["execution_order"], ["ingest", "train"])
        self.assertEqual(parsed["edges"], [["ingest", "train"]])

    def test_cli_pipeline_missing_dvc_yaml(self):
        res = self.runner.invoke(cli_main, ["pipeline", "-p", str(self.root)])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("No dvc.yaml", res.stdout)

    def test_cli_pipeline_cyclic_exit_code(self):
        content = {
            "stages": {
                "step1": {"deps": ["out2.txt"], "outs": ["out1.txt"]},
                "step2": {"deps": ["out1.txt"], "outs": ["out2.txt"]},
            }
        }
        (self.root / "dvc.yaml").write_text(yaml.dump(content), encoding="utf-8")
        res = self.runner.invoke(cli_main, ["pipeline", "-p", str(self.root)])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Cycle detected", res.stdout)

    def test_cli_lineage_unknown_run_exit_code(self):
        res = self.runner.invoke(cli_main, ["lineage", "unknown999", "-p", str(self.root)])
        self.assertEqual(res.exit_code, 1)

        res_json = self.runner.invoke(cli_main, ["lineage", "unknown999", "-p", str(self.root), "--json"])
        self.assertEqual(res_json.exit_code, 1)
        parsed = json.loads(res_json.stdout)
        self.assertIn("error", parsed)


class TestBackwardCompatibilityAndRealWineProject(unittest.TestCase):
    """Test backward compatibility against the real Wine demo repository and existing endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(r"D:\ML_CP")
        cls.ctx = ProjectContext.from_path(cls.repo_root)

    def test_real_project_dvc_pipeline_dag(self):
        """Verify dynamic DAG parser correctly models the real Wine demo DVC pipeline."""
        adapter = DVCAdapter(self.ctx)
        dag = adapter.get_pipeline_dag()

        self.assertTrue(dag.has_dvc_yaml)
        self.assertTrue(dag.has_dvc_lock)
        self.assertTrue(dag.is_valid)
        self.assertFalse(dag.is_cyclic)

        # 5 stages in Wine demo: prepare, featurize, train, evaluate, log_experiment
        self.assertEqual(
            set(dag.nodes),
            {"prepare", "featurize", "train", "evaluate", "log_experiment"}
        )
        self.assertEqual(dag.topological_sort()[0], "prepare")
        self.assertEqual(dag.topological_sort()[-1], "log_experiment")

        # Verify key outputs and parameters
        self.assertIn("data/prepared/train.csv", [o.path for o in dag.stages["prepare"].outs])
        self.assertIn("models/model.pkl", [o.path for o in dag.stages["train"].outs])
        self.assertEqual(dag.stages["train"].params.get("train.n_estimators"), 100)

    def test_real_run_lineage_trace(self):
        """Verify dynamic lineage tracing on real run b6b15235 produces verified evidence."""
        report = trace_run_lineage(self.ctx, "b6b15235")

        self.assertEqual(report.run_id, "b6b15235cabf4b1eaaa6b9773a9f15f7")
        self.assertEqual(report.experiment_name, "ML_Lineage_Experiments")
        self.assertTrue(report.git_provenance["commit_sha"].startswith("2e719abd"))

        # Verify evidence levels for stages
        stage_map = {s.stage_name: s for s in report.stages}
        self.assertEqual(stage_map["prepare"].evidence_level, EvidenceLevel.OBSERVED)
        self.assertEqual(stage_map["featurize"].evidence_level, EvidenceLevel.OBSERVED)
        self.assertEqual(stage_map["train"].evidence_level, EvidenceLevel.OBSERVED)
        self.assertEqual(stage_map["evaluate"].evidence_level, EvidenceLevel.INFERRED)

        # Output schema keys must match backward-compatible expectations
        rep_dict = report.to_dict()
        self.assertIn("run", rep_dict)
        self.assertIn("git_provenance", rep_dict)
        self.assertIn("metrics", rep_dict)
        self.assertIn("lineage_stages", rep_dict)
        self.assertIn("verification_summary", rep_dict)

        # Each stage must have description and inputs for app.py / chatbot compatibility
        for stage in rep_dict["lineage_stages"]:
            self.assertIn("stage", stage)
            self.assertIn("description", stage)
            self.assertIn("inputs", stage)
            self.assertIn("outputs", stage)
            self.assertIn("evidence_level", stage)


if __name__ == "__main__":
    unittest.main()
