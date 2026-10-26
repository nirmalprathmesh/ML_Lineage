#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_dashboard_data.py
============================
Automated unit tests for Stage 6C dashboard data processing,
metric analytics, comparison matrix, and lineage report helpers.
"""

import os
import sys
import unittest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app


class TestDashboardDataHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.enable_test_session_override()
        cls.runs = app.fetch_all_runs()
        cls.has_runs = len(cls.runs) > 0
        cls.first_run_id = cls.runs[0].info.run_id if cls.has_runs else None

    @classmethod
    def tearDownClass(cls):
        app.disable_test_session_override()

    def test_01_get_runs_dataframe_structure(self):
        df = app.get_runs_dataframe(self.runs)
        self.assertIsInstance(df, pd.DataFrame)
        if self.has_runs:
            self.assertEqual(len(df), len(self.runs))
            expected_cols = [
                "run_id", "short_id", "status", "experiment", "accuracy",
                "precision", "recall", "f1_score", "duration_seconds",
                "start_time", "git_commit", "model_type"
            ]
            for col in expected_cols:
                self.assertIn(col, df.columns, f"Missing expected column: {col}")
            # Check short ID is 8 characters
            self.assertEqual(len(df.iloc[0]["short_id"]), 8)

    def test_02_get_runs_dataframe_empty_handling(self):
        empty_df = app.get_runs_dataframe([])
        self.assertIsInstance(empty_df, pd.DataFrame)
        self.assertEqual(len(empty_df), 0)
        self.assertIn("run_id", empty_df.columns)
        self.assertIn("accuracy", empty_df.columns)

    def test_03_dataframe_filtering(self):
        df = app.get_runs_dataframe(self.runs)
        if self.has_runs:
            # Filter by status
            finished = df[df["status"] == "FINISHED"]
            self.assertGreaterEqual(len(finished), 1)

            # Filter by accuracy threshold
            high_acc = df[df["accuracy"] >= 0.99]
            self.assertGreaterEqual(len(high_acc), 1)

    def test_04_fetch_lineage_report_data_valid(self):
        if self.has_runs and self.first_run_id:
            data = app.fetch_lineage_report_data(self.first_run_id[:8])
            self.assertIsNotNone(data, "Expected valid lineage report dict.")
            self.assertIn("run", data)
            self.assertIn("lineage_stages", data)
            self.assertIn("verification_summary", data)
            self.assertEqual(data["run"]["run_id"], self.first_run_id)

    def test_05_fetch_lineage_report_data_invalid(self):
        data = app.fetch_lineage_report_data("nonexistent_run_prefix")
        self.assertIsNone(data, "Expected None for nonexistent run prefix.")

    def test_06_get_run_comparison_matrix_valid(self):
        if len(self.runs) >= 2:
            id_a = self.runs[0].info.run_id[:8]
            id_b = self.runs[1].info.run_id[:8]
            matrix = app.get_run_comparison_matrix(id_a, id_b)
            self.assertNotIn("error", matrix)
            self.assertIn("run_a", matrix)
            self.assertIn("run_b", matrix)
            self.assertIn("metrics", matrix)
            self.assertIn("parameters", matrix)
            self.assertIn("hashes", matrix)

            # Check metrics format
            for m in matrix["metrics"]:
                self.assertIn("metric", m)
                self.assertIn("run_a", m)
                self.assertIn("run_b", m)
                self.assertIn("delta", m)

    def test_07_get_run_comparison_matrix_invalid(self):
        matrix = app.get_run_comparison_matrix("invalid_1", "invalid_2")
        self.assertIn("error", matrix)


if __name__ == "__main__":
    unittest.main()
