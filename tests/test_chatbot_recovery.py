#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests/test_chatbot_recovery.py
==============================
Automated unit and integration test suite for Stage 6B chatbot recovery integration.
Covers all 22 required items from the specification:
1. Help routing
2. List-runs routing and real run retrieval
3. Status routing
4. Inspection of a real existing run
5. Comparison of two real existing runs
6. Lineage report generation for a real existing run
7. Verification command construction and output handling
8. Dry-run command construction and output handling
9. preview recover and preview restore routing
10. recover and restore performing dry-run only before confirmation
11. Successful matching confirmation invoking full recovery exactly once in a mocked test
12. Confirmation for a different run being rejected
13. Invalid, unknown, or ambiguous run IDs being rejected
14. Unknown text containing a run ID not silently triggering inspection
15. Subprocess timeout handling
16. Non-zero subprocess exit handling
17. Missing recovery script handling
18. Malformed or unexpected subprocess output handling
19. No shell execution from chat input
20. Existing lineage and run-list regression tests
21. Streamlit session-state and confirmation lifecycle where practical
22. Confirmation state cleared after cancellation or a failed recovery attempt
"""

import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app


class TestChatbotRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.enable_test_session_override()
        cls.real_runs = app.fetch_all_runs()
        cls.valid_run = cls.real_runs[0] if cls.real_runs else None
        cls.valid_run_id = cls.valid_run.info.run_id if cls.valid_run else "b6b15235cabf4b1eaaa6b9773a9f15f7"
        cls.valid_short_id = cls.valid_run_id[:8]

    @classmethod
    def tearDownClass(cls):
        app.disable_test_session_override()

    def setUp(self):
        # Always start each test with a clean confirmation state
        app.enable_test_session_override()
        app.clear_confirmation_state()

    def tearDown(self):
        app.clear_confirmation_state()


    # 1. Help routing
    def test_01_help_routing(self):
        for cmd in ["help", "/help", "?", "commands"]:
            resp = app.route_message(cmd)
            self.assertIn("ML Lineage Chatbot — Available Commands", resp)
            self.assertIn("confirm recover", resp)
            self.assertIn("preview recover", resp)
            self.assertIn("Overwrites Workspace Files", resp)
            self.assertIn("Read-Only", resp)

    # 2. List-runs routing and real run retrieval
    def test_02_list_runs_routing(self):
        for cmd in ["list runs", "list", "show runs", "runs"]:
            resp = app.route_message(cmd)
            self.assertIn("MLflow run(s)", resp)
            self.assertIn(self.valid_short_id, resp)

    # 3. Status routing
    def test_03_status_routing(self):
        for cmd in ["status", "system status", "overview"]:
            resp = app.route_message(cmd)
            self.assertIn("System Status", resp)
            self.assertIn("Git", resp)
            self.assertIn("DVC Workspace Pipeline", resp)
            self.assertIn("DVC Remote Storage Sync", resp)
            self.assertIn("MLflow Experiments", resp)

    # 4. Inspection of a real existing run
    def test_04_inspect_real_run(self):
        resp = app.route_message(f"inspect {self.valid_short_id}")
        self.assertIn("Run Inspection", resp)
        self.assertIn(self.valid_run_id, resp)
        self.assertIn("Metrics", resp)
        self.assertIn("Git Provenance", resp)
        self.assertIn("DVC Artifact Hashes", resp)

    # 5. Comparison of two real existing runs
    def test_05_compare_runs(self):
        if len(self.real_runs) >= 2:
            id1 = self.real_runs[0].info.run_id[:8]
            id2 = self.real_runs[1].info.run_id[:8]
            resp = app.route_message(f"compare {id1} {id2}")
            self.assertIn("Run Comparison", resp)
            self.assertIn(id1, resp)
            self.assertIn(id2, resp)
        # Test unknown run handling
        err_resp = app.route_message("compare 11112222 33334444")
        self.assertIn("Could not find run", err_resp)

    # 6. Lineage report generation for a real existing run
    def test_06_lineage_report_real_run(self):
        resp = app.route_message(f"lineage {self.valid_short_id}")
        self.assertTrue(
            "Lineage Report" in resp or "Pipeline Lineage Trace" in resp,
            f"Expected lineage report header, got: {resp[:200]}",
        )

    # 7. Verification command construction and output handling
    @patch("app.run_recovery_subprocess")
    def test_07_verification_command_construction(self, mock_subproc):
        mock_subproc.return_value = {
            "success": True,
            "exit_code": 0,
            "stdout": "Step 2: DVC Artifact Verification\n[OK] Workspace file matches expected hash.",
            "stderr": "",
            "error": None,
        }
        resp = app.handle_verify_run(self.valid_short_id)
        mock_subproc.assert_called_once_with(self.valid_run_id, ["--verify-only"])
        self.assertIn("Artifact Verification", resp)
        self.assertIn("All checked artifacts match their recorded hashes", resp)

    # 8. Dry-run command construction and output handling
    @patch("app.run_recovery_subprocess")
    def test_08_dry_run_command_construction(self, mock_subproc):
        mock_subproc.return_value = {
            "success": True,
            "exit_code": 0,
            "stdout": "Recovery Summary:\nArtifacts checked: 7\nWould restore: 0",
            "stderr": "",
            "error": None,
        }
        resp = app.route_message(f"dry-run {self.valid_short_id}")
        mock_subproc.assert_called_once_with(self.valid_run_id, ["--dry-run"])
        self.assertIn("Recovery Dry-Run Preview", resp)
        self.assertIn("No workspace files were modified", resp)

    # 9. preview recover and preview restore routing
    @patch("app.handle_dry_run")
    def test_09_preview_recover_restore_routing(self, mock_dry_run):
        mock_dry_run.return_value = "PREVIEW_OK"
        resp1 = app.route_message(f"preview recover {self.valid_short_id}")
        mock_dry_run.assert_called_with(self.valid_short_id)
        self.assertEqual(resp1, "PREVIEW_OK")

        resp2 = app.route_message(f"preview restore {self.valid_short_id}")
        mock_dry_run.assert_called_with(self.valid_short_id)
        self.assertEqual(resp2, "PREVIEW_OK")

    # 10. recover and restore performing dry-run only before confirmation
    @patch("app.run_recovery_subprocess")
    def test_10_recover_restore_dry_run_only_before_confirmation(self, mock_subproc):
        mock_subproc.return_value = {
            "success": True,
            "exit_code": 0,
            "stdout": "[--] Mode: DRY-RUN\n[OK] Dry-run complete.",
            "stderr": "",
            "error": None,
        }
        resp = app.route_message(f"recover {self.valid_short_id}")
        # Verify that only --dry-run was invoked, NEVER empty extra_args (full restore)
        mock_subproc.assert_called_once_with(self.valid_run_id, ["--dry-run"])
        self.assertIn("Confirmation Required", resp)
        self.assertIn("WARNING: Workspace Overwrite Risk", resp)
        self.assertIn(f"confirm recover {self.valid_short_id}", resp)

        # Check that confirmation state was saved
        pending = app.get_confirmation_state()
        self.assertIsNotNone(pending)
        self.assertEqual(pending["run_id"], self.valid_run_id)

    # 11. Successful matching confirmation invoking full recovery exactly once in a mocked test
    @patch("app.run_recovery_subprocess")
    def test_11_matching_confirmation_invokes_full_recovery_mocked(self, mock_subproc):
        # Set up a pending confirmation
        app.set_confirmation_state({
            "run_id": self.valid_run_id,
            "short_id": self.valid_short_id,
        })
        mock_subproc.return_value = {
            "success": True,
            "exit_code": 0,
            "stdout": "[OK] All artifacts successfully recovered and verified.",
            "stderr": "",
            "error": None,
        }
        resp = app.route_message(f"confirm recover {self.valid_short_id}")
        mock_subproc.assert_called_once_with(self.valid_run_id, [])
        self.assertIn("Recovery Completed Successfully", resp)
        # Confirmation state must be cleared after execution
        self.assertIsNone(app.get_confirmation_state())

    # 12. Confirmation for a different run being rejected
    @patch("app.run_recovery_subprocess")
    def test_12_confirmation_for_different_run_rejected(self, mock_subproc):
        app.set_confirmation_state({
            "run_id": self.valid_run_id,
            "short_id": self.valid_short_id,
        })
        resp = app.route_message("confirm recover 12345678")
        # Subprocess must NOT have been called
        mock_subproc.assert_not_called()
        self.assertIn("Confirmation Rejected", resp)
        self.assertIn("does not match", resp)
        # Pending confirmation should still be intact or can be inspected
        pending = app.get_confirmation_state()
        self.assertIsNotNone(pending)

    # 13. Invalid, unknown, or ambiguous run IDs being rejected
    def test_13_invalid_unknown_ambiguous_run_ids_rejected(self):
        resp_invalid = app.handle_verify_run("zzznothex!")
        self.assertIn("No unique run found", resp_invalid)

        resp_unknown = app.handle_verify_run("00000000")
        self.assertIn("No unique run found", resp_unknown)

        resp_recover_unk = app.handle_recover_prompt("00000000")
        self.assertIn("No unique run found", resp_recover_unk)

    # 14. Unknown text containing a run ID not silently triggering inspection
    def test_14_unknown_text_containing_run_id_not_silent_inspect(self):
        test_inputs = [
            f"can we check run {self.valid_short_id} please",
            f"delete {self.valid_short_id}",
            f"what happened in {self.valid_short_id} yesterday",
            f"run {self.valid_short_id} has issues",
        ]
        for user_text in test_inputs:
            resp = app.route_message(user_text)
            self.assertIn("I didn't recognize that command", resp)
            self.assertNotIn("Run Inspection", resp)

    # 15. Subprocess timeout handling
    @patch("subprocess.run")
    def test_15_subprocess_timeout_handling(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["python"], timeout=5)
        res = app.run_recovery_subprocess(self.valid_run_id, ["--verify-only"], timeout=5)
        self.assertFalse(res["success"])
        self.assertEqual(res["exit_code"], -1)
        self.assertEqual(res["error"], "TimeoutExpired")
        self.assertIn("timed out", res["stderr"])

    # 16. Non-zero subprocess exit handling
    @patch("subprocess.run")
    def test_16_nonzero_subprocess_exit_handling(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 2
        mock_proc.stdout = "Step 2: Mismatched hash"
        mock_proc.stderr = "Errors occurred"
        mock_run.return_value = mock_proc

        res = app.run_recovery_subprocess(self.valid_run_id, ["--verify-only"])
        self.assertFalse(res["success"])
        self.assertEqual(res["exit_code"], 2)
        self.assertIn("Mismatched hash", res["stdout"])

    # 17. Missing recovery script handling
    @patch("os.path.exists")
    def test_17_missing_recovery_script_handling(self, mock_exists):
        # Pretend scripts/recover_run.py does not exist
        def side_effect(path):
            if "recover_run.py" in str(path):
                return False
            return True

        mock_exists.side_effect = side_effect
        res = app.run_recovery_subprocess(self.valid_run_id, ["--verify-only"])
        self.assertFalse(res["success"])
        self.assertEqual(res["error"], "ScriptNotFound")
        self.assertIn("Recovery script not found", res["stderr"])

    # 18. Malformed or unexpected subprocess output handling
    @patch("subprocess.run")
    def test_18_malformed_subprocess_output_handling(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Corrupted\x00\xffbinary output"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        res = app.run_recovery_subprocess(self.valid_run_id, ["--verify-only"])
        self.assertTrue(res["success"])
        self.assertIn("Corrupted", res["stdout"])

    # 19. No shell execution from chat input
    @patch("subprocess.run")
    def test_19_no_shell_execution_from_chat_input(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "OK"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        app.run_recovery_subprocess(self.valid_run_id, ["--dry-run"])
        # Check call arguments
        self.assertTrue(mock_run.called)
        _, kwargs = mock_run.call_args
        self.assertNotIn("shell", kwargs)
        self.assertFalse(kwargs.get("shell", False))

    # 20. Existing lineage and run-list regression tests
    def test_20_existing_lineage_and_run_list_regression(self):
        runs = app.fetch_all_runs()
        self.assertIsInstance(runs, list)
        self.assertGreater(len(runs), 0)

        lineage_resp = app.handle_lineage_report(self.valid_short_id)
        self.assertIsInstance(lineage_resp, str)
        self.assertTrue(
            "Lineage Report" in lineage_resp or "Pipeline Lineage Trace" in lineage_resp,
            "Lineage report did not render expected sections.",
        )

    # 21. Streamlit session-state and confirmation lifecycle where practical
    def test_21_streamlit_session_state_confirmation_lifecycle(self):
        # Set confirmation state
        state = {"run_id": self.valid_run_id, "short_id": self.valid_short_id}
        app.set_confirmation_state(state)
        self.assertEqual(app.get_confirmation_state(), state)

        # Clear state
        app.clear_confirmation_state()
        self.assertIsNone(app.get_confirmation_state())

    # 22. Confirmation state cleared after cancellation or a failed recovery attempt
    def test_22_confirmation_state_cleared_after_cancel_or_failure(self):
        # 1. Cancel clears state
        app.set_confirmation_state({"run_id": self.valid_run_id, "short_id": self.valid_short_id})
        cancel_resp = app.route_message("cancel")
        self.assertIn("Recovery Cancelled", cancel_resp)
        self.assertIsNone(app.get_confirmation_state())

        # 2. Subsequent confirmation attempt fails because state is cleared
        subsequent_confirm = app.route_message(f"confirm recover {self.valid_short_id}")
        self.assertIn("No recovery operation is currently pending", subsequent_confirm)

    # 23. Cross-session isolation: pending recovery in Session A cannot authorize restore in Session B
    @patch("app.run_recovery_subprocess")
    def test_23_cross_session_isolation(self, mock_subproc):
        session_a = app.IsolatedTestSession()
        session_b = app.IsolatedTestSession()

        # In Session A: user initiates recovery prompt
        with session_a:
            app.set_confirmation_state({
                "run_id": self.valid_run_id,
                "short_id": self.valid_short_id,
                "timestamp": 123456789.0,
            })
            self.assertIsNotNone(app.get_confirmation_state())
            self.assertEqual(app.get_confirmation_state()["run_id"], self.valid_run_id)

        # In Session B: confirmation state MUST be empty
        with session_b:
            self.assertIsNone(app.get_confirmation_state())
            # Attempting confirm recover in Session B MUST be rejected
            resp_b = app.route_message(f"confirm recover {self.valid_short_id}")
            self.assertIn("No recovery operation is currently pending confirmation", resp_b)
            mock_subproc.assert_not_called()

        # Session A still retains its own pending confirmation until acted on or cancelled
        with session_a:
            self.assertIsNotNone(app.get_confirmation_state())
            app.clear_confirmation_state()
            self.assertIsNone(app.get_confirmation_state())

    # 24. No production global state leak outside Streamlit
    def test_24_no_production_global_state_leak(self):
        # When test override is disabled and running outside Streamlit,
        # get_confirmation_state() returns None and set_confirmation_state() does not leak to a global dict.
        app.disable_test_session_override()
        try:
            self.assertIsNone(app.get_confirmation_state())
            app.set_confirmation_state({"run_id": "test_run"})
            # Without active Streamlit context and no test override, nothing is stored in global state
            self.assertIsNone(app.get_confirmation_state())
        finally:
            app.enable_test_session_override()


if __name__ == "__main__":
    unittest.main()

