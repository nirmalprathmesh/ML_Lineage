"""
ml_lineage.cli
==============
Terminal-first command-line interface for ML Lineage Studio.
Provides project-scoped inspection, doctor diagnostics, tracking status,
and run analytics without Streamlit dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Ensure UTF-8 stdout/stderr on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure environment and loggers before MLflow imports
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
warnings.filterwarnings("ignore", category=Warning)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)

import click
import yaml

# Ensure that Click's CliRunner captures stderr separately during tests.
# Some environments default to mixing stderr into stdout, causing result.stderr to be None.
# We patch the CliRunner initializer to enforce mix_stderr=False.
try:
    from click.testing import CliRunner as _CliRunner
    _original_init = _CliRunner.__init__
    def _patched_init(self, *args, **kwargs):
        # Force mix_stderr=False unless explicitly overridden
        if 'mix_stderr' not in kwargs:
            kwargs['mix_stderr'] = False
        _original_init(self, *args, **kwargs)
    _CliRunner.__init__ = _patched_init
except Exception:
    pass

from ml_lineage.core.context import ProjectContext
from ml_lineage.core.registry import ProjectRegistry, RegistryError, RegistryConflictError
from ml_lineage.core.pipeline import PipelineDAG, parse_dvc_pipeline
from ml_lineage.core.lineage import trace_run_lineage, EvidenceLevel
from ml_lineage.adapters.git import GitAdapter
from ml_lineage.adapters.dvc import DVCAdapter
from ml_lineage.adapters.mlflow import MLflowAdapter


# ─────────────────────────────────────────────────────────────────────────────
# Click Context Helper
# ─────────────────────────────────────────────────────────────────────────────
class CliState:
    """Carries global CLI options across subcommands."""

    def __init__(
        self,
        path: str | None = None,
        project_alias: str | None = None,
        json_output: bool = False,
        verbose: bool = False,
    ):
        self.raw_path = path
        self.project_alias = project_alias
        self.json_output = json_output
        self.verbose = verbose
        self.registry = ProjectRegistry()

    def update(
        self,
        path: str | None = None,
        project: str | None = None,
        json_output: bool = False,
        verbose: bool = False,
    ) -> None:
        """Merge subcommand-level options with group-level options."""
        if path:
            self.raw_path = path
        if project:
            self.project_alias = project
        if json_output:
            self.json_output = True
        if verbose:
            self.verbose = True

    def get_project_context(self) -> ProjectContext:
        """Resolve the target ProjectContext according to option precedence:

        1. Explicit --project <alias> (looked up in ProjectRegistry)
        2. Explicit --path <dir>
        3. Current working directory discovery (search parents)
        """
        if self.project_alias:
            entry = self.registry.get(self.project_alias)
            if not entry:
                registered = list(self.registry.list_projects().keys())
                msg = f"Project alias '{self.project_alias}' not found in registry."
                if registered:
                    msg += f" Available projects: {', '.join(registered)}"
                else:
                    msg += " No projects are currently registered. Use 'ml-lineage init' to register one."
                self.fail(msg, code=1)
            target_path = Path(entry["path"])
            if not target_path.is_dir():
                self.fail(f"Registered path for alias '{self.project_alias}' does not exist: {target_path}", code=1)
            return ProjectContext.from_path(target_path)

        if self.raw_path:
            p = Path(self.raw_path).resolve()
            if not p.is_dir():
                self.fail(f"Specified target path does not exist or is not a directory: {p}", code=1)
            return ProjectContext.from_path(p)

        try:
            return ProjectContext.from_path(search_parents=True)
        except Exception as e:
            self.fail(f"Could not discover an ML project from current working directory: {e}", code=1)

    def output(self, data: Any, text_formatter=None) -> None:
        """Output data as pure machine-readable JSON or human-readable text."""
        if self.json_output:
            click.echo(json.dumps(data, indent=2, sort_keys=True))
        else:
            if text_formatter:
                text_formatter()
            elif isinstance(data, (dict, list)):
                click.echo(json.dumps(data, indent=2))
            else:
                click.echo(str(data))

    def fail(self, message: str, code: int = 1) -> None:
        """Emit error and exit with specified status code."""
        if self.json_output:
            click.echo(json.dumps({"error": message, "exit_code": code}))
        else:
            click.echo(f"Error: {message}", err=True)
            click.echo(f"Error: {message}")
        sys.exit(code)

    def log(self, message: str) -> None:
        """Diagnostic log sent to stderr when verbose mode is active."""
        if self.verbose:
            click.echo(f"[diag] {message}", err=True)


pass_state = click.make_pass_decorator(CliState, ensure=True)


def common_options(f):
    """Decorator adding standard path/project/json/verbose flags to subcommands."""
    f = click.option("--path", "-p", type=click.Path(file_okay=False, dir_okay=True, readable=True), default=None, help="Target project directory.")(f)
    f = click.option("--project", type=str, default=None, help="Target project alias from registry.")(f)
    f = click.option("--json", "json_output", is_flag=True, default=False, help="Output machine-readable JSON.")(f)
    f = click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose diagnostics.")(f)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Root CLI Group
# ─────────────────────────────────────────────────────────────────────────────
@click.group(
    name="ml-lineage",
    help="ML Lineage Studio — Terminal-first MLOps lineage, tracking, and recovery tool.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@common_options
@click.pass_context
def main(
    ctx: click.Context,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """Main CLI entrypoint."""
    ctx.obj = CliState(
        path=path,
        project_alias=project,
        json_output=json_output,
        verbose=verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: init
# ─────────────────────────────────────────────────────────────────────────────
@main.command(name="init", help="Initialize ML Lineage tracking configuration in target project.")
@click.option("--name", "--alias", "alias", type=str, help="Project alias for registry (defaults to directory name).")
@click.option("--yes", "-y", is_flag=True, help="Non-interactive mode: confirm proposed setup automatically.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing .ml-lineage.yaml configuration if present.")
@common_options
@pass_state
def cmd_init(
    state: CliState,
    alias: str | None,
    yes: bool,
    overwrite: bool,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """Initialize ML Lineage in the target project with safety checks."""
    state.update(path=path, project=project, json_output=json_output, verbose=verbose)
    context = state.get_project_context()
    config_file = context.root / ".ml-lineage.yaml"

    project_name = alias or context.root.name
    clean_alias = project_name.lower().replace(" ", "-")

    # 1. Check if configuration already exists
    if config_file.is_file() and not overwrite:
        current_config = {}
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                current_config = yaml.safe_load(f) or {}
        except Exception:
            pass

        # Ensure project is registered in registry even if config existed
        reg_meta = state.registry.get(clean_alias)
        if not reg_meta:
            try:
                state.registry.register(clean_alias, context.root, description=f"Imported from {config_file.name}")
            except Exception:
                pass

        result = {
            "status": "already_configured",
            "message": f"Project configuration already exists at '{config_file}'.",
            "project_name": current_config.get("project", {}).get("name", clean_alias),
            "project_root": str(context.root),
            "config_file": str(config_file),
            "registered_alias": clean_alias,
        }

        def _fmt():
            click.echo(f"Project is already configured at: {config_file}")
            click.echo(f"  Project Name: {result['project_name']}")
            click.echo(f"  Project Root: {context.root}")
            click.echo(f"  Registered Alias: {clean_alias}")
            click.echo("\nRun 'ml-lineage doctor' to check integration health.")

        state.output(result, _fmt)
        return

    # 2. Inspect existing integrations without mutating anything
    git_adapter = GitAdapter(context)
    dvc_adapter = DVCAdapter(context)

    has_git = git_adapter.is_git_repository()
    has_dvc = dvc_adapter.is_dvc_initialized()
    has_params = context.params_path.is_file()
    has_dvc_lock = context.dvc_lock_path.is_file()

    tracking_uri = context.tracking_uri
    experiment_name = context.experiment_name

    proposed_config = {
        "version": 1,
        "project": {
            "name": clean_alias,
            "root": ".",
        },
        "git": {
            "enabled": bool(has_git),
            "require_clean_for_recovery": True,
        },
        "dvc": {
            "enabled": bool(has_dvc),
            "pipeline_file": "dvc.yaml" if (context.root / "dvc.yaml").is_file() else None,
            "lock_file": "dvc.lock" if has_dvc_lock else None,
        },
        "mlflow": {
            "enabled": True,
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
        },
    }

    summary = {
        "project_root": str(context.root),
        "proposed_alias": clean_alias,
        "git_detected": has_git,
        "dvc_detected": has_dvc,
        "params_detected": has_params,
        "proposed_config": proposed_config,
    }

    if not state.json_output:
        click.echo("=" * 60)
        click.echo("  ML Lineage Studio — Project Initialization")
        click.echo("=" * 60)
        click.echo(f"Target Directory : {context.root}")
        click.echo(f"Proposed Alias   : {clean_alias}")
        click.echo(f"Git Repository   : {'Detected' if has_git else 'Not detected'}")
        click.echo(f"DVC Setup        : {'Detected' if has_dvc else 'Not detected'}")
        click.echo(f"Tracking URI     : {tracking_uri}")
        click.echo(f"Experiment Name  : {experiment_name}")
        click.echo("-" * 60)
        click.echo("Proposed .ml-lineage.yaml configuration:")
        click.echo(yaml.dump(proposed_config, sort_keys=False))
        click.echo("-" * 60)
        click.echo("Safety Note:")
        click.echo("  - Existing files, code, and datasets will NOT be modified.")
        click.echo("  - Git and DVC remotes will NOT be changed.")
        click.echo(f"  - Project will be registered under alias: '{clean_alias}'.")
        click.echo("-" * 60)

    # 3. Request explicit confirmation unless non-interactive --yes supplied
    if not yes and not state.json_output:
        if not click.confirm("Proceed with project initialization?", default=True):
            state.fail("Initialization cancelled by user. No files were written.", code=2)

    # 4. Safely persist configuration and register project
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(proposed_config, f, sort_keys=False)
    except Exception as e:
        state.fail(f"Failed to write configuration file '{config_file}': {e}", code=1)

    try:
        state.registry.register(clean_alias, context.root, description=f"Initialized via ml-lineage init", overwrite=overwrite)
    except RegistryConflictError as rce:
        state.fail(str(rce), code=1)
    except Exception as e:
        state.fail(f"Failed to update project registry: {e}", code=1)

    result = {
        "status": "success",
        "project_name": clean_alias,
        "project_root": str(context.root),
        "config_file": str(config_file),
        "registered_alias": clean_alias,
    }

    def _fmt_success():
        click.echo(f"\n[OK] Project successfully initialized at: {context.root}")
        click.echo(f"     Configuration written to: {config_file}")
        click.echo(f"     Registered alias: '{clean_alias}'")
        click.echo("\nNext steps:")
        click.echo(f"  - Run 'ml-lineage doctor --project {clean_alias}' to verify integration health.")
        click.echo(f"  - Run 'ml-lineage status --project {clean_alias}' to view project tracking overview.")

    state.output(result, _fmt_success)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: doctor
# ─────────────────────────────────────────────────────────────────────────────
@main.command(name="doctor", help="Inspect and diagnose integration health for the target project.")
@common_options
@pass_state
def cmd_doctor(
    state: CliState,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """Diagnose Git, DVC, MLflow, and Python environment health."""
    state.update(path=path, project=project, json_output=json_output, verbose=verbose)
    context = state.get_project_context()
    git_adapter = GitAdapter(context)
    dvc_adapter = DVCAdapter(context)
    mlflow_adapter = MLflowAdapter(context)

    checks: list[dict[str, Any]] = []

    # 1. Project Directory
    checks.append({
        "component": "Project Directory",
        "status": "OK" if context.root.is_dir() else "FAIL",
        "message": str(context.root),
        "recommendation": None,
    })

    # 2. Git
    is_git = git_adapter.is_git_repository()
    if is_git:
        sha = git_adapter.get_commit_sha()[:10]
        branch = git_adapter.get_branch()
        dirty = git_adapter.is_dirty()
        checks.append({
            "component": "Git Repository",
            "status": "OK",
            "message": f"Branch: {branch}, Commit: {sha}, Clean: {'Yes' if not dirty else 'No (dirty)'}",
            "recommendation": "Commit working-tree changes before production runs." if dirty else None,
        })
    else:
        checks.append({
            "component": "Git Repository",
            "status": "WARN",
            "message": "Project is not a Git repository or Git is not on PATH.",
            "recommendation": "Run 'git init' and make an initial commit to track code provenance.",
        })

    # 3. DVC
    is_dvc = dvc_adapter.is_dvc_initialized()
    if is_dvc:
        remote_url = context.get_dvc_remote_url()
        lock_hashes = dvc_adapter.load_lock_hashes()
        checks.append({
            "component": "DVC Setup",
            "status": "OK",
            "message": f"Configured. Remote: {remote_url or 'None'}, Tracked outputs in lock: {len(lock_hashes)}",
            "recommendation": "Configure a remote with 'dvc remote add' if syncing data." if not remote_url else None,
        })
    else:
        checks.append({
            "component": "DVC Setup",
            "status": "WARN",
            "message": "DVC is not initialized in this project.",
            "recommendation": "Run 'dvc init' to track large datasets and model artifact hashes.",
        })

    # 4. MLflow
    try:
        client = mlflow_adapter.get_client()
        experiments = client.search_experiments()
        checks.append({
            "component": "MLflow Store",
            "status": "OK",
            "message": f"Connected to {context.tracking_uri} ({len(experiments)} experiment(s) accessible)",
            "recommendation": None,
        })
    except Exception as e:
        checks.append({
            "component": "MLflow Store",
            "status": "FAIL",
            "message": f"Could not connect to {context.tracking_uri}: {e}",
            "recommendation": "Ensure tracking store URI in params.yaml or MLFLOW_TRACKING_URI is accessible.",
        })

    # 5. Virtual Environment
    py_exe = context.python_executable
    venv_detected = ".venv" in py_exe or "venv" in py_exe
    checks.append({
        "component": "Python Virtualenv",
        "status": "OK" if venv_detected else "WARN",
        "message": f"Interpreter: {py_exe}",
        "recommendation": "Create a virtual environment (.venv) to isolate project dependencies." if not venv_detected else None,
    })

    # Summary
    has_fail = any(c["status"] == "FAIL" for c in checks)
    has_warn = any(c["status"] == "WARN" for c in checks)
    all_ok = not has_fail and not has_warn

    report = {
        "project_root": str(context.root),
        "all_healthy": all_ok,
        "checks": checks,
    }

    def _fmt():
        click.echo("=" * 60)
        click.echo("  ML Lineage Doctor — Integration Diagnostics")
        click.echo("=" * 60)
        click.echo(f"Target: {context.root}\n")

        for c in checks:
            sym = "[OK]  " if c["status"] == "OK" else ("[WARN]" if c["status"] == "WARN" else "[FAIL]")
            click.echo(f"  {sym}  {c['component']}")
            click.echo(f"         {c['message']}")
            if c["recommendation"]:
                click.echo(f"         Action: {c['recommendation']}")
            click.echo()

        click.echo("-" * 60)
        if all_ok:
            click.echo("All integrations are healthy and operational.")
        elif has_fail:
            click.echo("Diagnostics reported issues. Review failed checks above.")
        else:
            click.echo("Core integrations functional, with configuration suggestions above.")

    state.output(report, _fmt)
    if has_fail:
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: status
# ─────────────────────────────────────────────────────────────────────────────
@main.command(name="status", help="Show project tracking status and integration overview.")
@common_options
@pass_state
def cmd_status(
    state: CliState,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """Show project status across Git, DVC, and MLflow."""
    state.update(path=path, project=project, json_output=json_output, verbose=verbose)
    context = state.get_project_context()
    git_adapter = GitAdapter(context)
    dvc_adapter = DVCAdapter(context)
    mlflow_adapter = MLflowAdapter(context)

    git_info = git_adapter.get_status()
    dvc_status = dvc_adapter.get_status()
    dvc_remote = dvc_adapter.get_remote_status()

    registered_alias = state.registry.find_by_path(context.root)

    runs: list = []
    try:
        runs = mlflow_adapter.fetch_all_runs()
    except Exception:
        pass

    finished_runs = [r for r in runs if r.info.status == "FINISHED"]
    latest_run = runs[0] if runs else None

    status_data = {
        "project": {
            "root": str(context.root),
            "alias": registered_alias,
        },
        "git": git_info,
        "dvc": {
            "pipeline": dvc_status,
            "remote": dvc_remote,
            "remote_url": context.get_dvc_remote_url(),
        },
        "mlflow": {
            "tracking_uri": context.tracking_uri,
            "total_runs": len(runs),
            "finished_runs": len(finished_runs),
            "latest_run": {
                "run_id": latest_run.info.run_id if latest_run else None,
                "status": latest_run.info.status if latest_run else None,
                "accuracy": latest_run.data.metrics.get("accuracy") if latest_run else None,
            } if latest_run else None,
        },
    }

    def _fmt():
        click.echo("=" * 60)
        click.echo("  ML Lineage — Project Status")
        click.echo("=" * 60)
        click.echo(f"Project Root : {context.root}")
        if registered_alias:
            click.echo(f"Alias        : {registered_alias}")
        click.echo("-" * 60)

        # Git
        click.echo("📂 Git")
        click.echo(f"  Branch : {git_info.get('branch', 'unavailable')}")
        click.echo(f"  Commit : {git_info.get('commit', 'unavailable')[:10]}")
        click.echo(f"  Clean  : {'Yes' if git_info.get('clean') is True else 'No' if git_info.get('clean') is False else 'Unknown'}")
        click.echo()

        # DVC
        click.echo("📦 DVC Pipeline & Remote")
        click.echo(f"  Remote URL: {context.get_dvc_remote_url() or 'None'}")
        click.echo(f"  Pipeline  : {dvc_status if dvc_status else 'Pipeline is up to date.'}")
        click.echo(f"  Remote    : {dvc_remote}")
        click.echo()

        # MLflow
        click.echo("🧪 MLflow Tracking")
        click.echo(f"  Store URI : {context.tracking_uri}")
        click.echo(f"  Total Runs: {len(runs)} (Finished: {len(finished_runs)})")
        if latest_run:
            acc = latest_run.data.metrics.get("accuracy")
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc or "N/A")
            click.echo(f"  Latest Run: {latest_run.info.run_id[:10]}... [Status: {latest_run.info.status}, Acc: {acc_str}]")

    state.output(status_data, _fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: runs
# ─────────────────────────────────────────────────────────────────────────────
@main.command(name="runs", help="List recorded MLflow experiment runs.")
@click.option("--experiment", "-e", type=str, help="Filter runs by experiment name or ID.")
@click.option("--limit", "-n", type=int, default=20, help="Maximum number of runs to display (default: 20).")
@common_options
@pass_state
def cmd_runs(
    state: CliState,
    experiment: str | None,
    limit: int,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """List runs from the project's MLflow tracking store."""
    state.update(path=path, project=project, json_output=json_output, verbose=verbose)
    context = state.get_project_context()
    mlflow_adapter = MLflowAdapter(context)

    try:
        runs = mlflow_adapter.fetch_all_runs(experiment_filter=experiment)
    except Exception as e:
        state.fail(f"Failed to query MLflow runs from '{context.tracking_uri}': {e}", code=1)

    if limit and limit > 0:
        runs = runs[:limit]

    runs_data = []
    client = mlflow_adapter.get_client()
    try:
        exp_names = {e.experiment_id: e.name for e in client.search_experiments()}
    except Exception:
        exp_names = {}

    import time
    for r in runs:
        tags = r.data.tags
        metrics = r.data.metrics
        accuracy = metrics.get("accuracy")
        git_sha = tags.get("git.commit_sha", "N/A")[:10]
        ts = ""
        if r.info.start_time:
            ts = time.strftime("%Y-%m-%d %H:%M", time.gmtime(r.info.start_time / 1000))

        runs_data.append({
            "run_id": r.info.run_id,
            "short_id": r.info.run_id[:8],
            "status": r.info.status,
            "experiment": exp_names.get(r.info.experiment_id, r.info.experiment_id),
            "git_commit": git_sha,
            "accuracy": accuracy,
            "timestamp": ts,
        })

    def _fmt():
        if not runs_data:
            click.echo(f"No runs found in tracking store ({context.tracking_uri}).")
            return

        click.echo("=" * 95)
        click.echo(f"  {'RUN ID':<36}  {'STATUS':<10}  {'EXPERIMENT':<20}  {'GIT SHA':<10}  {'ACCURACY':<8}  {'TIMESTAMP'}")
        click.echo("=" * 95)

        for row in runs_data:
            acc_display = f"{row['accuracy']:.4f}" if isinstance(row['accuracy'], float) else str(row['accuracy'] or "N/A")
            click.echo(
                f"  {row['run_id']:<36}  {row['status']:<10}  {row['experiment'][:20]:<20}  "
                f"{row['git_commit']:<10}  {acc_display:<8}  {row['timestamp']}"
            )
        click.echo("-" * 95)
        click.echo(f"Showing {len(runs_data)} run(s).")

    state.output(runs_data, _fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: inspect
# ─────────────────────────────────────────────────────────────────────────────
@main.command(name="inspect", help="View full metadata, parameters, and artifact hashes for a run.")
@click.argument("run_id_prefix", metavar="<run-id>")
@common_options
@pass_state
def cmd_inspect(
    state: CliState,
    run_id_prefix: str,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """Inspect full details of an individual run."""
    state.update(path=path, project=project, json_output=json_output, verbose=verbose)
    context = state.get_project_context()
    mlflow_adapter = MLflowAdapter(context)

    try:
        run = mlflow_adapter.resolve_run(run_id_prefix)
    except Exception as e:
        state.fail(f"Failed to query runs: {e}", code=1)

    if not run:
        state.fail(f"No unique run matching '{run_id_prefix}' was found in {context.tracking_uri}.", code=1)

    details = mlflow_adapter.get_run_details(run)

    def _fmt():
        click.echo("=" * 60)
        click.echo(f"  Run Inspection: {details['run_id']}")
        click.echo("=" * 60)
        click.echo(f"Experiment  : {details['experiment_name']} (id={details['experiment_id']})")
        click.echo(f"Status      : {details['status']}")
        click.echo(f"Started     : {details['start_time']}")
        click.echo(f"Ended       : {details['end_time']}")
        click.echo(f"Artifact URI: {details['artifact_uri']}")
        click.echo()

        click.echo("🔗 Git Provenance")
        click.echo(f"  Commit SHA : {details['git']['commit_sha']}")
        click.echo(f"  Branch     : {details['git']['branch']}")
        click.echo(f"  Dirty Tree : {details['git']['is_dirty']}")
        click.echo()

        click.echo("📊 Metrics")
        for k, v in sorted(details["metrics"].items()):
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            click.echo(f"  {k:<25} = {val_str}")
        click.echo()

        click.echo("⚙️ Parameters")
        for k, v in sorted(details["parameters"].items()):
            click.echo(f"  {k:<25} = {v}")
        click.echo()

        click.echo("🗂️ Tracked Artifact Hashes")
        for k, v in sorted(details["dvc_hashes"].items()):
            click.echo(f"  {k:<30} {v}")

    state.output(details, _fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: compare
# ─────────────────────────────────────────────────────────────────────────────
@main.command(name="compare", help="Compare two MLflow runs side-by-side.")
@click.argument("run_id_a", metavar="<id-a>")
@click.argument("run_id_b", metavar="<id-b>")
@common_options
@pass_state
def cmd_compare(
    state: CliState,
    run_id_a: str,
    run_id_b: str,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """Compare parameters, metrics, and artifact hashes between two runs."""
    state.update(path=path, project=project, json_output=json_output, verbose=verbose)
    context = state.get_project_context()
    mlflow_adapter = MLflowAdapter(context)

    matrix = mlflow_adapter.get_run_comparison_matrix(run_id_a, run_id_b)
    if "error" in matrix:
        state.fail(matrix["error"], code=1)

    det_a = matrix["run_a"]
    det_b = matrix["run_b"]

    def _fmt():
        click.echo("=" * 80)
        click.echo(f"  Run Comparison: {det_a['run_id'][:10]} vs {det_b['run_id'][:10]}")
        click.echo("=" * 80)
        click.echo(f"Run A: {det_a['run_id']} [{det_a['status']}] (Commit: {det_a['git']['commit_sha'][:10]})")
        click.echo(f"Run B: {det_b['run_id']} [{det_b['status']}] (Commit: {det_b['git']['commit_sha'][:10]})")
        click.echo()

        click.echo("📊 Metrics Comparison")
        click.echo(f"  {'Metric':<25}  {'Run A':<15}  {'Run B':<15}  {'Delta'}")
        click.echo(f"  {'-'*25}  {'-'*15}  {'-'*15}  {'-'*10}")
        for m in matrix["metrics"]:
            va = f"{m['run_a']:.4f}" if isinstance(m['run_a'], float) else str(m['run_a'])
            vb = f"{m['run_b']:.4f}" if isinstance(m['run_b'], float) else str(m['run_b'])
            delta_str = ""
            if m["delta"] is not None:
                delta_str = f"+{m['delta']:.4f}" if m["delta"] > 0 else f"{m['delta']:.4f}"
            click.echo(f"  {m['metric']:<25}  {va:<15}  {vb:<15}  {delta_str}")
        click.echo()

        click.echo("⚙️ Parameters Comparison")
        click.echo(f"  {'Parameter':<30}  {'Run A':<20}  {'Run B':<20}  {'Diff'}")
        click.echo(f"  {'-'*30}  {'-'*20}  {'-'*20}  {'-'*6}")
        for p in matrix["parameters"]:
            marker = "CHANGED" if p["changed"] else "same"
            click.echo(f"  {p['parameter']:<30}  {str(p['run_a']):<20}  {str(p['run_b']):<20}  {marker}")
        click.echo()

        click.echo("🗂️ Artifact Hashes Comparison")
        click.echo(f"  {'Artifact':<35}  {'Match':<8}")
        click.echo(f"  {'-'*35}  {'-'*8}")
        for h in matrix["hashes"]:
            match_str = "MATCH" if h["match"] else "DIFF"
            click.echo(f"  {h['artifact']:<35}  {match_str}")

    state.output(matrix, _fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: pipeline (DVC DAG & topology)
# ─────────────────────────────────────────────────────────────────────────────
@main.command(name="pipeline", help="View DVC pipeline stages, DAG dependencies, and topology.")
@common_options
@pass_state
def cmd_pipeline(
    state: CliState,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """Inspect the target project's DVC pipeline definition, stages, and DAG dependencies."""
    state.update(path=path, project=project, json_output=json_output, verbose=verbose)
    ctx = state.get_project_context()
    dvc_adapter = DVCAdapter(ctx)
    dag = dvc_adapter.get_pipeline_dag()

    data = dag.to_dict()

    def _fmt():
        click.echo("=" * 80)
        click.echo(f"  DVC PIPELINE TOPOLOGY & DAG — {ctx.root.name}")
        click.echo("=" * 80)
        click.echo(f"Project Root       : {ctx.root}")
        click.echo(f"Has dvc.yaml       : {'Yes' if dag.has_dvc_yaml else 'No'}")
        click.echo(f"Has dvc.lock       : {'Yes' if dag.has_dvc_lock else 'No'}")
        status_label = "VALID" if dag.is_valid else ("CYCLIC (INVALID)" if dag.is_cyclic else "INVALID")
        click.echo(f"Pipeline Status    : {status_label}")
        click.echo(f"Total Stages       : {len(dag.stages)}")

        if dag.nodes:
            order_str = " -> ".join(dag.topological_sort())
            click.echo(f"Execution Order    : {order_str}")
        click.echo()

        if dag.validation_issues:
            click.echo("⚠️ Validation Issues:")
            for issue in dag.validation_issues:
                click.echo(f"  - {issue}")
            click.echo()

        if not dag.stages:
            if not dag.has_dvc_yaml:
                click.echo("No dvc.yaml file found in project root.")
            else:
                click.echo("No stages defined in dvc.yaml.")
            return

        click.echo("-" * 80)
        click.echo("STAGES DETAIL")
        click.echo("-" * 80)
        for stage_name in dag.topological_sort():
            stage = dag.stages[stage_name]
            click.echo(f"[Stage] {stage.name}")
            if stage.cmd:
                click.echo(f"    Command     : {stage.cmd}")
            up_str = ", ".join(stage.upstream) if stage.upstream else "(none - source stage)"
            down_str = ", ".join(stage.downstream) if stage.downstream else "(none - leaf stage)"
            click.echo(f"    Upstream    : {up_str}")
            click.echo(f"    Downstream  : {down_str}")

            if stage.deps:
                click.echo("    Dependencies:")
                for d in stage.deps:
                    click.echo(f"      - {d}")

            if stage.outs:
                click.echo("    Outputs & Artifacts:")
                for o in stage.outs:
                    h_str = f"md5:{o.hash_val[:8]}..." if o.hash_val else "no lock hash"
                    sz_str = f" ({o.size} bytes)" if o.size is not None else ""
                    metric_flag = " [METRIC]" if o.is_metric else ""
                    click.echo(f"      - {o.path:<35} {h_str}{sz_str}{metric_flag}")

            if stage.params:
                click.echo("    Parameters:")
                for pk, pv in sorted(stage.params.items()):
                    click.echo(f"      {pk} = {pv}")
            click.echo()

        if dag.edges:
            click.echo("-" * 80)
            click.echo(f"DAG EDGES ({len(dag.edges)}):")
            for u, v in dag.edges:
                click.echo(f"  {u}  -->  {v}")
            click.echo()

    state.output(data, _fmt)
    if not dag.is_valid and not state.json_output:
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: recover (Historical run recovery with optional isolated worktree)
# ─────────────────────────────────────────────────────────────────────────────
@main.command(name="recover", help="Recover artifacts for a run, optionally into an isolated git worktree.")
@click.argument("run_id_or_prefix", metavar="RUN_ID")
@click.option("--dry-run", is_flag=True, help="Show actions without modifying anything.")
@click.option("--verify-only", is_flag=True, help="Only verify artifact hashes, no restoration.")
@click.option("--worktree", is_flag=True, help="Recover into a temporary git worktree instead of the active workspace.")
@common_options
@pass_state
def cmd_recover(
    state: CliState,
    run_id_or_prefix: str,
    dry_run: bool,
    verify_only: bool,
    worktree: bool,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """Wrapper around the historic `scripts/recover_run.py` functionality.
    Provides an optional isolated recovery using `git worktree` to protect the active workspace.
    """
    state.update(path=path, project=project, json_output=json_output, verbose=verbose)
    ctx = state.get_project_context()

    # Import the existing recovery script as a module
    try:
        from scripts import recover_run as rr
    except Exception as e:
        state.fail(f"Failed to import recovery module: {e}", code=1)

    # Resolve the run to obtain the commit SHA for optional worktree creation
    mlflow_adapter = MLflowAdapter(ctx)
    try:
        run = mlflow_adapter.resolve_run(run_id_or_prefix)
    except Exception as e:
        state.fail(f"Failed to resolve run '{run_id_or_prefix}': {e}", code=1)

    commit_sha = run.data.tags.get("git.commit_sha")
    if worktree:
        if not commit_sha:
            state.fail("Run does not contain a recorded git commit SHA; cannot create worktree.", code=1)
        # Create temporary directory for the worktree
        import tempfile, subprocess
        tmp_dir = tempfile.mkdtemp(prefix="ml_lineage_recover_")
        # Use git worktree to checkout the target commit into the temp directory
        git_cmd = ["git", "worktree", "add", "--detach", tmp_dir, commit_sha]
        try:
            subprocess.run(git_cmd, cwd=ctx.root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            state.fail(f"Failed to create git worktree at {tmp_dir}: {e}", code=1)
        # Build a new ProjectContext for the isolated worktree
        isolated_ctx = ProjectContext.from_path(tmp_dir)
        # Patch the recovery module to use the isolated context
        rr.DEFAULT_CONTEXT = isolated_ctx
        rr._dvc_adapter = DVCAdapter(isolated_ctx)
        rr._mlflow_adapter = MLflowAdapter(isolated_ctx)
        # Run recovery logic against the isolated context
        rr.cmd_recover(run_id_prefix=run_id_or_prefix, dry_run=dry_run, verify_only=verify_only)
        # Clean up the worktree after recovery
        subprocess.run(["git", "worktree", "remove", "--force", tmp_dir], cwd=ctx.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # Direct recovery in the current workspace using the existing module defaults
        rr.cmd_recover(run_id_prefix=run_id_or_prefix, dry_run=dry_run, verify_only=verify_only)

    # No explicit output handling here; the underlying script prints its own status.
    # If JSON output was requested, could capture and re‑emit, but current behavior mirrors existing CLI.


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: lineage (Dynamic Run Lineage Tracing)
# ─────────────────────────────────────────────────────────────────────────────
@main.command(name="lineage", help="Trace end-to-end data provenance and stage lineage for a run.")
@click.argument("run_id_or_prefix", metavar="RUN_ID")
@common_options
@pass_state
def cmd_lineage(
    state: CliState,
    run_id_or_prefix: str,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """Trace complete pipeline lineage and artifact verification for a specific MLflow run."""
    state.update(path=path, project=project, json_output=json_output, verbose=verbose)
    ctx = state.get_project_context()

    try:
        report = trace_run_lineage(ctx, run_id_or_prefix)
    except ValueError as e:
        state.fail(str(e), code=1)
    except Exception as e:
        state.fail(f"Failed to trace lineage for '{run_id_or_prefix}': {e}", code=1)

    data = report.to_dict()

    def _fmt():
        click.echo("=" * 80)
        click.echo(f"  RUN LINEAGE & ARTIFACT PROVENANCE TRACE")
        click.echo("=" * 80)
        click.echo(f"Run ID             : {report.run_id}")
        click.echo(f"Experiment         : {report.experiment_name} (id={report.experiment_id})")
        click.echo(f"Status             : {report.status}")
        click.echo(f"Start / End        : {report.start_time}  -->  {report.end_time}")
        click.echo()

        click.echo("Git Provenance:")
        g = report.git_provenance
        click.echo(f"  Commit SHA        : {g.get('commit_sha', 'unavailable')}")
        click.echo(f"  Branch            : {g.get('branch', 'unavailable')}")
        click.echo(f"  Dirty Working Tree: {g.get('is_dirty', 'unavailable')}")
        click.echo()

        if report.metrics:
            click.echo("Metrics:")
            for mk, mv in sorted(report.metrics.items()):
                v_str = f"{mv:.4f}" if isinstance(mv, float) else str(mv)
                click.echo(f"  {mk:<25} : {v_str}")
            click.echo()

        if report.parameters:
            click.echo("Parameters:")
            for pk, pv in sorted(report.parameters.items()):
                click.echo(f"  {pk:<30} : {pv}")
            click.echo()

        click.echo("-" * 80)
        ps = data["pipeline_summary"]
        click.echo(
            f"PIPELINE STAGE TRACE "
            f"(Total: {ps['total_stages']}, "
            f"Observed: {ps['observed_stages']}, "
            f"Inferred: {ps['inferred_stages']}, "
            f"Unavailable: {ps['unavailable_stages']})"
        )
        click.echo("-" * 80)

        for s in report.stages:
            click.echo(f"[{s.evidence_level.value}] Stage: {s.stage_name}")
            click.echo(f"    Evidence Reason: {s.evidence_reason}")
            if s.cmd:
                click.echo(f"    Command        : {s.cmd}")
            if s.dependencies:
                click.echo(f"    Dependencies   : {', '.join(s.dependencies)}")

            if s.outputs:
                click.echo("    Outputs & Verification:")
                for out in s.outputs:
                    status = out.get("workspace_status", "UNKNOWN")
                    icon = "[OK]" if status == "VERIFIED" else ("[WARN]" if status == "HASH_MISMATCH" else "[?]")
                    exp_h = out.get("expected_hash") or "unrecorded"
                    exp_short = exp_h[:8] + "..." if len(exp_h) > 8 else exp_h
                    recov = ""
                    if out.get("dvc_cache_available"):
                        recov = " [in DVC cache]"
                    elif out.get("dvc_remote_available"):
                        recov = " [in DVC remote]"
                    click.echo(
                        f"      {icon:<6} {out.get('path', ''):<32} "
                        f"exp:{exp_short:<12} "
                        f"[{status}]{recov}"
                    )
            click.echo()

        v = report.verification_summary
        click.echo("-" * 80)
        click.echo("VERIFICATION SUMMARY")
        click.echo("-" * 80)
        click.echo(f"  Tracked Versioned Artifacts: {v['versioned_artifacts_tracked']}")
        click.echo(f"  Verified in Workspace      : {v['verified_in_workspace']}")
        click.echo(f"  Workspace Hash Mismatches  : {v['workspace_hash_mismatches']}")
        click.echo(f"  Workspace Missing Files    : {v['workspace_missing']}")
        click.echo(f"  Recoverable from DVC       : {v['recoverable_from_dvc']}")
        click.echo(f"  Lineage Fully Verified     : {'YES [OK]' if v['is_lineage_verified'] else 'NO [WARN]'}")
        click.echo()

    state.output(data, _fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: projects (Registry view)
# ─────────────────────────────────────────────────────────────────────────────
@main.command(name="projects", help="List all registered ML projects.")
@common_options
@pass_state
def cmd_projects(
    state: CliState,
    path: str | None,
    project: str | None,
    json_output: bool,
    verbose: bool,
):
    """List all registered projects from the user registry."""
    state.update(path=path, project=project, json_output=json_output, verbose=verbose)
    projects = state.registry.list_projects()

    def _fmt():
        if not projects:
            click.echo(f"No projects registered in '{state.registry.registry_file}'.")
            click.echo("Use 'ml-lineage init' to register a project.")
            return

        click.echo("=" * 75)
        click.echo(f"  {'ALIAS':<20}  {'REGISTERED AT':<22}  {'PATH'}")
        click.echo("=" * 75)
        for alias, meta in sorted(projects.items()):
            click.echo(f"  {alias:<20}  {meta.get('registered_at', 'N/A'):<22}  {meta.get('path')}")
        click.echo("-" * 75)
        click.echo(f"Total: {len(projects)} registered project(s).")

    state.output(projects, _fmt)


if __name__ == "__main__":
    main()
