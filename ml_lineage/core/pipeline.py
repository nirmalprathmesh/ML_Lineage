"""
ml_lineage.core.pipeline
========================
Dynamic DVC DAG pipeline parser and topology representation.
Extracts stages, commands, parameters, dependencies, outputs, and lockfile metadata
from dvc.yaml and dvc.lock, constructing a deterministic directed acyclic graph (DAG).
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from ml_lineage.core.context import ProjectContext


def _normalize_relpath(path_str: str) -> str:
    """Normalize a relative path string to posix format without leading/trailing slashes."""
    if not path_str:
        return ""
    norm = os.path.normpath(path_str).replace("\\", "/")
    return norm.lstrip("/")


def _lookup_nested_key(data: dict[str, Any], key_path: str) -> Any:
    """Look up a dot-separated key in a nested dictionary, or direct key."""
    if key_path in data:
        return data[key_path]
    parts = key_path.split(".")
    curr: Any = data
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            return None
    return curr


@dataclass
class PipelineArtifact:
    """Represents a file or directory tracked as a stage dependency or output."""

    path: str
    role: str = "output"  # "output", "dependency", "metric", "plot"
    hash_type: str = "md5"
    hash_val: str | None = None
    size: int | None = None
    is_metric: bool = False
    is_persisted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "hash_type": self.hash_type,
            "hash": self.hash_val,
            "size": self.size,
            "is_metric": self.is_metric,
        }


@dataclass
class PipelineStage:
    """Represents a single DVC pipeline stage definition."""

    name: str
    cmd: str | None = None
    deps: list[str] = field(default_factory=list)
    outs: list[PipelineArtifact] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    always_changed: bool = False
    lock_entry: dict[str, Any] | None = None
    upstream: list[str] = field(default_factory=list)
    downstream: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cmd": self.cmd,
            "deps": self.deps,
            "outs": [out.to_dict() for out in self.outs],
            "params": self.params,
            "always_changed": self.always_changed,
            "has_lock": self.lock_entry is not None,
            "upstream": self.upstream,
            "downstream": self.downstream,
        }


@dataclass
class PipelineDAG:
    """Directed Acyclic Graph (DAG) representing the project's DVC pipeline."""

    project_root: str
    has_dvc_yaml: bool = False
    has_dvc_lock: bool = False
    stages: dict[str, PipelineStage] = field(default_factory=dict)
    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)  # (upstream, downstream)
    is_valid: bool = True
    is_cyclic: bool = False
    validation_issues: list[str] = field(default_factory=list)

    def topological_sort(self) -> list[str]:
        """Return stage names in a deterministic topological execution order."""
        if self.is_cyclic or not self.nodes:
            return list(self.nodes)

        in_degree: dict[str, int] = {node: 0 for node in self.nodes}
        for u, v in self.edges:
            if v in in_degree:
                in_degree[v] += 1

        queue: deque[str] = deque(sorted([node for node, deg in in_degree.items() if deg == 0]))
        order: list[str] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for neighbor in self.stages.get(curr, PipelineStage(name=curr)).downstream:
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # In case of disconnected nodes or unvisited due to unexpected structure
        for node in self.nodes:
            if node not in order:
                order.append(node)

        return order

    def get_upstream_stages(self, stage_name: str) -> list[str]:
        """Return names of direct upstream stages for the given stage."""
        stage = self.stages.get(stage_name)
        return list(stage.upstream) if stage else []

    def get_downstream_stages(self, stage_name: str) -> list[str]:
        """Return names of direct downstream stages for the given stage."""
        stage = self.stages.get(stage_name)
        return list(stage.downstream) if stage else []

    def find_stage_for_output(self, path: str) -> str | None:
        """Find the stage name that declares the given output path."""
        target = _normalize_relpath(path)
        for stage_name, stage in self.stages.items():
            for out in stage.outs:
                if out.path == target:
                    return stage_name
        return None

    def find_stages_for_dep(self, path: str) -> list[str]:
        """Find stage names that declare the given path as a dependency."""
        target = _normalize_relpath(path)
        matching: list[str] = []
        for stage_name, stage in self.stages.items():
            if target in stage.deps:
                matching.append(stage_name)
        return matching

    def to_dict(self) -> dict[str, Any]:
        """Serialize DAG into a clean, JSON-compatible dictionary."""
        return {
            "project_root": self.project_root,
            "has_dvc_yaml": self.has_dvc_yaml,
            "has_dvc_lock": self.has_dvc_lock,
            "is_valid": self.is_valid,
            "is_cyclic": self.is_cyclic,
            "validation_issues": list(self.validation_issues),
            "execution_order": self.topological_sort(),
            "stages": {name: s.to_dict() for name, s in self.stages.items()},
            "edges": [list(edge) for edge in self.edges],
        }


def parse_dvc_pipeline(context: ProjectContext) -> PipelineDAG:
    """Parse dvc.yaml and dvc.lock for the target project into a PipelineDAG.

    Args:
        context: The resolved target project context.

    Returns:
        A populated PipelineDAG with validated stages, dependencies, and topological ordering.
    """
    root_str = str(context.root)
    dag = PipelineDAG(
        project_root=root_str,
        has_dvc_yaml=context.dvc_yaml_path.is_file(),
        has_dvc_lock=context.dvc_lock_path.is_file(),
    )

    if not dag.has_dvc_yaml:
        dag.is_valid = False
        dag.validation_issues.append(f"No dvc.yaml found in target project root ({root_str}).")
        return dag

    # 1. Parse dvc.yaml
    try:
        with context.dvc_yaml_path.open("r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f)
    except Exception as e:
        dag.is_valid = False
        dag.validation_issues.append(f"Malformed YAML in {context.dvc_yaml_path.name}: {e}")
        return dag

    if not isinstance(raw_yaml, dict):
        dag.is_valid = False
        dag.validation_issues.append("dvc.yaml content must be a YAML mapping.")
        return dag

    raw_stages = raw_yaml.get("stages")
    if raw_stages is None:
        # Valid empty pipeline
        return dag

    if not isinstance(raw_stages, dict):
        dag.is_valid = False
        dag.validation_issues.append("'stages' section in dvc.yaml must be a dictionary.")
        return dag

    # 2. Parse dvc.lock if present
    lock_stages: dict[str, Any] = {}
    if dag.has_dvc_lock:
        try:
            with context.dvc_lock_path.open("r", encoding="utf-8") as f:
                raw_lock = yaml.safe_load(f) or {}
            if isinstance(raw_lock, dict):
                lock_stages = raw_lock.get("stages", {}) or {}
        except Exception as e:
            dag.validation_issues.append(f"Warning: Failed to parse dvc.lock: {e}")

    # 3. Cache project parameter values from params.yaml
    project_params = context.params

    # 4. Construct Stage Objects
    output_to_stage: dict[str, str] = {}
    stages: dict[str, PipelineStage] = {}

    for stage_name, stage_def in raw_stages.items():
        if not isinstance(stage_def, dict):
            dag.validation_issues.append(f"Stage '{stage_name}' definition must be a mapping.")
            continue

        cmd = stage_def.get("cmd")
        always_changed = bool(stage_def.get("always_changed", False))
        lock_entry = lock_stages.get(stage_name) if isinstance(lock_stages, dict) else None

        if dag.has_dvc_lock and lock_entry is None:
            dag.validation_issues.append(
                f"Stage '{stage_name}' is declared in dvc.yaml but missing from dvc.lock."
            )

        # Parse dependencies
        deps_list: list[str] = []
        raw_deps = stage_def.get("deps", [])
        if isinstance(raw_deps, list):
            for dep in raw_deps:
                if isinstance(dep, str):
                    deps_list.append(_normalize_relpath(dep))
                elif isinstance(dep, dict) and "path" in dep:
                    deps_list.append(_normalize_relpath(str(dep["path"])))

        # Parse outputs (outs, metrics, plots)
        outs_list: list[PipelineArtifact] = []

        # Lock hashes lookup for outputs
        lock_outs_map: dict[str, dict[str, Any]] = {}
        if lock_entry and isinstance(lock_entry.get("outs"), list):
            for lo in lock_entry["outs"]:
                if isinstance(lo, dict) and "path" in lo:
                    lock_outs_map[_normalize_relpath(str(lo["path"]))] = lo

        def _add_out_spec(spec: Any, role: str, is_metric: bool = False) -> None:
            if isinstance(spec, str):
                p = _normalize_relpath(spec)
                lo = lock_outs_map.get(p, {})
                outs_list.append(PipelineArtifact(
                    path=p,
                    role=role,
                    hash_type=lo.get("hash", "md5"),
                    hash_val=lo.get("md5"),
                    size=lo.get("size"),
                    is_metric=is_metric,
                ))
            elif isinstance(spec, dict):
                # Format: - path: {cache: false} or - path: ...
                for k, v in spec.items():
                    p = _normalize_relpath(str(k))
                    lo = lock_outs_map.get(p, {})
                    is_cached = True
                    if isinstance(v, dict) and "cache" in v:
                        is_cached = bool(v["cache"])
                    outs_list.append(PipelineArtifact(
                        path=p,
                        role=role,
                        hash_type=lo.get("hash", "md5"),
                        hash_val=lo.get("md5"),
                        size=lo.get("size"),
                        is_metric=is_metric,
                        is_persisted=is_cached,
                    ))

        for out_spec in stage_def.get("outs", []) or []:
            _add_out_spec(out_spec, role="output")

        for metric_spec in stage_def.get("metrics", []) or []:
            _add_out_spec(metric_spec, role="metric", is_metric=True)

        for plot_spec in stage_def.get("plots", []) or []:
            _add_out_spec(plot_spec, role="plot")

        # Map outputs to detect conflicting stages
        for out_art in outs_list:
            if out_art.path in output_to_stage:
                other = output_to_stage[out_art.path]
                dag.validation_issues.append(
                    f"Conflicting output path '{out_art.path}' declared in both '{other}' and '{stage_name}'."
                )
            else:
                output_to_stage[out_art.path] = stage_name

        # Parse parameters
        resolved_params: dict[str, Any] = {}
        raw_params = stage_def.get("params", [])
        if isinstance(raw_params, list):
            for p_item in raw_params:
                if isinstance(p_item, str):
                    val = _lookup_nested_key(project_params, p_item)
                    resolved_params[p_item] = val
                elif isinstance(p_item, dict):
                    # Format: - params_file: [key1, key2]
                    for p_file, keys in p_item.items():
                        # Try to load custom params file if different
                        p_file_path = context.resolve_path(p_file)
                        custom_dict = project_params
                        if p_file_path.is_file() and p_file_path != context.params_path:
                            try:
                                with p_file_path.open("r", encoding="utf-8") as pf:
                                    custom_dict = yaml.safe_load(pf) or {}
                            except Exception:
                                custom_dict = {}
                        if isinstance(keys, list):
                            for k in keys:
                                full_key = f"{p_file}:{k}" if p_file != "params.yaml" else k
                                resolved_params[full_key] = _lookup_nested_key(custom_dict, k)

        # Also pull lock parameter values if available
        if lock_entry and isinstance(lock_entry.get("params"), dict):
            for pfile, pvals in lock_entry["params"].items():
                if isinstance(pvals, dict):
                    for pk, pv in pvals.items():
                        if pk in resolved_params and resolved_params[pk] is None:
                            resolved_params[pk] = pv

        stage_obj = PipelineStage(
            name=stage_name,
            cmd=cmd,
            deps=deps_list,
            outs=outs_list,
            params=resolved_params,
            always_changed=always_changed,
            lock_entry=lock_entry,
        )
        stages[stage_name] = stage_obj

    # 5. Build Graph Edges & Validate Dependencies
    edges_set: set[tuple[str, str]] = set()

    for downstream_name, stage in stages.items():
        for dep in stage.deps:
            # Check direct match with stage outputs
            if dep in output_to_stage:
                upstream_name = output_to_stage[dep]
                if upstream_name != downstream_name:
                    edges_set.add((upstream_name, downstream_name))
            else:
                # Check if dep is inside a directory output of another stage
                matched_dir = False
                for out_path, upstream_name in output_to_stage.items():
                    if dep.startswith(out_path.rstrip("/") + "/") and upstream_name != downstream_name:
                        edges_set.add((upstream_name, downstream_name))
                        matched_dir = True
                        break

                if not matched_dir:
                    # Check if dependency exists in workspace
                    dep_path = context.resolve_path(dep)
                    if not dep_path.exists():
                        dag.validation_issues.append(
                            f"Unresolved dependency '{dep}' in stage '{downstream_name}': "
                            "file does not exist in workspace and is not produced by any upstream stage."
                        )

    # 6. Populate upstream and downstream lists on stages
    for u, v in sorted(edges_set):
        if u in stages and v in stages:
            if v not in stages[u].downstream:
                stages[u].downstream.append(v)
            if u not in stages[v].upstream:
                stages[v].upstream.append(u)

    dag.stages = stages
    dag.nodes = list(stages.keys())
    dag.edges = sorted(list(edges_set))

    # 7. Cycle Detection via DFS
    visited: dict[str, int] = {node: 0 for node in dag.nodes}  # 0: unvisited, 1: visiting, 2: visited

    def _dfs(node: str, path: list[str]) -> bool:
        visited[node] = 1
        for nbr in stages[node].downstream:
            if visited.get(nbr) == 1:
                # Cycle found
                cycle_slice = path[path.index(nbr):] + [nbr]
                dag.is_cyclic = True
                dag.is_valid = False
                dag.validation_issues.append(
                    f"Cycle detected in pipeline stages: {' -> '.join(cycle_slice)}"
                )
                return True
            elif visited.get(nbr) == 0:
                if _dfs(nbr, path + [nbr]):
                    return True
        visited[node] = 2
        return False

    for n in dag.nodes:
        if visited[n] == 0:
            if _dfs(n, [n]):
                break

    return dag
