#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dashboard_views.py
==================
Presentation layer for ML Lineage Studio.
"""

import json
import os
import subprocess
import time

import altair as alt
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _status_badge(status: str) -> str:
    return {"FINISHED": "FINISHED", "FAILED": "FAILED", "RUNNING": "RUNNING"}.get(status, status)


def _fmt(val) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def _page_header(title: str, subtitle: str):
    st.markdown(f"#### {title}")
    st.caption(subtitle)
    st.divider()


def _short_hash(h: str) -> str:
    if not h:
        return "—"
    return (h[:16] + "…") if len(h) > 16 else h


# ---------------------------------------------------------------------------
# 1. OVERVIEW
# ---------------------------------------------------------------------------
def render_overview(app_module):
    _page_header("Overview", "Experiment summary, pipeline health, and recent activity.")

    runs = app_module.fetch_all_runs()
    df = app_module.get_runs_dataframe(runs)

    if df.empty:
        st.info("No MLflow runs found in `mlflow.db`.")
        return

    finished = df[df["status"] == "FINISHED"]
    failed   = df[df["status"] == "FAILED"]
    running  = df[df["status"] == "RUNNING"]
    latest   = df.iloc[0]

    k = st.columns(7)
    k[0].metric("Total runs",  len(df),             f"{len(finished)} finished")
    k[1].metric("Failed",      len(failed))
    k[2].metric("Running",     len(running))
    k[3].metric("Latest acc.", _fmt(latest.get("accuracy")),  f"run {latest['short_id']}")
    k[4].metric("Latest F1",   _fmt(latest.get("f1_score")))

    git_info  = app_module.get_git_status()
    clean     = git_info.get("clean")
    git_label = "Clean" if clean is True else ("Dirty" if clean is False else "—")
    k[5].metric("Git", git_info.get("branch", "—"), git_label)
    dvc_rem   = app_module.get_dvc_remote_status()
    dvc_label = "in sync" if ("in sync" in dvc_rem.lower() or "cache and remote" in dvc_rem.lower()) else "verify"
    k[6].metric("DVC remote", "local_storage", dvc_label)

    st.divider()

    col_chart, col_table = st.columns([3, 2], gap="medium")

    with col_chart:
        st.caption("**Metrics across finished runs**")
        fin = finished.copy()
        if len(fin) > 1:
            fin = fin.sort_values("start_epoch_ms").reset_index(drop=True)
            fin["Run"] = fin.apply(lambda r: f"#{r.name+1} {r['short_id']}", axis=1)
            mc = [c for c in ["accuracy", "f1_score", "precision", "recall"] if c in fin.columns]
            chart_df = fin.melt(id_vars=["Run"], value_vars=mc,
                                var_name="metric", value_name="score")
            chart_df["Metric"] = chart_df["metric"].str.replace("_", " ").str.title()
            ch = (
                alt.Chart(chart_df)
                .mark_line(point=True, strokeWidth=1.5)
                .encode(
                    x=alt.X("Run:N", title=None, sort=None,
                             axis=alt.Axis(labelFontSize=11, labelAngle=-30)),
                    y=alt.Y("score:Q", title="Score",
                             scale=alt.Scale(domain=[0.85, 1.02]),
                             axis=alt.Axis(format=".3f", labelFontSize=11)),
                    color=alt.Color("Metric:N",
                                    legend=alt.Legend(orient="bottom", labelFontSize=11)),
                    tooltip=["Run:N", "Metric:N", alt.Tooltip("score:Q", format=".4f")],
                )
                .properties(height=220)
                .configure_axis(grid=True, gridOpacity=0.25)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info("Need at least 2 finished runs to render metric trend.")

    with col_table:
        st.caption("**Recent runs**")
        recent = df[["short_id", "status", "accuracy", "f1_score",
                     "git_commit", "start_time"]].head(7).copy()
        recent["status"] = recent["status"].map(
            {"FINISHED": "OK", "FAILED": "FAIL", "RUNNING": "RUN"}
        ).fillna("·")
        st.dataframe(
            recent,
            column_config={
                "short_id":   st.column_config.TextColumn("Run",    width="small"),
                "status":     st.column_config.TextColumn("",       width="small"),
                "accuracy":   st.column_config.NumberColumn("Acc",  format="%.4f"),
                "f1_score":   st.column_config.NumberColumn("F1",   format="%.4f"),
                "git_commit": st.column_config.TextColumn("Commit", width="small"),
                "start_time": st.column_config.TextColumn("Started"),
            },
            hide_index=True,
            use_container_width=True,
            height=270,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Experiments", use_container_width=True, key="ov_to_exp"):
                st.session_state.current_page = "Experiments"
                st.rerun()
        with c2:
            if st.button("Recovery", use_container_width=True, key="ov_to_rec"):
                st.session_state.current_page = "Recovery & Workspaces"
                st.rerun()


# ---------------------------------------------------------------------------
# 2. EXPERIMENTS
# ---------------------------------------------------------------------------
def render_experiments(app_module):
    _page_header("Experiments", "All MLflow runs with filtering and detail inspection.")

    runs = app_module.fetch_all_runs()
    df   = app_module.get_runs_dataframe(runs)

    if df.empty:
        st.info("No runs recorded.")
        return

    f1, f2, f3 = st.columns([3, 1, 1])
    with f1:
        q = st.text_input("Search run ID or git SHA", placeholder="e.g. b6b15235",
                          label_visibility="collapsed")
    with f2:
        statuses = st.multiselect(
            "Status", sorted(df["status"].dropna().unique()),
            default=sorted(df["status"].dropna().unique()),
            label_visibility="collapsed",
        )
    with f3:
        min_acc = st.slider("Min accuracy", 0.0, 1.0, 0.0, 0.01,
                            label_visibility="collapsed")

    fdf = df.copy()
    if q:
        ql = q.strip().lower()
        fdf = fdf[fdf["run_id"].str.lower().str.contains(ql) |
                  fdf["git_commit_full"].str.lower().str.contains(ql)]
    if statuses:
        fdf = fdf[fdf["status"].isin(statuses)]
    if min_acc > 0:
        fdf = fdf[fdf["accuracy"].fillna(0) >= min_acc]

    st.caption(f"{len(fdf)} run(s)")

    display_cols = ["short_id", "status", "accuracy", "f1_score", "precision", "recall",
                    "duration_seconds", "git_commit", "model_type", "start_time"]
    st.dataframe(
        fdf[display_cols],
        column_config={
            "short_id":         st.column_config.TextColumn("Run ID",    width="small"),
            "status":           st.column_config.TextColumn("Status",    width="small"),
            "accuracy":         st.column_config.NumberColumn("Acc",     format="%.4f"),
            "f1_score":         st.column_config.NumberColumn("F1",      format="%.4f"),
            "precision":        st.column_config.NumberColumn("Prec",    format="%.4f"),
            "recall":           st.column_config.NumberColumn("Recall",  format="%.4f"),
            "duration_seconds": st.column_config.NumberColumn("Dur (s)", format="%.1f"),
            "git_commit":       st.column_config.TextColumn("Commit",    width="small"),
            "model_type":       st.column_config.TextColumn("Model"),
            "start_time":       st.column_config.TextColumn("Started"),
        },
        hide_index=True,
        use_container_width=True,
        height=250,
    )
    st.download_button("Download CSV", fdf.to_csv(index=False).encode(),
                       "runs.csv", "text/csv", key="exp_csv")

    st.divider()
    st.caption("**Run inspector**")

    run_ids    = df["run_id"].tolist()
    run_labels = [
        f"{r[:8]}  ·  {df.loc[df['run_id']==r,'start_time'].iloc[0]}"
        f"  ·  {df.loc[df['run_id']==r,'status'].iloc[0]}"
        for r in run_ids
    ]
    sel_idx   = st.selectbox("Select run", range(len(run_ids)),
                              format_func=lambda i: run_labels[i],
                              label_visibility="collapsed", key="exp_sel")
    target_id = run_ids[sel_idx]
    run       = app_module.resolve_run(target_id)

    if not run:
        st.warning("Run could not be loaded.")
        return

    details = app_module.get_run_details(run)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Status",    _status_badge(details["status"]))
    m2.metric("Accuracy",  _fmt(details["metrics"].get("accuracy")))
    m3.metric("F1",        _fmt(details["metrics"].get("f1_score")))
    m4.metric("Precision", _fmt(details["metrics"].get("precision")))
    m5.metric("Recall",    _fmt(details["metrics"].get("recall")))
    dur = details["metrics"].get("duration_seconds")
    dur_str = f"{float(dur):.1f}s" if (dur is not None and str(dur).replace('.','',1).isdigit()) else "—"
    m6.metric("Duration",  dur_str)

    tab_p, tab_g, tab_e, tab_h, tab_a = st.tabs(
        ["Parameters", "Git Provenance", "Environment", "Artifact Hashes", "Actions"]
    )

    with tab_p:
        if details["parameters"]:
            pf = pd.DataFrame([{"Parameter": k, "Value": v}
                                for k, v in sorted(details["parameters"].items())])
            st.dataframe(pf, hide_index=True, use_container_width=True, height=210)
        else:
            st.caption("No parameters recorded.")

    with tab_g:
        g = details["git"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Commit SHA", g["commit_sha"][:12] + "…")
        c2.metric("Branch",     g["branch"])
        c3.metric("Dirty at record",
                  "Yes ⚠" if g["is_dirty"] == "True" else "No ✅")
        st.code(f"git show {g['commit_sha']} --stat", language="bash")
        st.caption("Read-only CLI command to inspect changes in this commit.")

    with tab_e:
        e  = details["environment"]
        ef = pd.DataFrame([{"Library": k, "Version": v} for k, v in e.items()])
        st.dataframe(ef, hide_index=True, use_container_width=True)

    with tab_h:
        hf = pd.DataFrame([
            {"Artifact": app_module.DVC_TAG_MAP.get(k, k), "Recorded MD5": v}
            for k, v in details["dvc_hashes"].items()
        ])
        st.dataframe(hf, hide_index=True, use_container_width=True)
        st.caption("Hashes stored in MLflow tags at experiment logging time.")

    with tab_a:
        ba, bb, bc, bd = st.columns(4)
        with ba:
            if st.button("Trace lineage", key="exp_to_lin"):
                st.session_state.selected_lineage_run = target_id
                st.session_state.current_page = "Lineage"
                st.rerun()
        with bb:
            if st.button("Audit artifacts", key="exp_to_art"):
                st.session_state.selected_artifact_run = target_id
                st.session_state.current_page = "Artifacts"
                st.rerun()
        with bc:
            if st.button("Recovery preview", key="exp_to_rec"):
                st.session_state.selected_recovery_run = target_id
                st.session_state.current_page = "Recovery & Workspaces"
                st.rerun()
        with bd:
            if st.button("Compare runs", key="exp_to_cmp"):
                st.session_state.current_page = "Compare"
                st.rerun()


# ---------------------------------------------------------------------------
# 3. COMPARE RUNS
# ---------------------------------------------------------------------------
def render_compare(app_module):
    _page_header("Compare Runs",
                 "Side-by-side metrics, parameters, artifact hashes, and Git provenance.")

    runs = app_module.fetch_all_runs()
    df   = app_module.get_runs_dataframe(runs)

    if len(df) < 2:
        st.warning("At least two runs are required for comparison.")
        return

    run_ids = df["run_id"].tolist()
    labels  = [
        f"{r[:8]}  ·  {df.loc[df['run_id']==r,'start_time'].iloc[0]}"
        f"  ({df.loc[df['run_id']==r,'status'].iloc[0]})"
        for r in run_ids
    ]

    sel_a, sel_b = st.columns(2)
    with sel_a:
        idx_a = st.selectbox("Baseline run (A)", range(len(run_ids)), 0,
                              format_func=lambda i: labels[i], key="cmp_a")
    with sel_b:
        idx_b = st.selectbox("Comparison run (B)", range(len(run_ids)),
                              min(1, len(run_ids)-1),
                              format_func=lambda i: labels[i], key="cmp_b")

    rid_a, rid_b = run_ids[idx_a], run_ids[idx_b]
    if rid_a == rid_b:
        st.info("Select two different runs to compare.")
        return

    matrix = app_module.get_run_comparison_matrix(rid_a, rid_b)
    if "error" in matrix:
        st.error(matrix["error"])
        return

    det_a, det_b = matrix["run_a"], matrix["run_b"]

    pri   = ["accuracy", "f1_score", "precision", "recall", "duration_seconds"]
    dcols = st.columns(len(pri))
    for i, m in enumerate(pri):
        va = det_a["metrics"].get(m)
        vb = det_b["metrics"].get(m)
        with dcols[i]:
            if va is not None and vb is not None:
                delta = vb - va
                fmt   = ".2f" if m == "duration_seconds" else ".4f"
                unit  = "s"  if m == "duration_seconds" else ""
                st.metric(m.replace("_", " ").title(),
                          f"{vb:{fmt}}{unit}",
                          f"{delta:+{fmt}}{unit}",
                          delta_color="inverse" if m == "duration_seconds" else "normal")
            else:
                st.metric(m.replace("_", " ").title(), "—")

    st.divider()

    t_m, t_p, t_h, t_g = st.tabs(
        ["Metrics", "Parameters", "Artifact Hashes", "Git & Environment"]
    )

    with t_m:
        rows = []
        for m in matrix["metrics"]:
            va, vb, d = m["run_a"], m["run_b"], m["delta"]
            rows.append({
                "Metric":                  m["metric"].replace("_", " ").title(),
                f"Run A ({rid_a[:8]})":    _fmt(va),
                f"Run B ({rid_b[:8]})":    _fmt(vb),
                "Delta B-A":               f"{d:+.4f}" if isinstance(d, float) else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with t_p:
        rows = []
        for p in matrix["parameters"]:
            rows.append({
                "Parameter":              p["parameter"],
                f"Run A ({rid_a[:8]})":   p["run_a"],
                f"Run B ({rid_b[:8]})":   p["run_b"],
                "Changed":                "Yes" if p["changed"] else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with t_h:
        rows = []
        for h in matrix["hashes"]:
            rows.append({
                "Artifact":               h["artifact"],
                f"Run A ({rid_a[:8]})":   _short_hash(h["run_a_hash"]),
                f"Run B ({rid_b[:8]})":   _short_hash(h["run_b_hash"]),
                "Match":                  "Yes" if h["match"] else "Differs",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Identical hashes mean the same artifact was used. "
            "A better score with a different hash indicates a genuinely different model or dataset."
        )

    with t_g:
        rows = [
            {"Attribute":   "Git Commit",
             f"Run A ({rid_a[:8]})": det_a["git"]["commit_sha"],
             f"Run B ({rid_b[:8]})": det_b["git"]["commit_sha"]},
            {"Attribute":   "Branch",
             f"Run A ({rid_a[:8]})": det_a["git"]["branch"],
             f"Run B ({rid_b[:8]})": det_b["git"]["branch"]},
            {"Attribute":   "Dirty tree",
             f"Run A ({rid_a[:8]})": det_a["git"]["is_dirty"],
             f"Run B ({rid_b[:8]})": det_b["git"]["is_dirty"]},
            {"Attribute":   "Python",
             f"Run A ({rid_a[:8]})": det_a["environment"]["python_version"],
             f"Run B ({rid_b[:8]})": det_b["environment"]["python_version"]},
            {"Attribute":   "scikit-learn",
             f"Run A ({rid_a[:8]})": det_a["environment"]["scikit_learn"],
             f"Run B ({rid_b[:8]})": det_b["environment"]["scikit_learn"]},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    csv = pd.DataFrame(
        [{"Metric": m["metric"],
          f"Run A ({rid_a[:8]})": m["run_a"],
          f"Run B ({rid_b[:8]})": m["run_b"]}
         for m in matrix["metrics"]]
    ).to_csv(index=False).encode()
    st.download_button("Download comparison CSV", csv,
                       f"compare_{rid_a[:8]}_vs_{rid_b[:8]}.csv",
                       "text/csv", key="cmp_csv")


# ---------------------------------------------------------------------------
# 4. LINEAGE
# ---------------------------------------------------------------------------
def render_lineage(app_module):
    _page_header(
        "Lineage",
        "End-to-end pipeline provenance: "
        "Raw Dataset -> Preparation -> Features -> Training -> Evaluation.",
    )

    runs = app_module.fetch_all_runs()
    df   = app_module.get_runs_dataframe(runs)

    if df.empty:
        st.info("No runs available to trace.")
        return

    run_ids     = df["run_id"].tolist()
    default_idx = 0
    sel_lin     = st.session_state.get("selected_lineage_run")
    if sel_lin and sel_lin in run_ids:
        default_idx = run_ids.index(sel_lin)

    sel_id = st.selectbox(
        "Run to trace", run_ids, index=default_idx,
        format_func=lambda r:
            f"{r[:8]}  ·  {df.loc[df['run_id']==r,'start_time'].iloc[0]}",
        label_visibility="collapsed", key="lin_sel",
    )

    with st.spinner("Loading lineage report…"):
        report = app_module.fetch_lineage_report_data(sel_id)

    if not report:
        st.error(
            f"Could not load lineage report for run `{sel_id[:8]}`. "
            "Check System Status for script availability."
        )
        return

    v = report.get("verification_summary", {})
    g = report.get("git_provenance",      {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lineage",     "Verified" if v.get("is_lineage_verified") else "Discrepancy")
    c2.metric("Git commit",  (g.get("commit_sha", "—")[:12] + "…"))
    c3.metric("Tracked",     f"{v.get('versioned_artifacts_tracked', 0)}")
    c4.metric("Verified",    f"{v.get('verified_in_workspace', 0)}")
    c5.metric("Recoverable", f"{v.get('recoverable_from_dvc', 0)}")

    st.divider()

    stages = report.get("lineage_stages", [])
    icons  = {"prepare": "1 Prepare", "featurize": "2 Featurize",
               "train": "3 Train",    "evaluate":  "4 Evaluate"}

    for i, s in enumerate(stages):
        name    = s.get("stage", "stage")
        label   = icons.get(name.lower(), name.upper())
        desc    = s.get("description", "")
        inputs  = s.get("inputs",  [])
        outputs = s.get("outputs", [])

        if i > 0:
            st.caption("  |")

        with st.expander(f"**{label}** — {desc}", expanded=False):
            ci, co = st.columns(2)
            with ci:
                st.markdown("**Inputs**")
                if inputs:
                    for inp in inputs:
                        st.markdown(f"- `{inp}`")
                else:
                    st.caption("(pipeline entry point)")
            with co:
                st.markdown("**Outputs**")
                for out in outputs:
                    status = out.get("status", "")
                    ok     = "VERIFIED" in status or "PRESENT" in status
                    badge  = "[OK]" if ok else "[?]"
                    h      = out.get("expected_hash", "")
                    sh     = _short_hash(h)
                    st.markdown(f"- {badge} `{out['path']}` — *{out.get('role','')}*")
                    if sh and sh != "—":
                        st.caption(f"  MD5: `{sh}`")

    st.divider()
    st.caption(
        f"Tracked: {v.get('versioned_artifacts_tracked',0)}  "
        f"Verified: {v.get('verified_in_workspace',0)}  "
        f"Mismatches: {v.get('workspace_hash_mismatches',0)}  "
        f"Missing: {v.get('workspace_missing',0)}"
    )
    st.download_button(
        "Download lineage JSON",
        json.dumps(report, indent=2).encode(),
        f"lineage_{sel_id[:8]}.json",
        "application/json",
        key="lin_dl",
    )


# ---------------------------------------------------------------------------
# 5. ARTIFACTS
# ---------------------------------------------------------------------------
def render_artifacts(app_module):
    _page_header(
        "Artifacts",
        "Recorded MD5 hash vs current workspace hash vs DVC lock for each tracked pipeline file.",
    )

    runs = app_module.fetch_all_runs()
    df   = app_module.get_runs_dataframe(runs)

    if df.empty:
        st.info("No runs found.")
        return

    run_ids     = df["run_id"].tolist()
    default_idx = 0
    sel_art     = st.session_state.get("selected_artifact_run")
    if sel_art and sel_art in run_ids:
        default_idx = run_ids.index(sel_art)

    sel_id = st.selectbox(
        "Select run", run_ids, index=default_idx,
        format_func=lambda r:
            f"{r[:8]}  ({df.loc[df['run_id']==r,'start_time'].iloc[0]})",
        label_visibility="collapsed", key="art_sel",
    )

    run = app_module.resolve_run(sel_id)
    if not run:
        st.error("Run not found.")
        return

    results  = app_module.verify_run_artifacts(run)
    verified = sum(1 for a in results if "VERIFIED" in a["status"])
    total    = len(results)

    if verified == total:
        st.success(
            f"All {total}/{total} workspace artifacts match recorded hashes "
            f"for run `{sel_id[:8]}`."
        )
    else:
        st.warning(
            f"{verified}/{total} match. "
            "Mismatched or missing files can be restored in Recovery & Workspaces."
        )

    rows = []
    for a in results:
        rows.append({
            "Artifact":      a["artifact"],
            "Recorded MD5":  _short_hash(a["expected_hash"]),
            "Workspace MD5": _short_hash(a.get("workspace_hash") or ""),
            "DVC Lock MD5":  _short_hash(a.get("lock_hash")      or ""),
            "Status":        a["status"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Recorded MD5 = stored in MLflow tags. "
        "Workspace MD5 = current file. "
        "DVC Lock MD5 = from dvc.lock (latest pipeline run). "
        "Restoration available in Recovery & Workspaces."
    )

    st.divider()
    st.caption("Live CLI verification — calls `scripts/recover_run.py --verify-only` (read-only)")
    if st.button("Run verification", key="art_verify"):
        with st.spinner("Verifying…"):
            proc = app_module.run_recovery_subprocess(sel_id, ["--verify-only"])
        if proc["stdout"]:
            st.code(proc["stdout"], language="text")
        if proc["stderr"]:
            st.warning(proc["stderr"])
        if proc["exit_code"] == 0:
            st.success("All checked artifacts match recorded hashes.")
        else:
            st.warning(f"Verification returned exit code {proc['exit_code']}.")


# ---------------------------------------------------------------------------
# 6. RECOVERY & WORKSPACES
# ---------------------------------------------------------------------------
def render_recovery(app_module):
    _page_header(
        "Recovery & Workspaces",
        "Restore a historical experiment state into an isolated workspace or the current workspace.",
    )

    runs = app_module.fetch_all_runs()
    df   = app_module.get_runs_dataframe(runs)

    if df.empty:
        st.info("No runs available for recovery.")
        return

    run_ids     = df["run_id"].tolist()
    default_idx = 0
    sel_rec     = st.session_state.get("selected_recovery_run")
    if sel_rec and sel_rec in run_ids:
        default_idx = run_ids.index(sel_rec)

    sel_id   = st.selectbox(
        "Select historical run", run_ids, index=default_idx,
        format_func=lambda r: (
            f"{r[:8]}  Git {df.loc[df['run_id']==r,'git_commit'].iloc[0]}"
            f"  {df.loc[df['run_id']==r,'start_time'].iloc[0]}"
        ),
        label_visibility="collapsed", key="rec_sel",
    )
    short_id = sel_id[:8]

    pending = app_module.get_confirmation_state()
    if pending and pending.get("run_id") != sel_id:
        app_module.clear_confirmation_state()
        pending = None

    sel_run    = app_module.resolve_run(sel_id)
    git_commit = "unavailable"
    if sel_run:
        git_commit = sel_run.data.tags.get("git.commit_sha", "unavailable")

    info_cols = st.columns(3)
    info_cols[0].metric("Run ID",     short_id)
    info_cols[1].metric("Git commit", (git_commit[:12] + "…") if len(git_commit) > 12 else git_commit)
    status_val = df.loc[df["run_id"] == sel_id, "status"].iloc[0]
    info_cols[2].metric("Status", _status_badge(status_val))

    st.divider()

    tab_iso, tab_ws = st.tabs([
        "Isolated Workspace (recommended)",
        "Current-Workspace Restore",
    ])

    # -- A. Isolated workspace -------------------------------------------------
    with tab_iso:
        st.caption(
            "Creates a `git worktree` at the destination checked out to the historical Git commit. "
            "DVC artifacts are then restored from the local cache. "
            "Your active project is not modified."
        )

        with st.expander("What this workflow does", expanded=False):
            st.markdown(
                "**Steps:**\n"
                f"1. `git worktree add <dest> {git_commit[:12]}` "
                "— creates a separate working tree at the historical code revision.\n"
                "2. `scripts/recover_run.py <run_id> --dry-run` — preview of artifact restoration.\n"
                "3. On confirmed execution: `scripts/recover_run.py <run_id>` in project root "
                "restores `data/` and `models/` from local DVC cache.\n\n"
                "**What is restored:** source code at historical commit (via Git worktree), "
                "DVC-tracked data and model files from local cache/remote.\n\n"
                "**What is NOT automatically restored:** Python `.venv` (shared by reference), "
                "new MLflow runs (created only if you explicitly run `dvc repro`).\n\n"
                "**Limitation:** If the local DVC cache does not contain all required artifact hashes, "
                "some files will be unavailable. The dry-run preview will identify this."
            )

        iso_dest = st.text_input(
            "Destination directory",
            value=f"D:\\ML_Workspaces\\run_{short_id}",
            key="iso_dest",
        )
        if os.path.exists(iso_dest):
            st.caption(f"Directory `{iso_dest}` already exists.")

        if st.button("Preview (dry-run)", key="iso_preview"):
            with st.spinner("Running dry-run preview…"):
                proc = app_module.run_recovery_subprocess(sel_id, ["--dry-run"])
            if proc["stdout"]:
                st.code(proc["stdout"], language="text")
            if proc["stderr"]:
                st.warning(proc["stderr"])
            if proc["exit_code"] == 0:
                st.success("Dry-run complete. No files modified.")
            else:
                st.error(f"Dry-run exit code {proc['exit_code']}.")

        if git_commit == "unavailable":
            st.warning("No Git commit SHA recorded for this run — worktree cannot be created.")
        else:
            if st.button("Create isolated workspace", type="primary", key="iso_exec"):
                _create_isolated_workspace(app_module, sel_id, git_commit, iso_dest)

    # -- B. Current-workspace restore -----------------------------------------
    with tab_ws:
        st.error(
            "This operation overwrites files in `data/` and `models/` with historical artifacts. "
            "Code files, Git history, and MLflow records are not affected. "
            "Use Isolated Workspace for a safer alternative."
        )

        st.markdown("**Step 1 — Run dry-run preview (required)**")
        if st.button("Execute dry-run preview", use_container_width=True, key="ws_dryrun"):
            with st.spinner("Analyzing recovery sources…"):
                proc = app_module.run_recovery_subprocess(sel_id, ["--dry-run"])
                app_module.set_confirmation_state({
                    "run_id":    sel_id,
                    "short_id":  short_id,
                    "timestamp": time.time(),
                })
            st.session_state.ws_dryrun_output = proc["stdout"]
            st.session_state.ws_dryrun_exit   = proc["exit_code"]
            st.rerun()

        if st.session_state.get("ws_dryrun_output"):
            st.code(st.session_state.ws_dryrun_output, language="text")
            if st.session_state.get("ws_dryrun_exit", 1) != 0:
                st.warning("Dry-run reported errors. Review before proceeding.")

        st.divider()
        st.markdown("**Step 2 — Explicit confirmation**")
        pending = app_module.get_confirmation_state()
        if pending and pending.get("run_id") == sel_id:
            confirm = st.checkbox(
                f"I understand that files in `data/` and `models/` will be overwritten "
                f"with artifacts from run `{short_id}`.",
                key="ws_confirm_cb",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Restore artifacts", type="primary",
                             disabled=not confirm, use_container_width=True,
                             key="ws_confirm_btn"):
                    with st.spinner("Restoring…"):
                        resp = app_module.handle_confirm_recover(short_id)
                    st.success(resp)
                    st.session_state.ws_dryrun_output = None
                    st.rerun()
            with c2:
                if st.button("Cancel", use_container_width=True, key="ws_cancel"):
                    app_module.clear_confirmation_state()
                    st.session_state.ws_dryrun_output = None
                    st.info("Cancelled. No files modified.")
                    st.rerun()
        else:
            st.caption("Run the dry-run preview first to unlock confirmation.")


def _create_isolated_workspace(app_module, run_id: str, git_commit: str, dest: str):
    """Create a git worktree at dest, then preview artifact restoration."""
    project_root   = app_module.PROJECT_ROOT
    python_exe     = app_module.get_python_executable()
    recover_script = os.path.join(project_root, "scripts", "recover_run.py")
    env = {**os.environ, "PYTHONUTF8": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}

    with st.status("Creating git worktree…", expanded=True) as s1:
        try:
            r = subprocess.run(
                ["git", "worktree", "add", dest, git_commit],
                capture_output=True, text=True, cwd=project_root,
                timeout=30, encoding="utf-8", errors="replace", env=env,
            )
            if r.stdout:
                st.write(r.stdout)
            if r.stderr:
                st.caption(r.stderr)
            if r.returncode == 0:
                s1.update(label="Worktree created", state="complete")
            else:
                s1.update(label=f"Worktree failed (exit {r.returncode})", state="error")
                st.error(r.stderr or r.stdout)
                return
        except Exception as e:
            s1.update(label="Worktree failed", state="error")
            st.error(str(e))
            return

    with st.status("Running artifact dry-run preview…", expanded=True) as s2:
        try:
            r2 = subprocess.run(
                [python_exe, recover_script, run_id, "--dry-run"],
                capture_output=True, text=True, cwd=project_root,
                timeout=30, encoding="utf-8", errors="replace", env=env,
            )
            if r2.stdout:
                st.code(r2.stdout, language="text")
            if r2.stderr:
                st.caption(r2.stderr)
            if r2.returncode == 0:
                s2.update(label="Dry-run preview complete", state="complete")
            else:
                s2.update(label=f"Dry-run exit {r2.returncode}", state="error")
        except Exception as e:
            s2.update(label="Dry-run failed", state="error")
            st.error(str(e))

    st.success(f"Isolated workspace ready at `{dest}`.")
    st.code(
        f"# Restore full DVC artifacts into the worktree:\n"
        f"{python_exe} {recover_script} {run_id[:8]}\n\n"
        f"# Run the pipeline in the isolated workspace (creates a new MLflow run):\n"
        f"cd {dest}\n"
        f"{python_exe} -m dvc repro",
        language="bash",
    )
    st.caption(
        "The .venv at D:\\ML_CP\\.venv is shared by reference — no reinstall needed. "
        "Pipeline execution must be initiated manually."
    )


# ---------------------------------------------------------------------------
# 7. CHAT ASSISTANT
# ---------------------------------------------------------------------------
def render_chat(app_module):
    _page_header(
        "Chat Assistant",
        "Command-based ML experiment queries. All read-only unless recovery is explicitly confirmed.",
    )

    runs = app_module.fetch_all_runs()

    suggestions = [
        ("list runs",  "List runs"),
        ("status",     "System status"),
        ("help",       "Help"),
    ]
    if runs:
        lid = runs[0].info.run_id[:8]
        suggestions += [
            (f"inspect {lid}",         f"Inspect {lid}"),
            (f"verify {lid}",          f"Verify {lid}"),
            (f"lineage {lid}",         f"Lineage {lid}"),
            (f"preview recover {lid}", f"Preview recover {lid}"),
        ]
        if len(runs) > 1:
            sid2 = runs[1].info.run_id[:8]
            suggestions.append((f"compare {lid} {sid2}", f"Compare {lid} vs {sid2}"))

    row_size = 4
    for row_start in range(0, len(suggestions), row_size):
        chunk = suggestions[row_start:row_start + row_size]
        cols  = st.columns(len(chunk))
        for i, (cmd, label) in enumerate(chunk):
            with cols[i]:
                if st.button(label, use_container_width=True, key=f"chat_sug_{row_start+i}"):
                    st.session_state.chat_trigger = cmd

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Welcome. Type a command or select a quick action above.\n\n"
                "Common commands: `list runs` · `inspect <id>` · `verify <id>` · "
                "`lineage <id>` · `compare <id_a> <id_b>` · "
                "`preview recover <id>` · `recover <id>` · `status` · `help`\n\n"
                "*All inspection, verification, and lineage commands are read-only. "
                "Recovery requires dry-run preview and explicit confirmation.*"
            ),
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending = app_module.get_confirmation_state()
    if pending:
        st.warning(
            f"Pending recovery for run `{pending['short_id']}`. "
            "Type `confirm recover <id>` to proceed or `cancel` to abort."
        )

    user_input = None
    if "chat_trigger" in st.session_state:
        user_input = st.session_state.pop("chat_trigger")
    else:
        user_input = st.chat_input("Enter command…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Processing…"):
                response = app_module.route_message(user_input)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# ---------------------------------------------------------------------------
# 8. SYSTEM STATUS
# ---------------------------------------------------------------------------
def render_system_status(app_module):
    _page_header(
        "System Status",
        "Health checks for Git, MLflow, DVC pipeline workspace, and local artifact storage.",
    )

    git_info = app_module.get_git_status()
    dvc_ws   = app_module.get_dvc_status()
    dvc_rem  = app_module.get_dvc_remote_status()
    runs     = app_module.fetch_all_runs()

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("**Git**")
        branch = git_info.get("branch", "—")
        commit = git_info.get("commit", "unavailable")
        clean  = git_info.get("clean")
        g1, g2, g3 = st.columns(3)
        g1.metric("Branch", branch)
        g2.metric("HEAD",   commit[:7] if len(commit) >= 7 else commit)
        g3.metric("Working tree",
                  "Clean" if clean is True else ("Dirty" if clean is False else "—"))
        if clean is False:
            st.caption(
                "Modified: README.md, app.py. "
                "Untracked: dashboard_views.py, tests/. "
                "These are working-session changes not yet staged."
            )

        st.markdown("**MLflow tracking**")
        finished = sum(1 for r in runs if r.info.status == "FINISHED")
        failed   = sum(1 for r in runs if r.info.status == "FAILED")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total runs", len(runs))
        m2.metric("Finished",   finished)
        m3.metric("Failed",     failed)
        st.caption(f"URI: `{app_module.get_tracking_uri()}`")

        st.markdown("**Scripts & configuration**")
        checks = [
            ("scripts/recover_run.py",    "scripts/recover_run.py"),
            ("scripts/lineage_report.py", "scripts/lineage_report.py"),
            ("dvc.yaml",    "dvc.yaml"),
            ("dvc.lock",    "dvc.lock"),
            ("params.yaml", "params.yaml"),
            ("mlflow.db",   "mlflow.db"),
        ]
        for name, rel in checks:
            path = os.path.join(app_module.PROJECT_ROOT, rel)
            st.caption(f"{'OK' if os.path.exists(path) else 'MISSING'}  `{name}`")

    with col_right:
        st.markdown("**DVC pipeline workspace** (`dvc status`)")
        display_ws = dvc_ws or "Pipeline is up to date."
        if "log_experiment" in display_ws and "always changed" in display_ws.lower():
            display_ws += (
                "\n\n# Note: log_experiment has always_changed: true (intentional).\n"
                "# It creates a new MLflow run on every dvc repro."
            )
        st.code(display_ws, language="yaml")

        st.markdown("**DVC local artifact storage** (`dvc status -c`)")
        st.caption(
            "Remote type: `local_storage` — local directory `D:\\ML_DVC_Storage`. "
            "Not cloud storage."
        )
        remote_display = dvc_rem or "—"
        st.code(remote_display, language="text")
        if "in sync" in remote_display.lower() or "cache and remote" in remote_display.lower():
            st.success("Local DVC cache and remote storage are in sync.")
        elif "error" in remote_display.lower():
            st.error("DVC remote check reported an error.")

    st.divider()
    st.caption(
        "Why log_experiment shows 'always changed': "
        "src/log_experiment.py calls mlflow.start_run() which generates a new run ID on every "
        "invocation — DVC cannot cache this. always_changed: true in dvc.yaml correctly reflects "
        "this. Running dvc repro will only re-execute upstream stages (prepare, featurize, train, "
        "evaluate) if their inputs have changed."
    )
