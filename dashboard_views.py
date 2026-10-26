#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dashboard_views.py
==================
Presentation layer for ML Lineage Studio.

Implements a dark, professional engineering dashboard with consistent
design system, modern navigation, and polished layouts for all 8 pages.
"""

import json
import os
import subprocess
import time

import altair as alt
import pandas as pd
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def inject_design_system():
    """Inject the complete CSS design system. Called once from main()."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global Reset ──────────────────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0f1117;
        color: #c9d1d9;
    }

    /* ── Main Content Area ─────────────────────────────────────────────── */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }

    /* ── Hide Streamlit Deploy & Branding ───────────────────────────── */
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
    button[kind="header"] { display: none; }

    /* ── Sidebar ───────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
        font-size: 0.82rem;
        color: #8b949e;
    }

    /* Hide radio circles for nav items */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.4rem 0.75rem !important;
        border-radius: 6px;
        margin-bottom: 1px;
        cursor: pointer;
        transition: background-color 0.15s ease;
        font-size: 0.84rem !important;
        color: #8b949e !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: rgba(56, 139, 253, 0.08);
        color: #c9d1d9 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: rgba(56, 139, 253, 0.15) !important;
        color: #58a6ff !important;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label span[data-testid="stMarkdownContainer"] p {
        color: inherit !important;
    }
    /* Hide the radio dot */
    section[data-testid="stSidebar"] div[role="radiogroup"] div[data-testid="stRadio"] > div > div > div {
        display: none !important;
    }

    /* ── Metric Cards ──────────────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 600;
        color: #e6edf3 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
    }

    /* ── Buttons ────────────────────────────────────────────────────────── */
    .stButton > button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 0.4rem 1rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #8b949e;
        color: #e6edf3;
    }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #1f6feb !important;
        border-color: #1f6feb !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #388bfd !important;
        border-color: #388bfd !important;
    }

    /* ── Data Tables ───────────────────────────────────────────────────── */
    .stDataFrame {
        font-size: 0.8rem;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Tabs ───────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-size: 0.82rem;
        color: #8b949e;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
        background: transparent !important;
    }

    /* ── Expanders ──────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        font-weight: 500;
        color: #c9d1d9;
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
    }

    /* ── Code / Pre ────────────────────────────────────────────────────── */
    pre, code {
        font-size: 0.78rem !important;
        background-color: #0d1117 !important;
        border: 1px solid #21262d;
        border-radius: 6px;
    }

    /* ── Chat Messages ─────────────────────────────────────────────────── */
    div[data-testid="stChatMessage"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stChatMessage"][data-testid-role="user"] {
        background-color: #1c2333;
    }

    /* ── Chat Input & Sticky Bottom Bar ───────────────────────────────── */
    /* Bottom sticky container wrapper & inner block container */
    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"],
    .stChatFloatingInputContainer {
        background-color: #0f1117 !important;
        border-top: 1px solid #21262d !important;
        box-shadow: none !important;
    }

    /* Chat input outer and inner container (removes red border) */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInputContainer"],
    div[data-testid="stChatInput"] > div {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stChatInput"]:focus-within,
    div[data-testid="stChatInputContainer"]:focus-within,
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 1px #58a6ff !important;
        outline: none !important;
    }

    /* Chat input textarea */
    div[data-testid="stChatInputContainer"] textarea,
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #f0f6fc !important;
        -webkit-text-fill-color: #f0f6fc !important;
        caret-color: #58a6ff !important;
        font-family: inherit !important;
        font-size: 0.88rem !important;
        line-height: 1.4 !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInputContainer"] textarea::placeholder,
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #8b949e !important;
        -webkit-text-fill-color: #8b949e !important;
        opacity: 1 !important;
    }

    /* Chat input submit button */
    div[data-testid="stChatInputContainer"] button,
    div[data-testid="stChatInput"] button {
        background-color: #1f6feb !important;
        border: 1px solid #1f6feb !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }

    div[data-testid="stChatInputContainer"] button:hover:not(:disabled),
    div[data-testid="stChatInput"] button:hover:not(:disabled) {
        background-color: #388bfd !important;
        border-color: #388bfd !important;
    }

    div[data-testid="stChatInputContainer"] button:disabled,
    div[data-testid="stChatInput"] button:disabled {
        background-color: #21262d !important;
        border-color: #30363d !important;
        color: #484f58 !important;
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }

    div[data-testid="stChatInputContainer"] button svg,
    div[data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    div[data-testid="stChatInputContainer"] button:disabled svg,
    div[data-testid="stChatInput"] button:disabled svg {
        fill: #484f58 !important;
        stroke: #484f58 !important;
    }

    /* ── Selectbox / Input ─────────────────────────────────────────────── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stMultiSelect > div > div {
        background-color: #0d1117;
        border-color: #30363d;
        color: #c9d1d9;
        font-size: 0.84rem;
    }

    /* ── Dividers ──────────────────────────────────────────────────────── */
    hr {
        border-color: #21262d !important;
        margin: 1rem 0 !important;
    }

    /* ── Checkbox ──────────────────────────────────────────────────────── */
    .stCheckbox label span {
        font-size: 0.82rem;
    }

    /* ── Slider ────────────────────────────────────────────────────────── */
    .stSlider label {
        font-size: 0.82rem;
    }

    /* ── Download Button ───────────────────────────────────────────────── */
    .stDownloadButton > button {
        background-color: #21262d;
        border: 1px solid #30363d;
        color: #8b949e;
        font-size: 0.78rem;
    }

    /* ── Alerts ─────────────────────────────────────────────────────────── */
    div[data-testid="stAlert"] {
        border-radius: 6px;
        font-size: 0.82rem;
    }

    /* ── Custom Components ─────────────────────────────────────────────── */
    .mls-page-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 0.15rem;
        line-height: 1.3;
    }
    .mls-page-desc {
        font-size: 0.82rem;
        color: #8b949e;
        margin-bottom: 1rem;
    }
    .mls-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .mls-section-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: #484f58;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
        padding-left: 0.75rem;
    }
    .mls-status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .mls-status-healthy { background-color: #3fb950; }
    .mls-status-warning { background-color: #d29922; }
    .mls-status-error { background-color: #f85149; }
    .mls-status-unknown { background-color: #484f58; }

    .mls-nav-icon {
        display: inline-block;
        width: 16px;
        text-align: center;
        margin-right: 8px;
        font-size: 0.8rem;
    }
    .mls-sidebar-footer {
        font-size: 0.7rem;
        color: #484f58;
        padding: 0.5rem 0.75rem;
        border-top: 1px solid #21262d;
    }
    .mls-sidebar-footer-item {
        display: flex;
        align-items: center;
        padding: 2px 0;
        gap: 6px;
    }
    .mls-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem 0.5rem 0.75rem 0.5rem;
    }
    .mls-brand-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #1f6feb, #a371f7);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        color: white;
        flex-shrink: 0;
    }
    .mls-brand-text {
        line-height: 1.2;
    }
    .mls-brand-name {
        font-size: 0.92rem;
        font-weight: 600;
        color: #e6edf3;
    }
    .mls-brand-sub {
        font-size: 0.68rem;
        color: #484f58;
    }
    .mls-chat-welcome {
        text-align: center;
        padding: 2rem 1rem;
    }
    .mls-chat-welcome h3 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 0.4rem;
    }
    .mls-chat-welcome p {
        font-size: 0.82rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    .mls-lineage-stage {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .mls-lineage-connector {
        text-align: center;
        color: #30363d;
        font-size: 1.2rem;
        margin: 0.15rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _page_header(title: str, desc: str):
    """Render a consistent page header with title and description."""
    st.markdown(f'<div class="mls-page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mls-page-desc">{desc}</div>', unsafe_allow_html=True)


def _fmt(val) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def _short_hash(h: str) -> str:
    if not h:
        return "—"
    return (h[:12] + "…") if len(h) > 12 else h


def _status_icon(status: str) -> str:
    return {"FINISHED": "✓", "FAILED": "✗", "RUNNING": "◉"}.get(status, "?")


def _status_color(status: str) -> str:
    return {"FINISHED": "#3fb950", "FAILED": "#f85149", "RUNNING": "#d29922"}.get(status, "#484f58")


def _health_dot(state: str) -> str:
    css = {"healthy": "mls-status-healthy", "warning": "mls-status-warning",
           "error": "mls-status-error"}.get(state, "mls-status-unknown")
    return f'<span class="mls-status-dot {css}"></span>'


def _navigate_to(page: str):
    """Helper to set navigation and rerun."""
    st.session_state.current_page = page
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar(app_module):
    """Render the professional sidebar navigation."""
    with st.sidebar:
        # Brand area
        st.markdown("""
        <div class="mls-brand">
            <div class="mls-brand-icon">◇</div>
            <div class="mls-brand-text">
                <div class="mls-brand-name">ML Lineage Studio</div>
                <div class="mls-brand-sub">Experiment Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="mls-section-label">Workspace</div>', unsafe_allow_html=True)

        pages_workspace = [
            "◈  Overview",
            "◉  Experiments",
            "⇆  Compare Runs",
            "⊶  Lineage Explorer",
            "▦  Artifacts",
        ]
        pages_ops = [
            "↻  Recovery Center",
            "▣  Chat Assistant",
            "♡  System Health",
        ]

        # Map display names to internal page keys
        page_map = {
            "◈  Overview": "Overview",
            "◉  Experiments": "Experiments",
            "⇆  Compare Runs": "Compare Runs",
            "⊶  Lineage Explorer": "Lineage Explorer",
            "▦  Artifacts": "Artifacts",
            "↻  Recovery Center": "Recovery Center",
            "▣  Chat Assistant": "Chat Assistant",
            "♡  System Health": "System Health",
        }
        reverse_map = {v: k for k, v in page_map.items()}

        all_pages = pages_workspace + pages_ops
        current = st.session_state.get("current_page", "Overview")
        current_display = reverse_map.get(current, "◈  Overview")
        current_idx = all_pages.index(current_display) if current_display in all_pages else 0

        selected_display = st.radio(
            "Navigation",
            all_pages,
            index=current_idx,
            label_visibility="collapsed",
            key="main_nav_radio",
        )

        selected_page = page_map.get(selected_display, "Overview")
        if selected_page != st.session_state.get("current_page"):
            st.session_state.current_page = selected_page
            st.rerun()

        # Insert operations label after workspace items
        # (We use a workaround since st.radio doesn't support groups)
        # The section label is placed before the radio
        st.markdown('<div class="mls-section-label" style="margin-top:0.75rem;">Operations</div>',
                    unsafe_allow_html=True)

        st.divider()

        # Footer status
        try:
            git_info = app_module.get_git_status()
            branch = git_info.get("branch", "—")
            commit = git_info.get("commit", "—")[:7]
            clean = git_info.get("clean")
            git_dot = "healthy" if clean is True else ("warning" if clean is False else "unknown")
        except Exception:
            branch, commit, git_dot = "—", "—", "unknown"

        try:
            runs = app_module.fetch_all_runs()
            mlflow_dot = "healthy" if runs is not None else "warning"
            run_count = len(runs) if runs else 0
        except Exception:
            mlflow_dot = "error"
            run_count = 0

        st.markdown(f"""
        <div class="mls-sidebar-footer">
            <div class="mls-sidebar-footer-item">{_health_dot(git_dot)} Git: {branch} · {commit}</div>
            <div class="mls-sidebar-footer-item">{_health_dot(mlflow_dot)} MLflow: {run_count} runs</div>
            <div class="mls-sidebar-footer-item">{_health_dot("healthy")} DVC: local_storage</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def render_overview(app_module):
    _page_header("Overview", "Experiment health, performance trends, and recent activity")

    runs = app_module.fetch_all_runs()
    df = app_module.get_runs_dataframe(runs)

    if df.empty:
        st.info("No MLflow runs found. Run `dvc repro` to create your first experiment.")
        return

    finished = df[df["status"] == "FINISHED"]
    failed = df[df["status"] == "FAILED"]
    running = df[df["status"] == "RUNNING"]
    latest = df.iloc[0]

    # ── Summary Metrics ────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Runs", len(df))
    c2.metric("Finished", len(finished))
    c3.metric("Failed", len(failed))
    c4.metric("Running", len(running))
    c5.metric("Latest Acc", _fmt(latest.get("accuracy")))
    c6.metric("Latest F1", _fmt(latest.get("f1_score")))

    st.markdown("")

    # ── Performance Chart & Recent Runs ────────────────────────────────────
    col_chart, col_table = st.columns([3, 2], gap="large")

    with col_chart:
        st.markdown("##### Performance Trends")
        fin = finished.copy()
        if len(fin) > 1:
            fin = fin.sort_values("start_epoch_ms").reset_index(drop=True)
            fin["run_label"] = fin.apply(lambda r: f"#{r.name+1} ({r['short_id']})", axis=1)

            metric_choice = st.selectbox(
                "Metric",
                ["All Metrics", "accuracy", "f1_score", "precision", "recall"],
                label_visibility="collapsed",
                key="ov_metric_sel",
            )

            if metric_choice == "All Metrics":
                mc = [c for c in ["accuracy", "f1_score", "precision", "recall"] if c in fin.columns]
            else:
                mc = [metric_choice] if metric_choice in fin.columns else []

            if mc:
                chart_df = fin.melt(
                    id_vars=["run_label"], value_vars=mc,
                    var_name="metric", value_name="score"
                )
                chart_df["Metric"] = chart_df["metric"].str.replace("_", " ").str.title()

                chart = (
                    alt.Chart(chart_df)
                    .mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2)
                    .encode(
                        x=alt.X("run_label:N", title=None, sort=None,
                                axis=alt.Axis(labelFontSize=10, labelAngle=0, labelColor="#8b949e")),
                        y=alt.Y("score:Q", title="Score",
                                axis=alt.Axis(format=".3f", labelFontSize=10, labelColor="#8b949e",
                                              gridColor="#21262d", gridOpacity=0.5)),
                        color=alt.Color("Metric:N",
                                        scale=alt.Scale(range=["#58a6ff", "#3fb950", "#d29922", "#f0883e"]),
                                        legend=alt.Legend(orient="bottom", labelFontSize=10, labelColor="#8b949e")),
                        tooltip=["run_label:N", "Metric:N", alt.Tooltip("score:Q", format=".4f")],
                    )
                    .properties(height=280)
                    .configure_view(strokeWidth=0)
                    .configure_axis(domainColor="#30363d")
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.caption("Selected metric not available.")
        elif len(fin) == 1:
            st.caption("Single finished run. Additional runs needed for trend visualization.")
            r = fin.iloc[0]
            mc = st.columns(4)
            mc[0].metric("Accuracy", _fmt(r.get("accuracy")))
            mc[1].metric("F1", _fmt(r.get("f1_score")))
            mc[2].metric("Precision", _fmt(r.get("precision")))
            mc[3].metric("Recall", _fmt(r.get("recall")))
        else:
            st.caption("No finished runs available for charting.")

    with col_table:
        st.markdown("##### Recent Runs")
        recent = df[["short_id", "status", "accuracy", "f1_score", "git_commit", "start_time"]].head(8).copy()
        recent.columns = ["Run", "Status", "Acc", "F1", "Commit", "Started"]
        st.dataframe(
            recent,
            hide_index=True,
            use_container_width=True,
            height=280,
        )

    # ── Quick Actions ──────────────────────────────────────────────────────
    st.markdown("")
    qa1, qa2, qa3, qa4 = st.columns(4)
    with qa1:
        if st.button("⇆ Compare Runs", use_container_width=True, key="ov_cmp"):
            _navigate_to("Compare Runs")
    with qa2:
        if st.button("▣ Chat Assistant", use_container_width=True, key="ov_chat"):
            _navigate_to("Chat Assistant")
    with qa3:
        if st.button("⊶ Lineage", use_container_width=True, key="ov_lin"):
            _navigate_to("Lineage Explorer")
    with qa4:
        if st.button("↻ Recovery", use_container_width=True, key="ov_rec"):
            _navigate_to("Recovery Center")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_experiments(app_module):
    _page_header("Experiments", "Browse, filter, and inspect all MLflow experiment runs")

    runs = app_module.fetch_all_runs()
    df = app_module.get_runs_dataframe(runs)

    if df.empty:
        st.info("No runs recorded. Execute `dvc repro` to create experiment runs.")
        return

    # ── Filters ────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([3, 1.5, 1.5])
    with f1:
        q = st.text_input("Search by run ID or commit SHA", placeholder="e.g. b6b15235",
                          label_visibility="collapsed", key="exp_search")
    with f2:
        statuses = st.multiselect(
            "Status filter", sorted(df["status"].dropna().unique()),
            default=sorted(df["status"].dropna().unique()),
            label_visibility="collapsed", key="exp_status_filter",
        )
    with f3:
        min_acc = st.slider("Min accuracy", 0.0, 1.0, 0.0, 0.01,
                            label_visibility="collapsed", key="exp_acc_slider")

    fdf = df.copy()
    if q:
        ql = q.strip().lower()
        fdf = fdf[fdf["run_id"].str.lower().str.contains(ql) |
                  fdf["git_commit_full"].str.lower().str.contains(ql)]
    if statuses:
        fdf = fdf[fdf["status"].isin(statuses)]
    if min_acc > 0:
        fdf = fdf[fdf["accuracy"].fillna(0) >= min_acc]

    st.caption(f"{len(fdf)} run(s) matching filters")

    # ── Run Table ──────────────────────────────────────────────────────────
    display_cols = ["short_id", "status", "accuracy", "f1_score", "precision", "recall",
                    "duration_seconds", "git_commit", "model_type", "start_time"]
    st.dataframe(
        fdf[display_cols],
        column_config={
            "short_id":         st.column_config.TextColumn("Run ID",   width="small"),
            "status":           st.column_config.TextColumn("Status",   width="small"),
            "accuracy":         st.column_config.NumberColumn("Acc",    format="%.4f"),
            "f1_score":         st.column_config.NumberColumn("F1",     format="%.4f"),
            "precision":        st.column_config.NumberColumn("Prec",   format="%.4f"),
            "recall":           st.column_config.NumberColumn("Recall", format="%.4f"),
            "duration_seconds": st.column_config.NumberColumn("Dur (s)", format="%.1f"),
            "git_commit":       st.column_config.TextColumn("Commit",   width="small"),
            "model_type":       st.column_config.TextColumn("Model"),
            "start_time":       st.column_config.TextColumn("Started"),
        },
        hide_index=True, use_container_width=True, height=260,
    )
    st.download_button("Export CSV", fdf.to_csv(index=False).encode(),
                       "runs.csv", "text/csv", key="exp_csv")

    st.divider()

    # ── Run Inspector ──────────────────────────────────────────────────────
    st.markdown("##### Run Inspector")
    run_ids = df["run_id"].tolist()
    run_labels = [
        f"{r[:8]}  ·  {df.loc[df['run_id']==r, 'start_time'].iloc[0]}"
        f"  ·  {df.loc[df['run_id']==r, 'status'].iloc[0]}"
        for r in run_ids
    ]
    sel_idx = st.selectbox("Select run", range(len(run_ids)),
                           format_func=lambda i: run_labels[i],
                           label_visibility="collapsed", key="exp_sel")
    target_id = run_ids[sel_idx]
    run = app_module.resolve_run(target_id)

    if not run:
        st.warning("Run could not be loaded.")
        return

    details = app_module.get_run_details(run)

    # Metrics row
    mc = st.columns(6)
    mc[0].metric("Status", details["status"])
    mc[1].metric("Accuracy", _fmt(details["metrics"].get("accuracy")))
    mc[2].metric("F1 Score", _fmt(details["metrics"].get("f1_score")))
    mc[3].metric("Precision", _fmt(details["metrics"].get("precision")))
    mc[4].metric("Recall", _fmt(details["metrics"].get("recall")))
    dur = details["metrics"].get("duration_seconds")
    dur_str = f"{float(dur):.1f}s" if dur is not None else "—"
    mc[5].metric("Duration", dur_str)

    # Detail tabs
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
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Commit", _short_hash(g["commit_sha"]))
        gc2.metric("Branch", g["branch"])
        gc3.metric("Dirty", "Yes" if g["is_dirty"] == "True" else "No")
        with st.expander("Git inspection command"):
            st.code(f"git show {g['commit_sha']} --stat", language="bash")

    with tab_e:
        e = details["environment"]
        ef = pd.DataFrame([{"Component": k.replace("_", " ").title(), "Version": v} for k, v in e.items()])
        st.dataframe(ef, hide_index=True, use_container_width=True)

    with tab_h:
        hf = pd.DataFrame([
            {"Artifact": app_module.DVC_TAG_MAP.get(k, k), "Recorded MD5": _short_hash(v)}
            for k, v in details["dvc_hashes"].items()
        ])
        st.dataframe(hf, hide_index=True, use_container_width=True)

    with tab_a:
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1:
            if st.button("Trace lineage", key="exp_to_lin"):
                st.session_state.selected_lineage_run = target_id
                _navigate_to("Lineage Explorer")
        with ac2:
            if st.button("Audit artifacts", key="exp_to_art"):
                st.session_state.selected_artifact_run = target_id
                _navigate_to("Artifacts")
        with ac3:
            if st.button("Recovery preview", key="exp_to_rec"):
                st.session_state.selected_recovery_run = target_id
                _navigate_to("Recovery Center")
        with ac4:
            if st.button("Compare runs", key="exp_to_cmp"):
                _navigate_to("Compare Runs")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COMPARE RUNS
# ═══════════════════════════════════════════════════════════════════════════════

def render_compare(app_module):
    _page_header("Compare Runs", "Side-by-side metrics, parameters, and artifact analysis")

    runs = app_module.fetch_all_runs()
    df = app_module.get_runs_dataframe(runs)

    if len(df) < 2:
        st.warning("At least two runs are required for comparison.")
        return

    run_ids = df["run_id"].tolist()
    labels = [
        f"{r[:8]}  ·  {df.loc[df['run_id']==r, 'start_time'].iloc[0]}"
        f"  ({df.loc[df['run_id']==r, 'status'].iloc[0]})"
        for r in run_ids
    ]

    sel_a, sel_b = st.columns(2)
    with sel_a:
        idx_a = st.selectbox("Baseline (A)", range(len(run_ids)), 0,
                             format_func=lambda i: labels[i], key="cmp_a")
    with sel_b:
        idx_b = st.selectbox("Comparison (B)", range(len(run_ids)),
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

    # ── Delta Metrics ──────────────────────────────────────────────────────
    pri = ["accuracy", "f1_score", "precision", "recall", "duration_seconds"]
    dcols = st.columns(len(pri))
    for i, m in enumerate(pri):
        va = det_a["metrics"].get(m)
        vb = det_b["metrics"].get(m)
        with dcols[i]:
            label = m.replace("_", " ").title()
            if va is not None and vb is not None:
                delta = vb - va
                fmt = ".2f" if m == "duration_seconds" else ".4f"
                unit = "s" if m == "duration_seconds" else ""
                st.metric(label, f"{vb:{fmt}}{unit}", f"{delta:+{fmt}}{unit}",
                          delta_color="inverse" if m == "duration_seconds" else "normal")
            else:
                st.metric(label, "—")

    st.divider()

    # ── Detail Tabs ────────────────────────────────────────────────────────
    t_m, t_p, t_h, t_g = st.tabs(["Metrics", "Parameters", "Artifact Hashes", "Git & Environment"])

    with t_m:
        rows = []
        for m in matrix["metrics"]:
            va, vb, d = m["run_a"], m["run_b"], m["delta"]
            rows.append({
                "Metric":               m["metric"].replace("_", " ").title(),
                f"Run A ({rid_a[:8]})": _fmt(va),
                f"Run B ({rid_b[:8]})": _fmt(vb),
                "Delta B−A":           f"{d:+.4f}" if isinstance(d, float) else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with t_p:
        rows = []
        for p in matrix["parameters"]:
            rows.append({
                "Parameter":             p["parameter"],
                f"Run A ({rid_a[:8]})":  p["run_a"],
                f"Run B ({rid_b[:8]})":  p["run_b"],
                "Changed":               "Yes" if p["changed"] else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with t_h:
        rows = []
        for h in matrix["hashes"]:
            rows.append({
                "Artifact":              h["artifact"],
                f"Run A ({rid_a[:8]})":  _short_hash(h["run_a_hash"]),
                f"Run B ({rid_b[:8]})":  _short_hash(h["run_b_hash"]),
                "Match":                 "Yes" if h["match"] else "Differs",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with t_g:
        rows = [
            {"Attribute": "Git Commit",
             f"Run A ({rid_a[:8]})": _short_hash(det_a["git"]["commit_sha"]),
             f"Run B ({rid_b[:8]})": _short_hash(det_b["git"]["commit_sha"])},
            {"Attribute": "Branch",
             f"Run A ({rid_a[:8]})": det_a["git"]["branch"],
             f"Run B ({rid_b[:8]})": det_b["git"]["branch"]},
            {"Attribute": "Python",
             f"Run A ({rid_a[:8]})": det_a["environment"]["python_version"],
             f"Run B ({rid_b[:8]})": det_b["environment"]["python_version"]},
            {"Attribute": "scikit-learn",
             f"Run A ({rid_a[:8]})": det_a["environment"]["scikit_learn"],
             f"Run B ({rid_b[:8]})": det_b["environment"]["scikit_learn"]},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Export ─────────────────────────────────────────────────────────────
    csv = pd.DataFrame(
        [{"Metric": m["metric"],
          f"Run A ({rid_a[:8]})": m["run_a"],
          f"Run B ({rid_b[:8]})": m["run_b"]}
         for m in matrix["metrics"]]
    ).to_csv(index=False).encode()
    st.download_button("Export comparison CSV", csv,
                       f"compare_{rid_a[:8]}_vs_{rid_b[:8]}.csv", "text/csv", key="cmp_csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LINEAGE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

def render_lineage(app_module):
    _page_header("Lineage Explorer",
                 "End-to-end pipeline provenance: data → preparation → features → training → evaluation")

    runs = app_module.fetch_all_runs()
    df = app_module.get_runs_dataframe(runs)

    if df.empty:
        st.info("No runs available to trace.")
        return

    run_ids = df["run_id"].tolist()
    default_idx = 0
    sel_lin = st.session_state.get("selected_lineage_run")
    if sel_lin and sel_lin in run_ids:
        default_idx = run_ids.index(sel_lin)

    sel_id = st.selectbox(
        "Select run", run_ids, index=default_idx,
        format_func=lambda r:
            f"{r[:8]}  ·  {df.loc[df['run_id']==r, 'start_time'].iloc[0]}",
        label_visibility="collapsed", key="lin_sel",
    )

    with st.spinner("Loading lineage report…"):
        report = app_module.fetch_lineage_report_data(sel_id)

    if not report:
        st.error(f"Could not load lineage report for run `{sel_id[:8]}`. Verify scripts are available in System Health.")
        return

    v = report.get("verification_summary", {})
    g = report.get("git_provenance", {})

    # ── Summary Metrics ────────────────────────────────────────────────────
    lc = st.columns(5)
    verified = v.get("is_lineage_verified", False)
    lc[0].metric("Lineage", "Verified" if verified else "Discrepancy")
    lc[1].metric("Git Commit", _short_hash(g.get("commit_sha", "—")))
    lc[2].metric("Tracked", f"{v.get('versioned_artifacts_tracked', 0)}")
    lc[3].metric("Verified", f"{v.get('verified_in_workspace', 0)}")
    lc[4].metric("Recoverable", f"{v.get('recoverable_from_dvc', 0)}")

    st.divider()

    # ── Pipeline Stage Trace ───────────────────────────────────────────────
    stages = report.get("lineage_stages", [])
    stage_labels = {"prepare": "Prepare", "featurize": "Featurize",
                    "train": "Train", "evaluate": "Evaluate"}

    for i, s in enumerate(stages):
        name = s.get("stage", "stage")
        label = stage_labels.get(name.lower(), name.upper())
        desc = s.get("description", "")
        inputs = s.get("inputs", [])
        outputs = s.get("outputs", [])

        if i > 0:
            st.markdown('<div class="mls-lineage-connector">↓</div>', unsafe_allow_html=True)

        with st.expander(f"**Stage {i+1}: {label}** — {desc}", expanded=(i == 0)):
            ci, co = st.columns(2)
            with ci:
                st.markdown("**Inputs**")
                if inputs:
                    for inp in inputs:
                        st.markdown(f"- `{inp}`")
                else:
                    st.caption("Pipeline entry point")
            with co:
                st.markdown("**Outputs**")
                for out in outputs:
                    status = out.get("status", "")
                    ok = "VERIFIED" in status or "PRESENT" in status
                    indicator = "✓" if ok else "?"
                    h = out.get("expected_hash", "")
                    st.markdown(f"- [{indicator}] `{out['path']}` — *{out.get('role', '')}*")
                    if h:
                        st.caption(f"MD5: `{_short_hash(h)}`")

    # ── Summary Footer ─────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"Tracked: {v.get('versioned_artifacts_tracked', 0)}  ·  "
        f"Verified: {v.get('verified_in_workspace', 0)}  ·  "
        f"Mismatches: {v.get('workspace_hash_mismatches', 0)}  ·  "
        f"Missing: {v.get('workspace_missing', 0)}"
    )

    c1, c2 = st.columns([1, 4])
    with c1:
        st.download_button(
            "Export JSON", json.dumps(report, indent=2).encode(),
            f"lineage_{sel_id[:8]}.json", "application/json", key="lin_dl",
        )
    with c2:
        with st.expander("Raw lineage report"):
            st.json(report)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_artifacts(app_module):
    _page_header("Artifacts",
                 "Verify pipeline artifact integrity against recorded hashes and DVC state")

    runs = app_module.fetch_all_runs()
    df = app_module.get_runs_dataframe(runs)

    if df.empty:
        st.info("No runs found.")
        return

    run_ids = df["run_id"].tolist()
    default_idx = 0
    sel_art = st.session_state.get("selected_artifact_run")
    if sel_art and sel_art in run_ids:
        default_idx = run_ids.index(sel_art)

    sel_id = st.selectbox(
        "Select run", run_ids, index=default_idx,
        format_func=lambda r:
            f"{r[:8]}  ({df.loc[df['run_id']==r, 'start_time'].iloc[0]})",
        label_visibility="collapsed", key="art_sel",
    )

    run = app_module.resolve_run(sel_id)
    if not run:
        st.error("Run not found.")
        return

    results = app_module.verify_run_artifacts(run)
    verified = sum(1 for a in results if "VERIFIED" in a["status"])
    total = len(results)

    # ── Verification Summary ───────────────────────────────────────────────
    vc = st.columns(3)
    vc[0].metric("Artifacts", total)
    vc[1].metric("Verified", verified)
    vc[2].metric("Issues", total - verified)

    if verified == total:
        st.success(f"All {total} artifacts match recorded hashes for run `{sel_id[:8]}`.")
    else:
        st.warning(f"{verified}/{total} verified. Use Recovery Center to restore missing or mismatched artifacts.")

    # ── Hash Table ─────────────────────────────────────────────────────────
    rows = []
    for a in results:
        rows.append({
            "Artifact":     a["artifact"],
            "Recorded":     _short_hash(a["expected_hash"]),
            "Workspace":    _short_hash(a.get("workspace_hash") or ""),
            "DVC Lock":     _short_hash(a.get("lock_hash") or ""),
            "Status":       a["status"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Recorded = MLflow tags · Workspace = current file · DVC Lock = from dvc.lock")

    # ── Live Verification ──────────────────────────────────────────────────
    st.divider()
    st.markdown("##### CLI Verification")
    st.caption("Runs `scripts/recover_run.py --verify-only` (read-only)")
    if st.button("Run verification", key="art_verify"):
        with st.spinner("Verifying…"):
            proc = app_module.run_recovery_subprocess(sel_id, ["--verify-only"])
        if proc["stdout"]:
            with st.expander("Verification output", expanded=True):
                st.code(proc["stdout"], language="text")
        if proc["stderr"]:
            st.warning(proc["stderr"])
        if proc["exit_code"] == 0:
            st.success("All checked artifacts match recorded hashes.")
        else:
            st.warning(f"Verification returned exit code {proc['exit_code']}.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RECOVERY CENTER
# ═══════════════════════════════════════════════════════════════════════════════

def render_recovery(app_module):
    _page_header("Recovery Center",
                 "Restore historical experiment states via isolated workspace or in-place recovery")

    runs = app_module.fetch_all_runs()
    df = app_module.get_runs_dataframe(runs)

    if df.empty:
        st.info("No runs available for recovery.")
        return

    run_ids = df["run_id"].tolist()
    default_idx = 0
    sel_rec = st.session_state.get("selected_recovery_run")
    if sel_rec and sel_rec in run_ids:
        default_idx = run_ids.index(sel_rec)

    sel_id = st.selectbox(
        "Select historical run", run_ids, index=default_idx,
        format_func=lambda r: (
            f"{r[:8]}  ·  Git {df.loc[df['run_id']==r, 'git_commit'].iloc[0]}"
            f"  ·  {df.loc[df['run_id']==r, 'start_time'].iloc[0]}"
        ),
        label_visibility="collapsed", key="rec_sel",
    )
    short_id = sel_id[:8]

    pending = app_module.get_confirmation_state()
    if pending and pending.get("run_id") != sel_id:
        app_module.clear_confirmation_state()
        pending = None

    sel_run = app_module.resolve_run(sel_id)
    git_commit = "unavailable"
    if sel_run:
        git_commit = sel_run.data.tags.get("git.commit_sha", "unavailable")

    # ── Run Info ───────────────────────────────────────────────────────────
    ic = st.columns(3)
    ic[0].metric("Run ID", short_id)
    ic[1].metric("Git Commit", _short_hash(git_commit))
    status_val = df.loc[df["run_id"] == sel_id, "status"].iloc[0]
    ic[2].metric("Status", status_val)

    st.divider()

    # ── Recovery Tabs ──────────────────────────────────────────────────────
    tab_iso, tab_ws = st.tabs([
        "Isolated Workspace (recommended)",
        "Current-Workspace Restore",
    ])

    # -- A. Isolated workspace ─────────────────────────────────────────────
    with tab_iso:
        st.caption(
            "Creates a `git worktree` at the destination checked out to the historical commit. "
            "DVC artifacts are then restored from the local cache. Your active project is not modified."
        )

        with st.expander("How this works"):
            st.markdown(
                f"1. `git worktree add <dest> {_short_hash(git_commit)}` — "
                "creates a separate working tree at the historical revision.\n"
                f"2. `scripts/recover_run.py {short_id} --dry-run` — preview of artifact restoration.\n"
                f"3. On confirmation: `scripts/recover_run.py {short_id}` restores `data/` and `models/`.\n\n"
                "**Restored:** source code (via worktree), DVC data/model artifacts.\n"
                "**Not restored:** Python `.venv` (shared), new MLflow runs.\n"
                "**Limitation:** requires local DVC cache to contain historical hashes."
            )

        iso_dest = st.text_input(
            "Destination directory",
            value=f"D:\\ML_Workspaces\\run_{short_id}",
            key="iso_dest",
        )
        if os.path.exists(iso_dest):
            st.caption(f"Directory `{iso_dest}` already exists.")

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("Preview (dry-run)", use_container_width=True, key="iso_preview"):
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

        with bc2:
            if git_commit == "unavailable":
                st.warning("No Git commit SHA recorded — worktree cannot be created.")
            else:
                if st.button("Create isolated workspace", type="primary",
                             use_container_width=True, key="iso_exec"):
                    _create_isolated_workspace(app_module, sel_id, git_commit, iso_dest)

    # -- B. Current-workspace restore ──────────────────────────────────────
    with tab_ws:
        st.error(
            "⚠ This operation overwrites files in `data/` and `models/` with historical artifacts. "
            "Code files, Git history, and MLflow records are not affected."
        )

        st.markdown("**Step 1 — Dry-run preview (required)**")
        if st.button("Execute dry-run preview", use_container_width=True, key="ws_dryrun"):
            with st.spinner("Analyzing recovery sources…"):
                proc = app_module.run_recovery_subprocess(sel_id, ["--dry-run"])
                app_module.set_confirmation_state({
                    "run_id": sel_id,
                    "short_id": short_id,
                    "timestamp": time.time(),
                })
            st.session_state.ws_dryrun_output = proc["stdout"]
            st.session_state.ws_dryrun_exit = proc["exit_code"]
            st.rerun()

        if st.session_state.get("ws_dryrun_output"):
            with st.expander("Dry-run output", expanded=True):
                st.code(st.session_state.ws_dryrun_output, language="text")
            if st.session_state.get("ws_dryrun_exit", 1) != 0:
                st.warning("Dry-run reported issues. Review before proceeding.")

        st.divider()
        st.markdown("**Step 2 — Confirm restoration**")
        pending = app_module.get_confirmation_state()
        if pending and pending.get("run_id") == sel_id:
            confirm = st.checkbox(
                f"I understand that `data/` and `models/` will be overwritten with artifacts from run `{short_id}`.",
                key="ws_confirm_cb",
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Restore artifacts", type="primary",
                             disabled=not confirm, use_container_width=True, key="ws_confirm_btn"):
                    with st.spinner("Restoring…"):
                        resp = app_module.handle_confirm_recover(short_id)
                    st.success(resp)
                    st.session_state.ws_dryrun_output = None
                    st.rerun()
            with rc2:
                if st.button("Cancel", use_container_width=True, key="ws_cancel"):
                    app_module.clear_confirmation_state()
                    st.session_state.ws_dryrun_output = None
                    st.info("Cancelled. No files modified.")
                    st.rerun()
        else:
            st.caption("Run the dry-run preview to unlock confirmation.")


def _create_isolated_workspace(app_module, run_id: str, git_commit: str, dest: str):
    """Create a git worktree at dest, then preview artifact restoration."""
    project_root = app_module.PROJECT_ROOT
    python_exe = app_module.get_python_executable()
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
    with st.expander("Next steps"):
        st.code(
            f"# Restore DVC artifacts into the worktree:\n"
            f"{python_exe} {recover_script} {run_id[:8]}\n\n"
            f"# Run the pipeline in the isolated workspace:\n"
            f"cd {dest}\n"
            f"{python_exe} -m dvc repro",
            language="bash",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — CHAT ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════

def render_chat(app_module):
    _page_header("Chat Assistant",
                 "Command-driven experiment queries · All operations read-only unless recovery is explicitly confirmed")

    runs = app_module.fetch_all_runs()

    # ── Suggested Commands ─────────────────────────────────────────────────
    suggestions = [
        ("list runs",  "List Runs"),
        ("status",     "System Status"),
        ("help",       "Command Help"),
    ]
    if runs:
        lid = runs[0].info.run_id[:8]
        suggestions += [
            (f"inspect {lid}",         f"Inspect {lid}"),
            (f"verify {lid}",          f"Verify {lid}"),
            (f"lineage {lid}",         f"Lineage {lid}"),
            (f"preview recover {lid}", f"Preview Recovery"),
        ]
        if len(runs) > 1:
            sid2 = runs[1].info.run_id[:8]
            suggestions.append((f"compare {lid} {sid2}", f"Compare Runs"))

    # Show suggestion chips
    row_size = 4
    for row_start in range(0, len(suggestions), row_size):
        chunk = suggestions[row_start:row_start + row_size]
        cols = st.columns(len(chunk))
        for i, (cmd, label) in enumerate(chunk):
            with cols[i]:
                if st.button(label, use_container_width=True, key=f"chat_sug_{row_start+i}"):
                    st.session_state.chat_trigger = cmd

    st.divider()

    # ── Message History ────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Welcome to ML Lineage Studio. Type a command or select an action above.\n\n"
                "**Common commands:** `list runs` · `inspect <id>` · `verify <id>` · "
                "`lineage <id>` · `compare <a> <b>` · `preview recover <id>` · `status` · `help`\n\n"
                "*All operations are read-only unless recovery is explicitly confirmed.*"
            ),
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Pending Recovery Warning ───────────────────────────────────────────
    pending = app_module.get_confirmation_state()
    if pending:
        st.warning(
            f"Recovery pending for run `{pending['short_id']}`. "
            "Type `confirm recover <id>` to proceed or `cancel` to abort."
        )

    # ── Chat Input ─────────────────────────────────────────────────────────
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

    # ── Clear Chat ─────────────────────────────────────────────────────────
    if len(st.session_state.messages) > 1:
        st.markdown("")
        if st.button("Clear conversation", key="chat_clear"):
            # Preserve any pending recovery state
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Conversation cleared. Type a command or select an action above."
            }]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — SYSTEM HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

def render_system_status(app_module):
    _page_header("System Health",
                 "Diagnostic checks for Git, MLflow, DVC pipeline, and project infrastructure")

    git_info = app_module.get_git_status()
    dvc_ws = app_module.get_dvc_status()
    dvc_rem = app_module.get_dvc_remote_status()
    runs = app_module.fetch_all_runs()

    col_l, col_r = st.columns(2, gap="large")

    # ── Left Column ────────────────────────────────────────────────────────
    with col_l:
        # Git
        st.markdown("##### Git Repository")
        branch = git_info.get("branch", "—")
        commit = git_info.get("commit", "unavailable")
        clean = git_info.get("clean")

        gc = st.columns(3)
        gc[0].metric("Branch", branch)
        gc[1].metric("HEAD", commit[:7] if len(commit) >= 7 else commit)
        wt_label = "Clean" if clean is True else ("Dirty" if clean is False else "Unknown")
        gc[2].metric("Working Tree", wt_label)

        if clean is False:
            st.caption("Working tree has uncommitted changes.")
        elif clean is True:
            st.caption("Working tree is clean — all changes committed.")

        st.markdown("")

        # MLflow
        st.markdown("##### MLflow Tracking")
        finished = sum(1 for r in runs if r.info.status == "FINISHED")
        failed = sum(1 for r in runs if r.info.status == "FAILED")

        mc = st.columns(3)
        mc[0].metric("Total Runs", len(runs))
        mc[1].metric("Finished", finished)
        mc[2].metric("Failed", failed)
        st.caption(f"URI: `{app_module.get_tracking_uri()}`")

        if runs:
            latest = runs[0]
            ts = time.strftime("%Y-%m-%d %H:%M", time.gmtime(latest.info.start_time / 1000))
            acc = latest.data.metrics.get("accuracy")
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            st.caption(f"Latest: `{latest.info.run_id[:8]}` ({ts}) · Accuracy: {acc_str}")

        st.markdown("")

        # Project Files
        st.markdown("##### Project Files")
        checks = [
            ("scripts/recover_run.py",    "Recovery utility"),
            ("scripts/lineage_report.py", "Lineage report"),
            ("dvc.yaml",    "Pipeline definition"),
            ("dvc.lock",    "Pipeline lock"),
            ("params.yaml", "Parameters"),
            ("mlflow.db",   "MLflow database"),
        ]
        for rel, label in checks:
            path = os.path.join(app_module.PROJECT_ROOT, rel)
            exists = os.path.exists(path)
            dot = _health_dot("healthy" if exists else "error")
            status = "Available" if exists else "Missing"
            st.markdown(f"{dot} **{label}** · `{rel}` · {status}", unsafe_allow_html=True)

    # ── Right Column ───────────────────────────────────────────────────────
    with col_r:
        # DVC Pipeline
        st.markdown("##### DVC Pipeline Status")
        st.caption("`dvc status` — checks if pipeline stages need re-execution")

        display_ws = dvc_ws or "Pipeline is up to date."
        with st.expander("Pipeline status output", expanded=True):
            st.code(display_ws, language="yaml")
        if "always changed" in (dvc_ws or "").lower():
            st.caption(
                "Stages marked `always changed` (e.g. log_experiment) intentionally run on every "
                "`dvc repro` because they generate unique outputs each time."
            )

        st.markdown("")

        # DVC Remote
        st.markdown("##### DVC Storage Sync")
        st.caption("`dvc status -c` — compares local cache with configured remote")
        st.caption("Remote type: **local** (`D:\\ML_DVC_Storage`) — not cloud storage.")

        remote_display = dvc_rem or "—"
        with st.expander("Remote sync output", expanded=True):
            st.code(remote_display, language="text")

        if "in sync" in remote_display.lower() or "cache and remote" in remote_display.lower():
            st.success("Local DVC cache and remote storage are synchronized.")
        elif "error" in remote_display.lower():
            st.error("DVC remote check reported an error.")
        else:
            st.caption("Review output above for sync details.")
