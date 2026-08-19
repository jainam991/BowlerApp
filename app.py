"""
Bowler Workload Platform — Real Data Edition v3
=====================================================
A significantly expanded, state-of-the-art dashboard built on real
ball-by-ball cricket data (via preprocess.py).

Highlights over v2:
  - Custom ACWR gauge chart (matplotlib semi-circle speedometer)
  - Radar/spider chart player comparison
  - Advanced multi-criteria search & filter page
  - Rising Stars (season-over-season improvement) leaderboard
  - Most Consistent bowlers (economy variance) leaderboard
  - Team vs Team historical head-to-head analysis
  - Player milestones (best/worst match performances)
  - Richer visual theme throughout (cards, gradients, badges)
  - Methodology / About page documenting the whole pipeline

Honesty note (unchanged from v1/v2): there is no public dataset of
real bowler injuries, so instead of a fabricated "injury prediction"
ML classifier, this app uses ACWR (Acute:Chronic Workload Ratio) — a
real, published sports-science method for flagging workload overload,
computed from real match data. It is not a validated injury predictor.

Run locally with:
    streamlit run app.py

Expects (in the same folder, or uploaded via the sidebar):
    bowler_match_summary.csv
    bowler_vs_team_summary.csv
    bowler_master.csv
(all produced by running preprocess.py on your full dataset)
"""

import os
import base64
import re
import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bowler_app")

st.set_page_config(page_title="Bowler Workload Platform (Real Data)", page_icon="🏏", layout="wide")

# ==================================================================
# THEME / CUSTOM CSS
# ==================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-0: #05070a;
        --bg-1: #0a0e13;
        --bg-2: #10141a;
        --panel: #12161d;
        --panel-2: #171c24;
        --border: #232a35;
        --border-soft: #1c222c;
        --text-hi: #f3f1e9;
        --text-mid: #c7cdd8;
        --text-dim: #838d9c;
        --pitch: #22b573;
        --pitch-deep: #12583a;
        --leather: #d8973c;
        --ball: #d1483f;
        --sky: #4f9fd8;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Headings + anything with visual authority use the display face */
    h1, h2, h3, h4,
    .stApp [data-testid="stMarkdownContainer"] h1,
    .stApp [data-testid="stMarkdownContainer"] h2,
    .stApp [data-testid="stMarkdownContainer"] h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.3px;
    }

    /* Numbers read like a scoreboard/terminal everywhere they appear */
    .num, .kpi-card .value, .tile-pill, .tile-footer, .badge,
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% -5%, rgba(34,181,115,0.10) 0%, transparent 45%),
            radial-gradient(circle at 90% 0%, rgba(216,151,60,0.06) 0%, transparent 40%),
            linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%);
    }

    /* Hide the dev-tool chrome (Deploy button / menu) for a standalone-product feel,
       while keeping the sidebar-collapse control intact. */
    [data-testid="stToolbar"] { visibility: hidden; }

    /* ---------- Seam-stitch signature motif ---------- */
    .seam {
        height: 14px; margin: 4px 0 22px 0; position: relative;
        background-image: repeating-linear-gradient(
            90deg, transparent, transparent 10px, rgba(216,151,60,0.55) 10px, rgba(216,151,60,0.55) 12px
        );
        -webkit-mask-image: linear-gradient(90deg, transparent, black 8%, black 92%, transparent);
                mask-image: linear-gradient(90deg, transparent, black 8%, black 92%, transparent);
        opacity: 0.55;
    }
    .seam::before, .seam::after {
        content: ""; position: absolute; left: 0; right: 0; height: 1px;
        background: rgba(216,151,60,0.35);
    }
    .seam::before { top: 3px; } .seam::after { bottom: 3px; }

    /* ---------- Header ---------- */
    .app-header {
        background:
            radial-gradient(circle at 85% -40%, rgba(255,255,255,0.10), transparent 55%),
            linear-gradient(120deg, #0a3826 0%, #135a3d 45%, #1c8455 80%, #22b573 100%);
        padding: 30px 34px; border-radius: 18px; margin-bottom: 4px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 34px rgba(0,0,0,0.40);
        position: relative; overflow: hidden;
    }
    .app-header::after {
        content: "🏏"; position: absolute; right: 24px; top: 50%; transform: translateY(-50%) rotate(-18deg);
        font-size: 84px; opacity: 0.10; pointer-events: none;
    }
    .app-header .eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
        letter-spacing: 1.6px; text-transform: uppercase; color: #bdeecf; opacity: 0.85;
    }
    .app-header h1 {
        color: #fbfbf8; margin: 4px 0 0 0; font-size: 33px; font-weight: 700;
        letter-spacing: -0.6px; font-family: 'Space Grotesk', sans-serif;
    }
    .app-header p { color: #d7e8dd; margin: 7px 0 0 0; font-size: 14.5px; font-weight: 400; max-width: 640px; }
    .app-header .filter-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }

    /* ---------- Injury & Fitness banner — the flagship page gets its
       own visual identity (medical amber/red vs the app's usual green),
       so it reads as the core of the project, not just another tab. ---------- */
    .injury-header {
        background:
            radial-gradient(circle at 85% -40%, rgba(255,255,255,0.10), transparent 55%),
            linear-gradient(120deg, #3d1414 0%, #5c2323 45%, #8a3a2a 80%, #b0562e 100%);
        padding: 30px 34px; border-radius: 18px; margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 34px rgba(0,0,0,0.40);
        position: relative; overflow: hidden;
    }
    .injury-header::after {
        content: "🚑"; position: absolute; right: 24px; top: 50%; transform: translateY(-50%) rotate(-10deg);
        font-size: 84px; opacity: 0.10; pointer-events: none;
    }
    .injury-header .eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
        letter-spacing: 1.6px; text-transform: uppercase; color: #ffd8c2; opacity: 0.9;
    }
    .injury-header h1 {
        color: #fbfbf8; margin: 4px 0 0 0; font-size: 30px; font-weight: 700;
        letter-spacing: -0.6px; font-family: 'Space Grotesk', sans-serif;
    }
    .injury-header p { color: #ffe3d5; margin: 7px 0 0 0; font-size: 14.5px; max-width: 680px; }

    .home-callout {
        background: linear-gradient(120deg, rgba(176,65,62,0.16), rgba(176,86,46,0.10));
        border: 1px solid rgba(240,133,124,0.35); border-radius: 16px;
        padding: 20px 24px; margin: 18px 0 22px 0;
    }
    .home-callout .tag {
        display: inline-block; background: rgba(240,133,124,0.18); color: #f0857c;
        border: 1px solid rgba(240,133,124,0.4); border-radius: 999px;
        padding: 2px 12px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
        text-transform: uppercase; margin-bottom: 8px;
    }
    .home-callout h3 { margin: 4px 0 6px 0; color: #fbfbf8; font-family: 'Space Grotesk', sans-serif; }
    .home-callout p { color: #d7d0cc; margin: 0 0 4px 0; font-size: 14px; }

    .alert-row {
        display: flex; align-items: center; gap: 12px; padding: 12px 16px;
        background: var(--bg-2); border: 1px solid var(--border-soft); border-left: 3px solid;
        border-radius: 10px; margin-bottom: 8px;
    }
    .alert-row .alert-type {
        font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px;
        opacity: 0.85;
    }
    .alert-row .alert-name { font-weight: 700; color: var(--text-hi); font-size: 14px; }
    .alert-row .alert-detail { color: var(--text-mid); font-size: 12.5px; }
    .fchip {
        background: rgba(0,0,0,0.24); border: 1px solid rgba(255,255,255,0.20);
        color: #eafaf0; font-size: 11.5px; font-weight: 700; padding: 5px 13px;
        border-radius: 999px; letter-spacing: 0.3px; font-family: 'JetBrains Mono', monospace;
    }

    /* ---------- KPI / scoreboard cards ---------- */
    .kpi-card {
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border); border-radius: 12px;
        padding: 16px 16px 14px 16px; text-align: left; position: relative;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card::before {
        content: ""; position: absolute; top: 14px; right: 14px; width: 6px; height: 6px;
        border-radius: 50%; background: var(--pitch); box-shadow: 0 0 8px 1px rgba(34,181,115,0.7);
    }
    .kpi-card:hover { border-color: var(--pitch); transform: translateY(-2px); }
    .kpi-card .value {
        font-size: 26px; font-weight: 700; color: var(--text-hi); display: block; line-height: 1.1;
    }
    .kpi-card .label {
        font-size: 11px; color: var(--text-dim); margin-top: 6px; letter-spacing: 0.4px;
        text-transform: uppercase; font-weight: 600;
    }

    .player-card {
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border); border-radius: 16px;
        padding: 18px 20px; margin-bottom: 12px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
        border-left: 3px solid var(--pitch);
    }
    .meta { color: var(--text-dim); font-size: 13.5px; }

    /* Real section cards — targets the stable class Streamlit attaches to
       st.container(key="sec_...") wrappers. Unlike the old .section-card
       div (which was split across two separate st.markdown calls and never
       actually nested around its contents), this styles a real container
       that genuinely wraps every child element placed inside it —
       charts, tables, columns, anything. */
    div[class*="st-key-sec_"] {
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--bg-2) 100%);
        border: 1px solid var(--border-soft); border-radius: 14px;
        padding: 18px 20px 14px 20px; margin-bottom: 18px;
        transition: border-color 0.15s ease;
    }
    div[class*="st-key-sec_"]:hover { border-color: var(--border); }
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 15px; font-weight: 600; color: #e4e8ee;
        margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
    }
    /* A lighter-weight variant for wrapping a single chart with no visible
       title bar (still gets real containment + hover, just no header row) */
    div[class*="st-key-chart_"] {
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--bg-2) 100%);
        border: 1px solid var(--border-soft); border-radius: 14px;
        padding: 14px 16px 6px 16px; margin-bottom: 18px;
    }

    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 6px;
        font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
        border: 1px solid transparent;
    }
    .badge-low { background: rgba(34,181,115,0.12); color: #5be3a3; border-color: rgba(34,181,115,0.35); }
    .badge-moderate { background: rgba(216,151,60,0.12); color: #eab962; border-color: rgba(216,151,60,0.35); }
    .badge-high { background: rgba(209,72,63,0.14); color: #f0857c; border-color: rgba(209,72,63,0.4); }
    .badge-under { background: rgba(79,159,216,0.13); color: #8ec6ee; border-color: rgba(79,159,216,0.35); }
    .badge-fit { background: rgba(34,181,115,0.12); color: #5be3a3; border-color: rgba(34,181,115,0.35); }
    .badge-managed { background: rgba(216,151,60,0.12); color: #eab962; border-color: rgba(216,151,60,0.35); }
    .badge-injured { background: rgba(209,72,63,0.14); color: #f0857c; border-color: rgba(209,72,63,0.4); }
    .badge-rehab { background: rgba(147,112,219,0.14); color: #c9a9f5; border-color: rgba(147,112,219,0.4); }

    .milestone-chip {
        display: inline-block; background: var(--panel-2); color: var(--text-mid);
        border: 1px solid var(--border); padding: 6px 14px; border-radius: 8px;
        font-size: 13px; margin: 4px 6px 4px 0;
    }
    .milestone-chip b { color: var(--pitch); font-family: 'JetBrains Mono', monospace; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border-soft); }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-2); border-radius: 8px 8px 0 0; padding: 9px 18px;
        font-family: 'Space Grotesk', sans-serif; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { color: var(--pitch) !important; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0f14 0%, #070a0d 100%);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ---------- Inputs ---------- */
    .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div,
    .stTextInput input {
        background-color: var(--panel) !important; border-color: var(--border) !important;
        border-radius: 9px !important;
    }
    .stButton button, .stDownloadButton button {
        border-radius: 9px !important; border: 1px solid var(--border) !important;
        font-family: 'Space Grotesk', sans-serif !important; font-weight: 500 !important;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        border-color: var(--pitch) !important; color: var(--pitch) !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background-color: var(--pitch) !important; }

    /* ---------- Market tiles ---------- */
    .tile {
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border); border-radius: 14px;
        padding: 18px 20px; margin-bottom: 16px;
        transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
    }
    .tile:hover {
        border-color: #34404f; transform: translateY(-2px);
        box-shadow: 0 10px 26px rgba(0,0,0,0.32);
    }
    .tile-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
    .tile-avatar {
        min-width: 34px; width: 34px; height: 34px; border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; font-weight: 700; color: white; flex-shrink: 0;
        font-family: 'Space Grotesk', sans-serif;
    }
    .tile-category {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px; font-weight: 700; letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .tile-title { font-family: 'Space Grotesk', sans-serif; font-size: 15.5px; font-weight: 600; color: var(--text-hi); margin-top: 1px; }
    .tile-subtitle { font-size: 12px; color: var(--text-dim); }
    .tile-row { display: flex; align-items: center; justify-content: space-between; margin: 10px 0 4px 0; gap: 10px; }
    .tile-bar-label { font-size: 12px; color: var(--text-mid); margin-bottom: 4px; }
    .tile-bar-track { background: var(--border-soft); border-radius: 6px; height: 5px; width: 100%; overflow: hidden; }
    .tile-bar-fill { height: 100%; border-radius: 6px; }
    .tile-pill {
        border: 1.5px solid; border-radius: 6px; padding: 2px 11px;
        font-size: 12px; font-weight: 700; white-space: nowrap;
    }
    .tile-footer {
        display: flex; justify-content: space-between; margin-top: 14px;
        padding-top: 10px; border-top: 1px dashed var(--border-soft);
        font-size: 11px; color: var(--text-dim);
    }

    /* ---------- Head-to-head compare bars (Compare Players page) ---------- */
    .cmp-row { margin-bottom: 16px; }
    .cmp-label {
        text-align: center; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
        text-transform: uppercase; color: var(--text-dim); margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    .cmp-values { display: flex; justify-content: space-between; margin-bottom: 4px; }
    .cmp-value {
        font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 14px;
    }
    .cmp-track {
        display: flex; height: 8px; border-radius: 999px; overflow: hidden;
        background: var(--border-soft);
    }
    .cmp-fill-a, .cmp-fill-b { height: 100%; }
    .cmp-fill-a { border-radius: 999px 0 0 999px; }
    .cmp-fill-b { border-radius: 0 999px 999px 0; }

    /* ================================================================
       FORMAT GATE — full-screen T20 / ODI / TEST / All selector
       ================================================================ */
    .gate-wrap { padding: 6px 0 0 0; }
    .gate-hero { text-align: center; margin: 10px 0 34px 0; }
    .gate-hero .eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 700;
        letter-spacing: 2px; text-transform: uppercase; color: var(--pitch); opacity: 0.9;
    }
    .gate-hero h1 {
        font-size: clamp(26px, 4.2vw, 42px); margin: 8px 0 0 0; color: var(--text-hi);
        letter-spacing: -0.8px;
    }
    .gate-hero p { color: var(--text-dim); font-size: 14.5px; margin: 10px auto 0 auto; max-width: 520px; }

    .gate-tile {
        border-radius: 18px 18px 0 0; padding: 26px 22px 20px 22px; position: relative;
        overflow: hidden; min-height: 210px; border: 1px solid var(--border); border-bottom: none;
        transition: transform 0.18s ease;
    }
    .gate-tile::after {
        content: attr(data-glyph); position: absolute; right: 10px; top: 8px;
        font-size: 88px; opacity: 0.12; transform: rotate(-12deg); pointer-events: none; line-height: 1;
    }
    .gate-tile .gate-tag {
        font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 700;
        letter-spacing: 1.4px; text-transform: uppercase; padding: 3px 10px; border-radius: 999px;
        display: inline-block; border: 1px solid rgba(255,255,255,0.25); color: #fff; opacity: 0.92;
    }
    .gate-tile .gate-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700;
        color: #fff; margin: 12px 0 4px 0; letter-spacing: -0.4px;
    }
    .gate-tile .gate-sub { color: rgba(255,255,255,0.82); font-size: 12.5px; max-width: 220px; }
    .gate-stats { display: flex; gap: 16px; margin-top: 18px; }
    .gate-stat .n {
        font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 17px; color: #fff;
        display: block; line-height: 1.1;
    }
    .gate-stat .l {
        font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.6px; color: rgba(255,255,255,0.65);
    }

    /* Per-format gradient themes */
    .gate-t20 { background: linear-gradient(150deg, #7a2b1f 0%, #c9522f 55%, #e8843c 100%); }
    .gate-odi { background: linear-gradient(150deg, #12395e 0%, #1f6ba8 55%, #4f9fd8 100%); }
    .gate-test { background: linear-gradient(150deg, #0a3826 0%, #135a3d 55%, #22b573 100%); }
    .gate-all { background: linear-gradient(150deg, #2b2f38 0%, #454c59 55%, #6b7385 100%); }

    /* The st.button rendered directly beneath each tile, restyled to look
       like the tile's attached CTA footer via stable key-based classes. */
    div[class*="st-key-fmt_"] button {
        border-radius: 0 0 18px 18px !important; border: 1px solid var(--border) !important;
        border-top: none !important; height: 46px !important; width: 100% !important;
        font-family: 'JetBrains Mono', monospace !important; font-weight: 700 !important;
        font-size: 12.5px !important; letter-spacing: 0.4px !important;
        background: var(--panel) !important; color: var(--text-mid) !important;
        transition: all 0.15s ease !important; margin-top: -1px !important;
    }
    div.st-key-fmt_t20 button:hover { background: #7a2b1f !important; color: #fff !important; border-color: #c9522f !important; }
    div.st-key-fmt_odi button:hover { background: #12395e !important; color: #fff !important; border-color: #4f9fd8 !important; }
    div.st-key-fmt_test button:hover { background: #0a3826 !important; color: #fff !important; border-color: #22b573 !important; }
    div.st-key-fmt_all button:hover { background: #2b2f38 !important; color: #fff !important; border-color: #6b7385 !important; }

    div[class*="st-key-fmt_"]:hover { transform: translateY(-3px); }
    div[class*="st-key-fmt_"] { transition: transform 0.18s ease; }

    /* Active-format pill shown in the app header once a format is chosen */
    .fmt-pill {
        display: inline-flex; align-items: center; gap: 6px; font-family: 'JetBrains Mono', monospace;
        font-size: 11px; font-weight: 700; letter-spacing: 0.5px; padding: 5px 12px 5px 10px;
        border-radius: 999px; border: 1px solid rgba(255,255,255,0.25); color: #fff;
        background: rgba(0,0,0,0.22); text-transform: uppercase;
    }

    /* ---------- Responsiveness ---------- */
    @media (max-width: 900px) {
        .gate-tile { min-height: 172px; padding: 20px 18px 16px 18px; }
        .gate-tile .gate-title { font-size: 20px; }
        .gate-tile::after { font-size: 64px; }
        .app-header { padding: 22px 20px; }
        .app-header h1 { font-size: 24px; }
    }
    @media (max-width: 640px) {
        .gate-stats { flex-wrap: wrap; gap: 10px; }
        .kpi-card .value { font-size: 21px; }
        .app-header::after { display: none; }
    }

    /* ================================================================
       ANALYZING ENGINE — brief transition screen played after picking
       a format, before landing in the app. Sells the scale of the data.
       ================================================================ */
    .an-wrap { display: flex; justify-content: center; padding: 60px 0 40px 0; }
    .an-panel {
        width: 100%; max-width: 640px; background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border); border-radius: 18px; padding: 30px 32px 26px 32px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    }
    .an-eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 700;
        letter-spacing: 1.8px; text-transform: uppercase; color: var(--pitch);
        display: flex; align-items: center; gap: 8px;
    }
    .an-dot {
        width: 7px; height: 7px; border-radius: 50%; background: var(--pitch);
        box-shadow: 0 0 10px 2px rgba(34,181,115,0.75); animation: an-pulse 1s ease-in-out infinite;
    }
    @keyframes an-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
    .an-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 21px; font-weight: 700;
        color: var(--text-hi); margin: 8px 0 22px 0; letter-spacing: -0.3px;
    }
    .an-counters { display: flex; gap: 14px; margin-bottom: 22px; }
    .an-counter {
        flex: 1; background: var(--bg-2); border: 1px solid var(--border-soft); border-radius: 12px;
        padding: 13px 14px; text-align: center;
    }
    .an-counter .n {
        font-family: 'JetBrains Mono', monospace; font-weight: 800; font-size: 19px;
        color: var(--pitch); display: block; font-variant-numeric: tabular-nums; line-height: 1.15;
    }
    .an-counter .l {
        font-size: 9.5px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 3px;
    }
    .an-progress-track {
        background: var(--border-soft); border-radius: 999px; height: 6px; width: 100%;
        overflow: hidden; margin-bottom: 20px;
    }
    .an-progress-fill {
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, var(--pitch-deep), var(--pitch));
        box-shadow: 0 0 12px 1px rgba(34,181,115,0.55);
        transition: width 0.05s linear;
    }
    .an-log { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .an-log-line {
        display: flex; align-items: center; gap: 9px; padding: 4px 0; color: var(--text-dim);
    }
    .an-log-line.done { color: var(--text-mid); }
    .an-log-line .icon { width: 14px; flex-shrink: 0; color: var(--pitch); }
    .an-log-line.pending .icon { color: var(--border); }

    /* ---------- Welcome / entry screen (plays before the format gate) ---------- */
    @keyframes ball-arc {
        0%   { transform: translate(0, 10px) rotate(0deg); opacity: 0; }
        8%   { opacity: 1; }
        50%  { transform: translate(46vw, -100px) rotate(360deg); }
        92%  { opacity: 1; }
        100% { transform: translate(92vw, 30px) rotate(680deg); opacity: 0; }
    }
    @keyframes floodlight-pulse {
        0%, 100% { opacity: 0.32; }
        50% { opacity: 0.62; }
    }
    .welcome-wrap {
        position: relative; min-height: 62vh; display: flex; flex-direction: column;
        align-items: center; justify-content: center; text-align: center;
        overflow: hidden; padding: 46px 20px 30px 20px;
    }
    .welcome-wrap::before, .welcome-wrap::after {
        content: ""; position: absolute; width: 360px; height: 360px; border-radius: 50%;
        background: radial-gradient(circle, rgba(34,181,115,0.30) 0%, transparent 70%);
        animation: floodlight-pulse 4.5s ease-in-out infinite; pointer-events: none;
    }
    .welcome-wrap::before { top: -130px; left: -90px; }
    .welcome-wrap::after {
        bottom: -150px; right: -110px; animation-delay: 2s;
        background: radial-gradient(circle, rgba(216,151,60,0.24) 0%, transparent 70%);
    }
    .welcome-ball {
        position: absolute; top: 42%; left: 0; font-size: 26px;
        animation: ball-arc 3.4s ease-in-out infinite;
        filter: drop-shadow(0 0 8px rgba(255,255,255,0.28));
    }
    .welcome-eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700;
        letter-spacing: 3px; color: var(--pitch); text-transform: uppercase;
    }
    .welcome-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: var(--text-hi);
        margin: 16px 0 14px 0; letter-spacing: -1.2px; line-height: 1.08;
        font-size: clamp(30px, 5vw, 56px);
    }
    .welcome-title span { color: var(--pitch); }
    .welcome-sub { font-size: 15.5px; color: var(--text-dim); max-width: 600px; margin: 0 auto 8px auto; line-height: 1.6; }
    .welcome-stats { display: flex; gap: 34px; justify-content: center; margin-top: 26px; flex-wrap: wrap; }
    .welcome-stat .n { font-family: 'JetBrains Mono', monospace; font-size: 23px; font-weight: 700; color: var(--text-hi); display: block; }
    .welcome-stat .l { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.6px; }

    div[class*="st-key-enter_app"] button {
        margin-top: 8px !important; padding: 14px 8px !important; font-size: 15px !important;
        background: linear-gradient(120deg, #145c3f, #22b573) !important; color: #fff !important;
        border: none !important; border-radius: 999px !important;
        font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important;
        box-shadow: 0 8px 26px rgba(34,181,115,0.35) !important; transition: all 0.15s ease !important;
    }
    div[class*="st-key-enter_app"] button:hover {
        transform: translateY(-2px); box-shadow: 0 12px 32px rgba(34,181,115,0.5) !important;
    }
    .welcome-cta-hint { font-size: 11px; color: var(--text-dim); margin-top: 10px; font-family: 'JetBrains Mono', monospace; }

    div[class*="st-key-skip_intro"] button {
        background: transparent !important; border: none !important; color: var(--text-dim) !important;
        font-size: 12.5px !important; text-decoration: underline; text-underline-offset: 3px;
        box-shadow: none !important; padding: 6px 4px !important;
    }
    div[class*="st-key-skip_intro"] button:hover { color: var(--pitch) !important; }

    /* Live data-driven insight pill */
    .insight-pill {
        display: inline-flex; align-items: center; gap: 8px; margin-top: 22px;
        background: rgba(216,151,60,0.08); border: 1px solid rgba(216,151,60,0.30);
        padding: 7px 16px; border-radius: 999px; font-size: 12.5px; color: var(--text-mid);
    }
    .insight-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--leather);
        box-shadow: 0 0 8px 1px rgba(216,151,60,0.7); flex-shrink: 0; animation: floodlight-pulse 2s ease-in-out infinite; }
    .insight-pill b { color: var(--text-hi); font-family: 'JetBrains Mono', monospace; }

    /* Single-line tagline strip — previously a CSS-only rotating/fading
       animation using absolutely-positioned stacked spans, which broke
       inside Streamlit's HTML rendering (collapsed into a vertical
       single-character column). Replaced with one guaranteed-safe
       static line: no animation, no absolute positioning, no risk. */
    .tagline-strip {
        text-align: center; margin: 2px 0 4px 0;
        font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--sky);
        white-space: normal;
    }
    .tagline-strip .sep { color: var(--text-dim); margin: 0 10px; }

    /* Three-step "how it works" strip */
    .onboard-row { display: flex; gap: 16px; justify-content: center; margin: 34px auto 6px auto; max-width: 900px; flex-wrap: wrap; }
    .onboard-card {
        flex: 1 1 240px; max-width: 280px; text-align: left;
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .onboard-card:hover { transform: translateY(-3px); border-color: var(--fc-accent, var(--pitch)); }
    .onboard-card .step-n {
        font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 700;
        color: var(--fc-accent, var(--pitch)); letter-spacing: 1px;
    }
    .onboard-card .step-icon { font-size: 22px; margin: 8px 0 8px 0; }
    .onboard-card .step-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 14.5px;
        color: var(--text-hi); margin-bottom: 5px;
    }
    .onboard-card .step-desc { font-size: 12px; color: var(--text-dim); line-height: 1.5; }

    /* ---------- ACWR Engine page ---------- */
    .acwr-hero {
        position: relative; overflow: hidden; border-radius: 18px; padding: 28px 30px;
        background: linear-gradient(120deg, #0a3826 0%, #135a3d 40%, #1c8455 75%, #22b573 100%);
        border: 1px solid rgba(255,255,255,0.10); margin-bottom: 18px;
    }
    .acwr-hero::after {
        content: "\1F9EC"; position: absolute; right: 26px; top: 50%; transform: translateY(-50%);
        font-size: 76px; opacity: 0.12; pointer-events: none;
    }
    .acwr-hero .eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
        letter-spacing: 1.6px; text-transform: uppercase; color: #bdeecf; opacity: 0.9;
    }
    .acwr-hero h2 {
        color: #fbfbf8; margin: 6px 0 8px 0; font-size: 26px; font-weight: 700;
        font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.4px;
    }
    .acwr-hero p { color: #d7e8dd; margin: 0; font-size: 14px; max-width: 640px; line-height: 1.55; }
    .acwr-formula {
        display: inline-block; margin-top: 14px; font-family: 'JetBrains Mono', monospace;
        font-size: 13.5px; color: #eafaf0; background: rgba(0,0,0,0.22);
        border: 1px solid rgba(255,255,255,0.18); border-radius: 8px; padding: 8px 16px;
    }

    .zone-bar-track {
        position: relative; height: 34px; border-radius: 8px; overflow: hidden; display: flex;
        margin: 10px 0 6px 0; border: 1px solid var(--border);
    }
    .zone-seg { display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px; font-weight: 700; color: rgba(255,255,255,0.85); letter-spacing: 0.3px; }
    .zone-marker {
        position: absolute; top: -7px; width: 0; height: 0; transform: translateX(-50%);
        border-left: 7px solid transparent; border-right: 7px solid transparent; border-top: 9px solid var(--text-hi);
        transition: left 0.15s ease;
    }
    .zone-labels { display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim);
        font-family: 'JetBrains Mono', monospace; margin-bottom: 18px; }

    .verdict-card {
        border-radius: 14px; padding: 18px 20px; margin-top: 4px; border: 1px solid var(--border);
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
    }
    .verdict-card .v-tier { font-family: 'Space Grotesk', sans-serif; font-size: 19px; font-weight: 700; }
    .verdict-card .v-text { font-size: 13px; color: var(--text-mid); margin-top: 6px; line-height: 1.55; }

    .trend-badge {
        display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px; font-weight: 700;
        padding: 3px 10px; border-radius: 999px; font-family: 'JetBrains Mono', monospace;
    }
    .trend-up { background: rgba(209,72,63,0.14); color: #f0857c; border: 1px solid rgba(209,72,63,0.4); }
    .trend-down { background: rgba(34,181,115,0.12); color: #5be3a3; border: 1px solid rgba(34,181,115,0.35); }
    .trend-flat { background: rgba(154,164,178,0.10); color: #9aa4b2; border: 1px solid rgba(154,164,178,0.3); }

    /* Squad-status count cards (tab_status) — CSS was previously missing
       entirely, so these rendered with no background/border at all. */
    .tier-card {
        border: 1px solid var(--border); border-radius: 14px; padding: 16px 14px;
        text-align: center; transition: transform 0.15s ease;
    }
    .tier-card:hover { transform: translateY(-2px); }
    .tier-count {
        font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 700; line-height: 1;
    }
    .tier-label {
        font-size: 11px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.5px; margin-top: 6px;
    }

    /* Return-to-Play ramp cards */
    .ramp-card {
        border: 1px solid var(--border); border-radius: 14px; padding: 16px;
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
        text-align: center; position: relative;
    }
    .ramp-card .ramp-phase {
        font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 700;
        color: var(--pitch); text-transform: uppercase; letter-spacing: 0.6px;
    }
    .ramp-card .ramp-overs { font-size: 26px; font-weight: 700; color: var(--text-hi); margin: 8px 0 2px 0; }
    .ramp-card .ramp-pct { font-size: 12px; color: var(--text-dim); }
    .ramp-card .ramp-date { font-size: 11px; color: var(--text-dim); margin-top: 8px; font-family: 'JetBrains Mono', monospace; }
    .ramp-arrow { text-align: center; color: var(--border); font-size: 20px; padding-top: 30px; }

    /* Fitness passport live preview frame */
    .passport-frame-wrap {
        border: 1px solid var(--border); border-radius: 14px; overflow: hidden; margin-top: 6px;
    }

    /* Home page callout pointing to Injury & Fitness */
    .injury-callout {
        display: flex; align-items: center; justify-content: space-between; gap: 18px; flex-wrap: wrap;
        background: linear-gradient(120deg, #3a1414 0%, #5c1f1f 50%, #7a2b1f 100%);
        border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; padding: 20px 26px;
        margin: 4px 0 22px 0;
    }
    .injury-callout .ic-text .eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
        letter-spacing: 1.6px; text-transform: uppercase; color: #ffcbb3; opacity: 0.9;
    }
    .injury-callout .ic-text h3 {
        color: #fff8f4; margin: 5px 0 4px 0; font-size: 19px; font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }
    .injury-callout .ic-text p { color: #ffe3d5; margin: 0; font-size: 13px; max-width: 480px; }
    div[class*="st-key-goto_injury"] button {
        background: rgba(0,0,0,0.25) !important; border: 1px solid rgba(255,255,255,0.28) !important;
        color: #fff !important; font-weight: 700 !important; white-space: nowrap !important;
    }
    div[class*="st-key-goto_injury"] button:hover { background: rgba(0,0,0,0.4) !important; }

    /* ---------- Assessment Guide chat interface ---------- */
    .chat-user-bubble {
        background: linear-gradient(120deg, #145c3f, #1f7a52); color: #fff;
        border-radius: 14px 14px 3px 14px; padding: 12px 16px; margin: 14px 0 6px auto;
        max-width: 78%; font-size: 13.5px; line-height: 1.5;
    }
    .chat-assistant-bubble {
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border); border-radius: 14px 14px 14px 3px;
        padding: 18px 20px; margin: 0 auto 22px 0; max-width: 92%; font-size: 13.5px; line-height: 1.6;
    }
    .chat-avatar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
    .chat-avatar {
        width: 26px; height: 26px; border-radius: 7px; display: flex; align-items: center;
        justify-content: center; font-size: 13px; background: rgba(34,181,115,0.16); flex-shrink: 0;
    }
    .chat-avatar-label {
        font-family: 'JetBrains Mono', monospace; font-size: 10.5px; font-weight: 700;
        letter-spacing: 0.6px; text-transform: uppercase; color: var(--pitch);
    }
    .redflag-banner {
        background: rgba(209,72,63,0.12); border: 1px solid rgba(209,72,63,0.45);
        border-radius: 10px; padding: 12px 16px; margin-bottom: 14px; color: #f0857c; font-size: 13px;
    }
    .redflag-banner b { color: #ff9d94; }
    .protocol-step {
        display: flex; gap: 10px; margin-bottom: 8px; align-items: flex-start;
    }
    .protocol-step .p-letter {
        width: 22px; height: 22px; border-radius: 6px; background: rgba(79,159,216,0.15);
        color: #4f9fd8; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 11px;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px;
    }
    .similar-case-row {
        border-top: 1px dashed var(--border-soft); padding: 8px 0; font-size: 12.5px; color: var(--text-mid);
    }
    .disclaimer-box {
        background: rgba(154,164,178,0.08); border: 1px solid var(--border-soft); border-radius: 8px;
        padding: 10px 14px; margin-top: 14px; font-size: 11.5px; color: var(--text-dim); line-height: 1.5;
    }

    /* ---------- Squad Impact Engine banner — violet/blue, distinct from
       both the green home theme and the red Injury & Fitness banner ---------- */
    .impact-header {
        background:
            radial-gradient(circle at 85% -40%, rgba(255,255,255,0.10), transparent 55%),
            linear-gradient(120deg, #1a1a3d 0%, #2d2360 45%, #3d2a7a 80%, #5138a8 100%);
        padding: 30px 34px; border-radius: 18px; margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 34px rgba(0,0,0,0.40);
        position: relative; overflow: hidden;
    }
    .impact-header::after {
        content: "🔄"; position: absolute; right: 24px; top: 50%; transform: translateY(-50%) rotate(-8deg);
        font-size: 84px; opacity: 0.10; pointer-events: none;
    }
    .impact-header .eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
        letter-spacing: 1.6px; text-transform: uppercase; color: #d8cfff; opacity: 0.9;
    }
    .impact-header h1 {
        color: #fbfbf8; margin: 4px 0 0 0; font-size: 30px; font-weight: 700;
        letter-spacing: -0.6px; font-family: 'Space Grotesk', sans-serif;
    }
    .impact-header p { color: #e5deff; margin: 7px 0 0 0; font-size: 14.5px; max-width: 700px; }

    /* Recovery Debt Ledger — deliberately looks like a financial
       statement (incurred / repayment / balance) rather than a tile
       grid, since the metaphor is the whole point of the feature. */
    .ledger-card {
        background: linear-gradient(160deg, var(--panel-2) 0%, var(--panel) 100%);
        border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; margin-bottom: 16px;
    }
    .ledger-row {
        display: flex; justify-content: space-between; align-items: baseline;
        padding: 9px 0; border-bottom: 1px dashed var(--border-soft); font-size: 13px;
    }
    .ledger-row:last-child { border-bottom: none; }
    .ledger-row .l-label { color: var(--text-mid); }
    .ledger-row .l-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--text-hi); }
    .ledger-balance {
        display: flex; justify-content: space-between; align-items: center;
        margin-top: 12px; padding-top: 14px; border-top: 2px solid var(--border);
    }
    .ledger-balance .l-label { font-size: 13.5px; font-weight: 700; color: var(--text-hi); }
    .ledger-balance .l-value { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 700; }
    .debt-bar-track { height: 6px; background: var(--border-soft); border-radius: 3px; overflow: hidden; margin: 4px 0 2px 0; }
    .debt-bar-fill { height: 100%; border-radius: 3px; }
    .cascade-team-tag {
        display: inline-block; background: rgba(81,56,168,0.16); color: #b8a4f5;
        border: 1px solid rgba(81,56,168,0.4); border-radius: 999px; padding: 3px 12px;
        font-size: 11px; font-weight: 700; letter-spacing: 0.4px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <div class="eyebrow">LIVE ANALYTICS · BALL-BY-BALL DATA</div>
  <h1>🏏 Bowler Workload Platform</h1>
  <p>Real ball-by-ball performance analytics + ACWR-based workload monitoring, powered by real match data.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="seam"></div>', unsafe_allow_html=True)

PALETTE = ["#1f7a52", "#b04a4a", "#2f7ab0", "#c9a227", "#7a52b0", "#52a3b0"]

# ==================================================================
# HELPER FUNCTIONS
# ==================================================================


def acwr_tier(acwr):
    if pd.isna(acwr):
        return "Unknown"
    if acwr < 0.8:
        return "Undertrained"
    elif acwr <= 1.3:
        return "Low"
    elif acwr <= 1.5:
        return "Moderate"
    else:
        return "High"


def tier_badge(tier):
    cls = {
        "Low": "badge-low", "Moderate": "badge-moderate",
        "High": "badge-high", "Undertrained": "badge-under", "Unknown": "badge-moderate"
    }.get(tier, "badge-moderate")
    return f'<span class="badge {cls}">{tier}</span>'


# ------------------------------------------------------------------
# INJURY LOG — persistent, real system-of-record (not just analytics)
#
# There is no public dataset of real bowler injuries (checked — see
# About This Project page), so this is the actual missing ground truth:
# a local CSV log that staff add real cases to over time. It persists
# on disk across app restarts (unlike st.session_state or
# @st.cache_data, which reset). A "seed demo data" option is provided
# separately so the UI isn't empty before real cases exist.
# ------------------------------------------------------------------

INJURY_LOG_PATH = "injury_log.csv"
INJURY_LOG_COLUMNS = [
    "log_id", "bowler", "display_name", "date_reported", "status",
    "injury_type", "body_part", "expected_return_date", "notes", "source"
]
STATUS_OPTIONS = ["Fit", "Managed", "Injured", "Rehab"]
BODY_PART_OPTIONS = [
    "Lower back", "Shoulder", "Knee", "Ankle", "Hamstring", "Quadriceps",
    "Groin", "Side/abdominal", "Elbow", "Foot/stress fracture", "Other"
]

# ==================================================================
# ASSESSMENT GUIDE — curated, general sports-medicine reference content.
# This is deliberately NOT an open-ended diagnostic chatbot: inputs are
# fixed dropdowns/checklists, not free text, and outputs are general
# educational protocol information — never a diagnosis or prescription.
# Content reflects well-established, publicly documented sports-medicine
# concepts (POLICE protocol, standard red-flag escalation criteria, and
# widely-published injury patterns specific to fast bowling). It is not a
# substitute for assessment by a qualified medical professional.
# ==================================================================

PROTOCOL_KNOWLEDGE = {
    "Lower back": {
        "mechanism": "Repeated hyperextension and rotation during the bowling action — especially "
                     "with a mixed or front-on action — is a well-documented cause of lumbar stress "
                     "injury (spondylolysis) in young fast bowlers.",
        "considerations": "Lumbar stress fractures often don't show on an early X-ray and may need an "
                           "MRI to confirm. Typical recovery is measured in months, not weeks, and is "
                           "usually managed by a sports physician, not just rest.",
        "red_flags": ["Pain radiating down one or both legs", "Numbness, tingling, or weakness in the legs",
                      "Any loss of bladder or bowel control (seek emergency care immediately)",
                      "Pain that wakes them at night"],
    },
    "Shoulder": {
        "mechanism": "High, repetitive overhead loading through the bowling action and follow-through "
                     "places cumulative stress on the rotator cuff and shoulder labrum.",
        "considerations": "Usually develops gradually from cumulative load rather than one single event — "
                           "a technique review alongside workload management is standard alongside rest.",
        "red_flags": ["Visible deformity", "Sudden inability to lift the arm", "Numbness down the arm",
                      "Night pain that prevents sleep"],
    },
    "Knee": {
        "mechanism": "High impact loading through the front (landing) leg at back-foot contact and "
                     "through delivery is the main driver of front-knee issues in fast bowlers.",
        "considerations": "Recurring front-knee soreness is often linked to landing technique (a very "
                           "stiff front leg increases impact force) as much as to overall workload.",
        "red_flags": ["Locking or the knee giving way", "Visible swelling within hours", "Inability to bear weight"],
    },
    "Ankle": {
        "mechanism": "Landing forces and lateral movement during the delivery stride and follow-through, "
                     "either as an acute roll/sprain or a repetitive-stress issue.",
        "considerations": "Recurring ankle issues often trace back to landing mechanics, footwear, or "
                           "the bowling surface rather than being isolated bad luck.",
        "red_flags": ["Inability to bear any weight at all", "Visible deformity", "Significant swelling within minutes"],
    },
    "Hamstring": {
        "mechanism": "The explosive front-leg braking action at high run-up speed is a classic mechanism "
                     "for hamstring strain.",
        "considerations": "Hamstring strains have a genuinely high re-injury rate if a bowler returns "
                           "before full rehabilitation — one of the clearest cases for a graded, "
                           "staged return rather than an all-or-nothing comeback.",
        "red_flags": ["An audible pop or snap at the moment of injury", "Significant bruising appearing quickly",
                      "Inability to walk normally"],
    },
    "Quadriceps": {
        "mechanism": "Deceleration forces through the front leg at landing place high eccentric load "
                     "on the quadriceps.",
        "considerations": "Shares a similar re-injury risk profile to hamstring strains — a graded "
                           "return is important here too.",
        "red_flags": ["Significant swelling", "Inability to bend or straighten the knee"],
    },
    "Groin": {
        "mechanism": "Rotational load and the wide delivery stride combine to place stress through the "
                     "hip and groin region during the bowling action.",
        "considerations": "Often connects to hip and pelvis stability and technique rather than being "
                           "an isolated muscle strain — worth a broader assessment, not just the sore spot.",
        "red_flags": ["Severe pain with any hip movement", "Inability to walk"],
    },
    "Side/abdominal": {
        "mechanism": "The classic fast-bowler 'side strain' — injury to the internal oblique muscles "
                     "from the rapid trunk rotation and lateral flexion in the bowling action. One of "
                     "the most well-documented bowling-specific injuries in the sports-medicine literature.",
        "considerations": "Side strains are notorious for long recovery (commonly 4–8+ weeks) and a high "
                           "recurrence rate if a bowler returns to full loading too early.",
        "red_flags": ["Sharp pain with breathing", "Pain that prevents any twisting or reaching overhead"],
    },
    "Elbow": {
        "mechanism": "Repetitive extension load at release, which can be more pronounced with certain "
                     "grip types or bowling actions.",
        "considerations": "Usually a gradual-onset overuse pattern rather than a single acute event.",
        "red_flags": ["Visible swelling or deformity", "The joint locking"],
    },
    "Foot/stress fracture": {
        "mechanism": "Repetitive high-impact loading through the front and back foot at delivery, "
                     "particularly when bowling volume increases quickly.",
        "considerations": "A classic overuse injury directly linked to workload spikes — exactly the "
                           "pattern ACWR is designed to flag. A sudden jump in overs bowled is a "
                           "well-known risk factor for stress fractures specifically.",
        "red_flags": ["Pain that persists at rest or worsens over consecutive sessions despite reduced load",
                      "One specific point that's tender to touch"],
    },
    "Other": {
        "mechanism": "General guidance only — the specific mechanism depends on the area involved.",
        "considerations": "For anything outside the areas listed here, a professional assessment is "
                           "especially important since general guidance can't cover every case.",
        "red_flags": ["Any sudden, severe, or worsening pain", "Any loss of function"],
    },
}

SYMPTOM_OPTIONS = [
    "Sudden onset during a delivery",
    "Gradual onset over several sessions",
    "Pain only while bowling",
    "Pain persists at rest / after bowling",
    "Swelling or visible change in the area",
    "Numbness, tingling, or weakness",
    "Reduced range of motion",
    "Audible pop or snap at the moment of injury",
    "Pain wakes them at night",
    "Same area has been injured before",
]

RED_FLAG_SYMPTOMS = {
    "Pain persists at rest / after bowling",
    "Swelling or visible change in the area",
    "Numbness, tingling, or weakness",
    "Audible pop or snap at the moment of injury",
    "Pain wakes them at night",
}


def generate_assessment_response(body_part, severity, symptoms, bowler_name, injury_log_df):
    """Builds the structured, rule-based 'assistant' response — general
    educational protocol content plus a lookup of similar logged cases.
    Nothing here is generated by a language model or trained on the log;
    it's a fixed knowledge base keyed by the selections, so there's no
    risk of a fabricated or hallucinated recommendation."""
    kb = PROTOCOL_KNOWLEDGE.get(body_part, PROTOCOL_KNOWLEDGE["Other"])
    triggered_flags = [s for s in symptoms if s in RED_FLAG_SYMPTOMS]
    is_urgent = severity == "Severe" or len(triggered_flags) >= 2

    parts = ['<div class="chat-assistant-bubble">']
    parts.append(
        '<div class="chat-avatar-row"><div class="chat-avatar">🩺</div>'
        '<div class="chat-avatar-label">Assessment Guide</div></div>'
    )

    if is_urgent or triggered_flags:
        flag_list = "".join(f"<li>{f}</li>" for f in triggered_flags) if triggered_flags else "<li>Severity marked as Severe</li>"
        parts.append(
            '<div class="redflag-banner"><b>⚠️ This combination includes signs worth prompt medical attention:</b>'
            f'<ul style="margin:6px 0 0 18px;">{flag_list}</ul>'
            'This tool cannot rule anything in or out — please have this assessed by a doctor or '
            'physiotherapist rather than relying on the guidance below.</div>'
        )

    parts.append(f'<p><b>{body_part} — likely mechanism in bowling</b><br>{kb["mechanism"]}</p>')
    parts.append(f'<p><b>Specific considerations</b><br>{kb["considerations"]}</p>')

    parts.append('<p><b>General first-response protocol (POLICE)</b></p>')
    police_steps = [
        ("P", "Protection", "Stop bowling. Avoid movements that reproduce the pain."),
        ("O", "Optimal Loading", "Gentle, pain-free movement is usually better than complete immobility once the acute phase passes — a medical/physio assessment should guide how much."),
        ("I", "Ice", "Ice for ~15–20 minutes at a time in the first 24–48 hours can help with acute pain and swelling."),
        ("C", "Compression", "A light compression wrap can help manage swelling where appropriate for the area."),
        ("E", "Elevation", "Elevate the area above heart level where practical, especially in the first day or two."),
    ]
    for letter, name, desc in police_steps:
        parts.append(
            f'<div class="protocol-step"><div class="p-letter">{letter}</div>'
            f'<div><b>{name}</b> — {desc}</div></div>'
        )

    severity_notes = {
        "Minor": "Minor + no red flags: monitor over the next 1–2 sessions, keep workload conservative, "
                 "and re-assess before returning to full bowling load.",
        "Moderate": "Moderate: recommend reducing bowling load significantly and arranging a "
                    "physiotherapy assessment rather than self-managing through it.",
        "Severe": "Severe: stop all bowling immediately and arrange a medical assessment promptly — "
                  "don't wait to see if it settles on its own.",
    }
    parts.append(f'<p><b>Given the selected severity</b><br>{severity_notes.get(severity, severity_notes["Moderate"])}</p>')

    if kb["red_flags"]:
        rf_list = "".join(f"<li>{f}</li>" for f in kb["red_flags"])
        parts.append(
            f'<p><b>Signs that would always warrant urgent professional review for this area</b>'
            f'<ul style="margin:6px 0 0 18px;">{rf_list}</ul></p>'
        )

    # ---- Similar cases from the log ----
    same_part = injury_log_df[injury_log_df["body_part"] == body_part] if not injury_log_df.empty else pd.DataFrame()
    if bowler_name and bowler_name != "General / no specific bowler":
        bowler_history = same_part[same_part["display_name"] == bowler_name]
    else:
        bowler_history = pd.DataFrame()

    parts.append('<p><b>Similar cases in your log</b></p>')
    if not bowler_history.empty:
        rows_html = "".join(
            f'<div class="similar-case-row">{r.date_reported} — <b>{r.display_name}</b>, {r.injury_type or "unspecified"} '
            f'({r.status}){" — " + r.notes if r.notes else ""}</div>'
            for r in bowler_history.sort_values("date_reported", ascending=False).itertuples()
        )
        parts.append(f'<div>{rows_html}</div>')
    elif not same_part.empty:
        rows_html = "".join(
            f'<div class="similar-case-row">{r.date_reported} — <b>{r.display_name}</b>, {r.injury_type or "unspecified"} '
            f'({r.status})</div>'
            for r in same_part.sort_values("date_reported", ascending=False).head(5).itertuples()
        )
        parts.append(f'<div>No history for this specific bowler at {body_part}. Other logged {body_part} cases in your squad:</div><div>{rows_html}</div>')
    else:
        parts.append('<div style="color:var(--text-dim);">No logged cases for this body part yet — this would be the first.</div>')

    parts.append(
        '<div class="disclaimer-box">This is general educational reference information, not a medical '
        'diagnosis or treatment plan, and it is not based on a validated clinical model — it\'s fixed '
        'reference content plus a lookup of your own logged cases. Always have real injuries assessed '
        'by a qualified doctor or physiotherapist.</div>'
    )
    parts.append('</div>')
    return "".join(parts)


def load_injury_log():
    if os.path.exists(INJURY_LOG_PATH):
        try:
            df = pd.read_csv(INJURY_LOG_PATH)
            for col in INJURY_LOG_COLUMNS:
                if col not in df.columns:
                    df[col] = "" if col != "log_id" else 0
            return df[INJURY_LOG_COLUMNS]
        except Exception as e:
            logger.error(f"Failed to read {INJURY_LOG_PATH}, starting from an empty log: {e}")
            st.warning(
                f"⚠️ Couldn't read the existing injury log ({e}) — starting from an empty "
                "log for this session rather than losing the page entirely. The file on "
                "disk hasn't been overwritten yet, so it may still be recoverable."
            )
    return pd.DataFrame(columns=INJURY_LOG_COLUMNS)


def save_injury_log(df):
    try:
        df.to_csv(INJURY_LOG_PATH, index=False)
        logger.info(f"Saved injury log ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Failed to save {INJURY_LOG_PATH}: {e}")
        st.error(f"⚠️ Couldn't save the injury log to disk: {e}. Your last change may not have persisted.")


def add_injury_log_entry(bowler, display_name, status, injury_type, body_part,
                          expected_return_date, notes, source="manual"):
    df = load_injury_log()
    next_id = int(df["log_id"].max()) + 1 if not df.empty else 1
    new_row = pd.DataFrame([{
        "log_id": next_id, "bowler": bowler, "display_name": display_name,
        "date_reported": pd.Timestamp.today().strftime("%Y-%m-%d"), "status": status,
        "injury_type": injury_type, "body_part": body_part,
        "expected_return_date": str(expected_return_date) if expected_return_date else "",
        "notes": notes, "source": source,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_injury_log(df)
    return df


def get_current_status(bowler_key, injury_log_df):
    """The latest logged status for a bowler, defaulting to Fit if no
    entries exist yet. This is deliberately a lookup of the most recent
    human-entered record, not an inference — staff log a new 'Fit' entry
    when a player is cleared, same as a real medical system of record."""
    if injury_log_df.empty:
        return "Fit"
    entries = injury_log_df[injury_log_df["bowler"] == bowler_key]
    if entries.empty:
        return "Fit"
    entries = entries.sort_values("date_reported")
    return entries.iloc[-1]["status"]


def status_badge(status):
    cls = {"Fit": "badge-fit", "Managed": "badge-managed",
           "Injured": "badge-injured", "Rehab": "badge-rehab"}.get(status, "badge-fit")
    icon = {"Fit": "✅", "Managed": "🟡", "Injured": "🚑", "Rehab": "🧑‍⚕️"}.get(status, "✅")
    return f'<span class="badge {cls}">{icon} {status}</span>'


def safe_overs_ceiling(chronic_avg_overs):
    """A concrete, actionable number instead of just a risk label: the
    upper edge of the ACWR 'sweet spot' (1.3x chronic average) — bowling
    up to roughly this many overs keeps this match's ACWR out of the
    risk zone, given this player's recent normal workload."""
    if pd.isna(chronic_avg_overs) or chronic_avg_overs <= 0:
        return None
    return round(chronic_avg_overs * 1.3, 1)


def seed_demo_injury_data(bowler_master_df, n=8, seed=123):
    """Populates the injury log with clearly-labeled synthetic cases so
    the status badges / correlation views aren't empty for a demo or
    pitch, before real cases exist. Marked source='demo' so they can be
    told apart from real entries and cleared independently."""
    rng = np.random.RandomState(seed)
    if bowler_master_df.empty:
        return load_injury_log()
    sample_n = min(n, len(bowler_master_df))
    sample = bowler_master_df.sample(sample_n, random_state=seed)
    injury_types = ["Stress fracture", "Muscle strain", "Ligament sprain",
                     "Tendinitis", "Soreness / fatigue", "Impact injury"]
    statuses = ["Injured", "Rehab", "Managed", "Fit"]
    df = load_injury_log()
    next_id = int(df["log_id"].max()) + 1 if not df.empty else 1
    rows = []
    for _, prow in sample.iterrows():
        status = rng.choice(statuses, p=[0.3, 0.25, 0.25, 0.2])
        days_ago = int(rng.randint(1, 60))
        return_days = int(rng.randint(7, 45))
        report_date = pd.Timestamp.today() - pd.Timedelta(days=days_ago)
        return_date = pd.Timestamp.today() + pd.Timedelta(days=return_days)
        rows.append({
            "log_id": next_id, "bowler": prow["bowler"], "display_name": prow["full_name"],
            "date_reported": report_date.strftime("%Y-%m-%d"), "status": status,
            "injury_type": rng.choice(injury_types), "body_part": rng.choice(BODY_PART_OPTIONS),
            "expected_return_date": return_date.strftime("%Y-%m-%d"),
            "notes": "Demo/seed record for presentation purposes.", "source": "demo",
        })
        next_id += 1
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    save_injury_log(df)
    return df


def clear_injury_log(source_filter=None):
    """Clears the whole log, or only rows matching a given source
    (e.g. clear only 'demo' rows and keep real entries intact)."""
    if source_filter is None:
        save_injury_log(pd.DataFrame(columns=INJURY_LOG_COLUMNS))
    else:
        df = load_injury_log()
        df = df[df["source"] != source_filter]
        save_injury_log(df)


# ------------------------------------------------------------------
# MULTI-FACTOR RISK — ACWR alone is one signal. Real sports science
# (Foster, 1998) pairs it with "monotony" (how repetitive recent load
# is — low variation is itself a risk factor, not just high load) and
# "strain" (monotony x total load). Computed here from match-to-match
# overs bowled, since day-by-day training data isn't available.
# ------------------------------------------------------------------

def compute_monotony_strain(player_matches, window=7):
    """Returns (monotony, strain) from a player's last `window` matches,
    or (None, None) if there's too little data or zero variance (which
    would make the ratio undefined rather than meaningfully infinite)."""
    if player_matches is None or player_matches.empty:
        return None, None
    recent = player_matches.sort_values('start_date').tail(window)
    if len(recent) < 2:
        return None, None
    mean_load = recent['overs_bowled'].mean()
    std_load = recent['overs_bowled'].std()
    monotony = (mean_load / std_load) if (std_load and std_load > 0) else None
    total_load = recent['overs_bowled'].sum()
    strain = (monotony * total_load) if monotony is not None else None
    return monotony, strain


@st.cache_data
def compute_monotony_strain_all(match_summary_df, window=7):
    """Monotony/strain for every bowler in one pass via groupby, instead
    of filtering the full match-history dataframe once per bowler (which
    was O(bowlers x records) and, at real dataset scale — thousands of
    bowlers, well over a hundred thousand match records — was a major,
    silent source of slowness on every single page interaction)."""
    results = {}
    for bowler, grp in match_summary_df.sort_values('start_date').groupby('bowler'):
        recent = grp.tail(window)
        if len(recent) < 2:
            results[bowler] = (None, None)
            continue
        mean_load = recent['overs_bowled'].mean()
        std_load = recent['overs_bowled'].std()
        monotony = (mean_load / std_load) if (std_load and std_load > 0) else None
        total_load = recent['overs_bowled'].sum()
        strain = (monotony * total_load) if monotony is not None else None
        results[bowler] = (monotony, strain)
    return results


def monotony_tier(monotony):
    if monotony is None or pd.isna(monotony):
        return "Unknown"
    if monotony < 1.5:
        return "Varied"
    elif monotony <= 2.0:
        return "Moderate"
    else:
        return "High"


# ------------------------------------------------------------------
# ALERTS — one prioritized list of everything that needs a human
# decision this week, instead of staff having to check every page.
# ------------------------------------------------------------------

@st.cache_data
def compute_alerts(latest_state_df, match_summary_df, injury_log_df):
    monotony_map = compute_monotony_strain_all(match_summary_df)
    alerts = []
    for row in latest_state_df.itertuples():
        monotony, strain = monotony_map.get(row.bowler, (None, None))

        if row.acwr_tier == "High":
            alerts.append({
                "priority": 1, "type": "High ACWR", "display_name": row.display_name,
                "detail": f"ACWR {row.acwr:.2f} — bowling well above their recent normal.",
                "color": "#f08080"
            })
        if pd.notna(getattr(row, "rest_days_before", None)) and row.rest_days_before > 20:
            alerts.append({
                "priority": 2, "type": "Deconditioning risk", "display_name": row.display_name,
                "detail": f"{row.rest_days_before:.0f} days since their last recorded match — "
                          "returning to full load too fast after a long gap is itself a risk.",
                "color": "#8ec6ee"
            })
        if monotony is not None and monotony > 2.0:
            alerts.append({
                "priority": 2, "type": "High monotony", "display_name": row.display_name,
                "detail": f"Monotony {monotony:.2f} — recent workload has been unusually "
                          "repetitive match to match, which compounds strain.",
                "color": "#e8c15a"
            })
        status = getattr(row, "current_status", "Fit")
        if status in ("Injured", "Rehab") and row.acwr_tier in ("Moderate", "High"):
            alerts.append({
                "priority": 0, "type": "Status conflict", "display_name": row.display_name,
                "detail": f"Logged as {status}, but their most recent recorded workload "
                          f"(ACWR {row.acwr:.2f}) suggests they're still bowling meaningful overs. Verify status.",
                "color": "#c9a9f5"
            })
    return sorted(alerts, key=lambda a: a["priority"])


# ------------------------------------------------------------------
# RETURN-TO-PLAY RAMP PLANNER — a graded overs progression back to
# full workload, instead of jumping straight back to normal (which is
# exactly the ACWR-spike pattern this app already flags as risky).
# ------------------------------------------------------------------

def generate_ramp_plan(base_overs, start_date=None):
    if start_date is None or pd.isna(start_date):
        start_date = pd.Timestamp.today()
    else:
        start_date = pd.Timestamp(start_date)
    phases = [("Week 1", 0.25), ("Week 2", 0.50), ("Week 3", 0.75), ("Week 4 (full)", 1.00)]
    rows = []
    for i, (label, pct) in enumerate(phases):
        rows.append({
            "phase": label,
            "target_date": (start_date + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "target_overs": round((base_overs or 0) * pct, 1),
            "pct_of_normal": f"{pct*100:.0f}%",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# SQUAD IMPACT ENGINE — three functions that go beyond assessing one
# bowler in isolation: what an absence does to the rest of the team
# (cascade), what a layoff actually costs in lost conditioning (debt),
# and what to concretely do about either (substitution). All three are
# explicit projections built from real workload numbers already in the
# dataset — not machine-learned, and clearly framed as estimates.
# ------------------------------------------------------------------

def compute_cascade_risk(latest_state_df, injury_log_df):
    """When a bowler is out, their overs don't disappear — someone else
    on the team absorbs them, usually the bowlers already carrying the
    most load. This projects, per team, how much extra work each fit
    teammate would likely take on and whether that pushes their own ACWR
    into a worse tier — the ripple effect that's invisible until it shows
    up as *their* injury a few weeks later."""
    if latest_state_df.empty or "bowling_team" not in latest_state_df.columns:
        return pd.DataFrame()

    results = []
    for team, team_df in latest_state_df.groupby("bowling_team"):
        out_bowlers = team_df[team_df["current_status"].isin(["Injured", "Managed"])]
        if out_bowlers.empty:
            continue
        fit_bowlers = team_df[team_df["current_status"] == "Fit"].copy()
        fit_bowlers = fit_bowlers.dropna(subset=["chronic_avg_overs"])
        fit_bowlers = fit_bowlers[fit_bowlers["chronic_avg_overs"] > 0]
        if fit_bowlers.empty:
            continue

        gap_overs = out_bowlers["chronic_avg_overs"].fillna(0).sum()
        total_fit_load = fit_bowlers["chronic_avg_overs"].sum()
        if total_fit_load <= 0:
            continue

        for row in fit_bowlers.itertuples():
            share = row.chronic_avg_overs / total_fit_load
            extra_overs = gap_overs * share
            projected_overs = row.chronic_avg_overs + extra_overs
            projected_acwr = projected_overs / row.chronic_avg_overs if row.chronic_avg_overs > 0 else None
            current_tier = row.acwr_tier
            projected_tier = acwr_tier(projected_acwr) if projected_acwr is not None else current_tier
            tier_rank = {"Undertrained": 0, "Low": 1, "Moderate": 2, "High": 3}
            worsens = tier_rank.get(projected_tier, 0) > tier_rank.get(current_tier, 0)
            results.append({
                "bowling_team": team, "display_name": row.display_name, "bowler": row.bowler,
                "out_bowlers": ", ".join(out_bowlers["display_name"].tolist()),
                "gap_overs": round(gap_overs, 1), "current_chronic": round(row.chronic_avg_overs, 1),
                "extra_overs": round(extra_overs, 1), "projected_overs": round(projected_overs, 1),
                "current_acwr": row.acwr, "current_tier": current_tier,
                "projected_acwr": projected_acwr, "projected_tier": projected_tier,
                "worsens": worsens,
            })
    return pd.DataFrame(results).sort_values("worsens", ascending=False) if results else pd.DataFrame()


def compute_recovery_debt(case_row, match_summary_df):
    """Frames time out as a debt to be repaid, not just days to wait out.
    A bowler who was carrying a heavy workload before a long layoff has
    lost more bowling-specific conditioning than a lightly-used bowler
    out for the same number of days — 'pain-free' and 'conditioned' are
    not the same thing, and treating them as the same is exactly how
    re-injuries happen on the standard fixed-length ramp."""
    bowler_key = case_row.get("bowler")
    injury_date = pd.Timestamp(case_row.get("date_reported"))
    expected_return = case_row.get("expected_return_date")
    if pd.isna(expected_return) or str(expected_return).strip() in ("", "nan", "NaT", "None"):
        return_date = pd.Timestamp.today()
    else:
        return_date = pd.Timestamp(expected_return)
    days_out = max((return_date - injury_date).days, 1)

    pre_injury = match_summary_df[
        (match_summary_df["bowler"] == bowler_key) & (match_summary_df["start_date"] < injury_date)
    ].sort_values("start_date")
    baseline_overs = pre_injury["overs_bowled"].tail(4).mean() if not pre_injury.empty else 0

    # Debt units: days out weighted by how much conditioning they had to
    # lose relative to a nominal 6-overs/match baseline bowler. Arbitrary
    # units by design (there's no published standard here) but internally
    # consistent, so it's useful for ranking/comparison, not an absolute claim.
    debt_score = days_out * (baseline_overs / 6) if baseline_overs > 0 else days_out * 0.5
    extra_phases = max(0, round(debt_score / 15) - 4)  # standard ramp already covers ~4 units of debt
    total_ramp_weeks = 4 + extra_phases

    return {
        "bowler": bowler_key, "days_out": days_out, "baseline_overs": round(baseline_overs, 1),
        "debt_score": round(debt_score, 1), "extra_phases": extra_phases,
        "total_ramp_weeks": total_ramp_weeks, "injury_date": injury_date, "return_date": return_date,
    }


def generate_debt_aware_ramp(base_overs, total_weeks, start_date=None):
    """Like generate_ramp_plan, but the number of graded-load weeks
    stretches with the recovery debt instead of always being fixed at 4 —
    a higher-debt layoff gets a longer, more cautious ramp back."""
    if start_date is None or pd.isna(start_date):
        start_date = pd.Timestamp.today()
    else:
        start_date = pd.Timestamp(start_date)
    total_weeks = max(total_weeks, 2)
    rows = []
    for i in range(total_weeks):
        pct = (i + 1) / total_weeks
        label = f"Week {i + 1}" + (" (full)" if i == total_weeks - 1 else "")
        rows.append({
            "phase": label,
            "target_date": (start_date + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"),
            "target_overs": round((base_overs or 0) * pct, 1),
            "pct_of_normal": f"{pct * 100:.0f}%",
        })
    return pd.DataFrame(rows)


def find_substitutes(target_bowler_row, latest_state_df, top_n=3):
    """Turns a risk flag into an actual decision: concrete, ranked
    replacement options on the same team, instead of leaving 'this bowler
    is high-risk' as a dead end with no next step."""
    team = target_bowler_row.get("bowling_team")
    if not team:
        return pd.DataFrame()
    candidates = latest_state_df[
        (latest_state_df["bowling_team"] == team) &
        (latest_state_df["current_status"] == "Fit") &
        (latest_state_df["bowler"] != target_bowler_row.get("bowler"))
    ].copy()
    if candidates.empty:
        return candidates

    candidates["safety_margin"] = candidates["chronic_avg_overs"].apply(safe_overs_ceiling) - candidates["chronic_avg_overs"]
    candidates["safety_margin"] = candidates["safety_margin"].fillna(0)
    # Rank by ACWR headroom first (further from the risk zone is safer to load up),
    # then by economy as a form tiebreak.
    candidates["headroom"] = 1.3 - candidates["acwr"].fillna(1.3)
    candidates = candidates.sort_values(["headroom", "economy"], ascending=[False, True])
    return candidates.head(top_n)


# ------------------------------------------------------------------
# FITNESS PASSPORT — a single-player exportable summary meant to
# actually be handed to medical/coaching staff, not just viewed in-app.
# Self-contained HTML (own inline styles) so it looks right even opened
# outside the running app.
# ------------------------------------------------------------------

def generate_fitness_passport_html(player_row, player_matches, injury_history):
    name = player_row.get("display_name", "Unknown")
    country = player_row.get("country", "Unknown")
    style = player_row.get("bowling_style", "Unknown")
    status = player_row.get("current_status", "Fit")
    acwr = player_row.get("acwr")
    acwr_str = f"{acwr:.2f}" if pd.notna(acwr) else "N/A"

    recent_rows = ""
    for r in player_matches.sort_values("start_date", ascending=False).head(10).itertuples():
        recent_rows += (
            f"<tr><td>{r.start_date.date() if pd.notna(r.start_date) else '-'}</td>"
            f"<td>{r.batting_team}</td><td>{r.overs_bowled:.1f}</td>"
            f"<td>{r.economy:.2f}</td><td>{int(r.wickets)}</td></tr>"
        )

    injury_rows = ""
    if injury_history is not None and not injury_history.empty:
        for r in injury_history.sort_values("date_reported", ascending=False).itertuples():
            injury_rows += (
                f"<tr><td>{r.date_reported}</td><td>{r.status}</td>"
                f"<td>{r.injury_type}</td><td>{r.body_part}</td>"
                f"<td>{r.expected_return_date}</td><td>{r.notes}</td></tr>"
            )
    else:
        injury_rows = "<tr><td colspan='6' style='text-align:center;color:#888;'>No cases logged.</td></tr>"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body {{ font-family: -apple-system, Arial, sans-serif; background:#0e1117; color:#e4e8ee; padding:32px; }}
h1 {{ margin-bottom:2px; }} .meta {{ color:#9aa4b2; margin-bottom:20px; }}
.badge {{ display:inline-block; padding:4px 14px; border-radius:999px; font-weight:700; font-size:13px; }}
table {{ border-collapse:collapse; width:100%; margin:14px 0 26px 0; }}
th, td {{ border:1px solid #2a2f3a; padding:8px 10px; font-size:13px; text-align:left; }}
th {{ background:#161b22; }} h2 {{ border-bottom:1px solid #2a2f3a; padding-bottom:6px; }}
.stat {{ display:inline-block; margin-right:28px; }} .stat b {{ font-size:20px; display:block; }}
</style></head><body>
<h1>🏏 Fitness Passport — {name}</h1>
<div class="meta">{country} • {style} • Generated {pd.Timestamp.today().strftime('%Y-%m-%d')}</div>
<span class="badge" style="background:#22222240;">{status}</span>
<h2>Current workload</h2>
<div class="stat"><b>{acwr_str}</b>ACWR</div>
<div class="stat"><b>{int(player_matches['wickets'].sum())}</b>Wickets on record</div>
<div class="stat"><b>{player_matches['overs_bowled'].sum():.1f}</b>Total overs on record</div>
<h2>Recent match log</h2>
<table><tr><th>Date</th><th>Opponent</th><th>Overs</th><th>Economy</th><th>Wickets</th></tr>{recent_rows}</table>
<h2>Injury / status history</h2>
<table><tr><th>Date</th><th>Status</th><th>Type</th><th>Body part</th><th>Exp. return</th><th>Notes</th></tr>{injury_rows}</table>
<p style="color:#7a8494; font-size:12px; margin-top:30px;">
ACWR is a workload heuristic, not a validated injury predictor — no public dataset of real
outcomes exists to validate it against. Use alongside real clinical judgment.</p>
</body></html>"""


def kpi_card(value, label):
    return f"""
    <div class="kpi-card">
      <div class="value">{value}</div>
      <div class="label">{label}</div>
    </div>
    """


def page_intro(text):
    """A short, plain-English 'what is this page for' line shown under
    every page title, so the app is understandable without any cricket-
    analytics or ML background."""
    st.markdown(
        f'<div style="color:var(--text-mid); font-size:14.5px; max-width:760px; '
        f'margin:-6px 0 20px 0; line-height:1.5;">{text}</div>',
        unsafe_allow_html=True
    )


_section_slug_counts = {}


def _section_key(title):
    """Turn a section title into a short, stable, unique CSS-targetable key."""
    slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')[:36]
    n = _section_slug_counts.get(slug, 0)
    _section_slug_counts[slug] = n + 1
    return f"sec_{slug}_{n}" if n else f"sec_{slug}"


@contextmanager
def section(title, icon=""):
    """A real card container. Unlike the old section_start()/section_end()
    pair — which opened a <div> in one st.markdown() call and closed it in
    another — this uses st.container(key=...), which Streamlit actually
    nests its children inside in the real DOM. The previous approach only
    ever visually bordered the title, because two separate st.markdown()
    calls never nest in the browser; every chart/table 'inside' a section
    was actually rendering as an unstyled sibling below the tiny title box.
    Styling is applied via CSS targeting the stable `st-key-sec_*` class
    Streamlit attaches to the container (see the [class*="st-key-sec_"]
    rule in the stylesheet) rather than raw HTML, so it can safely wrap
    any number of child Streamlit elements — charts, tables, columns."""
    key = _section_key(title)
    with st.container(key=key):
        st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)
        yield


_chart_key_counts = {}


@contextmanager
def chart_container(tag="chart"):
    """Same real-containment fix as section(), for wrapping a standalone
    chart that doesn't need its own title bar (the surrounding st.subheader
    already labels it) — just consistent card framing instead of a chart
    floating loose on the dark background."""
    n = _chart_key_counts.get(tag, 0)
    _chart_key_counts[tag] = n + 1
    key = f"chart_{tag}_{n}" if n else f"chart_{tag}"
    with st.container(key=key):
        yield


CATEGORY_PALETTE = ['#6fd18a', '#7ab8e8', '#e8c15a', '#f08080', '#c39bd3', '#76d7c4', '#f5b7b1', '#85c1e9']


def category_color(label):
    idx = abs(hash(str(label))) % len(CATEGORY_PALETTE)
    return CATEGORY_PALETTE[idx]


def pct_from_range(value, vmin, vmax, invert=False):
    """Map a value into a 0-100 bar-fill percentage given a range."""
    if pd.isna(value):
        return 0
    if vmax == vmin:
        pct = 50
    else:
        pct = (value - vmin) / (vmax - vmin) * 100
    pct = max(0, min(100, pct))
    return 100 - pct if invert else pct


def render_tile(avatar_text, avatar_bg, category_label, title, subtitle, rows, footer_left, footer_right):
    """Renders a Kalshi-style 'market tile' card: avatar + category tag +
    title, one or more progress-bar rows each with a colored pill, and a
    footer stats row.

    IMPORTANT: every fragment here is built with NO leading whitespace and
    NO embedded newlines. st.markdown() runs content through a Markdown
    parser before displaying it, and Markdown treats indented multi-line
    text as a literal code block instead of HTML — which is exactly what
    was happening before (rows_html built line-by-line in a loop, then
    spliced into an indented outer template). Keeping every piece on one
    flat line sidesteps that entirely.
    """
    rows_html = ""
    for r in rows:
        rows_html += (
            '<div class="tile-row">'
            '<div style="flex:1;">'
            f'<div class="tile-bar-label">{r["label"]}</div>'
            f'<div class="tile-bar-track"><div class="tile-bar-fill" style="width:{r["pct"]:.0f}%; background:{r["bar_color"]};"></div></div>'
            '</div>'
            f'<div class="tile-pill" style="border-color:{r["pill_color"]}; color:{r["pill_color"]};">{r["pill_text"]}</div>'
            '</div>'
        )
    cat_color = category_color(category_label)
    subtitle_html = f'<div class="tile-subtitle">{subtitle}</div>' if subtitle else ""
    return (
        '<div class="tile">'
        '<div class="tile-header">'
        f'<span class="tile-avatar" style="background:{avatar_bg};">{avatar_text}</span>'
        '<div>'
        f'<div class="tile-category" style="color:{cat_color};">{category_label}</div>'
        f'<div class="tile-title">{title}</div>'
        f'{subtitle_html}'
        '</div>'
        '</div>'
        f'{rows_html}'
        '<div class="tile-footer">'
        f'<span>{footer_left}</span><span>{footer_right}</span>'
        '</div>'
        '</div>'
    )


def tile_grid(html_list, n_cols=2):
    """Lay out a list of tile HTML strings in an n-column grid.

    Batches all tiles assigned to a given column into a single
    st.markdown() call (wrapped in one flex container) instead of one
    call per tile. Streamlit's per-element overhead, not raw HTML size,
    is what actually gets slow with long lists — a 20-tile grid used to
    mean 20 separate component instantiations; now it's just `n_cols`,
    regardless of how many tiles are in it."""
    if not html_list:
        return
    cols = st.columns(n_cols)
    buckets = [[] for _ in range(n_cols)]
    for i, html in enumerate(html_list):
        buckets[i % n_cols].append(html)

    for col, bucket in zip(cols, buckets):
        if not bucket:
            continue
        with col:
            st.markdown(
                '<div style="display:flex; flex-direction:column;">' + "".join(bucket) + '</div>',
                unsafe_allow_html=True
            )


def render_compare_bar(label, a_disp, a_val, b_disp, b_val, higher_is_better=True):
    """A two-color proportional bar for head-to-head numeric comparisons
    (e.g. Player A's wickets vs Player B's), used on Compare Players.
    Player A is always green, Player B is always blue (matching the header
    labels above); the better value gets a ✓ marker rather than a color
    swap, so the A/B color mapping stays consistent throughout."""
    total = abs(a_val) + abs(b_val)
    if total == 0:
        a_pct = b_pct = 50
    else:
        a_pct = (abs(a_val) / total) * 100
        b_pct = 100 - a_pct
    a_color, b_color = "#6fd18a", "#4f9fd8"
    a_wins = (a_val >= b_val) if higher_is_better else (a_val <= b_val)
    a_mark = " ✓" if a_wins and a_val != b_val else ""
    b_mark = " ✓" if (not a_wins) and a_val != b_val else ""
    return (
        '<div class="cmp-row">'
        f'<div class="cmp-label">{label}</div>'
        '<div class="cmp-values">'
        f'<span class="cmp-value" style="color:{a_color};">{a_disp}{a_mark}</span>'
        f'<span class="cmp-value" style="color:{b_color};">{b_disp}{b_mark}</span>'
        '</div>'
        '<div class="cmp-track">'
        f'<div class="cmp-fill-a" style="width:{a_pct:.0f}%; background:{a_color};"></div>'
        f'<div class="cmp-fill-b" style="width:{b_pct:.0f}%; background:{b_color};"></div>'
        '</div>'
        '</div>'
    )


# ------------------------------------------------------------------
# FORMAT GATE — landing screen: pick T20 / ODI / Test / All before
# entering the analytics app. Selection lives in st.session_state so
# it persists across reruns/page navigation until explicitly changed.
# ------------------------------------------------------------------

FORMAT_THEME = {
    "T20": {"css": "gate-t20", "glyph": "⚡", "tag": "20 OVERS · FAST",
            "sub": "Explosive scoring, death-over economy, and boundary pressure.", "accent": "#e8843c"},
    "ODI": {"css": "gate-odi", "glyph": "🏏", "tag": "50 OVERS · CLASSIC",
            "sub": "Powerplay control, middle-overs squeeze, and closing spells.", "accent": "#4f9fd8"},
    "TEST": {"css": "gate-test", "glyph": "🎖️", "tag": "5 DAYS · ATTRITION",
             "sub": "Long spells, workload endurance, and session-by-session control.", "accent": "#22b573"},
    "All": {"css": "gate-all", "glyph": "🌐", "tag": "COMBINED VIEW",
            "sub": "Every format blended together — the full career picture.", "accent": "#8891a1"},
}


def gate_tile_html(fmt_key, stats):
    theme = FORMAT_THEME[fmt_key]
    label = "All Formats" if fmt_key == "All" else fmt_key
    stats_html = "".join(
        f'<div class="gate-stat"><span class="n">{v}</span><span class="l">{l}</span></div>'
        for l, v in stats
    )
    return (
        f'<div class="gate-tile {theme["css"]}" data-glyph="{theme["glyph"]}">'
        f'<span class="gate-tag">{theme["tag"]}</span>'
        f'<div class="gate-title">{label}</div>'
        f'<div class="gate-sub">{theme["sub"]}</div>'
        f'<div class="gate-stats">{stats_html}</div>'
        f'</div>'
    )


def render_format_gate(all_data):
    """Full-screen responsive format-selector landing page. Sets
    st.session_state.app_format and reruns once the user picks a card."""
    st.markdown('<div class="gate-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="gate-hero">
      <div class="eyebrow">STEP 1 OF 1 · CHOOSE YOUR LENS</div>
      <h1>Which format are you analyzing?</h1>
      <p>Pick a match format to tailor every page — KPIs, leaderboards, ACWR workload
      monitoring, and matchups — to that format's rhythm. You can switch anytime from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)

    formats_present = [f for f in ["T20", "ODI", "TEST"] if f in all_data["format"].astype(str).str.upper().unique()]
    # normalize casing to match actual values in the data (e.g. "TEST" vs "Test")
    actual_values = {v.upper(): v for v in all_data["format"].dropna().unique()}

    cols = st.columns(3)
    for i, fmt_key in enumerate(["T20", "ODI", "TEST"]):
        real_val = actual_values.get(fmt_key, fmt_key)
        subset = all_data[all_data["format"] == real_val]
        stats = [
            ("Bowlers", f'{subset["bowler"].nunique():,}'),
            ("Matches", f'{subset["match_id"].nunique():,}'),
            ("Avg Econ", f'{(subset["runs_conceded"].sum() / max(subset["overs_bowled"].sum(), 1e-9)):.1f}'),
        ]
        with cols[i]:
            st.markdown(gate_tile_html(fmt_key, stats), unsafe_allow_html=True)
            if st.button(f"Analyze {fmt_key} →", key=f"fmt_{fmt_key.lower()}", width='stretch'):
                st.session_state.pending_format = real_val
                st.rerun()

    st.write("")
    left, mid, right = st.columns([1, 1.4, 1])
    with mid:
        all_stats = [
            ("Bowlers", f'{all_data["bowler"].nunique():,}'),
            ("Matches", f'{all_data["match_id"].nunique():,}'),
            ("Formats", f'{all_data["format"].nunique()}'),
        ]
        st.markdown(gate_tile_html("All", all_stats), unsafe_allow_html=True)
        if st.button("Analyze All Formats →", key="fmt_all", width='stretch'):
            st.session_state.pending_format = "All"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_analyzing_screen(target_format, all_data):
    """Brief terminal-style 'crunching the numbers' transition, played once
    after a format is picked and before landing in the app. Real dataset
    scale (not placeholder numbers) sells the sense of a large corpus."""
    import time

    if target_format == "All":
        subset = all_data
        label = "ALL FORMATS"
    else:
        subset = all_data[all_data["format"] == target_format]
        label = target_format

    deliveries = int(subset["legal_balls"].sum()) if "legal_balls" in subset.columns else int(subset["overs_bowled"].sum() * 6)
    matches = int(subset["match_id"].nunique())
    bowlers = int(subset["bowler"].nunique())
    records = len(subset)

    targets = [deliveries, matches, bowlers, records]
    labels = ["Deliveries indexed", "Matches parsed", "Bowlers profiled", "Records aggregated"]

    log_lines = [
        f"Loading bowler_match_summary.csv \u2192 {label}",
        f"Parsing {deliveries:,} ball-by-ball deliveries",
        f"Indexing {bowlers:,} bowlers across {matches:,} matches",
        "Computing ACWR workload profiles (Gabbett et al.)",
        "Building market tiles \u2192 ready",
    ]

    st.markdown('<div class="an-wrap">', unsafe_allow_html=True)
    slot = st.empty()

    # Kept intentionally brief: enough frames for the counters to visibly
    # animate, not so many that picking a format feels like a real wait.
    # (Previously 26 frames @ 0.035s — ~0.9s of blocking sleep plus 26
    # separate component re-renders — was the single biggest contributor
    # to the app feeling slow to load.)
    n_frames = 7
    for frame in range(1, n_frames + 1):
        pct = frame / n_frames
        eased = 1 - (1 - pct) ** 3  # ease-out cubic — fast start, settles at the end
        progress = round(pct * 100)

        counters_html = "".join(
            f'<div class="an-counter"><span class="n">{int(t * eased):,}</span><span class="l">{l}</span></div>'
            for t, l in zip(targets, labels)
        )

        lines_revealed = min(len(log_lines), int(pct * (len(log_lines) + 0.5)))
        log_html = ""
        for i, line in enumerate(log_lines):
            if i < lines_revealed:
                log_html += f'<div class="an-log-line done"><span class="icon">\u2713</span>{line}</div>'
            elif i == lines_revealed:
                log_html += f'<div class="an-log-line pending"><span class="icon">\u25cb</span>{line}</div>'
            else:
                log_html += f'<div class="an-log-line pending"><span class="icon">\u00b7</span>{line}</div>'

        slot.markdown(
            '<div class="an-panel">'
            '<div class="an-eyebrow"><span class="an-dot"></span>INITIALIZING ANALYTICS ENGINE</div>'
            f'<div class="an-title">Crunching {label} data\u2026</div>'
            f'<div class="an-counters">{counters_html}</div>'
            f'<div class="an-progress-track"><div class="an-progress-fill" style="width:{progress}%;"></div></div>'
            f'<div class="an-log">{log_html}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.01)

    st.markdown('</div>', unsafe_allow_html=True)


def render_welcome_screen(all_data):
    """The very first thing a visitor sees — brand hero + dataset scale,
    a live data-driven insight, a 3-step orientation strip, and a single
    CTA that plays a cricket-themed transition into the format gate
    (T20/ODI/Test/All). A muted 'skip intro' link keeps it approachable
    for anyone who just wants to get straight to the data."""
    bowlers = int(all_data["bowler"].nunique())
    matches = int(all_data["match_id"].nunique())
    deliveries = int(all_data["legal_balls"].sum()) if "legal_balls" in all_data.columns else int(all_data["overs_bowled"].sum() * 6)
    formats_n = int(all_data["format"].nunique())

    # Live, data-driven insight — computed fresh from the actual dataset,
    # not a placeholder, so the homepage feels alive rather than templated.
    wkt_totals = all_data.groupby("display_name")["wickets"].sum()
    top_name = wkt_totals.idxmax()
    top_wkts = int(wkt_totals.max())
    high_risk_n = int((all_data.sort_values("start_date").groupby("bowler").last()["acwr_tier"] == "High").sum())

    st.markdown(
        '<div class="welcome-wrap">'
        '<div class="welcome-ball">\U0001F3CF</div>'
        '<div class="welcome-eyebrow">REAL BALL-BY-BALL CRICKET ANALYTICS</div>'
        '<div class="welcome-title">Know every bowler.<br>Manage every <span>workload</span>.</div>'
        '<div class="welcome-sub">Performance, matchups, and ACWR-based workload monitoring built '
        'from real delivery-level data — not estimates. Step onto the analytics pitch.</div>'
        '<div class="tagline-strip">'
        '▸ No black-box scores<span class="sep">·</span>'
        '▸ Real ACWR sports-science math<span class="sep">·</span>'
        '▸ Every format, understood'
        '</div>'
        '<div class="welcome-stats">'
        f'<div class="welcome-stat"><span class="n">{bowlers:,}</span><span class="l">Bowlers</span></div>'
        f'<div class="welcome-stat"><span class="n">{matches:,}</span><span class="l">Matches</span></div>'
        f'<div class="welcome-stat"><span class="n">{deliveries:,}</span><span class="l">Deliveries</span></div>'
        f'<div class="welcome-stat"><span class="n">{formats_n}</span><span class="l">Formats</span></div>'
        '</div>'
        '<div class="insight-pill"><span class="dot"></span>'
        f'<span>Live from this dataset: <b>{top_name}</b> leads all bowlers with <b>{top_wkts:,} wickets</b> — '
        f'<b>{high_risk_n}</b> currently flagged high ACWR risk</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1.3, 1, 1.3])
    with mid:
        if st.button("Enter the Analytics Pitch →", key="enter_app", width='stretch'):
            st.session_state.entering_app = True
            st.rerun()
        st.markdown('<div class="welcome-cta-hint">Takes about 3 seconds</div>', unsafe_allow_html=True)
        if st.button("Skip the intro, take me straight in →", key="skip_intro", width='stretch'):
            st.session_state.entered_app = True
            st.rerun()

    st.markdown(
        '<div class="onboard-row">'
        '<div class="onboard-card" style="--fc-accent:#e8843c;">'
        '<div class="step-n">STEP 1</div><div class="step-icon">\U0001F3AF</div>'
        '<div class="step-title">Pick your format</div>'
        '<div class="step-desc">Start with T20, ODI, Test, or blend every format together for the full career picture.</div>'
        '</div>'
        '<div class="onboard-card" style="--fc-accent:#4f9fd8;">'
        '<div class="step-n">STEP 2</div><div class="step-icon">\U0001F50D</div>'
        '<div class="step-title">Explore any bowler</div>'
        '<div class="step-desc">Search, compare head-to-head with a radar chart, and track season-by-season trends.</div>'
        '</div>'
        '<div class="onboard-card" style="--fc-accent:#22b573;">'
        '<div class="step-n">STEP 3</div><div class="step-icon">\U0001F4C8</div>'
        '<div class="step-title">Watch the workload</div>'
        '<div class="step-desc">ACWR flags a rising spike in bowling load before it turns into an injury risk.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_entry_transition(all_data):
    """One-time cinematic 'walking out to the middle' loading sequence,
    played once between the welcome screen and the format gate. Reuses the
    same terminal-panel visual language as render_analyzing_screen for
    consistency, with its own narrative beats and whole-dataset stats."""
    import time

    deliveries = int(all_data["legal_balls"].sum()) if "legal_balls" in all_data.columns else int(all_data["overs_bowled"].sum() * 6)
    matches = int(all_data["match_id"].nunique())
    bowlers = int(all_data["bowler"].nunique())
    formats_n = int(all_data["format"].nunique())

    targets = [deliveries, matches, bowlers, formats_n]
    labels = ["Deliveries indexed", "Matches parsed", "Bowlers profiled", "Formats available"]

    log_lines = [
        "Walking out to the middle\u2026",
        f"Reading {deliveries:,} deliveries of match history",
        f"Profiling {bowlers:,} bowlers across {matches:,} matches",
        "Warming up the ACWR workload engine (Gabbett et al.)",
        "Pitch report ready \u2192 let's go",
    ]

    st.markdown('<div class="an-wrap">', unsafe_allow_html=True)
    slot = st.empty()

    # Kept intentionally brief — see render_analyzing_screen for why.
    n_frames = 7
    for frame in range(1, n_frames + 1):
        pct = frame / n_frames
        eased = 1 - (1 - pct) ** 3
        progress = round(pct * 100)

        counters_html = "".join(
            f'<div class="an-counter"><span class="n">{int(t * eased):,}</span><span class="l">{l}</span></div>'
            for t, l in zip(targets, labels)
        )

        lines_revealed = min(len(log_lines), int(pct * (len(log_lines) + 0.5)))
        log_html = ""
        for i, line in enumerate(log_lines):
            if i < lines_revealed:
                log_html += f'<div class="an-log-line done"><span class="icon">\u2713</span>{line}</div>'
            elif i == lines_revealed:
                log_html += f'<div class="an-log-line pending"><span class="icon">\u25cb</span>{line}</div>'
            else:
                log_html += f'<div class="an-log-line pending"><span class="icon">\u00b7</span>{line}</div>'

        slot.markdown(
            '<div class="an-panel">'
            '<div class="an-eyebrow"><span class="an-dot"></span>WELCOME TO THE PLATFORM</div>'
            '<div class="an-title">Setting the field\u2026</div>'
            f'<div class="an-counters">{counters_html}</div>'
            f'<div class="an-progress-track"><div class="an-progress-fill" style="width:{progress}%;"></div></div>'
            f'<div class="an-log">{log_html}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.01)

    st.markdown('</div>', unsafe_allow_html=True)


@st.cache_data
def compute_vs_team(ms):
    """Recompute bowler-vs-opponent aggregates live from match_summary so the
    global format filter applies correctly everywhere without needing to
    rerun preprocess.py."""
    if ms.empty:
        return pd.DataFrame(columns=['bowler', 'opponent_team', 'avg_economy', 'avg_wickets',
                                       'matches_played', 'performance_score'])
    vs = ms.groupby(['bowler', 'batting_team']).agg(
        avg_economy=('economy', 'mean'),
        avg_wickets=('wickets', 'mean'),
        matches_played=('match_id', 'nunique'),
    ).reset_index().rename(columns={'batting_team': 'opponent_team'})
    vs['performance_score'] = vs['avg_wickets'] * 2 - vs['avg_economy'] * 0.5
    return vs


def draw_acwr_gauge(value, vmax=2.2):
    """Custom semi-circle speedometer gauge for ACWR."""
    fig, ax = plt.subplots(figsize=(4, 2.7), subplot_kw={'aspect': 'equal'})

    zones = [(0, 0.8, '#2f6fa8'), (0.8, 1.3, '#2e8b4f'), (1.3, 1.5, '#c9a227'), (1.5, vmax, '#b0413e')]
    for start, end, color in zones:
        theta1 = 180 - (start / vmax) * 180
        theta2 = 180 - (end / vmax) * 180
        wedge = mpatches.Wedge((0.5, 0), 0.45, theta2, theta1, width=0.15, facecolor=color, edgecolor='none')
        ax.add_patch(wedge)

    val_clamped = min(max(value, 0), vmax) if pd.notna(value) else 0
    angle = np.radians(180 - (val_clamped / vmax) * 180)
    x = 0.5 + 0.38 * np.cos(angle)
    y = 0 + 0.38 * np.sin(angle)
    ax.plot([0.5, x], [0, y], color='white', linewidth=2.5, solid_capstyle='round')
    ax.add_patch(plt.Circle((0.5, 0), 0.025, color='white'))

    label = f"{value:.2f}" if pd.notna(value) else "N/A"
    ax.text(0.5, -0.22, label, ha='center', fontsize=20, color='white', weight='bold')
    ax.text(0.5, -0.38, "ACWR", ha='center', fontsize=10, color='#9aa4b2')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.48, 0.55)
    ax.axis('off')
    fig.patch.set_alpha(0)
    return fig


def draw_radar(players_stats, categories):
    """players_stats: dict of {name: [0-1 normalized values]}"""
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='#cdd5e0', size=10)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.spines['polar'].set_color('#2a2f3a')
    ax.grid(color='#2a2f3a')

    for i, (name, values) in enumerate(players_stats.items()):
        vals = values + values[:1]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, vals, linewidth=2, label=name, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), facecolor='#12161e', edgecolor='#2a2f3a',
              labelcolor='#e4e8ee')
    return fig


def normalize_series(s, invert=False):
    if s.max() == s.min():
        return pd.Series([0.5] * len(s), index=s.index)
    norm = (s - s.min()) / (s.max() - s.min())
    return 1 - norm if invert else norm


@st.cache_data
def compute_consistency(ms, min_matches=3):
    """Lower std-dev of economy across matches = more consistent."""
    stats = ms.groupby('display_name').agg(
        economy_std=('economy', 'std'),
        matches=('match_id', 'nunique'),
        avg_economy=('economy', 'mean'),
    ).reset_index()
    stats = stats[stats['matches'] >= min_matches].dropna(subset=['economy_std'])
    return stats.sort_values('economy_std')


@st.cache_data
def compute_rising_stars(ms, min_seasons=2):
    """Season-over-season economy improvement (decrease = improvement)."""
    if 'season' not in ms.columns:
        return pd.DataFrame()
    season_stats = ms.groupby(['display_name', 'season'])['economy'].mean().reset_index()
    results = []
    for name, grp in season_stats.groupby('display_name'):
        grp = grp.sort_values('season')
        if len(grp) < min_seasons:
            continue
        first_econ = grp.iloc[0]['economy']
        last_econ = grp.iloc[-1]['economy']
        improvement = first_econ - last_econ  # positive = got better (lower economy)
        results.append({
            'display_name': name, 'first_season': grp.iloc[0]['season'],
            'last_season': grp.iloc[-1]['season'], 'first_economy': first_econ,
            'last_economy': last_econ, 'improvement': improvement
        })
    return pd.DataFrame(results).sort_values('improvement', ascending=False)


# ------------------------------------------------------------------
# ML MATCHUP PREDICTOR — the one genuinely trained model in this app.
#
# Everything else here (ACWR, selection scores, leaderboards) is real
# math on real data, but it's all hand-specified formulas — not a model
# that learned anything. This is: a RandomForestRegressor trained on
# real historical bowler-match rows, predicting expected economy and
# wickets for a bowler against a specific opponent/format.
#
# Leakage handling: every feature is deliberately a "trailing" value —
# each bowler's career average economy/wickets is computed via
# shift(1).expanding().mean(), i.e. only using matches strictly BEFORE
# the one being predicted. rest_days_before / matches_last_30_days /
# chronic_avg_overs are already pre-match-known by construction from
# preprocess.py. No feature here can see its own target.
# ------------------------------------------------------------------

ML_FEATURE_COLS = [
    "career_avg_economy_before", "career_avg_wickets_before", "matches_played_before",
    "rest_days_before", "matches_last_30_days", "chronic_avg_overs",
    "format_enc", "opponent_enc",
]


@st.cache_data
def prepare_ml_training_data(match_summary_df, min_history=3):
    """Engineers leakage-safe trailing-form features and returns the
    subset of rows with enough real history to train/predict on."""
    ms = match_summary_df.sort_values(["bowler", "start_date"]).copy()
    ms["career_avg_economy_before"] = ms.groupby("bowler")["economy"].transform(
        lambda s: s.shift(1).expanding().mean())
    ms["career_avg_wickets_before"] = ms.groupby("bowler")["wickets"].transform(
        lambda s: s.shift(1).expanding().mean())
    ms["matches_played_before"] = ms.groupby("bowler").cumcount()

    le_format = LabelEncoder()
    le_opponent = LabelEncoder()
    ms["format_enc"] = le_format.fit_transform(ms["format"].astype(str))
    ms["opponent_enc"] = le_opponent.fit_transform(ms["batting_team"].astype(str))

    training_df = ms[ms["matches_played_before"] >= min_history].dropna(
        subset=ML_FEATURE_COLS + ["economy", "wickets"]
    )
    return training_df, le_format, le_opponent


@st.cache_resource
def train_performance_models(training_df, min_rows=30):
    """Trains two RandomForestRegressors (economy, wickets) on real
    match data and returns them alongside honest held-out test metrics —
    shown in the UI rather than hidden, consistent with how this app
    already treats ACWR as a heuristic to be transparent about."""
    if len(training_df) < min_rows:
        return None

    X = training_df[ML_FEATURE_COLS]
    y_econ = training_df["economy"]
    y_wkts = training_df["wickets"]

    X_train, X_test, ye_train, ye_test, yw_train, yw_test = train_test_split(
        X, y_econ, y_wkts, test_size=0.2, random_state=42
    )

    econ_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    econ_model.fit(X_train, ye_train)
    econ_pred = econ_model.predict(X_test)

    wkts_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    wkts_model.fit(X_train, yw_train)
    wkts_pred = wkts_model.predict(X_test)

    metrics = {
        "economy_mae": mean_absolute_error(ye_test, econ_pred),
        "economy_r2": r2_score(ye_test, econ_pred),
        "wickets_mae": mean_absolute_error(yw_test, wkts_pred),
        "wickets_r2": r2_score(yw_test, wkts_pred),
        "n_train": len(X_train), "n_test": len(X_test),
    }
    return {"econ_model": econ_model, "wkts_model": wkts_model, "metrics": metrics}


def predict_matchup(bowler_key, opponent_team, format_choice, match_summary_df,
                     models_bundle, le_format, le_opponent):
    """Predicts expected economy/wickets for a bowler against a specific
    opponent in a given format, using their real career-to-date form.
    Returns None if the bowler has no match history or the model isn't
    trained; falls back gracefully (rather than crashing) on an
    opponent/format the encoders never saw during training."""
    if models_bundle is None:
        return None
    bowler_hist = match_summary_df[match_summary_df["bowler"] == bowler_key].sort_values("start_date")
    if bowler_hist.empty:
        return None

    latest = bowler_hist.iloc[-1]
    career_avg_econ = bowler_hist["economy"].mean()
    career_avg_wkts = bowler_hist["wickets"].mean()
    matches_played = len(bowler_hist)

    try:
        format_enc = le_format.transform([format_choice])[0]
    except ValueError:
        format_enc = 0
    try:
        opponent_enc = le_opponent.transform([opponent_team])[0]
    except ValueError:
        opponent_enc = 0

    features = pd.DataFrame([{
        "career_avg_economy_before": career_avg_econ,
        "career_avg_wickets_before": career_avg_wkts,
        "matches_played_before": matches_played,
        "rest_days_before": latest["rest_days_before"],
        "matches_last_30_days": latest["matches_last_30_days"],
        "chronic_avg_overs": latest["chronic_avg_overs"],
        "format_enc": format_enc, "opponent_enc": opponent_enc,
    }])[ML_FEATURE_COLS]

    pred_econ = models_bundle["econ_model"].predict(features)[0]
    pred_wkts = models_bundle["wkts_model"].predict(features)[0]
    return {"predicted_economy": round(float(pred_econ), 2), "predicted_wickets": round(float(pred_wkts), 2)}


TIER_RANK = {"Undertrained": 0, "Low": 1, "Moderate": 2, "High": 3, "Unknown": -1}


def compute_acwr_trend(ms):
    """For every bowler with 2+ matches in the current view, compare their
    most recent ACWR/tier to their previous match. Flags anyone whose risk
    tier just got worse (e.g. Low -> High) — the 'just happened' signal
    that a static snapshot table can't show."""
    if ms.empty or 'display_name' not in ms.columns:
        return pd.DataFrame()
    results = []
    for name, grp in ms.sort_values('start_date').groupby('display_name'):
        if len(grp) < 2:
            continue
        prev, curr = grp.iloc[-2], grp.iloc[-1]
        prev_tier, curr_tier = prev['acwr_tier'], curr['acwr_tier']
        prev_rank, curr_rank = TIER_RANK.get(prev_tier, -1), TIER_RANK.get(curr_tier, -1)
        if prev_rank < 0 or curr_rank < 0:
            continue
        results.append({
            'display_name': name, 'bowler': curr.get('bowler'),
            'prev_acwr': prev['acwr'], 'curr_acwr': curr['acwr'],
            'prev_tier': prev_tier, 'curr_tier': curr_tier,
            'delta': curr_rank - prev_rank,
            'worsened': curr_rank > prev_rank,
            'newly_high': curr_tier == "High" and prev_tier != "High",
        })
    return pd.DataFrame(results)


# ==================================================================
# CHATBOT ENGINE — "Ask the Data"
# ==================================================================
# Deliberately rule-based, not LLM-backed — same reasoning as the
# Assessment Guide on the Injury & Fitness page (see methodology):
# every answer here is a direct pandas lookup against the real CSVs,
# so it can be wrong about what you asked, but it can never hallucinate
# a number that isn't in the data. It parses keywords + player names
# out of the question with regex/difflib (both stdlib), then routes to
# a small set of concrete lookups. No ML, no external API calls.

import difflib

CHATBOT_GREETING = (
    "👋 Ask me about the data — wickets, economy, ACWR/workload, monotony, rest days, fitness "
    "status, or how anyone did against a specific team, plus squad-wide questions like top "
    "wicket-takers, most consistent, or who's currently at risk. Type **help** any time for examples."
)

CHATBOT_HELP_TEXT = """
Here's what I can answer directly from the data:

- **A player's stats** — *"wickets for Bumrah"*, *"what's Rashid Khan's economy?"*, *"how many overs has Shami bowled?"*, *"rest days for Starc"*
- **Workload / risk** — *"ACWR for Starc"*, *"is Boult at risk?"*, *"monotony for Rabada"*, *"how many overs can Shami safely bowl?"*
- **Fitness status** — *"is Archer injured?"*, *"status of Rabada"*
- **Matchups** — *"how does Bumrah do against Australia?"*
- **Comparisons** — *"compare Bumrah and Starc"*
- **Squad leaderboards** — *"top 10 wicket takers"*, *"best economy"*, *"most consistent bowlers"*, *"rising stars"*, *"who is highest risk"*, *"who is injured"*
- **Dataset facts** — *"how many bowlers are there?"*, *"how many matches?"*, *"which teams?"*
- **Definitions** — *"what is ACWR?"*, *"what does monotony mean?"*

Once you've asked about someone, you can follow up with just **"and his ACWR?"** or **"what about wickets?"** — I'll remember who you meant. I only answer from what's actually in your loaded CSVs (and the currently selected format), so if I don't recognize a name or a question, I'll say so instead of guessing.
"""

STAT_DEFINITIONS = {
    "acwr": (
        "**ACWR (Acute:Chronic Workload Ratio)** compares a bowler's most recent match workload "
        "(acute = overs bowled in that match) to their rolling average over their last 4 matches "
        "(chronic). Below 0.8 = Undertrained, 0.8–1.3 = the 'sweet spot', 1.3–1.5 = Moderate risk, "
        "above 1.5 = High risk. It's a published sports-science heuristic, not a validated injury "
        "predictor — there's no public dataset of real injury outcomes to train one on."
    ),
    "monotony": (
        "**Monotony** is the mean of a bowler's recent workload divided by its standard deviation — "
        "high monotony means their workload has been unusually repetitive (little variation match to "
        "match), which sports-science research (Foster's training-load model) links to higher injury risk "
        "even at moderate overall loads."
    ),
    "strain": (
        "**Strain** = monotony × total recent workload — it combines *how repetitive* the workload has "
        "been with *how much* of it there's been, so a bowler can have high strain either by bowling a lot "
        "or by bowling the same amount every single match with no variation."
    ),
    "economy": (
        "**Economy** is runs conceded divided by overs bowled — lower means the bowler is giving away "
        "fewer runs per over."
    ),
    "performance score": (
        "**Performance score** (used on the matchup pages) = (average wickets × 2) − (average economy × 0.5) "
        "— a simple weighted blend rewarding wicket-taking and penalizing expensive bowling."
    ),
    "consistency": (
        "**Consistency** is measured as the standard deviation of a bowler's economy across their matches — "
        "a lower number means their economy barely changes match to match (more predictable), a higher "
        "number means it swings a lot."
    ),
    "chronic workload": (
        "**Chronic workload** is a bowler's rolling average overs bowled over their last 4 matches — "
        "their recent 'normal' workload, used as the baseline ACWR compares each new match against."
    ),
}

# Vocabulary used for lightweight typo-correction on intent keywords only —
# never touches player names, which have their own fuzzy-match path below.
_INTENT_VOCAB = [
    "wickets", "economy", "acwr", "workload", "risk", "status", "injured", "injury", "fit",
    "rehab", "managed", "overs", "matches", "rest", "monotony", "strain", "consistency",
    "consistent", "rising", "stars", "teams", "bowlers", "players", "compare", "versus",
    "best", "top", "most", "highest", "lowest", "least", "help", "define", "explain", "mean",
    "dataset", "squad", "safe", "ceiling", "chronic", "acute", "against", "improved",
]

_PRONOUNS = {"he", "him", "his", "she", "her", "they", "them", "their", "that", "this", "guy", "player", "one"}

# --------------------------------------------------------------------
# Per-page chat context — same engine everywhere, but the greeting,
# placeholder, quick-question chips, and the "no stat specified"
# default all adapt to whichever page the chatbot is opened from, so
# it feels tuned to what you're actually looking at rather than being
# one generic bot bolted onto every screen.
# --------------------------------------------------------------------
PAGE_CHAT_CONTEXT = {
    "default": {
        "greeting": CHATBOT_GREETING,
        "placeholder": "Ask about wickets, economy, ACWR, injuries…",
        "chips": ["Top wicket takers", "Who is at risk?", "How many bowlers are there?"],
        "focus": None,
    },
    "home": {
        "greeting": (
            "👋 Ask me anything about the squad — I can pull leaderboards, workload, or fitness "
            "status straight from the data. Type **help** for examples."
        ),
        "placeholder": "e.g. \"top wicket takers\", \"who is at risk?\"",
        "chips": ["Top wicket takers", "Who is at risk?", "Who is injured?", "How many matches?"],
        "focus": None,
    },
    "acwr_engine": {
        "greeting": (
            "👋 You're on the ACWR Engine page — ask about any bowler and I'll default to their "
            "current ACWR reading. Try **\"what is ACWR?\"** if you want the definition first."
        ),
        "placeholder": "e.g. \"ACWR for Bumrah\", \"who is at risk?\"",
        "chips": ["Who is at risk?", "What is ACWR?", "Most consistent bowlers"],
        "focus": "acwr",
    },
    "dataset_overview": {
        "greeting": "👋 Ask me about what's actually in the loaded dataset — counts, teams, coverage.",
        "placeholder": "e.g. \"how many matches?\", \"which teams?\"",
        "chips": ["How many bowlers are there?", "How many matches?", "Which teams?"],
        "focus": None,
    },
    "player_profile": {
        "greeting": "👋 Ask about the player you've got selected below — or anyone else in the squad.",
        "placeholder": "e.g. \"ACWR for this player\", \"rest days for them\"",
        "chips": ["Compare wickets leaderboard", "Who is at risk?"],
        "player_chip_templates": ["ACWR for {0}", "Rest days for {0}", "Is {0} injured?"],
        "focus": "summary",
    },
    "compare_players": {
        "greeting": "👋 Ask me to compare any two bowlers, or dig into one of the players you've selected below.",
        "placeholder": "e.g. \"compare Bumrah and Starc\"",
        "chips": ["Most consistent bowlers", "Top wicket takers"],
        "player_chip_templates": ["Compare {0} and {1}"],
        "focus": "compare",
    },
    "advanced_search": {
        "greeting": "👋 Ask me squad-wide questions to narrow things down before you filter below.",
        "placeholder": "e.g. \"top wicket takers\", \"most economical bowlers\"",
        "chips": ["Top wicket takers", "Best economy", "Which teams?"],
        "focus": None,
    },
    "leaderboard": {
        "greeting": "👋 Ask me for any leaderboard cut — wickets, economy, consistency, or rising stars.",
        "placeholder": "e.g. \"top 10 wicket takers\", \"most consistent bowlers\"",
        "chips": ["Top wicket takers", "Best economy", "Most consistent bowlers", "Rising stars"],
        "focus": "wickets",
    },
    "bowler_selection": {
        "greeting": "👋 Ask about candidates for selection — wickets, economy, or current risk status.",
        "placeholder": "e.g. \"best economy\", \"is this bowler at risk?\"",
        "chips": ["Top wicket takers", "Best economy", "Who is at risk?"],
        "focus": "wickets",
    },
    "workload_monitor": {
        "greeting": "👋 Ask about workload — ACWR, monotony, strain, or who needs a rest right now.",
        "placeholder": "e.g. \"who is at risk?\", \"monotony for Rabada\"",
        "chips": ["Who is at risk?", "What is monotony?", "What is ACWR?"],
        "focus": "acwr",
    },
    "injury_fitness": {
        "greeting": "👋 Ask about fitness status — who's injured, in rehab, or cleared to play.",
        "placeholder": "e.g. \"is Archer injured?\", \"who is injured?\"",
        "chips": ["Who is injured?", "Status of a player", "Who is at risk?"],
        "focus": "status",
    },
    "squad_impact": {
        "greeting": "👋 Ask about workload knock-ons — chronic load, ACWR, or who's carrying the most.",
        "placeholder": "e.g. \"chronic workload for Bumrah\", \"who is at risk?\"",
        "chips": ["Who is at risk?", "What is chronic workload?", "Top wicket takers"],
        "focus": "acwr",
    },
    "team_overview": {
        "greeting": "👋 Ask how any bowler performs against a specific team, or about the squad overall.",
        "placeholder": "e.g. \"how does Bumrah do against Australia?\"",
        "chips": ["Which teams?", "Top wicket takers", "How many matches?"],
        "focus": None,
    },
    "ask_data": {
        "greeting": CHATBOT_GREETING,
        "placeholder": "Ask about wickets, economy, ACWR, injuries…",
        "chips": ["Top wicket takers", "Best economy", "Most consistent bowlers", "Rising stars"],
        "focus": None,
    },
}


def _normalize_q(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _typo_correct(q):
    """Nudges obviously-misspelled intent words toward the nearest vocab
    term (e.g. 'wcikets' -> 'wickets', 'ijured' -> 'injured') so small
    typos don't silently fall through to the fallback response. Only
    replaces a token when it's a close-but-not-exact match, so real
    player-name tokens (already checked elsewhere) pass through untouched."""
    out = []
    for w in q.split(" "):
        if not w or w in _INTENT_VOCAB or len(w) < 4:
            out.append(w)
            continue
        close = difflib.get_close_matches(w, _INTENT_VOCAB, n=1, cutoff=0.82)
        out.append(close[0] if close else w)
    return " ".join(out)


def _find_player_names(query, names):
    """Match player display_names mentioned in the question, in strict
    priority order so a confident match never gets diluted by noise —
    this matters a lot once the roster has hundreds/thousands of players
    who share common surname components (e.g. 'Singh', 'Kaur', 'Sran'):

    Tier 1 — exact full name appears verbatim in the query. If so, that's
      the answer; nothing else is even considered.
    Tier 2 — every one of a name's significant tokens appears as a whole
      word in the query (word-boundary match, not substring), so
      'wickets for Bumrah' finds 'Jasprit Bumrah' from the surname alone.
      Only names hitting the *most* tokens are kept, so a name matching
      2 tokens beats one that only coincidentally matches 1.
    Tier 3 — fuzzy fallback for typos, but only run when nothing matched
      exactly, and only on longer (5+ char) tokens with a high similarity
      cutoff, specifically to avoid short common surname pieces colliding
      across a large roster.
    """
    q = _normalize_q(query)
    q_words = re.findall(r"[a-z0-9]+", q)
    q_word_set = set(q_words)

    # --- Tier 1: exact full-name substring ---
    exact = [n for n in names if n and n.lower() in q]
    if exact:
        exact.sort(key=len, reverse=True)
        deduped = []
        for h in exact:
            if not any(h != other and h.lower() in other.lower() for other in deduped):
                deduped.append(h)
        return deduped[:5]

    # --- Tier 2: whole-word token match, best coverage wins ---
    token_hits = []
    for n in names:
        if not n:
            continue
        tokens = [t for t in re.findall(r"[a-z0-9]+", n.lower()) if len(t) >= 3]
        if not tokens:
            continue
        hit = sum(1 for t in tokens if t in q_word_set)
        if hit:
            token_hits.append((n, hit, len(tokens)))
    if token_hits:
        best = max(h for _, h, _ in token_hits)
        # keep only names tied for the strongest evidence — if exactly one
        # name matched the most tokens (even just one, e.g. a unique
        # surname), that's the answer; only genuinely tied names disambiguate.
        strong = [n for n, hit, total in token_hits if hit == best]
        return strong[:5]

    # --- Tier 3: fuzzy typo tolerance, last resort only ---
    fuzzy_hits = []
    for n in names:
        if not n:
            continue
        name_tokens = [t for t in re.findall(r"[a-z0-9]+", n.lower()) if len(t) >= 5]
        for qt in q_words:
            if len(qt) < 5:
                continue
            for nt in name_tokens:
                if difflib.SequenceMatcher(None, qt, nt).ratio() >= 0.8:
                    fuzzy_hits.append(n)
                    break
    if fuzzy_hits:
        seen = []
        for n in fuzzy_hits:
            if n not in seen:
                seen.append(n)
        return seen[:5]
    return []


def _extract_top_n(q, default=5, cap=15):
    m = re.search(r"\btop\s+(\d+)\b", q)
    if m:
        return max(1, min(cap, int(m.group(1))))
    return default


def _find_team(q, team_names):
    """Looks for a known team name mentioned in the question, for
    'against <team>' style matchup queries."""
    q_low = q
    hits = [t for t in team_names if t and t.lower() in q_low]
    if hits:
        hits.sort(key=len, reverse=True)
        return hits[0]
    return None


def _player_snapshot(name, match_summary, latest_state, injury_log_df):
    """Pulls together everything known about one player into a dict,
    or None if they have no rows in the current (filtered) view."""
    rows = match_summary[match_summary["display_name"] == name]
    if rows.empty:
        return None
    latest_row = latest_state[latest_state["display_name"] == name]
    latest_match = rows.sort_values("start_date").iloc[-1] if "start_date" in rows.columns else rows.iloc[-1]
    monotony, strain = (None, None)
    try:
        monotony, strain = compute_monotony_strain(rows, window=7)
    except Exception:
        pass
    return {
        "name": name,
        "matches": rows["match_id"].nunique(),
        "wickets": int(rows["wickets"].sum()),
        "overs": round(rows["overs_bowled"].sum(), 1),
        "economy": round(rows["economy"].mean(), 2) if "economy" in rows.columns else None,
        "acwr": round(float(latest_row.iloc[0]["acwr"]), 2) if not latest_row.empty and pd.notna(latest_row.iloc[0]["acwr"]) else None,
        "acwr_tier": latest_row.iloc[0]["acwr_tier"] if not latest_row.empty else "Unknown",
        "status": latest_row.iloc[0]["current_status"] if not latest_row.empty and "current_status" in latest_row.columns else get_current_status(rows.iloc[-1]["bowler"], injury_log_df),
        "rest_days": latest_match.get("rest_days_before") if hasattr(latest_match, "get") else None,
        "chronic_avg_overs": latest_match.get("chronic_avg_overs") if hasattr(latest_match, "get") else None,
        "monotony": round(monotony, 2) if monotony is not None else None,
        "strain": round(strain, 1) if strain is not None else None,
        "bowler_key": rows.iloc[-1]["bowler"],
    }


def _fmt_snapshot_line(snap):
    econ = f"{snap['economy']:.2f}" if snap["economy"] is not None else "n/a"
    acwr = f"{snap['acwr']:.2f} ({snap['acwr_tier']})" if snap["acwr"] is not None else "not enough recent matches"
    return (
        f"**{snap['name']}** — {snap['matches']} match(es), {snap['wickets']} wicket(s), "
        f"{snap['overs']} overs, economy {econ}, ACWR {acwr}, status **{snap['status']}**."
    )


def chatbot_answer(query, names, match_summary_all, match_summary, vs_team, master,
                    latest_state, injury_log_df, format_choice, page_focus=None):
    raw_q = _normalize_q(query)
    if not raw_q:
        return "Ask me something about the data — try **help** for examples."

    fmt_note = f" _(within your current filter: {format_choice})_" if format_choice != "All" else ""

    # --- small talk ---
    if raw_q in {"help", "hi", "hello", "hey", "?"} or "what can you" in raw_q or "what can i ask" in raw_q:
        return CHATBOT_HELP_TEXT
    if raw_q in {"thanks", "thank you", "thx", "ty", "cheers"}:
        return "Anytime — ask away if you've got more questions."
    if raw_q in {"bye", "goodbye", "see ya"}:
        return "👋 See you around — I'll be here whenever you need another number."

    q = _typo_correct(raw_q)

    # --- definitions ---
    for term, definition in STAT_DEFINITIONS.items():
        if f"what is {term}" in q or f"what's {term}" in q or f"define {term}" in q or f"explain {term}" in q or f"{term} mean" in q:
            return definition
    if "acwr" in q and any(w in q for w in ["what", "mean", "explain", "define"]) and "for" not in q:
        return STAT_DEFINITIONS["acwr"]

    # --- dataset-wide facts ---
    if any(p in q for p in ["how many bowlers", "how many players", "squad size"]):
        n = match_summary["display_name"].nunique()
        return f"There are **{n} bowlers** in the current view{fmt_note}."
    if any(p in q for p in ["how many matches", "total matches", "matches in the dataset", "matches in this dataset"]):
        n = match_summary["match_id"].nunique()
        return f"There are **{n} matches** in the current view{fmt_note}."
    if any(p in q for p in ["which teams", "list teams", "what teams", "how many teams"]):
        teams = sorted(set(match_summary["bowling_team"].dropna().unique()) | set(match_summary.get("batting_team", pd.Series(dtype=str)).dropna().unique()))
        if not teams:
            return "I couldn't find any team data in the current view."
        return f"Teams in the current view{fmt_note}: " + ", ".join(f"**{t}**" for t in teams)

    # --- squad-wide leaderboards ---
    if re.search(r"\btop\s*\d*\s*wickets?\b", q) or any(p in q for p in ["most wickets", "leading wicket", "highest wickets"]):
        n = _extract_top_n(q)
        top = match_summary.groupby("display_name")["wickets"].sum().sort_values(ascending=False).head(n)
        lines = "\n".join(f"{i}. **{nm}** — {int(w)} wickets" for i, (nm, w) in enumerate(top.items(), 1))
        return f"Top wicket-takers{fmt_note}:\n\n{lines}"
    if "econom" in q and any(w in q for w in ["best", "lowest", "top", "most economical", "most economic"]) and not _find_player_names(raw_q, names):
        n = _extract_top_n(q)
        agg = match_summary.groupby("display_name").agg(matches=("match_id", "nunique"), econ=("economy", "mean"))
        agg = agg[agg["matches"] >= 3].sort_values("econ").head(n)
        if agg.empty:
            return "Not enough players with 3+ matches in the current view to rank economy."
        lines = "\n".join(f"{i}. **{nm}** — {r.econ:.2f} economy ({int(r.matches)} matches)" for i, (nm, r) in enumerate(agg.iterrows(), 1))
        return f"Most economical bowlers (min. 3 matches){fmt_note}:\n\n{lines}"
    if any(p in q for p in ["most consistent", "least consistent", "consistency leaderboard", "most stable economy"]):
        cons = compute_consistency(match_summary, min_matches=3)
        if cons.empty:
            return "Not enough players with 3+ matches in the current view to rank consistency."
        ascending = "least consistent" not in q
        cons = cons.sort_values("economy_std", ascending=ascending).head(_extract_top_n(q))
        lines = "\n".join(f"{i}. **{r.display_name}** — economy std {r.economy_std:.2f} ({int(r.matches)} matches)" for i, r in enumerate(cons.itertuples(), 1))
        label = "Most consistent" if ascending else "Least consistent"
        return f"{label} bowlers (lower std = more consistent){fmt_note}:\n\n{lines}"
    if any(p in q for p in ["rising star", "most improved", "improved the most"]):
        rs = compute_rising_stars(match_summary)
        if rs.empty:
            return "Not enough season-over-season data in the current view for a rising-stars ranking."
        rs = rs.sort_values("improvement", ascending=False).head(_extract_top_n(q))
        lines = "\n".join(f"{i}. **{r.display_name}** — economy improved {r.improvement:+.2f} ({r.first_season} → {r.last_season})" for i, r in enumerate(rs.itertuples(), 1))
        return f"Rising stars (biggest season-over-season economy improvement){fmt_note}:\n\n{lines}"
    if any(p in q for p in ["highest acwr", "highest risk", "most at risk", "at risk players", "who is at risk", "who's at risk"]):
        risky = latest_state[latest_state["acwr_tier"] == "High"].sort_values("acwr", ascending=False).head(max(8, _extract_top_n(q, default=8)))
        if risky.empty:
            return f"Nobody is currently flagged **High** risk on ACWR{fmt_note}."
        lines = "\n".join(f"- **{r.display_name}** — ACWR {r.acwr:.2f}" for r in risky.itertuples())
        return f"Currently flagged **High** risk{fmt_note}:\n\n{lines}"
    if any(p in q for p in ["who is injured", "who's injured", "list injured", "injured players", "currently injured"]):
        hurt = latest_state[latest_state["current_status"].isin(["Injured", "Rehab"])]
        if hurt.empty:
            return "Nobody in the current view is logged as Injured or in Rehab right now."
        lines = "\n".join(f"- **{r.display_name}** — {r.current_status}" for r in hurt.itertuples())
        return f"Currently Injured / Rehab:\n\n{lines}"

    # --- player identification, with pronoun/follow-up memory ---
    all_names = list(match_summary_all["display_name"].dropna().unique())
    matched = _find_player_names(raw_q, all_names)

    last_player = st.session_state.get("chat_last_player")
    q_tokens = set(re.findall(r"[a-z0-9]+", q))
    if not matched and last_player and (q_tokens & _PRONOUNS):
        matched = [last_player]
    elif not matched and last_player and len(q_tokens) <= 6 and (q_tokens & set(_INTENT_VOCAB)):
        # short, name-less follow-up that still mentions a known stat/intent word,
        # e.g. "what about wickets?" or "and overs?" — but not unrelated chatter
        matched = [last_player]

    if len(matched) > 2:
        listed = ", ".join(f"**{m}**" for m in matched[:5])
        return f"That could be a few different players — did you mean one of: {listed}? Try asking again with the full name."

    if len(matched) == 2:
        a, b = matched[0], matched[1]
        sa = _player_snapshot(a, match_summary, latest_state, injury_log_df)
        sb = _player_snapshot(b, match_summary, latest_state, injury_log_df)
        if not sa or not sb:
            missing = a if not sa else b
            return f"I found **{missing}** in the dataset, but they have no matches in the current filter{fmt_note}."
        st.session_state["chat_last_player"] = a
        return (
            f"**{sa['name']}** vs **{sb['name']}**{fmt_note}\n\n"
            f"| | {sa['name']} | {sb['name']} |\n|---|---|---|\n"
            f"| Matches | {sa['matches']} | {sb['matches']} |\n"
            f"| Wickets | {sa['wickets']} | {sb['wickets']} |\n"
            f"| Overs | {sa['overs']} | {sb['overs']} |\n"
            f"| Economy | {sa['economy'] if sa['economy'] is not None else 'n/a'} | {sb['economy'] if sb['economy'] is not None else 'n/a'} |\n"
            f"| ACWR | {sa['acwr'] if sa['acwr'] is not None else 'n/a'} ({sa['acwr_tier']}) | {sb['acwr'] if sb['acwr'] is not None else 'n/a'} ({sb['acwr_tier']}) |\n"
            f"| Status | {sa['status']} | {sb['status']} |"
        )

    if len(matched) == 1:
        name = matched[0]
        snap = _player_snapshot(name, match_summary, latest_state, injury_log_df)
        if not snap:
            return f"I found **{name}** in the dataset, but they have no matches in the current filter{fmt_note}. Try switching format in the sidebar."
        st.session_state["chat_last_player"] = name

        # matchup vs a specific team
        team_names = sorted(set(match_summary["bowling_team"].dropna().unique()) | set(match_summary.get("batting_team", pd.Series(dtype=str)).dropna().unique()))
        team = _find_team(raw_q, team_names) if any(w in q for w in ["against", "vs", "versus"]) else None
        if team:
            vt = vs_team[(vs_team["display_name"] == name) & (vs_team["opponent_team"] == team)] if not vs_team.empty else pd.DataFrame()
            if vt.empty:
                return f"I don't have any recorded matches for **{snap['name']}** against **{team}**{fmt_note}."
            r = vt.iloc[0]
            return (
                f"Against **{team}**, **{snap['name']}** averages **{r['avg_wickets']:.2f} wickets** and "
                f"**{r['avg_economy']:.2f} economy** across {int(r['matches_played'])} match(es){fmt_note}."
            )

        if any(p in q for p in ["wicket", "wkts"]):
            return f"**{snap['name']}** has taken **{snap['wickets']} wickets** across {snap['matches']} match(es){fmt_note}."
        if "econom" in q or "econ" in q:
            econ = f"{snap['economy']:.2f}" if snap["economy"] is not None else "not available"
            return f"**{snap['name']}**'s average economy is **{econ}**{fmt_note}."
        if "safe" in q and "over" in q:
            ceiling = safe_overs_ceiling(snap["chronic_avg_overs"]) if snap["chronic_avg_overs"] is not None else None
            if ceiling is None:
                return f"Not enough recent match history to estimate a safe overs ceiling for **{snap['name']}**."
            return f"Based on recent workload, **{snap['name']}** can likely bowl up to **~{ceiling} overs** in the next match without pushing ACWR into risk territory{fmt_note}."
        if "over" in q:
            return f"**{snap['name']}** has bowled **{snap['overs']} overs**{fmt_note}."
        if "rest" in q:
            rd = snap["rest_days"]
            if rd is None or pd.isna(rd):
                return f"No rest-day data available for **{snap['name']}**{fmt_note}."
            return f"**{snap['name']}** had **{int(rd)} rest day(s)** before their most recent match{fmt_note}."
        if "chronic" in q or ("workload" in q and "acwr" not in q):
            cw = snap["chronic_avg_overs"]
            if cw is None or pd.isna(cw):
                return f"Not enough match history to compute chronic workload for **{snap['name']}**."
            return f"**{snap['name']}**'s chronic (rolling 4-match average) workload is **{cw:.1f} overs**{fmt_note}."
        if "strain" in q:
            if snap["strain"] is None:
                return f"Not enough recent matches to compute strain for **{snap['name']}**."
            return f"**{snap['name']}**'s current strain score is **{snap['strain']}**{fmt_note}."
        if "monoton" in q:
            if snap["monotony"] is None:
                return f"Not enough recent matches to compute monotony for **{snap['name']}**."
            return f"**{snap['name']}**'s current monotony is **{snap['monotony']}** ({monotony_tier(snap['monotony'])}){fmt_note}."
        if "acwr" in q or "workload" in q or "risk" in q:
            if snap["acwr"] is None:
                return f"**{snap['name']}** doesn't have enough recent matches for an ACWR reading{fmt_note}."
            return f"**{snap['name']}**'s current ACWR is **{snap['acwr']}**, tier **{snap['acwr_tier']}**{fmt_note}."
        if "status" in q or "injur" in q or "fit" in q or "hurt" in q:
            return f"**{snap['name']}**'s current logged status is **{snap['status']}**{fmt_note}."
        if "match" in q:
            return f"**{snap['name']}** has played **{snap['matches']} match(es)**{fmt_note}."

        # no specific stat asked in the question itself — fall back to
        # whatever this page is about, so the same bare "tell me about
        # Bumrah" gives an ACWR answer on the workload pages and a status
        # answer on Injury & Fitness, instead of always dumping everything.
        if page_focus == "acwr":
            if snap["acwr"] is None:
                return f"**{snap['name']}** doesn't have enough recent matches for an ACWR reading{fmt_note}. Here's what else I have: {_fmt_snapshot_line(snap)}"
            return f"**{snap['name']}**'s current ACWR is **{snap['acwr']}**, tier **{snap['acwr_tier']}**{fmt_note}."
        if page_focus == "status":
            return f"**{snap['name']}**'s current logged status is **{snap['status']}**{fmt_note}."
        if page_focus == "wickets":
            return f"**{snap['name']}** has taken **{snap['wickets']} wickets** at economy **{snap['economy'] if snap['economy'] is not None else 'n/a'}** across {snap['matches']} match(es){fmt_note}."
        if page_focus == "economy":
            econ = f"{snap['economy']:.2f}" if snap["economy"] is not None else "not available"
            return f"**{snap['name']}**'s average economy is **{econ}**{fmt_note}."
        if page_focus == "compare":
            return _fmt_snapshot_line(snap) + f" Try *\"compare {snap['name']} and \u2039another player\u203a\"* for a side-by-side." + fmt_note

        # no page-specific default either — give the full snapshot
        return _fmt_snapshot_line(snap) + fmt_note

    return (
        "I couldn't match that to a player or a question I know how to answer from the data. "
        "Try a player's name plus what you want to know (e.g. *\"ACWR for Bumrah\"*), or type "
        "**help** to see example questions."
    )


def render_chatbot_ui(key_prefix, names, match_summary_all, match_summary, vs_team, master,
                       latest_state, injury_log_df, format_choice, page="home",
                       context_players=None, height=None):
    """Renders the chat history + input box. Same session-state history key
    regardless of where it's called from, so the floating widget and the
    dedicated 'Ask the Data' page share one running conversation — but the
    greeting, placeholder, quick-question chips, and default answer focus
    all adapt to `page`, and the chips get filled in with whichever
    player(s) are already selected on that page via `context_players`."""
    ctx = PAGE_CHAT_CONTEXT.get(page, PAGE_CHAT_CONTEXT["default"])
    history_key = "chat_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = [{"role": "assistant", "content": ctx["greeting"]}]

    top_l, top_r = st.columns([5, 1])
    with top_r:
        if st.button("🗑️ Clear", key=f"{key_prefix}_clear_chat", width='stretch'):
            st.session_state[history_key] = [{"role": "assistant", "content": ctx["greeting"]}]
            st.session_state.pop("chat_last_player", None)
            st.rerun()

    chat_box = st.container(height=height) if height else st.container()
    with chat_box:
        for msg in st.session_state[history_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    def _ask(q):
        st.session_state[history_key].append({"role": "user", "content": q})
        answer = chatbot_answer(q, names, match_summary_all, match_summary, vs_team, master,
                                 latest_state, injury_log_df, format_choice, page_focus=ctx.get("focus"))
        st.session_state[history_key].append({"role": "assistant", "content": answer})
        st.rerun()

    # Quick-question chips: player-specific ones (built from whatever's
    # already selected on this page) come first, then the page's generic
    # ones, capped at 4 so the row doesn't crowd the page.
    chips = []
    if context_players:
        for tmpl in ctx.get("player_chip_templates", []):
            try:
                chips.append(tmpl.format(*context_players))
            except (IndexError, KeyError):
                pass
    chips += [c for c in ctx.get("chips", []) if c not in chips]
    chips = chips[:4]
    if chips:
        chip_cols = st.columns(len(chips))
        for i, (c, chip_q) in enumerate(zip(chip_cols, chips)):
            with c:
                if st.button(chip_q, key=f"{key_prefix}_chip_{i}_{page}", width='stretch'):
                    _ask(chip_q)

    user_q = st.chat_input(ctx.get("placeholder", "Ask about wickets, economy, ACWR, injuries…"), key=f"{key_prefix}_chat_input")
    if user_q:
        _ask(user_q)


# ==================================================================
# DATA LOADING
# ==================================================================

REQUIRED_FILES = ["bowler_match_summary.csv", "bowler_vs_team_summary.csv", "bowler_master.csv"]


@st.cache_data
def load_csv(path_or_buffer):
    """Reads a CSV, letting exceptions propagate to the caller — callers
    are responsible for catching and surfacing a friendly message, since
    what "friendly" means differs by context (missing file vs a bad
    upload vs a corrupt local file)."""
    return pd.read_csv(path_or_buffer)


def get_data():
    data = {}
    missing = []
    load_errors = {}

    for fname in REQUIRED_FILES:
        if os.path.exists(fname):
            try:
                data[fname] = load_csv(fname)
                logger.info(f"Loaded {fname} ({len(data[fname])} rows)")
            except Exception as e:
                logger.error(f"Failed to load {fname}: {e}")
                load_errors[fname] = str(e)
        else:
            missing.append(fname)
            logger.warning(f"{fname} not found locally")

    if load_errors:
        for fname, err in load_errors.items():
            st.sidebar.error(f"⚠️ Couldn't read {fname} — it may be corrupted or malformed.\n\n`{err}`")

    if missing:
        st.sidebar.warning(f"Missing locally: {', '.join(missing)}. Upload them below.")
        for fname in missing:
            uploaded = st.sidebar.file_uploader(f"Upload {fname}", type="csv", key=fname)
            if uploaded is not None:
                try:
                    data[fname] = load_csv(uploaded)
                    logger.info(f"Loaded {fname} from upload ({len(data[fname])} rows)")
                except Exception as e:
                    logger.error(f"Failed to load uploaded {fname}: {e}")
                    st.sidebar.error(f"⚠️ Couldn't read the uploaded {fname}: {e}")

    return data


st.sidebar.title("🏏 Navigation")

try:
    data = get_data()
except Exception as e:
    logger.error(f"Unexpected error during data loading: {e}")
    st.error(
        "Something went wrong loading the dataset. This shouldn't normally happen — "
        f"here's the technical detail: `{e}`"
    )
    st.stop()

if len(data) < 3:
    st.info(
        "👋 Waiting for data. Run `preprocess.py` on your full dataset locally, "
        "then either place the 3 output CSVs next to `app.py`, or upload them "
        "using the sidebar uploaders."
    )
    st.stop()

match_summary_all = data["bowler_match_summary.csv"].copy()
master = data["bowler_master.csv"].copy()

match_summary_all["start_date"] = pd.to_datetime(match_summary_all["start_date"], errors="coerce")
match_summary_all["acwr_tier"] = match_summary_all["acwr"].apply(acwr_tier)
match_summary_all = match_summary_all.merge(master, on="bowler", how="left")
match_summary_all["display_name"] = match_summary_all["full_name"].fillna(match_summary_all["bowler"])
has_venue = "venue" in match_summary_all.columns

# ==================================================================
# WELCOME SCREEN — plays once, before the format gate
# ==================================================================

if "entered_app" not in st.session_state:
    st.session_state.entered_app = False
if "entering_app" not in st.session_state:
    st.session_state.entering_app = False

if st.session_state.entering_app:
    render_entry_transition(match_summary_all)
    st.session_state.entered_app = True
    st.session_state.entering_app = False
    st.rerun()

if not st.session_state.entered_app:
    render_welcome_screen(match_summary_all)
    st.stop()

# ==================================================================
# FORMAT GATE — must pick T20 / ODI / Test / All before anything else
# ==================================================================

if "app_format" not in st.session_state:
    st.session_state.app_format = None
if "pending_format" not in st.session_state:
    st.session_state.pending_format = None

if st.session_state.pending_format is not None:
    render_analyzing_screen(st.session_state.pending_format, match_summary_all)
    st.session_state.app_format = st.session_state.pending_format
    st.session_state.pending_format = None
    st.rerun()

if st.session_state.app_format is None:
    render_format_gate(match_summary_all)
    st.stop()

format_choice = st.session_state.app_format

# ==================================================================
# GLOBAL FILTERS
# ==================================================================

top_l, top_r = st.columns([5, 1])
with top_l:
    fmt_glyph = FORMAT_THEME.get(format_choice.upper() if format_choice != "All" else "All", FORMAT_THEME["All"])["glyph"]
    st.markdown(
        f'<span class="fmt-pill" style="background:{FORMAT_THEME.get(format_choice.upper() if format_choice != "All" else "All", FORMAT_THEME["All"])["accent"]}22;'
        f'border-color:{FORMAT_THEME.get(format_choice.upper() if format_choice != "All" else "All", FORMAT_THEME["All"])["accent"]}66;">'
        f'{fmt_glyph} Viewing: {format_choice if format_choice != "All" else "All Formats"}</span>',
        unsafe_allow_html=True,
    )
with top_r:
    if st.button("⇄ Switch format", width='stretch'):
        st.session_state.app_format = None
        st.rerun()

st.sidebar.subheader("Filters")
available_formats = sorted(match_summary_all["format"].dropna().unique())
sidebar_index = (["All"] + available_formats).index(format_choice) if format_choice in (["All"] + available_formats) else 0
sidebar_format_choice = st.sidebar.selectbox("Match format", ["All"] + available_formats, index=sidebar_index)
if sidebar_format_choice != st.session_state.app_format:
    st.session_state.pending_format = sidebar_format_choice
    st.rerun()

available_countries = sorted(match_summary_all["country"].dropna().unique()) if "country" in match_summary_all.columns else []
country_choice = st.sidebar.multiselect("Country (optional)", available_countries)

match_summary = match_summary_all
if format_choice != "All":
    match_summary = match_summary[match_summary["format"] == format_choice]
if country_choice:
    match_summary = match_summary[match_summary["country"].isin(country_choice)]

vs_team = compute_vs_team(match_summary)
vs_team = vs_team.merge(master, on="bowler", how="left")
vs_team["display_name"] = vs_team["full_name"].fillna(vs_team["bowler"])

latest_state = (
    match_summary.sort_values("start_date")
    .groupby("bowler", as_index=False)
    .last()
) if not match_summary.empty else match_summary.copy()

# Injury log is read fresh each run (small file, no caching needed) so
# writes from the logging form are reflected immediately.
injury_log_df = load_injury_log()
latest_state["current_status"] = latest_state["bowler"].apply(lambda b: get_current_status(b, injury_log_df))

# ML matchup predictor — trained once on the full, unfiltered dataset
# (format is itself a model feature, so training on everything gives it
# more real examples to learn from than retraining per format filter).
# Both calls are cached, so this is only genuinely computed once per
# dataset, not on every rerun.
ml_training_df, ml_le_format, ml_le_opponent = prepare_ml_training_data(match_summary_all)
ml_models = train_performance_models(ml_training_df)

# ==================================================================
# SIDEBAR NAV — custom icon-based nav, not the native radio widget.
# Streamlit's radio only supports plain-text labels (which is why emoji
# were used before — they're just Unicode glyphs). Real per-item SVG
# icons need actual HTML, so this renders each nav entry as its own
# button with a CSS-injected icon, giving full control over a clean,
# monochrome, single-accent-color look instead of colorful emoji.
# ==================================================================

NAV_ICONS = {
    "home": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M9 22V12h6v10"/>',
    "injury_fitness": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "squad_impact": '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10"/><path d="M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    "acwr_engine": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10"/>',
    "dataset_overview": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "player_profile": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "compare_players": '<path d="M12 3v18"/><path d="M5 7h14"/><path d="M5 7l-2.5 5.5a3 3 0 0 0 5.9 0z"/><path d="M19 7l-2.5 5.5a3 3 0 0 0 5.9 0z"/>',
    "advanced_search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "leaderboard": '<circle cx="12" cy="8" r="6"/><polyline points="8.2 13.9 7 22 12 19 17 22 15.8 13.9"/>',
    "bowler_selection": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "workload_monitor": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "team_overview": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M15.5 3.13a4 4 0 0 1 0 7.75"/>',
    "export_data": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "ask_data": '<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>',
    "methodology": '<circle cx="12" cy="12" r="9"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
}

NAV_ITEMS = [
    ("home", "Home"),
    ("injury_fitness", "Injury & Fitness"),
    ("squad_impact", "Squad Impact Engine"),
    ("acwr_engine", "ACWR Engine"),
    ("dataset_overview", "Dataset Overview"),
    ("player_profile", "Player Profile"),
    ("compare_players", "Compare Players"),
    ("advanced_search", "Advanced Search"),
    ("leaderboard", "Leaderboard"),
    ("bowler_selection", "Bowler Selection"),
    ("workload_monitor", "Workload Monitor"),
    ("team_overview", "Team Overview"),
    ("ask_data", "Ask the Data"),
    ("export_data", "Export Data"),
    ("methodology", "Methodology"),
]


def _svg_data_uri(inner_path, color):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{inner_path}</svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


_nav_icon_css = ["<style>"]
for slug, inner in NAV_ICONS.items():
    icon_inactive = _svg_data_uri(inner, "%239aa4b2")   # muted gray, URL-encoded '#'
    icon_active = _svg_data_uri(inner, "%23ffffff")     # white on filled accent background
    _nav_icon_css.append(f"""
    div[class*="st-key-nav_{slug}"] button {{
        display: flex !important; align-items: center !important; justify-content: flex-start !important;
        gap: 11px !important; text-align: left !important; font-weight: 500 !important;
        background: transparent !important; border: 1px solid transparent !important;
        padding: 8px 10px !important; border-radius: 8px !important; box-shadow: none !important;
    }}
    div[class*="st-key-nav_{slug}"] button::before {{
        content: ""; display: inline-block; width: 17px; height: 17px; flex-shrink: 0;
        background-image: url("{icon_inactive}"); background-size: contain; background-repeat: no-repeat;
    }}
    div[class*="st-key-nav_{slug}"] button:hover {{
        background: rgba(255,255,255,0.05) !important; border-color: var(--border) !important;
    }}
    div[class*="st-key-nav_{slug}"] button[data-testid="stBaseButton-primary"] {{
        background: linear-gradient(120deg, #145c3f, #1f7a52) !important;
        border-color: #22b573 !important; color: #fff !important;
    }}
    div[class*="st-key-nav_{slug}"] button[data-testid="stBaseButton-primary"]::before {{
        background-image: url("{icon_active}");
    }}
    """)
_nav_icon_css.append("</style>")
st.sidebar.markdown("\n".join(_nav_icon_css), unsafe_allow_html=True)

if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

st.sidebar.subheader("Go to")
for slug, label in NAV_ITEMS:
    is_active = st.session_state.current_page == slug
    if st.sidebar.button(
        label, key=f"nav_{slug}", use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state.current_page = slug
        st.rerun()

page = st.session_state.current_page

st.sidebar.markdown("---")
st.sidebar.caption(
    "Performance stats are computed from real ball-by-ball data. Workload risk "
    "uses ACWR, a published sports-science heuristic — not a validated injury "
    "prediction model, since no public dataset contains real injury outcomes."
)
if not has_venue:
    st.sidebar.caption("ℹ️ Venue analysis needs `preprocess.py` rerun with venue capture — optional.")

if match_summary.empty:
    st.warning("No records found for the current filters. Try loosening the format/country filter.")
    st.stop()

# ==================================================================
# PAGE: HOME
# ==================================================================

if page == "home":
    st.title("Home")
    filt_label = format_choice if format_choice != "All" else "All formats"
    if country_choice:
        filt_label += f" • {', '.join(country_choice)}"
    st.caption(f"Showing: **{filt_label}**")

    home_open_alerts = compute_alerts(latest_state, match_summary, injury_log_df)
    alert_note = (
        f"{len(home_open_alerts)} open alert{'s' if len(home_open_alerts) != 1 else ''} right now"
        if home_open_alerts else "No open alerts right now"
    )
    ic1, ic2 = st.columns([4, 1])
    with ic1:
        st.markdown(f"""
        <div class="injury-callout">
          <div class="ic-text">
            <div class="eyebrow">Core of this project</div>
            <h3>🚑 Injury &amp; Fitness Management</h3>
            <p>Real, dated case logging, return-to-play ramp plans, multi-factor risk alerts,
            and exportable fitness passports — the ground truth ACWR alone can't give you. {alert_note}.</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with ic2:
        st.write("")
        st.write("")
        if st.button("Open →", key="goto_injury", use_container_width=True):
            st.session_state.current_page = "injury_fitness"
            st.rerun()

    cols = st.columns(5)
    with cols[0]:
        st.markdown(kpi_card(match_summary["bowler"].nunique(), "Bowlers"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(kpi_card(match_summary["match_id"].nunique(), "Matches"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(kpi_card(int(match_summary["wickets"].sum()), "Total wickets"), unsafe_allow_html=True)
    with cols[3]:
        avg_econ = match_summary["runs_conceded"].sum() / max(match_summary["overs_bowled"].sum(), 1e-9)
        st.markdown(kpi_card(f"{avg_econ:.2f}", "Avg economy"), unsafe_allow_html=True)
    with cols[4]:
        high_risk_n = (latest_state["acwr_tier"] == "High").sum()
        st.markdown(kpi_card(high_risk_n, "High ACWR risk now"), unsafe_allow_html=True)

    st.write("")
    st.caption("💡 Tip: click any bar below to jump straight to that bowler's full profile.")
    col1, col2 = st.columns(2)

    with col1:
        with section("Top wicket-takers", "🎯"):
            top_wickets = (
                match_summary.groupby("display_name")["wickets"].sum()
                .sort_values(ascending=False).head(8)
            )
            fig = go.Figure(go.Bar(
                x=top_wickets.values[::-1], y=top_wickets.index[::-1], orientation='h',
                marker_color="#22b573",
                hovertemplate="<b>%{y}</b><br>%{x} wickets<extra></extra>",
            ))
            fig.update_layout(
                height=320, margin=dict(l=4, r=10, t=4, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c7cdd8", family="Inter"),
                xaxis=dict(title="Total wickets", gridcolor="#1c222c"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                clickmode="event+select",
            )
            event = st.plotly_chart(fig, key="home_wickets_chart", on_select="rerun", use_container_width=True, config={"displayModeBar": False})
            pts = event.get("selection", {}).get("points", []) if event else []
            if pts:
                st.session_state.jump_to_player = pts[0]["y"]
                st.session_state.current_page = "player_profile"
                st.rerun()

    with col2:
        with section("Most economical (min 3 matches)", "💰"):
            econ_stats = match_summary.groupby("display_name").agg(
                avg_economy=("economy", "mean"), matches=("match_id", "nunique")
            )
            econ_stats = econ_stats[econ_stats["matches"] >= 3].sort_values("avg_economy").head(8)
            fig = go.Figure(go.Bar(
                x=econ_stats["avg_economy"].values[::-1], y=econ_stats.index[::-1], orientation='h',
                marker_color="#4f9fd8",
                hovertemplate="<b>%{y}</b><br>Economy %{x:.2f}<extra></extra>",
            ))
            fig.update_layout(
                height=320, margin=dict(l=4, r=10, t=4, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c7cdd8", family="Inter"),
                xaxis=dict(title="Avg economy", gridcolor="#1c222c"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                clickmode="event+select",
            )
            event = st.plotly_chart(fig, key="home_economy_chart", on_select="rerun", use_container_width=True, config={"displayModeBar": False})
            pts = event.get("selection", {}).get("points", []) if event else []
            if pts:
                st.session_state.jump_to_player = pts[0]["y"]
                st.session_state.current_page = "player_profile"
                st.rerun()

    col3, col4 = st.columns(2)
    with col3:
        with section("Bowling style distribution", "🎳"):
            if "bowling_style" in match_summary.columns:
                style_counts = match_summary.drop_duplicates("bowler")["bowling_style"].value_counts().head(8)
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_alpha(0)
                ax.pie(style_counts.values, labels=style_counts.index, autopct='%1.0f%%',
                       colors=PALETTE * 2, textprops={'color': "#cdd5e0", 'fontsize': 8})
                st.pyplot(fig)

    with col4:
        with section("Matches per season", "📅"):
            if "season" in match_summary.columns:
                season_counts = match_summary.groupby("season")["match_id"].nunique().sort_index()
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_alpha(0)
                ax.set_facecolor('none')
                ax.bar(season_counts.index.astype(str), season_counts.values, color="#1f7a52")
                ax.tick_params(colors="#cdd5e0")
                plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)

# ==================================================================
# PAGE: ACWR ENGINE  (hero feature)
# ==================================================================

elif page == "acwr_engine":
    st.markdown(
        '<div class="acwr-hero">'
        '<div class="eyebrow">THE ENGINE BEHIND EVERY WORKLOAD NUMBER IN THIS APP</div>'
        '<h2>🧬 ACWR — Acute:Chronic Workload Ratio</h2>'
        '<p>Every risk badge, gauge, and "high risk" flag you see in this platform comes from one number: '
        'how much a bowler bowled in their most recent match, relative to their own recent normal. '
        'It is a published sports-science heuristic (Gabbett et al.) used by real strength & conditioning '
        'staff — not a black-box score invented for this app.</p>'
        '<div class="acwr-formula">ACWR &nbsp;=&nbsp; Acute workload (this match) &nbsp;÷&nbsp; Chronic workload (avg of last 4 matches)</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    tier_colors = {"Undertrained": "#7ab8e8", "Low": "#6fd18a", "Moderate": "#e8c15a", "High": "#f08080"}
    tier_verdicts = {
        "Undertrained": "Bowling noticeably less than their own recent normal. Not dangerous by itself, but a "
                         "sudden jump back up from here is exactly the kind of spike ACWR is designed to catch.",
        "Low": "This is the published \u201csweet spot.\u201d Current workload is close to what this bowler is "
               "already conditioned for \u2014 associated with the lowest injury risk in the research.",
        "Moderate": "Workload is climbing faster than their recent normal. Not an emergency, but worth watching "
                    "over the next match or two.",
        "High": "A rapid spike relative to recent normal \u2014 the workload pattern most strongly associated "
                "with elevated injury risk in sports-science research. Worth a real conversation with the "
                "bowler and medical staff, not just this app.",
    }

    col1, col2 = st.columns([1.15, 1])

    with col1:
        with section("Try any ACWR value", "\U0001F39B\uFE0F"):
            st.caption("Drag the slider — this is the exact zone logic used everywhere else in the app.")
            demo_val = st.slider("Hypothetical ACWR", 0.0, 2.4, 1.0, 0.01, key="acwr_demo_slider")
            demo_tier = acwr_tier(demo_val)
            demo_pct = pct_from_range(demo_val, 0, 2.4)

            st.markdown(
                '<div class="zone-bar-track">'
                '<div class="zone-seg" style="flex-grow:0.8; background:#2f6fa8;">UNDER</div>'
                '<div class="zone-seg" style="flex-grow:0.5; background:#22b573;">SWEET SPOT</div>'
                '<div class="zone-seg" style="flex-grow:0.2; background:#c9a227;">MODERATE</div>'
                '<div class="zone-seg" style="flex-grow:0.9; background:#b0413e;">HIGH</div>'
                f'<div class="zone-marker" style="left:{demo_pct}%;"></div>'
                '</div>'
                '<div class="zone-labels"><span>0.0</span><span>0.8</span><span>1.3</span><span>1.5</span><span>2.4</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="verdict-card"><div class="v-tier" style="color:{tier_colors[demo_tier]};">{demo_tier}</div>'
                f'<div class="v-text">{tier_verdicts[demo_tier]}</div></div>',
                unsafe_allow_html=True,
            )

    with col2:
        with section("Squad risk breakdown", "\U0001F4CA"):
            st.caption("Every currently-filtered bowler's latest ACWR tier, right now.")
            tier_counts = latest_state["acwr_tier"].value_counts()
            total = int(tier_counts.sum())
            bar_colors = {"Low": "#6fd18a", "Moderate": "#e8c15a", "High": "#f08080", "Undertrained": "#7ab8e8"}
            for tier in ["Low", "Moderate", "High", "Undertrained"]:
                n = int(tier_counts.get(tier, 0))
                pct = (n / total * 100) if total else 0
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; font-size:12.5px; color:var(--text-mid); margin-bottom:3px;">'
                    f'<span>{tier}</span><span style="font-family:\'JetBrains Mono\',monospace;">{n} ({pct:.0f}%)</span></div>'
                    f'<div class="zone-bar-track" style="height:8px; margin-bottom:14px;">'
                    f'<div style="flex:none; width:{pct}%; background:{bar_colors[tier]};"></div></div>',
                    unsafe_allow_html=True,
                )

    with section("What-if simulator \u2014 plan a bowler's next match", "\U0001F52E"):
        st.caption("Pick a real bowler, then try a hypothetical overs count for their next match to see the ACWR before it happens.")
        sim_names = sorted(latest_state["display_name"].dropna().unique())
        if sim_names:
            sim_selected = st.selectbox("Bowler", sim_names, key="acwr_sim_player")
            sim_row = latest_state[latest_state["display_name"] == sim_selected].iloc[0]
            player_hist = match_summary[match_summary["display_name"] == sim_selected].sort_values("start_date")
            chronic_avg = float(player_hist["overs_bowled"].tail(4).mean()) if not player_hist.empty else 0.0

            max_overs = 40.0 if str(sim_row.get("format", "")).upper() == "TEST" else 20.0
            hypo_overs = st.slider(
                "Hypothetical overs bowled next match", 0.0, max_overs,
                float(round(min(chronic_avg, max_overs), 1)), 0.5, key="acwr_sim_overs"
            )
            hypo_acwr = hypo_overs / chronic_avg if chronic_avg > 0 else 1.0
            hypo_tier = acwr_tier(hypo_acwr)

            sc1, sc2 = st.columns([1, 1.4])
            with sc1:
                fig = draw_acwr_gauge(hypo_acwr)
                st.pyplot(fig)
            with sc2:
                st.markdown(
                    f'<div class="verdict-card">'
                    f'<div class="v-tier" style="color:{tier_colors[hypo_tier]};">Projected: {hypo_tier}</div>'
                    f'<div class="v-text">Their recent average workload (chronic) is <b>{chronic_avg:.1f} overs</b> per match. '
                    f'Bowling <b>{hypo_overs:.1f} overs</b> next time would put their ACWR at <b>{hypo_acwr:.2f}</b>. '
                    f'{tier_verdicts[hypo_tier]}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No bowlers available in the current filter to simulate.")

    with section("Newly flagged \u2014 risk that just changed", "\U0001F6A8"):
        st.caption("Bowlers whose risk tier got worse between their last two matches on record \u2014 the earliest possible warning sign.")
        trend_df = compute_acwr_trend(match_summary)
        newly_high = trend_df[trend_df["newly_high"]] if not trend_df.empty else pd.DataFrame()
        if newly_high.empty:
            st.success("No bowler has freshly crossed into High risk between their last two matches in this view.")
        else:
            tiles = []
            for row in newly_high.itertuples():
                tiles.append(render_tile(
                    avatar_text=row.display_name[:2].upper(), avatar_bg="#b0413e",
                    category_label="JUST CHANGED", title=row.display_name,
                    subtitle=f"{row.prev_tier} \u2192 {row.curr_tier}",
                    rows=[{"label": "ACWR now", "pct": pct_from_range(row.curr_acwr, 0, 2.4), "bar_color": "#f08080",
                           "pill_text": f"{row.curr_acwr:.2f}", "pill_color": "#f08080"}],
                    footer_left=f"was {row.prev_acwr:.2f}",
                    footer_right='<span class="trend-badge trend-up">\u2191 worsened</span>'
                ))
            tile_grid(tiles, n_cols=2)

    st.markdown('<div class="seam"></div>', unsafe_allow_html=True)
    st.caption(
        "ACWR is a published heuristic, not a validated injury predictor \u2014 no public dataset contains "
        "real bowler injury outcomes to train or validate one against. Treat it as an early-warning signal "
        "that should inform, not replace, real medical and coaching judgment. Full formula and citation on "
        "the Methodology page."
    )

# ==================================================================
# PAGE: DATASET OVERVIEW
# ==================================================================

elif page == "dataset_overview":
    st.title("Dataset Overview")
    page_intro(
        "A quick snapshot of the raw data behind everything else in this app — how many "
        "bowlers, how many matches, and where they're spread across formats, seasons, and countries."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bowlers", match_summary["bowler"].nunique())
    c2.metric("Matches", match_summary["match_id"].nunique())
    c3.metric("Bowler-match records", len(match_summary))
    date_range = f"{match_summary['start_date'].min().date()} → {match_summary['start_date'].max().date()}"
    c4.metric("Date range", date_range)

    tab1, tab2, tab3, tab4 = st.tabs(["By format", "By season", "By country", "Raw sample"])

    with tab1:
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_alpha(0); ax.set_facecolor('none')
        match_summary_all["format"].value_counts().plot(kind="bar", ax=ax, color="#1f7a52")
        ax.set_ylabel("Bowler-match records", color="#9aa4b2")
        ax.tick_params(colors="#cdd5e0")
        st.pyplot(fig)

    with tab2:
        if "season" in match_summary.columns:
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_alpha(0); ax.set_facecolor('none')
            match_summary.groupby("season")["match_id"].nunique().sort_index().plot(kind="bar", ax=ax, color="#1f7a52")
            ax.set_ylabel("Matches", color="#9aa4b2")
            ax.tick_params(colors="#cdd5e0")
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)

    with tab3:
        if "country" in match_summary.columns:
            country_counts = match_summary.drop_duplicates("bowler")["country"].value_counts().head(12)
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_alpha(0); ax.set_facecolor('none')
            ax.barh(country_counts.index[::-1], country_counts.values[::-1], color="#2f7ab0")
            ax.tick_params(colors="#cdd5e0")
            st.pyplot(fig)

    with tab4:
        st.dataframe(
            match_summary[["display_name", "country", "start_date", "format", "batting_team",
                            "overs_bowled", "economy", "wickets", "acwr", "acwr_tier"]].head(20),
            width='stretch'
        )

# ==================================================================
# PAGE: PLAYER PROFILE
# ==================================================================

elif page == "player_profile":
    st.title("Player Profile")
    page_intro(
        "Search for any bowler to see their real career numbers, how their workload has "
        "trended over time, and whether their current bowling load looks safe or risky."
    )

    jump_target = st.session_state.pop("jump_to_player", None)
    search = st.text_input("🔍 Search player", "")
    names = sorted(latest_state["display_name"].dropna().unique())
    if search:
        names = [n for n in names if search.lower() in n.lower()]

    if not names:
        st.warning("No players match that search (in the current filters).")
    else:
        default_idx = names.index(jump_target) if jump_target in names else 0
        selected = st.selectbox("Select player", names, index=default_idx)
        prow = latest_state[latest_state["display_name"] == selected].iloc[0]
        player_matches = match_summary[match_summary["display_name"] == selected].sort_values("start_date")

        tier = prow["acwr_tier"]
        current_status = prow.get("current_status", "Fit")
        c1, c2, c3 = st.columns([1, 1, 1.3])
        with c1:
            if isinstance(prow.get("image_url"), str) and prow["image_url"].startswith("http"):
                st.image(prow["image_url"], width=140)
            st.markdown(f"""
            <div class="player-card">
              <b style="font-size:18px;">{selected}</b><br>
              <span class="meta">{prow.get('country','Unknown')} • {prow.get('bowling_style','Unknown')}</span><br><br>
              {tier_badge(tier)} {status_badge(current_status)}
            </div>
            """, unsafe_allow_html=True)

        with c2:
            fig = draw_acwr_gauge(prow['acwr'])
            st.pyplot(fig)
            safe_overs = safe_overs_ceiling(prow.get('chronic_avg_overs'))
            if safe_overs is not None:
                st.caption(f"📏 **Safe ceiling this match: ~{safe_overs} overs**, based on their recent average.")
            with st.expander("What does this gauge mean?"):
                st.caption(
                    "It compares this player's most recent workload to their own recent "
                    "average. Green (around 1.0) means their current load is normal for "
                    "them. Red means they've bowled a lot more than usual very recently — "
                    "a pattern sports scientists associate with higher injury risk."
                )

        with c3:
            m1, m2 = st.columns(2)
            m1.metric("Matches on record", len(player_matches))
            m2.metric("Wickets (filtered)", int(player_matches['wickets'].sum()))
            m3, m4 = st.columns(2)
            m3.metric("Economy (filtered)",
                      f"{player_matches['runs_conceded'].sum() / max(player_matches['overs_bowled'].sum(), 1e-9):.2f}")
            m4.metric("Rest days (latest)", f"{prow.get('rest_days_before', 0):.0f}")

        # Milestones
        if not player_matches.empty:
            best_wickets_row = player_matches.loc[player_matches['wickets'].idxmax()]
            qualifying = player_matches[player_matches['overs_bowled'] >= 2]
            best_econ_row = qualifying.loc[qualifying['economy'].idxmin()] if not qualifying.empty else None

            st.markdown(
                f'<span class="milestone-chip">🏅 Best match: <b>{int(best_wickets_row["wickets"])} wkts</b> '
                f'vs {best_wickets_row["batting_team"]} ({best_wickets_row["start_date"].date()})</span>',
                unsafe_allow_html=True
            )
            if best_econ_row is not None:
                st.markdown(
                    f'<span class="milestone-chip">💎 Best economy: <b>{best_econ_row["economy"]:.2f}</b> '
                    f'vs {best_econ_row["batting_team"]} ({best_econ_row["start_date"].date()})</span>',
                    unsafe_allow_html=True
                )

        st.subheader("Workload over time (overs bowled + ACWR)")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        fig.patch.set_alpha(0)
        for ax in (ax1, ax2):
            ax.set_facecolor('none')
            ax.tick_params(colors="#cdd5e0")
        ax1.plot(player_matches["start_date"], player_matches["overs_bowled"], marker="o", color="#1f7a52")
        ax1.set_ylabel("Overs bowled", color="#9aa4b2")
        ax2.plot(player_matches["start_date"], player_matches["acwr"], marker="o", color="#b04a4a")
        if not player_matches.empty:
            ax2.axhspan(0.8, 1.3, color="green", alpha=0.1)
            ax2.axhspan(1.3, 1.5, color="orange", alpha=0.1)
            ax2.axhspan(1.5, player_matches["acwr"].max() + 0.5, color="red", alpha=0.1)
        ax2.set_ylabel("ACWR", color="#9aa4b2")
        ax2.set_xlabel("Match date", color="#9aa4b2")
        st.pyplot(fig)

        if "season" in player_matches.columns and not player_matches.empty:
            st.subheader("Career trend: wickets & economy by season")
            season_trend = player_matches.groupby("season").agg(
                wickets=("wickets", "sum"), economy=("economy", "mean"), matches=("match_id", "nunique"),
            ).reset_index()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
            fig.patch.set_alpha(0)
            for ax in (ax1, ax2):
                ax.set_facecolor('none')
                ax.tick_params(colors="#cdd5e0")
            ax1.bar(season_trend["season"].astype(str), season_trend["wickets"], color="#1f7a52")
            ax1.set_title("Wickets per season", color="#cdd5e0")
            plt.setp(ax1.get_xticklabels(), rotation=60, ha="right", fontsize=8)
            ax2.plot(season_trend["season"].astype(str), season_trend["economy"], marker="o", color="#b04a4a")
            ax2.set_title("Avg economy per season", color="#cdd5e0")
            plt.setp(ax2.get_xticklabels(), rotation=60, ha="right", fontsize=8)
            st.pyplot(fig)

            season_trend["cumulative_wickets"] = season_trend["wickets"].cumsum()
            fig, ax = plt.subplots(figsize=(9, 3.5))
            fig.patch.set_alpha(0); ax.set_facecolor('none'); ax.tick_params(colors="#cdd5e0")
            ax.plot(season_trend["season"].astype(str), season_trend["cumulative_wickets"], marker="o", color="#2f7ab0")
            ax.set_ylabel("Cumulative wickets", color="#9aa4b2")
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)

        st.subheader("Performance by opponent")
        player_vs_team = vs_team[vs_team["display_name"] == selected].sort_values("performance_score", ascending=False)
        if not player_vs_team.empty:
            econ_min, econ_max = player_vs_team["avg_economy"].min(), player_vs_team["avg_economy"].max()
            wkt_min, wkt_max = player_vs_team["avg_wickets"].min(), player_vs_team["avg_wickets"].max()

            tiles = []
            for row in player_vs_team.itertuples():
                econ_pct = pct_from_range(row.avg_economy, econ_min, econ_max, invert=True)
                wkt_pct = pct_from_range(row.avg_wickets, wkt_min, wkt_max)
                perf_good = row.performance_score >= player_vs_team["performance_score"].median()
                pill_color = "#6fd18a" if perf_good else "#e8c15a"
                tiles.append(render_tile(
                    avatar_text=row.opponent_team[:2].upper(),
                    avatar_bg=category_color(row.opponent_team),
                    category_label=row.opponent_team,
                    title=f"Performance score: {row.performance_score:.2f}",
                    subtitle=None,
                    rows=[
                        {"label": "Economy", "pct": econ_pct, "bar_color": "#2f7ab0",
                         "pill_text": f"{row.avg_economy:.2f}", "pill_color": pill_color},
                        {"label": "Avg wickets", "pct": wkt_pct, "bar_color": "#1f7a52",
                         "pill_text": f"{row.avg_wickets:.1f}", "pill_color": pill_color},
                    ],
                    footer_left=f"{row.matches_played} matches", footer_right=f"vs {row.opponent_team}"
                ))
            tile_grid(tiles, n_cols=2)

# ==================================================================
# PAGE: COMPARE PLAYERS
# ==================================================================

elif page == "compare_players":
    st.title("Compare Players")
    page_intro(
        "Put two or three bowlers side by side — wickets, economy, workload safety, and "
        "consistency — to see who's the stronger pick at a glance."
    )

    names = sorted(latest_state["display_name"].dropna().unique())
    if len(names) < 2:
        st.warning("Need at least 2 players in the current filters to compare.")
    else:
        num_players = st.radio("How many players?", [2, 3], horizontal=True)
        cols = st.columns(num_players)
        selected_players = []
        for i, col in enumerate(cols):
            with col:
                p = st.selectbox(f"Player {i+1}", names, index=min(i, len(names) - 1), key=f"cmp_{i}")
                selected_players.append(p)

        rows = [latest_state[latest_state["display_name"] == p].iloc[0] for p in selected_players]

        cards = st.columns(num_players)
        for col, row, pname in zip(cards, rows, selected_players):
            with col:
                tier = row["acwr_tier"]
                st.markdown(f"""
                <div class="player-card">
                  <b>{pname}</b><br>
                  <span class="meta">{row.get('country','-')} • {row.get('bowling_style','-')}</span><br><br>
                  {tier_badge(tier)}
                </div>
                """, unsafe_allow_html=True)

        st.subheader("Head-to-head stats")

        if num_players == 2:
            row_a, row_b = rows[0], rows[1]
            name_a, name_b = selected_players[0], selected_players[1]
            wkts_a = int(match_summary[match_summary['display_name'] == name_a]['wickets'].sum())
            wkts_b = int(match_summary[match_summary['display_name'] == name_b]['wickets'].sum())

            bars_html = "".join([
                render_compare_bar("Wickets (filtered)", f"{wkts_a}", wkts_a, f"{wkts_b}", wkts_b, higher_is_better=True),
                render_compare_bar("Economy (lower is better)", f"{row_a['economy']:.2f}", row_a['economy'],
                                    f"{row_b['economy']:.2f}", row_b['economy'], higher_is_better=False),
                render_compare_bar("Latest overs bowled", f"{row_a['overs_bowled']:.1f}", row_a['overs_bowled'],
                                    f"{row_b['overs_bowled']:.1f}", row_b['overs_bowled'], higher_is_better=True),
                render_compare_bar("ACWR (closer to 1.0 is safer)",
                                    f"{row_a['acwr']:.2f}" if pd.notna(row_a['acwr']) else "N/A",
                                    -abs(row_a['acwr'] - 1.0) if pd.notna(row_a['acwr']) else 0,
                                    f"{row_b['acwr']:.2f}" if pd.notna(row_b['acwr']) else "N/A",
                                    -abs(row_b['acwr'] - 1.0) if pd.notna(row_b['acwr']) else 0,
                                    higher_is_better=True),
            ])
            st.markdown(
                f'<div style="display:flex; justify-content:space-between; padding:0 4px; '
                f'margin-bottom:10px;"><b style="color:#6fd18a;">{name_a}</b>'
                f'<b style="color:#4f9fd8;">{name_b}</b></div>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div class="section-card">{bars_html}</div>', unsafe_allow_html=True)
        else:
            metrics_rows = ["Country", "Bowling style", "Latest ACWR", "Risk tier",
                             "Latest economy", "Latest overs bowled", "Wickets (filtered)"]
            compare_data = {"Metric": metrics_rows}
            for row, pname in zip(rows, selected_players):
                compare_data[pname] = [
                    row.get("country", "-"), row.get("bowling_style", "-"),
                    f"{row['acwr']:.2f}", row["acwr_tier"],
                    f"{row['economy']:.2f}", f"{row['overs_bowled']:.1f}",
                    str(int(match_summary[match_summary['display_name'] == pname]['wickets'].sum()))
                ]
            st.dataframe(pd.DataFrame(compare_data), width='stretch', hide_index=True)

        st.subheader("Radar comparison")
        radar_categories = ["Wickets", "Economy (inv)", "Overs bowled", "ACWR safety", "Consistency"]
        all_players_agg = match_summary.groupby("display_name").agg(
            wickets=("wickets", "sum"), economy=("economy", "mean"),
            overs=("overs_bowled", "sum"), economy_std=("economy", "std")
        ).fillna(0)

        radar_data = {}
        for pname, row in zip(selected_players, rows):
            if pname not in all_players_agg.index:
                continue
            prow_agg = all_players_agg.loc[pname]
            wickets_n = normalize_series(all_players_agg["wickets"]).get(pname, 0.5)
            economy_n = normalize_series(all_players_agg["economy"], invert=True).get(pname, 0.5)
            overs_n = normalize_series(all_players_agg["overs"]).get(pname, 0.5)
            acwr_safety = 1 - min(abs(row["acwr"] - 1.0) / 1.2, 1.0) if pd.notna(row["acwr"]) else 0.5
            consistency_n = normalize_series(all_players_agg["economy_std"], invert=True).get(pname, 0.5)
            radar_data[pname] = [wickets_n, economy_n, overs_n, acwr_safety, consistency_n]

        if radar_data:
            fig = draw_radar(radar_data, radar_categories)
            st.pyplot(fig)
            with st.expander("What does this shape mean?"):
                st.caption(
                    "Each axis is a different strength, scaled 0–1 relative to the players "
                    "shown — bigger reach on an axis is better on that trait. A player whose "
                    "shape covers more overall area is stronger across more dimensions at "
                    "once, rather than being a one-trick performer."
                )

        combined = vs_team[vs_team["display_name"].isin(selected_players)]
        if not combined.empty:
            st.subheader("Performance by opponent")
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_alpha(0); ax.set_facecolor('none'); ax.tick_params(colors="#cdd5e0")
            sns.barplot(data=combined, x="opponent_team", y="performance_score", hue="display_name", ax=ax)
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)

# ==================================================================
# PAGE: ADVANCED SEARCH
# ==================================================================

elif page == "advanced_search":
    st.title("Advanced Search")
    page_intro(
        "Narrow the whole squad down using any combination of filters — country, bowling "
        "style, minimum wickets, economy range, or current workload risk — and export the results."
    )
    st.write("Filter the squad across multiple criteria at once.")

    agg = match_summary.groupby(["display_name", "bowler"]).agg(
        matches=("match_id", "nunique"),
        wickets=("wickets", "sum"),
        avg_economy=("economy", "mean"),
    ).reset_index()
    agg = agg.merge(latest_state[["bowler", "acwr", "acwr_tier", "country", "bowling_style"]], on="bowler", how="left")

    c1, c2, c3 = st.columns(3)
    with c1:
        country_filter = st.multiselect("Country", sorted(agg["country"].dropna().unique()))
        style_filter = st.multiselect("Bowling style", sorted(agg["bowling_style"].dropna().unique()))
    with c2:
        min_matches = st.slider("Minimum matches", 0, int(agg["matches"].max()), 1)
        min_wickets = st.slider("Minimum wickets", 0, int(agg["wickets"].max()), 0)
    with c3:
        econ_range = st.slider("Economy range", 0.0, float(agg["avg_economy"].max()) + 1,
                                 (0.0, float(agg["avg_economy"].max()) + 1))
        tier_filter = st.multiselect("ACWR risk tier", ["Low", "Moderate", "High", "Undertrained", "Unknown"])

    filtered = agg[
        (agg["matches"] >= min_matches) &
        (agg["wickets"] >= min_wickets) &
        (agg["avg_economy"].between(econ_range[0], econ_range[1]))
    ]
    if country_filter:
        filtered = filtered[filtered["country"].isin(country_filter)]
    if style_filter:
        filtered = filtered[filtered["bowling_style"].isin(style_filter)]
    if tier_filter:
        filtered = filtered[filtered["acwr_tier"].isin(tier_filter)]

    st.subheader(f"Results ({len(filtered)} players)")
    st.dataframe(
        filtered[["display_name", "country", "bowling_style", "matches", "wickets", "avg_economy", "acwr", "acwr_tier"]]
        .sort_values("wickets", ascending=False)
        .rename(columns={"display_name": "Player"}),
        width='stretch', hide_index=True
    )

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered results", data=csv_bytes, file_name="search_results.csv", mime="text/csv")

# ==================================================================
# PAGE: LEADERBOARD
# ==================================================================

elif page == "leaderboard":
    st.title("Leaderboard")
    page_intro(
        "Six different ways to rank the squad — from raw wicket-takers to the steadiest "
        "performers to who's improved the most season over season."
    )

    tabs = st.tabs(["Best performers", "Safest ACWR", "Highest workload", "Most wickets",
                     "Most consistent", "Rising stars"])

    with tabs[0]:
        avg_perf = vs_team.groupby("display_name")["performance_score"].mean().reset_index()
        avg_perf = avg_perf.sort_values("performance_score", ascending=False).head(10)
        if avg_perf.empty:
            st.info("No matchup data available for this filter.")
        else:
            vmin, vmax = avg_perf["performance_score"].min(), avg_perf["performance_score"].max()
            player_meta = latest_state.set_index("display_name")
            tiles = []
            for i, row in enumerate(avg_perf.itertuples(), start=1):
                meta = player_meta.loc[row.display_name] if row.display_name in player_meta.index else None
                country = meta["country"] if meta is not None else "-"
                style = meta["bowling_style"] if meta is not None else "-"
                pct = pct_from_range(row.performance_score, vmin, vmax)
                tiles.append(render_tile(
                    avatar_text=f"#{i}", avatar_bg="#1f7a52", category_label=style,
                    title=row.display_name, subtitle=country,
                    rows=[{"label": "Performance score", "pct": pct, "bar_color": "#1f7a52",
                           "pill_text": f"{row.performance_score:.2f}", "pill_color": "#6fd18a"}],
                    footer_left="Avg vs all opponents", footer_right=f"Rank #{i}"
                ))
            tile_grid(tiles, n_cols=2)

    with tabs[1]:
        safest = latest_state[latest_state["acwr_tier"] == "Low"].sort_values("acwr").head(10)
        if safest.empty:
            st.info("No bowlers currently in the Low-risk ACWR tier for this filter.")
        else:
            tiles = []
            for i, row in enumerate(safest.itertuples(), start=1):
                pct = pct_from_range(row.acwr, 0, 2.2)
                tiles.append(render_tile(
                    avatar_text=f"#{i}", avatar_bg="#2f6fa8", category_label=row.acwr_tier,
                    title=row.display_name, subtitle=row.country if hasattr(row, 'country') else None,
                    rows=[{"label": "ACWR", "pct": pct, "bar_color": "#6fd18a",
                           "pill_text": f"{row.acwr:.2f}", "pill_color": "#6fd18a"}],
                    footer_left=f"{row.overs_bowled:.1f} overs (latest)", footer_right="Safest workload"
                ))
            tile_grid(tiles, n_cols=2)

    with tabs[2]:
        high_load = latest_state.sort_values("acwr", ascending=False).head(10)
        tier_colors = {"Low": "#6fd18a", "Moderate": "#e8c15a", "High": "#f08080", "Undertrained": "#7ab8e8"}
        tiles = []
        for i, row in enumerate(high_load.itertuples(), start=1):
            pct = pct_from_range(row.acwr, 0, 2.2)
            pc = tier_colors.get(row.acwr_tier, "#999")
            tiles.append(render_tile(
                avatar_text=f"#{i}", avatar_bg="#b0413e", category_label=row.acwr_tier,
                title=row.display_name, subtitle=row.country if hasattr(row, 'country') else None,
                rows=[{"label": "ACWR", "pct": pct, "bar_color": pc,
                       "pill_text": f"{row.acwr:.2f}", "pill_color": pc}],
                footer_left=f"{row.overs_bowled:.1f} overs (latest)",
                footer_right="⚠️ High" if row.acwr_tier == "High" else row.acwr_tier
            ))
        tile_grid(tiles, n_cols=2)

    with tabs[3]:
        top_wickets = match_summary.groupby("display_name")["wickets"].sum().sort_values(ascending=False).head(10)
        if top_wickets.empty:
            st.info("No wicket data available for this filter.")
        else:
            vmax = top_wickets.max()
            player_meta = latest_state.set_index("display_name")
            tiles = []
            for i, (name, wkts) in enumerate(top_wickets.items(), start=1):
                meta = player_meta.loc[name] if name in player_meta.index else None
                style = meta["bowling_style"] if meta is not None else "-"
                country = meta["country"] if meta is not None else "-"
                pct = pct_from_range(wkts, 0, vmax)
                tiles.append(render_tile(
                    avatar_text=f"#{i}", avatar_bg="#1f7a52", category_label=style,
                    title=name, subtitle=country,
                    rows=[{"label": "Total wickets", "pct": pct, "bar_color": "#1f7a52",
                           "pill_text": f"{int(wkts)}", "pill_color": "#6fd18a"}],
                    footer_left="Wickets (filtered)", footer_right=f"Rank #{i}"
                ))
            tile_grid(tiles, n_cols=2)

    with tabs[4]:
        st.write("Lowest economy variance across matches (min 3 matches) — steadiest performers.")
        consistent = compute_consistency(match_summary).head(10)
        if consistent.empty:
            st.info("Not enough matches per player yet to compute consistency (need 3+ each).")
        else:
            vmin, vmax = consistent["economy_std"].min(), consistent["economy_std"].max()
            player_meta = latest_state.set_index("display_name")
            tiles = []
            for i, row in enumerate(consistent.itertuples(), start=1):
                meta = player_meta.loc[row.display_name] if row.display_name in player_meta.index else None
                style = meta["bowling_style"] if meta is not None else "-"
                country = meta["country"] if meta is not None else "-"
                pct = pct_from_range(row.economy_std, vmin, vmax, invert=True)
                tiles.append(render_tile(
                    avatar_text=f"#{i}", avatar_bg="#7a52b0", category_label=style,
                    title=row.display_name, subtitle=country,
                    rows=[{"label": "Economy std-dev", "pct": pct, "bar_color": "#7a52b0",
                           "pill_text": f"{row.economy_std:.2f}", "pill_color": "#6fd18a"}],
                    footer_left=f"{row.matches} matches", footer_right=f"Avg econ {row.avg_economy:.2f}"
                ))
            tile_grid(tiles, n_cols=2)

    with tabs[5]:
        st.write("Biggest improvement in average economy between their first and most recent season on record.")
        rising = compute_rising_stars(match_summary).head(10)
        if rising.empty:
            st.info("Not enough multi-season data available to compute this yet.")
        else:
            vmin, vmax = rising["improvement"].min(), rising["improvement"].max()
            player_meta = latest_state.set_index("display_name")
            tiles = []
            for i, row in enumerate(rising.itertuples(), start=1):
                meta = player_meta.loc[row.display_name] if row.display_name in player_meta.index else None
                style = meta["bowling_style"] if meta is not None else "-"
                pct = pct_from_range(row.improvement, vmin, vmax)
                improved = row.improvement > 0
                tiles.append(render_tile(
                    avatar_text=f"#{i}", avatar_bg="#c9a227" if improved else "#7a8494",
                    category_label=style, title=row.display_name,
                    subtitle=f"{row.first_season} → {row.last_season}",
                    rows=[{"label": "Economy change", "pct": pct,
                           "bar_color": "#6fd18a" if improved else "#f08080",
                           "pill_text": f"{'▼' if improved else '▲'} {abs(row.improvement):.2f}",
                           "pill_color": "#6fd18a" if improved else "#f08080"}],
                    footer_left=f"{row.first_economy:.2f} → {row.last_economy:.2f}",
                    footer_right="Improved" if improved else "Declined"
                ))
            tile_grid(tiles, n_cols=2)

# ==================================================================
# PAGE: BOWLER SELECTION
# ==================================================================

elif page == "bowler_selection":
    st.title("Matchup-Aware Bowler Selection")
    page_intro(
        "Picking a bowler for a specific opponent isn't just about who takes the most "
        "wickets — it's also about who's not overworked right now. This page blends both "
        "into one ranked recommendation, and you control how much weight each side gets."
    )

    opponents = sorted(vs_team["opponent_team"].dropna().unique())
    if not opponents:
        st.warning("No opponent data available for this filter.")
    else:
        opponent = st.selectbox("Opponent team", opponents)
        perf_weight = st.slider("Performance weight", 0.0, 1.0, 0.6, 0.05)
        safety_weight = 1 - perf_weight
        st.caption(f"Safety weight (auto): {safety_weight:.2f}")

        candidates = vs_team[vs_team["opponent_team"] == opponent].merge(
            latest_state[["bowler", "acwr"]], on="bowler", how="left"
        )
        if candidates.empty:
            st.warning("No bowlers found with recorded matches against this opponent.")
        else:
            min_p, max_p = candidates["performance_score"].min(), candidates["performance_score"].max()
            candidates["performance_norm"] = (candidates["performance_score"] - min_p) / (max_p - min_p + 1e-9)
            candidates["acwr_penalty"] = candidates["acwr"].apply(
                lambda a: 0 if pd.isna(a) else max(0, a - 1.3, 0.8 - a)
            )
            max_penalty = candidates["acwr_penalty"].max() or 1
            candidates["safety_score"] = 1 - (candidates["acwr_penalty"] / (max_penalty + 1e-9))
            candidates["selection_score"] = (
                perf_weight * candidates["performance_norm"] + safety_weight * candidates["safety_score"]
            )
            ranked = candidates.sort_values("selection_score", ascending=False).head(10)

            st.subheader(f"Recommended bowlers vs {opponent}")
            if ml_models is not None:
                st.caption(
                    "🤖 Each card includes a machine-learning prediction — a RandomForestRegressor "
                    "trained on real historical matches, not a formula — estimating expected economy "
                    "and wickets for this specific matchup based on the bowler's actual career form."
                )
            player_meta = latest_state.set_index("bowler")
            tiles = []
            unavailable_flagged = []
            for i, row in enumerate(ranked.itertuples(), start=1):
                meta = player_meta.loc[row.bowler] if row.bowler in player_meta.index else None
                style = meta["bowling_style"] if meta is not None else "-"
                country = meta["country"] if meta is not None else "-"
                status = meta["current_status"] if meta is not None else "Fit"

                score_color = "#6fd18a" if row.selection_score >= 0.6 else (
                    "#e8c15a" if row.selection_score >= 0.4 else "#f08080")
                safety_pct = row.safety_score * 100

                subtitle = f"{country} • {style} &nbsp; {status_badge(status)}"
                if status in ("Injured", "Rehab"):
                    unavailable_flagged.append(row.display_name)

                tile_rows = [
                    {"label": "Selection score", "pct": row.selection_score * 100, "bar_color": "#1f7a52",
                     "pill_text": f"{row.selection_score*100:.0f}%", "pill_color": score_color},
                    {"label": "Workload safety", "pct": safety_pct, "bar_color": "#2f7ab0",
                     "pill_text": f"ACWR {row.acwr:.2f}" if pd.notna(row.acwr) else "N/A",
                     "pill_color": "#6fd18a" if safety_pct >= 60 else "#e8c15a"},
                ]

                ml_pred = predict_matchup(row.bowler, opponent, format_choice, match_summary_all,
                                            ml_models, ml_le_format, ml_le_opponent)
                if ml_pred is not None:
                    pred_econ_pct = pct_from_range(ml_pred["predicted_economy"], 2.5, 15, invert=True)
                    tile_rows.append({
                        "label": "🤖 ML-predicted", "pct": pred_econ_pct, "bar_color": "#c39bd3",
                        "pill_text": f"{ml_pred['predicted_economy']} econ / {ml_pred['predicted_wickets']} wkts",
                        "pill_color": "#c9a9f5"
                    })

                tiles.append(render_tile(
                    avatar_text=f"#{i}", avatar_bg=category_color(opponent),
                    category_label=f"vs {opponent}", title=row.display_name, subtitle=subtitle,
                    rows=tile_rows,
                    footer_left=f"{row.avg_economy:.2f} econ / {row.avg_wickets:.1f} wkts (history)",
                    footer_right=f"{row.matches_played} matches"
                ))
            if unavailable_flagged:
                st.warning(
                    f"⚠️ Recommended by the numbers, but currently unavailable per the injury log: "
                    f"**{', '.join(unavailable_flagged)}**. Check Injury & Fitness before selecting."
                )
            tile_grid(tiles, n_cols=2)

            if ml_models is not None:
                with st.expander("🤖 About the ML model behind these predictions"):
                    m = ml_models["metrics"]
                    st.markdown(f"""
                    **Model**: RandomForestRegressor (scikit-learn), one for economy and one for wickets,
                    trained on **{m['n_train']:,}** real historical bowler-match rows and evaluated on
                    **{m['n_test']:,}** held-out rows it never saw during training.

                    | Target | Test MAE | Test R² |
                    |---|---|---|
                    | Economy | {m['economy_mae']:.2f} runs/over | {m['economy_r2']:.3f} |
                    | Wickets | {m['wickets_mae']:.2f} wickets | {m['wickets_r2']:.3f} |

                    **Features used** (all "trailing" — computed only from matches *before* the one being
                    predicted, so the model can't see the answer): the bowler's career-average economy and
                    wickets up to that point, how many matches they'd played, days of rest beforehand,
                    matches in the last 30 days, their rolling recent workload, the opponent team, and the
                    format.

                    This is the one place in the app that's an actual trained model rather than a formula —
                    everything else (ACWR, selection score, leaderboards) is real math on real data, but
                    hand-specified rather than learned.
                    """)
            elif ml_training_df is not None and len(ml_training_df) > 0:
                st.info(
                    f"ℹ️ ML predictions need more historical data to train reliably — only "
                    f"{len(ml_training_df)} eligible rows in the current dataset. This will activate "
                    "automatically once there's enough match history."
                )

# ==================================================================
# PAGE: WORKLOAD MONITOR
# ==================================================================

elif page == "workload_monitor":
    st.title("Workload Monitor (ACWR)")
    page_intro(
        "Overworking a bowler too quickly is a well-known injury risk factor in real "
        "sports science. This page flags who's currently bowling much more than their "
        "recent normal, using ACWR — explained below."
    )
    st.write(
        "ACWR (Acute:Chronic Workload Ratio) compares a bowler's most recent workload "
        "to their rolling average. Ratios in the 0.8–1.3 range are associated with "
        "lower injury risk in sports-science research; ratios above 1.5 indicate a "
        "rapid, potentially risky spike in workload."
    )

    col1, col2 = st.columns(2)
    with col1:
        with section("Risk tier distribution", "📊"):
            tier_counts = latest_state["acwr_tier"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_alpha(0); ax.set_facecolor('none'); ax.tick_params(colors="#cdd5e0")
            colors_map = {"Low": "#6fd18a", "Moderate": "#e8c15a", "High": "#f08080", "Undertrained": "#7ab8e8"}
            tier_counts.plot(kind="bar", ax=ax, color=[colors_map.get(t, "#999") for t in tier_counts.index])
            ax.set_ylabel("Number of bowlers", color="#9aa4b2")
            st.pyplot(fig)

    with col2:
        with section("ACWR distribution (all bowlers)", "📉"):
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_alpha(0); ax.set_facecolor('none'); ax.tick_params(colors="#cdd5e0")
            sns.histplot(latest_state["acwr"].dropna(), bins=20, ax=ax, color="#1f7a52", kde=True)
            ax.axvspan(0.8, 1.3, color="green", alpha=0.1)
            ax.set_xlabel("ACWR", color="#9aa4b2")
            st.pyplot(fig)

    st.subheader("Bowlers currently in the High-risk zone (ACWR > 1.5)")
    high_risk = latest_state[latest_state["acwr_tier"] == "High"].sort_values("acwr", ascending=False)
    if high_risk.empty:
        st.success("No bowlers currently in the high-risk ACWR zone.")
    else:
        tiles = []
        for row in high_risk.itertuples():
            pct = pct_from_range(row.acwr, 0, 2.2)
            tiles.append(render_tile(
                avatar_text=row.display_name[:2].upper(), avatar_bg="#b0413e",
                category_label=row.country if hasattr(row, 'country') and pd.notna(row.country) else "Unknown",
                title=row.display_name, subtitle="⚠️ Overworked relative to recent normal",
                rows=[{"label": "ACWR", "pct": pct, "bar_color": "#f08080",
                       "pill_text": f"{row.acwr:.2f}", "pill_color": "#f08080"}],
                footer_left=f"{row.overs_bowled:.1f} overs (latest match)",
                footer_right=f"{row.rest_days_before:.0f} rest days before"
            ))
        tile_grid(tiles, n_cols=2)

# ==================================================================
# PAGE: INJURY & FITNESS
# ==================================================================

elif page == "injury_fitness":
    st.markdown("""
    <div class="injury-header">
      <div class="eyebrow">Core of this project</div>
      <h1>🚑 Injury &amp; Fitness</h1>
      <p>Every other page here analyzes performance data that already exists publicly.
      This page is different — it's the one place building the ground truth that
      doesn't exist anywhere else: real, dated cases, tied to real players, that the
      rest of the app's workload flags can actually be checked against.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI summary strip
    status_counts_top = latest_state["current_status"].value_counts()
    real_entries = int((injury_log_df["source"] == "manual").sum()) if not injury_log_df.empty else 0
    demo_entries = int((injury_log_df["source"] == "demo").sum()) if not injury_log_df.empty else 0
    open_alerts = compute_alerts(latest_state, match_summary, injury_log_df)

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    with kc1:
        st.markdown(kpi_card(int(status_counts_top.get("Fit", 0)), "Fit"), unsafe_allow_html=True)
    with kc2:
        st.markdown(kpi_card(
            int(status_counts_top.get("Managed", 0)) + int(status_counts_top.get("Rehab", 0))
            + int(status_counts_top.get("Injured", 0)), "Not fully fit"), unsafe_allow_html=True)
    with kc3:
        st.markdown(kpi_card(len(open_alerts), "Open alerts"), unsafe_allow_html=True)
    with kc4:
        st.markdown(kpi_card(real_entries, "Real logged cases"), unsafe_allow_html=True)
    with kc5:
        st.markdown(kpi_card(demo_entries, "Demo cases"), unsafe_allow_html=True)

    st.write("")

    tab_alerts, tab_log, tab_status, tab_assess, tab_ramp, tab_passport, tab_full = st.tabs([
        "🔔 Alerts", "📝 Log a case", "🩺 Squad status", "🩹 Assessment Guide",
        "📈 Return-to-Play", "📇 Fitness Passport", "📋 Full log & export"
    ])

    # ---------------- ALERTS ----------------
    with tab_alerts:
        page_intro(
            "Everything that needs a human decision this week, in one place — instead of "
            "checking every player individually. Combines workload risk, deconditioning "
            "from long gaps, repetitive-load patterns, and status/workload contradictions."
        )
        if not open_alerts:
            st.success("✅ No open alerts right now — nothing currently needs a decision.")
        else:
            alerts_html = "".join(f"""
                <div class="alert-row" style="border-left-color:{a['color']};">
                  <div style="flex:1;">
                    <div class="alert-type" style="color:{a['color']};">{a['type']}</div>
                    <div class="alert-name">{a['display_name']}</div>
                    <div class="alert-detail">{a['detail']}</div>
                  </div>
                </div>
                """ for a in open_alerts)
            st.markdown(alerts_html, unsafe_allow_html=True)
            with st.expander("What do these alert types mean?"):
                st.markdown("""
                - **Status conflict** — logged as Injured/Rehab, but their recorded workload
                  suggests otherwise. Worth double-checking the log is current.
                - **High ACWR** — bowling well above their own recent normal right now.
                - **Deconditioning risk** — a long gap since their last recorded match;
                  ramping straight back to full load after a break is its own risk pattern.
                - **High monotony** — recent workload has been unusually repetitive
                  match-to-match, which research links to elevated strain even without a
                  high total load.
                """)

    # ---------------- LOG A CASE ----------------
    with tab_log:
        with section("Report a new case", "📝"):
            name_options = sorted(latest_state["display_name"].dropna().unique())
            with st.form("injury_log_form", clear_on_submit=True):
                fc1, fc2 = st.columns(2)
                with fc1:
                    sel_name = st.selectbox("Bowler", name_options)
                    sel_status = st.selectbox("Status", STATUS_OPTIONS)
                    sel_body_part = st.selectbox("Body part (if applicable)", [""] + BODY_PART_OPTIONS)
                with fc2:
                    sel_injury_type = st.text_input("Injury / issue type", placeholder="e.g. Stress fracture")
                    sel_return_date = st.date_input("Expected return date (optional)", value=None)
                sel_notes = st.text_area("Notes", placeholder="Any additional context for medical/coaching staff")
                submitted = st.form_submit_button("Save entry", use_container_width=True)

            if submitted:
                match_row = latest_state[latest_state["display_name"] == sel_name]
                bowler_key = match_row.iloc[0]["bowler"] if not match_row.empty else sel_name
                add_injury_log_entry(
                    bowler=bowler_key, display_name=sel_name, status=sel_status,
                    injury_type=sel_injury_type, body_part=sel_body_part,
                    expected_return_date=sel_return_date, notes=sel_notes, source="manual"
                )
                st.success(f"Logged: {sel_name} → {sel_status}")
                st.rerun()

        with section("Demo data (for presentations before real cases exist)", "🌱"):
            st.caption(
                "Seeds a handful of clearly-labeled synthetic cases so the status views below "
                "aren't empty. These are marked as demo data and can be cleared independently "
                "of anything you log for real."
            )
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("🌱 Seed demo injury data", use_container_width=True):
                    seed_demo_injury_data(master, n=8)
                    st.success("Seeded 8 demo cases.")
                    st.rerun()
            with dc2:
                if st.button("🗑️ Clear demo data only", use_container_width=True):
                    clear_injury_log(source_filter="demo")
                    st.success("Demo cases cleared. Real entries kept.")
                    st.rerun()

    # ---------------- CURRENT SQUAD STATUS ----------------
    with tab_status:
        status_counts = latest_state["current_status"].value_counts()
        status_colors = {"Fit": "#6fd18a", "Managed": "#e8c15a", "Injured": "#f08080", "Rehab": "#c9a9f5"}
        cols = st.columns(4)
        for col, s in zip(cols, STATUS_OPTIONS):
            count = int(status_counts.get(s, 0))
            color = status_colors[s]
            with col:
                st.markdown(f"""
                <div class="tier-card" style="background:{color}18; border-color:{color}55;">
                  <div class="tier-count" style="color:{color};">{count}</div>
                  <div class="tier-label" style="color:{color};">{s}</div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")
        not_fit = latest_state[latest_state["current_status"] != "Fit"].sort_values("current_status")
        if not_fit.empty:
            st.success("Everyone on record is currently marked Fit.")
        else:
            st.subheader("Currently not fully available")
            tiles = []
            for row in not_fit.itertuples():
                latest_entry = injury_log_df[injury_log_df["bowler"] == row.bowler].sort_values("date_reported")
                last = latest_entry.iloc[-1] if not latest_entry.empty else None
                subtitle = f"{row.country} • {row.bowling_style} &nbsp; {status_badge(row.current_status)}"
                footer_left = f"Since {last['date_reported']}" if last is not None else ""
                footer_right = f"Return ~{last['expected_return_date']}" if (
                    last is not None and str(last['expected_return_date']).strip()) else ""
                tiles.append(render_tile(
                    avatar_text=row.display_name[:2].upper(),
                    avatar_bg=status_colors.get(row.current_status, "#999"),
                    category_label=last["injury_type"] if (last is not None and str(last["injury_type"]).strip()) else "Status update",
                    title=row.display_name, subtitle=subtitle,
                    rows=[{"label": "ACWR", "pct": pct_from_range(row.acwr, 0, 2.2), "bar_color": "#2f7ab0",
                           "pill_text": f"{row.acwr:.2f}" if pd.notna(row.acwr) else "N/A", "pill_color": "#8ec6ee"}],
                    footer_left=footer_left, footer_right=footer_right
                ))
            tile_grid(tiles, n_cols=2)

    # ---------------- ASSESSMENT GUIDE ----------------
    with tab_assess:
        page_intro(
            "A structured reference tool, not a diagnosis engine — pick a body part, severity, and "
            "symptoms from fixed options, and it returns general sports-medicine protocol guidance "
            "plus any similar cases already in your log. Every input is a dropdown or checklist, "
            "never free text, so there's nothing here that's guessing or generating medical advice "
            "on the fly."
        )
        st.markdown(
            '<div class="disclaimer-box">🩹 Educational reference only — not a diagnosis, not a '
            'treatment plan, and not a substitute for a qualified doctor or physiotherapist. If '
            'anything here looks concerning, that\'s exactly the signal to get it checked in person.</div>',
            unsafe_allow_html=True
        )
        st.write("")

        if "assessment_chat" not in st.session_state:
            st.session_state.assessment_chat = []

        squad_names_assess = sorted(latest_state["display_name"].dropna().unique())
        bowler_options = ["General / no specific bowler"] + squad_names_assess

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            assess_bowler = st.selectbox("Bowler (optional, for case history)", bowler_options, key="assess_bowler")
        with ac2:
            assess_part = st.selectbox("Body part", BODY_PART_OPTIONS, key="assess_part")
        with ac3:
            assess_severity = st.selectbox("Severity", ["Minor", "Moderate", "Severe"], key="assess_severity")

        assess_symptoms = st.multiselect("Symptoms (select all that apply)", SYMPTOM_OPTIONS, key="assess_symptoms")

        bc1, bc2 = st.columns([1, 1])
        with bc1:
            ask_clicked = st.button("🩹 Get assessment guidance →", key="assess_ask_btn", use_container_width=True)
        with bc2:
            if st.button("Clear conversation", key="assess_clear_btn", use_container_width=True):
                st.session_state.assessment_chat = []
                st.rerun()

        if ask_clicked:
            user_summary = (
                f"{assess_part} · {assess_severity} severity"
                + (f" · {assess_bowler}" if assess_bowler != "General / no specific bowler" else "")
                + (f" · Symptoms: {', '.join(assess_symptoms)}" if assess_symptoms else " · No specific symptoms selected")
            )
            response_html = generate_assessment_response(
                assess_part, assess_severity, assess_symptoms, assess_bowler, injury_log_df
            )
            st.session_state.assessment_chat.append((user_summary, response_html))
            st.rerun()

        if not st.session_state.assessment_chat:
            st.caption("No questions asked yet this session — fill in the options above and click \"Get assessment guidance\".")
        else:
            for user_summary, response_html in st.session_state.assessment_chat:
                st.markdown(f'<div class="chat-user-bubble">{user_summary}</div>', unsafe_allow_html=True)
                st.markdown(response_html, unsafe_allow_html=True)

    # ---------------- RETURN-TO-PLAY PLANNER ----------------
    with tab_ramp:
        page_intro(
            "A graded overs progression back to full workload instead of jumping straight back to "
            "normal — which recreates exactly the acute workload spike the rest of this app already "
            "flags as high-risk ACWR. Standard graded-loading practice: roughly a quarter of normal "
            "load in week 1, building to full load by week 4."
        )
        ramp_names = sorted(latest_state["display_name"].dropna().unique())
        if not ramp_names:
            st.info("No bowlers available in the current filter.")
        else:
            ramp_name = st.selectbox("Bowler", ramp_names, key="ramp_bowler_select")
            match_row = latest_state[latest_state["display_name"] == ramp_name]
            bowler_key = match_row.iloc[0]["bowler"] if not match_row.empty else ramp_name
            player_matches = match_summary[match_summary["bowler"] == bowler_key].sort_values("start_date")

            # If there's a logged injury/rehab case for this bowler, use their
            # workload from *before* that case as the "normal" baseline —
            # matches during/after an injury are exactly the reduced-load
            # matches we don't want to anchor the ramp to. Otherwise fall
            # back to their overall recent chronic average.
            bowler_cases = injury_log_df[injury_log_df["bowler"] == bowler_key].sort_values("date_reported")
            baseline_note = "recent chronic average (no logged case on file)"
            start_date = None
            if not bowler_cases.empty:
                last_case = bowler_cases.iloc[-1]
                try:
                    case_date = pd.Timestamp(last_case["date_reported"])
                    pre_case = player_matches[player_matches["start_date"] < case_date]
                    if not pre_case.empty:
                        base_overs = pre_case["overs_bowled"].tail(4).mean()
                        baseline_note = f"average overs in the 4 matches before their {last_case['date_reported']} {last_case['status'].lower()} entry"
                    else:
                        base_overs = player_matches["overs_bowled"].tail(4).mean()
                except (ValueError, TypeError):
                    base_overs = player_matches["overs_bowled"].tail(4).mean()
                if str(last_case.get("expected_return_date", "")).strip():
                    start_date = last_case["expected_return_date"]
            else:
                base_overs = player_matches["overs_bowled"].tail(4).mean() if not player_matches.empty else None

            if base_overs is None or pd.isna(base_overs) or base_overs <= 0:
                st.warning("Not enough match history on record for this bowler to compute a baseline workload.")
            else:
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    st.markdown(kpi_card(f"{base_overs:.1f}", "Baseline overs/match"), unsafe_allow_html=True)
                with bc2:
                    st.markdown(kpi_card(status_badge(get_current_status(bowler_key, injury_log_df)), "Current status"), unsafe_allow_html=True)
                with bc3:
                    st.markdown(kpi_card(str(start_date) if start_date else "Today", "Ramp start date"), unsafe_allow_html=True)
                st.caption(f"Baseline computed from: {baseline_note}.")

                ramp_df = generate_ramp_plan(base_overs, start_date)
                phase_colors = ["#8ec6ee", "#e8c15a", "#c9a9f5", "#6fd18a"]
                cols = st.columns(len(ramp_df))
                for i, (col, r) in enumerate(zip(cols, ramp_df.itertuples())):
                    with col:
                        st.markdown(f"""
                        <div class="ramp-card" style="border-top:3px solid {phase_colors[i % len(phase_colors)]};">
                          <div class="ramp-phase" style="color:{phase_colors[i % len(phase_colors)]};">{r.phase}</div>
                          <div class="ramp-overs">{r.target_overs}</div>
                          <div class="ramp-pct">overs &middot; {r.pct_of_normal} of normal</div>
                          <div class="ramp-date">🗓️ {r.target_date}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown('<div class="seam"></div>', unsafe_allow_html=True)
                st.caption(
                    "This is a general template, not a medical prescription — actual return timelines "
                    "should always follow your own medical staff's assessment of the specific injury."
                )

    # ---------------- FITNESS PASSPORT ----------------
    with tab_passport:
        page_intro(
            "One exportable summary for a single bowler — status, current workload, recent match "
            "log, and full case history — meant to actually be handed to medical or coaching staff, "
            "not just viewed on screen."
        )
        passport_names = sorted(latest_state["display_name"].dropna().unique())
        if not passport_names:
            st.info("No bowlers available in the current filter.")
        else:
            passport_name = st.selectbox("Bowler", passport_names, key="passport_bowler_select")
            match_row = latest_state[latest_state["display_name"] == passport_name]
            if match_row.empty:
                st.warning("No record found for this bowler in the current filter.")
            else:
                player_row = match_row.iloc[0].to_dict()
                bowler_key = player_row["bowler"]
                player_matches = match_summary[match_summary["bowler"] == bowler_key]
                injury_history = injury_log_df[injury_log_df["bowler"] == bowler_key]

                passport_html = generate_fitness_passport_html(player_row, player_matches, injury_history)

                dc1, dc2 = st.columns([1, 1])
                with dc1:
                    st.download_button(
                        "⬇️ Download passport (HTML — opens in any browser, printable to PDF)",
                        data=passport_html, file_name=f"{passport_name.replace(' ', '_')}_fitness_passport.html",
                        mime="text/html", use_container_width=True, key="passport_download_btn"
                    )
                with dc2:
                    st.caption(f"{int(len(injury_history))} case(s) on file for {passport_name}.")

                st.markdown('<div class="passport-frame-wrap">', unsafe_allow_html=True)
                components.html(passport_html, height=650, scrolling=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- FULL LOG & EXPORT ----------------
    with tab_full:
        if injury_log_df.empty:
            st.info("No cases logged yet — use the 'Log a case' tab, or seed demo data to preview this view.")
        else:
            st.dataframe(
                injury_log_df.sort_values("date_reported", ascending=False),
                use_container_width=True, hide_index=True
            )
            csv_bytes = injury_log_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download injury log CSV", data=csv_bytes,
                                 file_name="injury_log.csv", mime="text/csv")

            st.markdown('<div class="seam"></div>', unsafe_allow_html=True)
            st.caption("⚠️ Clears the entire log, including real entries — use with care.")
            if st.button("🗑️ Clear entire log (real + demo)"):
                clear_injury_log()
                st.success("Injury log cleared.")
                st.rerun()

# ==================================================================
# PAGE: SQUAD IMPACT ENGINE
# ==================================================================

elif page == "squad_impact":
    st.markdown(
        '<div class="impact-header">'
        '<div class="eyebrow">BEYOND SINGLE-PLAYER RISK</div>'
        '<h1>🔄 Squad Impact Engine</h1>'
        '<p>What an absence actually does to the rest of the team, what a layoff really costs in lost '
        "conditioning, and concrete replacement options — not just a risk label with nowhere to go.</p>"
        '</div>', unsafe_allow_html=True
    )

    tab_cascade, tab_debt, tab_sub = st.tabs(["🌊 Cascade Risk", "💳 Recovery Debt Ledger", "🔁 Substitution Finder"])

    # ---------------- CASCADE RISK ----------------
    with tab_cascade:
        page_intro(
            "When a bowler is logged Injured or Managed, their overs don't vanish — the team's other "
            "bowlers absorb them, usually in proportion to how much they already bowl. This projects "
            "who inherits the load and whether it would push their own ACWR into a worse tier — the "
            "ripple effect that's normally invisible until it shows up as someone else's injury weeks later."
        )
        cascade_df = compute_cascade_risk(latest_state, injury_log_df)
        if cascade_df.empty:
            st.success("No currently Injured or Managed bowlers in this view, so there's no cascade to project.")
        else:
            worsening = cascade_df[cascade_df["worsens"]]
            cc1, cc2 = st.columns(2)
            cc1.metric("Teammates projected to worsen a tier", int(len(worsening)))
            cc2.metric("Teams affected", int(cascade_df["bowling_team"].nunique()))
            st.write("")

            for team, team_df in cascade_df.groupby("bowling_team"):
                st.markdown(f'<span class="cascade-team-tag">🏟️ {team}</span>', unsafe_allow_html=True)
                out_names = team_df["out_bowlers"].iloc[0]
                st.caption(f"Covering for: {out_names} — {team_df['gap_overs'].iloc[0]:.1f} overs/match to redistribute across fit teammates")
                tier_color_map = {"Undertrained": "#8ec6ee", "Low": "#6fd18a", "Moderate": "#e8c15a", "High": "#f08080"}
                tiles = []
                for row in team_df.sort_values("worsens", ascending=False).itertuples():
                    proj_color = tier_color_map.get(row.projected_tier, "#8ec6ee")
                    tiles.append(render_tile(
                        avatar_text=row.display_name[:2].upper(), avatar_bg=proj_color,
                        category_label="⚠️ TIER WORSENS" if row.worsens else "Manageable",
                        title=row.display_name,
                        subtitle=f"+{row.extra_overs:.1f} overs projected next match",
                        rows=[
                            {"label": "Current ACWR", "pct": pct_from_range(row.current_acwr, 0, 2.2), "bar_color": "#4f9fd8",
                             "pill_text": f"{row.current_acwr:.2f}" if pd.notna(row.current_acwr) else "N/A", "pill_color": "#8ec6ee"},
                            {"label": "Projected ACWR", "pct": pct_from_range(row.projected_acwr, 0, 2.2), "bar_color": proj_color,
                             "pill_text": f"{row.projected_acwr:.2f}" if row.projected_acwr is not None else "N/A", "pill_color": proj_color},
                        ],
                        footer_left=f"{row.current_tier} → {row.projected_tier}",
                        footer_right=f"{row.projected_overs:.1f} ov projected",
                    ))
                tile_grid(tiles, n_cols=2)
                st.write("")
            st.caption(
                "This is a projection based on proportional redistribution of the missing bowler's usual "
                "workload — real selection decisions on the day may spread the load differently."
            )

    # ---------------- RECOVERY DEBT LEDGER ----------------
    with tab_debt:
        page_intro(
            "Standard return-to-play plans give everyone the same fixed ramp regardless of how long "
            "they were out or how much they used to bowl. This treats time out as a debt — quantified "
            "from real pre-injury workload and days missed — and extends the ramp when the debt is "
            "larger, instead of treating 'pain-free' and 'fully conditioned' as the same thing."
        )
        debt_candidates = injury_log_df[injury_log_df["status"].isin(["Injured", "Rehab", "Managed"])] if not injury_log_df.empty else pd.DataFrame()
        if debt_candidates.empty:
            st.info("No bowlers currently Injured, Rehab, or Managed — no active recovery debt to show. Log a case on the Injury & Fitness page first.")
        else:
            for row in debt_candidates.sort_values("date_reported", ascending=False).itertuples():
                case_dict = row._asdict()
                debt = compute_recovery_debt(case_dict, match_summary)
                ramp = generate_debt_aware_ramp(debt["baseline_overs"], debt["total_ramp_weeks"], debt["return_date"])
                debt_pct = min(100, debt["debt_score"] / 60 * 100)
                debt_color = "#f08080" if debt["debt_score"] > 45 else ("#e8c15a" if debt["debt_score"] > 20 else "#6fd18a")

                st.markdown(f"""
                <div class="ledger-card">
                  <div style="font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:700; margin-bottom:2px;">{row.display_name}</div>
                  <div style="color:var(--text-dim); font-size:12px; margin-bottom:10px;">{row.body_part} &middot; logged {row.date_reported} &middot; status {row.status}</div>
                  <div class="ledger-row"><span class="l-label">Days out</span><span class="l-value">{debt['days_out']}</span></div>
                  <div class="ledger-row"><span class="l-label">Pre-injury workload (baseline)</span><span class="l-value">{debt['baseline_overs']:.1f} overs/match</span></div>
                  <div class="ledger-row"><span class="l-label">Conditioning debt incurred</span><span class="l-value">{debt['debt_score']:.1f} units</span></div>
                  <div class="debt-bar-track"><div class="debt-bar-fill" style="width:{debt_pct:.0f}%; background:{debt_color};"></div></div>
                  <div class="ledger-row"><span class="l-label">Standard ramp</span><span class="l-value">4 weeks</span></div>
                  <div class="ledger-row"><span class="l-label">Extra caution phases recommended</span><span class="l-value">+{debt['extra_phases']}</span></div>
                  <div class="ledger-balance"><span class="l-label">Recommended full ramp length</span><span class="l-value" style="color:{debt_color};">{debt['total_ramp_weeks']} weeks</span></div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander(f"View {row.display_name}'s repayment schedule"):
                    st.dataframe(ramp, use_container_width=True, hide_index=True)

            st.caption(
                "Debt units are an internally-consistent scoring system for comparing cases against each "
                "other, not a published clinical measure — always defer to real medical guidance on actual return timelines."
            )

    # ---------------- SUBSTITUTION FINDER ----------------
    with tab_sub:
        page_intro(
            "Turns a risk flag into an actual decision. Pick any bowler who's injured, managed, or "
            "high-risk, and get concrete, ranked replacement options from the same team — bowlers "
            "with real headroom below their safe ACWR ceiling, not just a warning with nowhere to go."
        )
        at_risk_pool = latest_state[
            (latest_state["current_status"].isin(["Injured", "Managed"])) |
            (latest_state["acwr_tier"] == "High")
        ]
        if at_risk_pool.empty:
            st.info("No bowlers currently flagged Injured, Managed, or High ACWR risk in this view.")
        else:
            target_names = sorted(at_risk_pool["display_name"].dropna().unique())
            target_name = st.selectbox("Bowler needing coverage", target_names, key="sub_target_select")
            target_row = at_risk_pool[at_risk_pool["display_name"] == target_name].iloc[0]

            reason_bits = []
            if target_row["current_status"] in ("Injured", "Managed"):
                reason_bits.append(target_row["current_status"])
            if target_row["acwr_tier"] == "High":
                reason_bits.append("High ACWR")
            st.caption(f"{target_name} — {', '.join(reason_bits)} · team: {target_row.get('bowling_team', 'Unknown')}")

            subs = find_substitutes(target_row, latest_state)
            if subs.empty:
                st.warning("No fit teammates found on the same team in the current filter to suggest as cover.")
            else:
                tiles = []
                for i, row in enumerate(subs.itertuples(), start=1):
                    headroom_pct = max(0, min(100, row.headroom * 100))
                    tiles.append(render_tile(
                        avatar_text=row.display_name[:2].upper(), avatar_bg="#22b573",
                        category_label=f"OPTION {i}", title=row.display_name,
                        subtitle=f"{row.bowling_style if pd.notna(getattr(row, 'bowling_style', None)) else ''}",
                        rows=[
                            {"label": "ACWR headroom", "pct": headroom_pct, "bar_color": "#22b573",
                             "pill_text": f"{row.acwr:.2f}" if pd.notna(row.acwr) else "N/A", "pill_color": "#6fd18a"},
                            {"label": "Recent economy", "pct": pct_from_range(row.economy, 3, 12, invert=True), "bar_color": "#4f9fd8",
                             "pill_text": f"{row.economy:.2f}" if pd.notna(row.economy) else "N/A", "pill_color": "#8ec6ee"},
                        ],
                        footer_left=f"Safe ceiling ~{safe_overs_ceiling(row.chronic_avg_overs) or 'N/A'} ov",
                        footer_right=row.acwr_tier,
                    ))
                tile_grid(tiles, n_cols=3)
                st.caption(
                    "Ranked by ACWR headroom (furthest from the risk zone first), then recent economy as a form tiebreak. "
                    "This surfaces who has capacity — final selection should still weigh matchup and tactical fit."
                )

# ==================================================================
# PAGE: TEAM OVERVIEW
# ==================================================================

elif page == "team_overview":
    st.title("Team Overview")
    page_intro(
        "Zoom out from individual players to a whole team's bowling attack — who they rely "
        "on most, and how that team has actually performed against a specific rival."
    )

    teams = sorted(match_summary["bowling_team"].dropna().unique())
    if not teams:
        st.warning("No team data available for this filter.")
    else:
        tab1, tab2 = st.tabs(["Single team", "Head-to-head"])

        with tab1:
            team = st.selectbox("Bowling team", teams)
            team_matches = match_summary[match_summary["bowling_team"] == team]

            c1, c2, c3 = st.columns(3)
            c1.metric("Bowlers used", team_matches["bowler"].nunique())
            c2.metric("Matches", team_matches["match_id"].nunique())
            c3.metric("Total wickets", int(team_matches["wickets"].sum()))

            st.subheader(f"Top bowlers for {team}")
            team_bowlers = team_matches.groupby("display_name").agg(
                wickets=("wickets", "sum"), avg_economy=("economy", "mean"), matches=("match_id", "nunique"),
            ).reset_index().sort_values("wickets", ascending=False)

            top10 = team_bowlers.head(10)
            if top10.empty:
                st.info("No bowler records for this team in the current filter.")
            else:
                vmax = top10["wickets"].max()
                tiles = []
                for i, row in enumerate(top10.itertuples(), start=1):
                    pct = pct_from_range(row.wickets, 0, vmax)
                    tiles.append(render_tile(
                        avatar_text=f"#{i}", avatar_bg=category_color(team),
                        category_label=team, title=row.display_name, subtitle=None,
                        rows=[{"label": "Wickets", "pct": pct, "bar_color": "#1f7a52",
                               "pill_text": f"{int(row.wickets)}", "pill_color": "#6fd18a"}],
                        footer_left=f"Econ {row.avg_economy:.2f}", footer_right=f"{row.matches} matches"
                    ))
                tile_grid(tiles, n_cols=2)

            if "season" in team_matches.columns:
                st.subheader(f"{team} economy trend by season")
                season_econ = team_matches.groupby("season")["economy"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(8, 3))
                fig.patch.set_alpha(0); ax.set_facecolor('none'); ax.tick_params(colors="#cdd5e0")
                ax.plot(season_econ["season"].astype(str), season_econ["economy"], marker="o", color="#b04a4a")
                plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                team_a = st.selectbox("Team A", teams, key="h2h_a")
            with c2:
                team_b = st.selectbox("Team B", [t for t in teams if t != team_a] or teams, key="h2h_b")

            h2h = match_summary[
                ((match_summary["bowling_team"] == team_a) & (match_summary["batting_team"] == team_b)) |
                ((match_summary["bowling_team"] == team_b) & (match_summary["batting_team"] == team_a))
            ]
            if h2h.empty:
                st.info(f"No recorded matches between {team_a} and {team_b} in the current filter.")
            else:
                st.metric("Matches between these teams", h2h["match_id"].nunique())
                h2h_summary = h2h.groupby("bowling_team").agg(
                    wickets=("wickets", "sum"), avg_economy=("economy", "mean")
                ).reset_index()

                if len(h2h_summary) >= 2:
                    row_x, row_y = h2h_summary.iloc[0], h2h_summary.iloc[1]
                    bars_html = "".join([
                        render_compare_bar("Total wickets", f"{int(row_x['wickets'])}", row_x['wickets'],
                                            f"{int(row_y['wickets'])}", row_y['wickets'], higher_is_better=True),
                        render_compare_bar("Avg economy (lower is better)", f"{row_x['avg_economy']:.2f}",
                                            row_x['avg_economy'], f"{row_y['avg_economy']:.2f}",
                                            row_y['avg_economy'], higher_is_better=False),
                    ])
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; padding:0 4px; '
                        f'margin-bottom:10px;"><b style="color:#6fd18a;">{row_x["bowling_team"]}</b>'
                        f'<b style="color:#4f9fd8;">{row_y["bowling_team"]}</b></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f'<div class="section-card">{bars_html}</div>', unsafe_allow_html=True)

                st.subheader(f"Top wicket-takers: {team_a} vs {team_b}")
                top_h2h_bowlers = h2h.groupby("display_name")["wickets"].sum().sort_values(ascending=False).head(8)
                if not top_h2h_bowlers.empty:
                    vmax = top_h2h_bowlers.max()
                    tiles = []
                    for i, (name, wkts) in enumerate(top_h2h_bowlers.items(), start=1):
                        pct = pct_from_range(wkts, 0, vmax)
                        tiles.append(render_tile(
                            avatar_text=f"#{i}", avatar_bg="#c9a227",
                            category_label=f"{team_a} vs {team_b}", title=name, subtitle=None,
                            rows=[{"label": "Wickets in these matches", "pct": pct, "bar_color": "#c9a227",
                                   "pill_text": f"{int(wkts)}", "pill_color": "#e8c15a"}],
                            footer_left="Head-to-head record", footer_right=""
                        ))
                    tile_grid(tiles, n_cols=2)

# ==================================================================
# PAGE: ASK THE DATA
# ==================================================================

elif page == "ask_data":
    st.title("Ask the Data")
    page_intro(
        "A rule-based chatbot, not an AI model — every answer here is a direct lookup against "
        "the real CSVs your other pages already use. It won't guess or hallucinate a number "
        "that isn't in the data; if it doesn't recognize your question, it'll say so."
    )
    _ask_names = list(match_summary_all["display_name"].dropna().unique())
    render_chatbot_ui(
        "askdata_page", _ask_names, match_summary_all, match_summary, vs_team, master,
        latest_state, injury_log_df, format_choice, page="ask_data", height=480
    )

# ==================================================================
# PAGE: EXPORT DATA
# ==================================================================

elif page == "export_data":
    st.title("Export Data")
    page_intro(
        "Every table this app computes is yours to take with you — download any of them as "
        "a plain CSV file, already matching whatever filters are currently active."
    )
    st.write("Download any of the processed tables as CSV, reflecting the current filters.")

    export_choice = st.selectbox(
        "Choose a table to export",
        ["Bowler match summary (filtered)", "Bowler vs team summary (filtered)", "Player master list",
         "Advanced search aggregate (filtered)"]
    )

    if export_choice == "Bowler match summary (filtered)":
        export_df = match_summary
    elif export_choice == "Bowler vs team summary (filtered)":
        export_df = vs_team
    elif export_choice == "Player master list":
        export_df = master
    else:
        export_df = match_summary.groupby(["display_name", "bowler"]).agg(
            matches=("match_id", "nunique"), wickets=("wickets", "sum"), avg_economy=("economy", "mean"),
        ).reset_index()

    st.dataframe(export_df.head(50), width='stretch')
    st.caption(f"Showing first 50 of {len(export_df):,} rows. Full data included in download.")

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV", data=csv_bytes,
        file_name=f"{export_choice.split(' (')[0].lower().replace(' ', '_')}.csv", mime="text/csv",
    )

# ==================================================================
# PAGE: METHODOLOGY
# ==================================================================

elif page == "methodology":
    st.title("About This Project")
    page_intro(
        "What this app actually does, in plain English — then the technical details for "
        "anyone who wants to dig deeper."
    )

    with section("🚑 The flagship feature: Injury & Fitness Management", "🏆"):
        st.markdown("""
        Every other page in this app analyzes performance data that already exists publicly —
        ball-by-ball match records anyone could download. **The Injury & Fitness page is
        different.** It's the one place in this project building something that doesn't exist
        anywhere else: a real, dated record of actual player status, tied to the same
        bowlers the workload analytics already track.

        That matters because ACWR — the workload-risk metric behind most of this app — is a
        published heuristic, not a validated predictor. No public dataset of real injury
        outcomes exists to check it against, for this project or anyone else's. Injury & Fitness
        is the piece that starts closing that gap: every case logged there is real ground truth
        that the rest of the app's risk flags can eventually be checked against.

        It brings together everything a workload-monitoring tool should actually do something
        with: a live squad status register, a prioritized alerts feed combining four real risk
        signals, a graded return-to-play ramp planner, multi-factor risk scoring beyond ACWR
        alone, and an exportable per-player fitness passport for handing to medical staff.
        """)

    with section("The idea in one paragraph", "💡"):
        st.markdown("""
        Bowling too much, too fast, is one of the most well-established risk factors for
        injury in cricket — but most teams only find out a bowler was overworked *after*
        they get hurt. This project takes real, historical ball-by-ball match data and
        turns it into a live dashboard that shows exactly how much each bowler has been
        bowling recently, how that compares to their own normal, and who might be
        approaching risky territory right now — alongside how well they're actually
        performing, so workload and form can be weighed together.
        """)

    with section("Who this is for", "👥"):
        st.markdown("""
        - **Coaches / team analysts** deciding who to select for an upcoming match
        - **Fans and cricket enthusiasts** who want a deeper, data-driven view of players
        - **Students or recruiters** interested in a real end-to-end data science project —
          from raw data, to feature engineering, to an interactive product
        """)

    with section("Where the data comes from", "📄"):
        st.markdown("""
        Every number in this app is computed from **real ball-by-ball delivery data** —
        not simulated or made up. Each row of the original dataset is a single ball bowled
        in a real match; this app aggregates millions of those deliveries into per-bowler,
        per-match summaries (overs bowled, runs conceded, wickets taken), then layers
        workload and matchup analysis on top.
        """)

    st.markdown('<div class="seam"></div>', unsafe_allow_html=True)
    st.subheader("Technical details")
    st.caption("For anyone who wants the exact definitions and formulas behind every metric.")

    with section("Data source", "📄"):
        st.markdown("""
        All performance numbers come from real ball-by-ball delivery data (processed via
        `preprocess.py`), aggregated per bowler per match: legal deliveries, runs conceded,
        and wickets credited (excluding run-outs and other non-bowler dismissals).
        """)

    with section("Workload metrics", "📈"):
        st.markdown("""
        - **Overs bowled**: legal deliveries ÷ 6, computed per match
        - **Economy**: runs conceded ÷ overs bowled
        - **Rest days before**: real gap between a bowler's consecutive match dates
        - **Matches in last 30 days**: rolling count of that bowler's matches in the
          30 days prior to each match date
        """)

    with section("ACWR — Acute:Chronic Workload Ratio", "⚖️"):
        st.markdown("""
        A published sports-science method (Gabbett et al.) comparing a bowler's most
        recent workload (acute) to their rolling average workload (chronic):

        - **Acute** = overs bowled in the current match
        - **Chronic** = rolling average overs bowled over the player's last 4 matches

        | ACWR range | Interpretation |
        |---|---|
        | < 0.8 | Undertrained — possible detraining risk |
        | 0.8 – 1.3 | "Sweet spot" — associated with lower injury risk |
        | 1.3 – 1.5 | Moderate risk — workload rising faster than usual |
        | > 1.5 | High risk — rapid workload spike |

        **This is a heuristic, not a validated injury predictor.** No public dataset
        contains real bowler injury outcomes, so there's no ground truth to train or
        validate a true injury-prediction model against. ACWR is a legitimate,
        published proxy used by sports scientists and coaches, but it should inform —
        not replace — real medical and coaching judgment.
        """)

    with section("Other computed metrics", "🧮"):
        st.markdown("""
        - **Performance score** (matchup pages) = (avg wickets × 2) − (avg economy × 0.5)
        - **Consistency** = standard deviation of economy across a player's matches
          (lower = more consistent)
        - **Rising stars** = change in average economy between a player's first and
          most recent season on record (positive = improved, i.e. economy decreased)
        - **Radar chart axes** are min-max normalized within the currently filtered
          player pool, so values are relative to the current view, not absolute
          career benchmarks
        """)

# ==================================================================
# FLOATING CHATBOT WIDGET — pinned bottom-right, available on every
# page regardless of which nav item is selected. Placed after the page
# routing (not before it) so it can pick up whichever player(s) that
# page already has selected — e.g. Player Profile's `selected`, or
# Compare Players' `selected_players` — and use them in its quick
# chips. Uses the same `st-key-*` CSS-targeting trick as the sidebar
# nav buttons: st.container(key=...) gets a stable `st-key-<key>`
# class Streamlit attaches in the real DOM, which position:fixed can
# then target to pin it in place.
# ==================================================================

st.markdown("""
<style>
    div[class*="st-key-floating_chatbot"] {
        position: fixed; bottom: 22px; right: 22px; z-index: 9999; width: auto;
    }
    div[class*="st-key-floating_chatbot"] > div { width: auto; }
    div[class*="st-key-floating_chatbot"] button {
        border-radius: 999px !important; padding: 10px 18px !important;
        background: linear-gradient(120deg, #145c3f, #1f7a52) !important;
        border-color: #22b573 !important; color: #fff !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.45) !important; font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

_page_locals = locals()
_float_context_players = None
if page == "player_profile" and "selected" in _page_locals:
    _float_context_players = [_page_locals["selected"]]
elif page == "compare_players" and "selected_players" in _page_locals:
    _float_context_players = _page_locals["selected_players"][:2]

with st.container(key="floating_chatbot"):
    with st.popover("💬 Ask the data"):
        st.caption("Rule-based lookups against your loaded CSVs — no AI model, no hallucinated numbers.")
        _float_names = list(match_summary_all["display_name"].dropna().unique())
        render_chatbot_ui(
            "floating", _float_names, match_summary_all, match_summary, vs_team, master,
            latest_state, injury_log_df, format_choice, page=page,
            context_players=_float_context_players, height=320
        )
