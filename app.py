"""ReviewForge: AI-Assisted Academic Review Paper Drafting & Verification."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
import time
import warnings
from typing import Any, Dict, List, Optional

from PIL import Image

warnings.filterwarnings("ignore", message=".*use_container_width.*")

import streamlit as st

from config import (
    AI_PROVIDER,
    DEFAULT_GROQ_RPM,
    DEFAULT_RPM,
    GROQ_PRIMARY_MODEL,
    LENGTH_OPTIONS,
    PRIMARY_MODEL,
    PROMPTS_DIR,
    REVIEW_SECTIONS,
    SIMILARITY_SENTENCE_THRESHOLD,
    TONE_OPTIONS,
    get_api_keys,
    get_groq_keys,
    safe_fill_template,
    save_keys_to_env,
)
from core.analysis import (
    build_comparison_table,
    extract_key_claims,
    generate_taxonomy_and_timeline,
)
from core.db import (
    clear_chat_history,
    create_project,
    delete_paper,
    delete_project,
    get_analysis,
    get_chat_history,
    get_or_create_default_project,
    get_paper_pages,
    get_sections,
    init_db,
    list_papers,
    list_projects,
    reorder_citations,
    save_analysis,
    save_chat_message,
    save_paper,
    save_section,
    update_paper_card,
    update_paper_metadata,
    update_project,
)
from core.gaps import find_research_gaps, generate_research_objectives
from core.ieee_export import (
    generate_bibtex_export,
    generate_ieee_docx,
    generate_markdown_export,
    generate_similarity_report_markdown,
)
from core.llm import (
    RateLimiter,
    call_gemini,
    call_groq,
    call_llm,
    get_groq_key_manager,
    get_key_manager,
    update_llm_keys,
)
from core.paper_card import generate_paper_card
from core.pdf_extract import process_pdf
from core.similarity import (
    CorpusIndex,
    check_section_similarity,
    check_self_overlap,
)
from core.writer import (
    format_ieee_references_section,
    generate_section_draft,
    run_auto_rewrite_loop,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FULL_LOGO = str(ASSETS_DIR / "full-logo.png")
SMALL_LOGO = str(ASSETS_DIR / "small-logo.png")
FAVICON = str(ASSETS_DIR / "favicon.png")

# Configure Streamlit page
st.set_page_config(
    page_title="ReviewForge | Academic Review Paper Engine",
    page_icon=FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Top Navbar & Collapsed Brand Logo
st.logo(
    image=FULL_LOGO,
    icon_image=SMALL_LOGO,
    size="large",
)

# Custom CSS for FinScope-inspired academic design system
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&display=swap');

    /* Global Canvas & Base Typography */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        color: #111111 !important;
        background-color: #ffffff !important;
    }

    /* Eliminate Streamlit's default 10rem bottom padding so scrolling terminates exactly at the footer */
    .block-container,
    [data-testid="stMainBlockContainer"],
    .stMainBlockContainer {
        padding-bottom: 0px !important;
    }

    /* Suppress Streamlit bottom spacers, anchors and default footer */
    footer,
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stAppIframeResizerAnchor"] {
        display: none !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }

    /* Top Navbar: Clean White with FinScope 1px Hairline Border */
    [data-testid="stHeader"] {
        background-color: #ffffff !important;
        border-bottom: 1px solid #dfdfdf !important;
        box-shadow: none !important;
    }

    /* Enforce Dark Ink Text (#111111) Across All Text Containers */
    p, span, label, li, a, td, th, div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] label,
    div[data-testid="stMarkdownContainer"] strong {
        color: #111111 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Explicitly preserve Streamlit's Material Symbols and Icon fonts */
    [data-testid="stIconMaterial"],
    [data-testid*="stIcon"],
    [data-testid*="Icon"],
    .material-symbols-rounded,
    .material-symbols-outlined,
    span[class*="material-symbols"],
    i[class*="material-icons"],
    .stIcon {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-weight: normal !important;
        font-style: normal !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        line-break: auto !important;
        display: inline-block !important;
        color: inherit !important;
    }

    /* Headings: Inter, Weight 400 with negative tracking */
    h1, h2, h3, .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 400 !important;
        letter-spacing: -1.2px !important;
        color: #111111 !important;
    }

    h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.4px !important;
        color: #111111 !important;
    }

    /* FinScope Eyebrow Label */
    .eyebrow-label {
        font-family: 'Inter', system-ui, sans-serif !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 1.8px !important;
        text-transform: uppercase !important;
        color: #4f525d !important;
        margin-bottom: 4px;
    }

    .main-header {
        font-size: 2.6rem !important;
        font-weight: 400 !important;
        letter-spacing: -1.8px !important;
        line-height: 1.15 !important;
        color: #111111 !important;
        margin-bottom: 0.2rem !important;
    }

    .sub-header {
        color: #4f525d !important;
        font-size: 0.98rem !important;
        font-weight: 400 !important;
        margin-bottom: 1.8rem !important;
        line-height: 1.5 !important;
    }
    .sub-header * {
        color: #4f525d !important;
    }

    /* Primary Button: FinScope Ink-Black Fill + 2px Mint Voltage Border */
    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 2px solid #31d891 !important;
        border-radius: 70px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0px !important;
        padding: 0.55rem 1.6rem !important;
        box-shadow: none !important;
        transition: all 0.2s ease-in-out !important;
    }

    div.stButton > button[kind="primary"] *,
    div.stDownloadButton > button[kind="primary"] * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {
        background-color: #111111 !important;
        border-color: #111111 !important;
        color: #31d891 !important;
        box-shadow: none !important;
    }
    div.stButton > button[kind="primary"]:hover *,
    div.stDownloadButton > button[kind="primary"]:hover * {
        color: #31d891 !important;
        fill: #31d891 !important;
    }

    /* Secondary Button: Transparent + 1.5px Hairline Border */
    div.stButton > button[kind="secondary"],
    div.stButton > button:not([kind="primary"]),
    div.stDownloadButton > button:not([kind="primary"]) {
        background-color: transparent !important;
        color: #111111 !important;
        border: 1.5px solid #dfdfdf !important;
        border-radius: 70px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        box-shadow: none !important;
        transition: all 0.2s ease-in-out !important;
    }

    div.stButton > button:not([kind="primary"]) *,
    div.stDownloadButton > button:not([kind="primary"]) * {
        color: #111111 !important;
        fill: #111111 !important;
    }

    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button:not([kind="primary"]):hover,
    div.stDownloadButton > button:not([kind="primary"]):hover {
        border-color: #31d891 !important;
        color: #111111 !important;
        background-color: #f9fafb !important;
        box-shadow: none !important;
    }

    /* Navigation Tabs: Flat Strip with Mint Voltage Underline */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px !important;
        border-bottom: 1.5px solid #dfdfdf !important;
        padding-bottom: 4px !important;
        background-color: transparent !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 0.90rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.2px !important;
        color: #4f525d !important;
        border-radius: 4px 4px 0 0 !important;
        padding: 8px 14px !important;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"] * {
        color: #4f525d !important;
    }

    .stTabs [aria-selected="true"] {
        color: #111111 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #31d891 !important;
    }
    .stTabs [aria-selected="true"] * {
        color: #111111 !important;
    }

    /* Cards & Containers: Flat Editorial, 1.5px Hairline, Zero Shadow */
    .metric-card, .paper-card {
        background: #ffffff !important;
        border: 1.5px solid #dfdfdf !important;
        border-radius: 10px !important;
        padding: 18px !important;
        margin-bottom: 12px !important;
        box-shadow: none !important;
        color: #111111 !important;
    }
    .metric-card *, .paper-card * {
        color: #111111 !important;
    }
    .metric-card small, .metric-card em, .paper-card small, .paper-card em {
        color: #4f525d !important;
    }

    /* Autonomous Synthesis Island (FinScope #100841 Deep Navy) */
    .dark-island, .autonomous-hero-card {
        background-color: #100841 !important;
        border: 1px solid rgba(49, 216, 145, 0.25) !important;
        border-radius: 4px !important;
        padding: 24px 28px !important;
        margin-bottom: 20px !important;
        box-shadow: none !important;
    }

    .dark-island-eyebrow {
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: #31d891 !important;
        margin-bottom: 6px !important;
    }

    .dark-island-title {
        font-family: 'Inter', sans-serif !important;
        font-size: 20px !important;
        font-weight: 400 !important;
        letter-spacing: -0.5px !important;
        line-height: 1.3 !important;
        color: #ffffff !important;
        margin: 0 0 8px 0 !important;
    }

    .dark-island-desc {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 14px !important;
        color: #c4c9ce !important;
        line-height: 1.6 !important;
        margin: 0 !important;
        max-width: 860px !important;
    }

    .dark-island *, .autonomous-hero-card * {
        color: #ffffff !important;
    }
    .dark-island .dark-island-eyebrow, .autonomous-hero-card .dark-island-eyebrow {
        color: #31d891 !important;
    }
    .dark-island .dark-island-desc, .autonomous-hero-card .dark-island-desc {
        color: #c4c9ce !important;
    }

    /* Success Card */
    .success-card {
        background-color: #ffffff !important;
        border: 1.5px solid #31d891 !important;
        border-radius: 10px !important;
        padding: 20px 24px !important;
        margin-bottom: 20px !important;
    }
    .success-card * {
        color: #111111 !important;
    }

    /* Evidence Snippet Callout: Academic Italic with Mint Accent */
    .evidence-box {
        background: #f9fafb !important;
        border-left: 3px solid #31d891 !important;
        border-radius: 0 4px 4px 0 !important;
        padding: 10px 14px !important;
        margin: 8px 0 !important;
        font-family: 'Plus Jakarta Sans', serif !important;
        font-style: italic !important;
        font-size: 0.90rem !important;
        color: #111111 !important;
        line-height: 1.6 !important;
    }
    .evidence-box * {
        color: #111111 !important;
    }

    /* Academic Badges: Tight 5px Rounding with Scientific Status Palette */
    .badge-green {
        background-color: #f0fdf4 !important;
        color: #15803d !important;
        border: 1px solid #bbf7d0 !important;
        padding: 2px 8px !important;
        border-radius: 5px !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
    }

    .badge-yellow {
        background-color: #fefce8 !important;
        color: #b45309 !important;
        border: 1px solid #fef08a !important;
        padding: 2px 8px !important;
        border-radius: 5px !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
    }

    .badge-red {
        background-color: #fef2f2 !important;
        color: #df0134 !important;
        border: 1px solid #fecaca !important;
        padding: 2px 8px !important;
        border-radius: 5px !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px !important;
    }

    .badge-mint {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1.5px solid #31d891 !important;
        padding: 2px 8px !important;
        border-radius: 5px !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
    }

    /* Form Inputs, Selectboxes, Textareas: Explicit White Background & Ink Text */
    input, textarea, select {
        background-color: #ffffff !important;
        color: #111111 !important;
    }

    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1.5px solid #dfdfdf !important;
        border-radius: 4px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        box-shadow: none !important;
    }

    .stTextInput div[data-testid="stTextInputRootElement"],
    .stTextInput div[data-testid="stTextInputRootElement"] input {
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        border-radius: 4px !important;
        box-sizing: border-box !important;
    }

    /* Straight-Line Alignment for Wizard Inputs and Trigger Button */
    .stTextInput div[data-testid="stTextInputRootElement"],
    .stTextInput div[data-testid="stTextInputRootElement"] input {
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        border-radius: 4px !important;
        box-sizing: border-box !important;
    }

    div.stButton > button {
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        border-radius: 4px !important;
        box-sizing: border-box !important;
        margin: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #31d891 !important;
        box-shadow: none !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1.5px solid #dfdfdf !important;
        border-radius: 4px !important;
    }

    .stSelectbox div[data-baseweb="select"] * {
        color: #111111 !important;
        background-color: transparent !important;
    }

    /* Dropdown Menus / Popovers */
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1.5px solid #dfdfdf !important;
    }

    ul[role="listbox"] li, ul[role="listbox"] li * {
        background-color: #ffffff !important;
        color: #111111 !important;
    }

    ul[role="listbox"] li:hover, ul[role="listbox"] li[aria-selected="true"] {
        background-color: #f0fdf4 !important;
        color: #111111 !important;
    }

    /* Expanders: Flat White Card with Crisp Text */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1.5px solid #dfdfdf !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: #111111 !important;
    }
    [data-testid="stExpander"] summary * {
        color: #111111 !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {
        color: #111111 !important;
    }

    /* Metrics: White Card with Dark Ink Values */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1.5px solid #dfdfdf !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }
    [data-testid="stMetricLabel"] * {
        color: #4f525d !important;
    }
    [data-testid="stMetricValue"] * {
        color: #111111 !important;
    }

    /* Sidebar Styling & Brand Logo */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #f9fafb !important;
        border-right: 1.5px solid #dfdfdf !important;
        color: #111111 !important;
    }
    [data-testid="stSidebarHeader"] {
        padding: 14px 8px 6px 8px !important;
        background-color: #f9fafb !important;
        height: auto !important;
        min-height: 75px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        position: relative !important;
    }
    [data-testid="stSidebarHeader"] > div,
    [data-testid="stSidebarHeader"] [data-testid="stLogoLink"],
    [data-testid="stSidebarHeader"] [data-testid="stLogoSpacer"] {
        height: auto !important;
        max-height: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    [data-testid="stSidebarHeader"] img,
    [data-testid="stSidebar"] [data-testid="stLogo"] {
        height: auto !important;
        max-height: 80px !important;
        width: auto !important;
        max-width: 215px !important;
        object-fit: contain !important;
        object-position: left center !important;
        margin: 0 !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
    }

    /* Sidebar Collapse Icon: Right Edge, Visible Grey, Turns Black on Navbar Hover */
    [data-testid="stSidebarCollapseButton"] {
        margin-left: auto !important;
        margin-right: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebarCollapseButton"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 6px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] * {
        visibility: visible !important;
        opacity: 1 !important;
        color: #888888 !important;
        fill: #888888 !important;
        transition: color 0.15s ease, fill 0.15s ease, background-color 0.15s ease !important;
    }
    [data-testid="stSidebar"]:hover [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stSidebarHeader"]:hover [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stSidebarCollapseButton"]:hover * {
        color: #111111 !important;
        fill: #111111 !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover {
        background-color: rgba(0, 0, 0, 0.05) !important;
        border-radius: 4px !important;
    }

    /* Collapsed Header Logo & Toggle */
    [data-testid="stHeader"] [data-testid="stLogo"], [data-testid="stHeader"] img {
        max-height: 34px !important;
        max-width: 34px !important;
        object-fit: contain !important;
        object-position: left center !important;
        margin-left: 2px !important;
    }
    [data-testid="stHeader"] [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stHeader"] button[aria-label*="sidebar"] * {
        color: #888888 !important;
        fill: #888888 !important;
        transition: color 0.15s ease, fill 0.15s ease !important;
    }
    [data-testid="stHeader"]:hover [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stHeader"]:hover button[aria-label*="sidebar"] *,
    [data-testid="stHeader"] button[aria-label*="sidebar"]:hover * {
        color: #111111 !important;
        fill: #111111 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        color: #111111 !important;
    }
    [data-testid="stSidebar"] label {
        color: #111111 !important;
    }
    [data-testid="stSidebar"] p {
        color: #111111 !important;
    }

    .provider-card {
        background: #ffffff !important;
        border: 1.5px solid #dfdfdf !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 0.85rem !important;
        margin-bottom: 12px !important;
        color: #111111 !important;
    }
    .provider-card * {
        color: #111111 !important;
    }
    .provider-card code {
        color: #111111 !important;
        background-color: #f1f5f9 !important;
    }
    .provider-sub {
        color: #4f525d !important;
        font-size: 0.8rem !important;
    }

    .provider-hr {
        border: 0 !important;
        border-top: 1px solid #dfdfdf !important;
        margin: 8px 0 !important;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background-color: #f9fafb !important;
        border: 1.5px solid #dfdfdf !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatMessage"] * {
        color: #111111 !important;
    }

    /* Minimalist Professional Workstation Footer */
    .app-footer {
        background-color: transparent !important;
        border-top: 1px solid #dfdfdf !important;
        padding-top: 20px !important;
        padding-bottom: 24px !important;
        margin-top: 48px !important;
        margin-bottom: 0px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        flex-wrap: wrap !important;
        gap: 12px !important;
    }

    .footer-left {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    .footer-brand {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #111111 !important;
        letter-spacing: -0.2px !important;
    }

    .footer-sep {
        color: #dfdfdf !important;
        font-size: 12px !important;
    }

    .footer-desc {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        color: #868c95 !important;
    }

    .footer-right {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 12px !important;
        color: #868c95 !important;
    }

    .app-footer, .app-footer * {
        color: #868c95 !important;
    }
    .app-footer .footer-brand {
        color: #111111 !important;
    }

    @media (max-width: 640px) {
        .app-footer {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 8px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Database
init_db()

# State Initialization
if "project_id" not in st.session_state:
    default_proj = get_or_create_default_project()
    st.session_state.project_id = default_proj["id"]
    st.session_state.project_name = default_proj["name"]

km = get_key_manager()
groq_km = get_groq_key_manager()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.caption("AI-Assisted Systematic Review Paper Synthesizer")
    st.markdown("<hr style='margin: 8px 0 16px 0; border: none; border-top: 1px solid #dfdfdf;'>", unsafe_allow_html=True)

    # Project Management
    st.markdown("#### Active Project")
    projects = list_projects()
    proj_names = [p["name"] for p in projects]
    proj_map = {p["name"]: p["id"] for p in projects}

    current_proj_name = next(
        (p["name"] for p in projects if p["id"] == st.session_state.project_id),
        proj_names[0] if proj_names else "Default Project",
    )

    selected_proj = st.selectbox(
        "Select Project",
        proj_names,
        index=proj_names.index(current_proj_name) if current_proj_name in proj_names else 0,
    )
    if selected_proj and proj_map[selected_proj] != st.session_state.project_id:
        st.session_state.project_id = proj_map[selected_proj]
        st.session_state.project_name = selected_proj
        st.rerun()

    with st.expander("New Project / Manage"):
        new_name = st.text_input("New Project Name", placeholder="e.g. Multimodal LLM Survey")
        if st.button("Create Project", icon=":material/add:", use_container_width=True):
            if new_name.strip():
                new_id = create_project(new_name.strip())
                st.session_state.project_id = new_id
                st.session_state.project_name = new_name.strip()
                st.success("Project created!")
                st.rerun()

        if len(projects) > 1:
            if st.button("Delete Current Project", icon=":material/delete:", use_container_width=True, type="secondary"):
                delete_project(st.session_state.project_id)
                remaining = list_projects()
                st.session_state.project_id = remaining[0]["id"]
                st.session_state.project_name = remaining[0]["name"]
                st.rerun()

    st.markdown("---")

    # AI Engine & Provider Status
    st.markdown("#### AI Engine & Provider")
    provider_options = ["auto", "groq", "gemini"]
    provider_labels = {
        "auto": "Auto (Gemini with Groq Fallback)",
        "groq": "Groq (High-Throughput Llama 3.3 70B)",
        "gemini": "Gemini (Google Flash 2.0)",
    }
    import os
    current_prov = os.getenv("AI_PROVIDER", "auto")
    if current_prov not in provider_options:
        current_prov = "auto"

    selected_prov = st.selectbox(
        "Active AI Provider",
        provider_options,
        index=provider_options.index(current_prov),
        format_func=lambda x: provider_labels.get(x, x),
    )
    if selected_prov != current_prov:
        update_llm_keys(provider=selected_prov)
        st.rerun()

    gemini_display, gemini_model = km.get_status_display()
    groq_display, groq_model = groq_km.get_status_display()

    st.markdown(
        f"""
        <div class="provider-card">
            <strong>Gemini:</strong> {gemini_display} <br>
            <span class="provider-sub">Model: <code>{gemini_model}</code></span><br>
            <hr class="provider-hr">
            <strong>Groq:</strong> {groq_display} <br>
            <span class="provider-sub">Model: <code>{groq_model}</code></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dynamic Key Management
    with st.expander("API Key Configuration"):
        st.markdown(
            "Configure your keys. Keys rotate automatically on rate limits and save persistently into `.env`."
        )

        st.markdown("**Groq Keys (High-Throughput)**")
        st.caption("[Get free Groq key](https://console.groq.com/keys) (300+ tok/sec, generous free tier)")
        user_groq_input = st.text_input(
            "Groq API Keys (comma-separated)",
            placeholder="gsk_..., gsk_...",
            type="password",
            help="High-quota provider. Recommended when Gemini free quota is exhausted.",
            key="ui_groq_keys_input",
        )

        st.markdown("**Google Gemini Keys**")
        st.caption("[Get Gemini keys at Google AI Studio](https://aistudio.google.com/app/apikey)")
        user_gemini_input = st.text_input(
            "Gemini API Keys (comma-separated)",
            placeholder="AIzaSy..., AIzaSy...",
            type="password",
            help="Keys will rotate round-robin with automatic cooldowns on 429 quota errors.",
            key="ui_gemini_keys_input",
        )

        if st.button("Save & Apply Keys", icon=":material/save:", type="primary", use_container_width=True):
            g_keys = [k.strip() for k in user_gemini_input.split(",") if k.strip() and not k.startswith("your_")] if user_gemini_input else None
            q_keys = [k.strip() for k in user_groq_input.split(",") if k.strip() and not k.startswith("your_")] if user_groq_input else None

            if (g_keys and len(g_keys) > 0) or (q_keys and len(q_keys) > 0):
                update_llm_keys(gemini_keys=g_keys, groq_keys=q_keys)
                st.success("API keys updated and saved to .env successfully!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.warning("Please enter at least one valid key.")

    # Global Settings
    st.markdown("#### Generation Settings")
    lite_mode = st.toggle(
        "Lite Mode (Single-Pass Summarization)",
        value=False,
        help="Summarizes papers using single-step calls instead of multi-chunk map-reduce to conserve quota.",
    )

    selected_tone = st.selectbox("Paper Tone", TONE_OPTIONS, index=0)
    selected_length = st.selectbox(
        "Target Length",
        list(LENGTH_OPTIONS.keys()),
        index=1,
        help="Target length per review section.",
    )

    allow_direct_quotes = st.checkbox(
        "Allow Direct Quotes",
        value=True,
        help="If enabled, quoted sentences (\"...\") with citations are excluded from the plagiarism penalty.",
    )

    sim_threshold = st.slider(
        "Similarity Flag Threshold",
        min_value=0.50,
        max_value=0.95,
        value=SIMILARITY_SENTENCE_THRESHOLD,
        step=0.05,
        help="Flag sentences matching source text with ratio above this cutoff.",
    )

    st.markdown("---")
    st.caption("ReviewForge • Academic Plagiarism-Safe Engine")


# Load papers for current project
papers = list_papers(st.session_state.project_id)
corpus_index = CorpusIndex(papers) if papers else None
sections = get_sections(st.session_state.project_id)

# Header Section
st.markdown('<div class="eyebrow-label">SYSTEMATIC LITERATURE REVIEW WORKSTATION</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">ReviewForge</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Publication-grade literature synthesis & plagiarism verification • Project: <strong>{st.session_state.project_name}</strong> ({len(papers)} papers indexed)</div>',
    unsafe_allow_html=True,
)

# ==================== 1-CLICK AUTOMATION WIZARD ====================
with st.container():
    st.markdown(
        """
        <div class="dark-island">
            <div class="dark-island-eyebrow">AUTONOMOUS SYNTHESIS</div>
            <div class="dark-island-title">1-Click Review Paper Generator</div>
            <p class="dark-island-desc">
                Synthesizes structured Paper Cards, discovers empirical research gaps, drafts 10 scholarly sections with IEEE in-text citations, and audits similarity against indexed PDFs before compiling an IEEE-formatted Word document.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    w_c1, w_c2, w_c3 = st.columns([3, 2, 2], vertical_alignment="bottom")
    with w_c1:
        auto_topic = st.text_input(
            "Review Paper Topic / Title",
            value=st.session_state.get("auto_topic", f"A Systematic Review of {st.session_state.project_name}"),
            placeholder="e.g. A Survey on Self-Attention in Computer Vision",
            key="wizard_auto_topic",
        )
    with w_c2:
        auto_focus = st.text_input(
            "Analytical Focus / Angle (Optional)",
            value=st.session_state.get("auto_focus", ""),
            placeholder="e.g. Focus on computational latency and resource constraints",
            key="wizard_auto_focus",
        )
    with w_c3:
        can_run_wizard = len(papers) > 0 and (km.total_keys > 0 or groq_km.total_keys > 0)
        btn_label = f"Generate Full Paper ({len(papers)} PDFs)" if papers else "Upload PDFs First"
        run_full_auto = st.button(
            btn_label,
            icon=":material/auto_stories:",
            type="primary",
            use_container_width=True,
            disabled=not can_run_wizard,
            help="Extracts cards, finds gaps, drafts all 10 sections, verifies plagiarism, and compiles IEEE .docx in one click.",
            key="wizard_run_btn",
        )

    if run_full_auto:
        st.session_state["auto_topic"] = auto_topic
        st.session_state["auto_focus"] = auto_focus
        with st.status("ReviewForge Autonomous Paper Generator...", expanded=True) as wizard_status:
            wizard_progress = st.progress(0.0)

            # Step 1: Extract cards for any papers missing them
            wizard_status.write(f"Step 1/5: Extracting structured Paper Cards for all {len(papers)} uploaded PDFs...")
            for p_idx, p in enumerate(papers, start=1):
                card = p.get("paper_card_json") or {}
                if not card.get("problem") or not card.get("methodology"):
                    pages_data = get_paper_pages(p["id"])
                    new_card = generate_paper_card(p, pages_data, lite_mode=lite_mode, key_manager=km)
                    update_paper_card(p["id"], new_card)
                    wizard_status.write(f"Card synthesized: [{p.get('citation_id')}] {p.get('title', p.get('filename'))[:35]}")
                wizard_progress.progress((p_idx / len(papers)) * 0.20)

            # Refresh papers after card updates
            papers = list_papers(st.session_state.project_id)

            # Step 2: Cross-paper analysis & gaps
            wizard_status.write("Step 2/5: Synthesizing comparative matrix, taxonomy, and research gaps...")
            tax_data = generate_taxonomy_and_timeline(papers, key_manager=km)
            save_analysis(st.session_state.project_id, "taxonomy", tax_data)
            wizard_progress.progress(0.28)

            gaps_data = find_research_gaps(papers, key_manager=km)
            save_analysis(st.session_state.project_id, "gaps", gaps_data)
            wizard_progress.progress(0.35)

            claims_list = extract_key_claims(papers, key_manager=km)
            save_analysis(st.session_state.project_id, "key_claims", {"claims": claims_list})
            wizard_progress.progress(0.40)
            wizard_status.write("Taxonomy, timeline evolution, and research gaps synthesized.")

            # Step 3: Draft all 10 sections
            combined_analysis = {"taxonomy": tax_data, "gaps": gaps_data, "key_claims": claims_list}
            corpus_index = CorpusIndex(papers)
            custom_directive = f"Paper Topic: {auto_topic}. " + (f"Specific Angle: {auto_focus}." if auto_focus else "")

            for s_i, sec_name in enumerate(REVIEW_SECTIONS, start=1):
                wizard_status.write(f"Step 3/5: Drafting section {s_i}/10: '{sec_name}'...")
                sec_text = generate_section_draft(
                    section_name=sec_name,
                    papers=papers,
                    analysis_data=combined_analysis,
                    previous_sections=sections,
                    custom_instructions=custom_directive,
                    tone=selected_tone,
                    target_length=selected_length,
                    key_manager=km,
                )
                sim_rep = check_section_similarity(
                    sec_text, corpus_index, threshold=sim_threshold, allow_direct_quotes=allow_direct_quotes
                )
                save_section(
                    project_id=st.session_state.project_id,
                    section_name=sec_name,
                    content=sec_text,
                    similarity_score=sim_rep["overall_score"],
                    similarity_report=sim_rep,
                )
                wizard_progress.progress(0.40 + (s_i / len(REVIEW_SECTIONS)) * 0.40)

            # Step 4: Verification and Auto-rewrite
            wizard_status.write("Step 4/5: Running anti-plagiarism verification & auto-rewrite loop...")
            sections = get_sections(st.session_state.project_id)
            for sec_name, sec_obj in sections.items():
                rep = sec_obj.get("similarity_report_json", {})
                if rep.get("flagged_items"):
                    rewritten_text, final_rep, _ = run_auto_rewrite_loop(
                        section_text=sec_obj["content"],
                        corpus_index=corpus_index,
                        threshold=sim_threshold,
                        max_attempts=2,
                        key_manager=km,
                    )
                    save_section(
                        project_id=st.session_state.project_id,
                        section_name=sec_name,
                        content=rewritten_text,
                        similarity_score=final_rep["overall_score"],
                        similarity_report=final_rep,
                    )
            wizard_progress.progress(0.92)

            # Step 5: Export IEEE DOCX
            wizard_status.write("Step 5/5: Compiling IEEE two-column Word document (.docx)...")
            sections = get_sections(st.session_state.project_id)
            docx_stream = generate_ieee_docx(
                sections=sections,
                papers=papers,
                paper_title=auto_topic,
                authors="Lead Researcher and Co-Authors",
                affiliations="Department of Computer Science & Engineering",
                include_ai_appendix=True,
            )
            st.session_state["generated_docx"] = docx_stream.getvalue()
            wizard_progress.progress(1.0)
            wizard_status.update(label="Complete Review Paper successfully generated", state="complete", expanded=False)
            st.success("Complete Review Paper successfully generated! Ready for download below.")

    # Display Download Card if generated
    if "generated_docx" in st.session_state:
        st.markdown(
            """
            <div class="success-card">
                <div class="eyebrow-label" style="color: #15803d; margin-bottom: 4px;">SYNTHESIS COMPLETE</div>
                <h4 style="color: #111111; font-family: 'Inter', sans-serif; font-weight: 500; margin: 0 0 6px 0;">Your Review Paper is Ready for Download</h4>
                <p style="color: #4f525d; margin: 0 0 12px 0; font-size: 0.92rem;">All 10 sections have been synthesized, cited, verified for similarity, and formatted in IEEE two-column layout.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        dl_c1, dl_c2, dl_c3 = st.columns(3)
        with dl_c1:
            st.download_button(
                "Download IEEE Word Document (.docx)",
                icon=":material/description:",
                data=st.session_state["generated_docx"],
                file_name=f"{st.session_state.project_name.lower().replace(' ', '_')}_ieee.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True,
            )
        with dl_c2:
            md_out = generate_markdown_export(sections, papers, paper_title=st.session_state.get("auto_topic", "Review Paper"))
            st.download_button(
                "Download Markdown (.md)",
                icon=":material/article:",
                data=md_out,
                file_name=f"{st.session_state.project_name.lower().replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_c3:
            bib_out = generate_bibtex_export(papers)
            st.download_button(
                "Download BibTeX References (.bib)",
                icon=":material/format_quote:",
                data=bib_out,
                file_name=f"{st.session_state.project_name.lower().replace(' ', '_')}_references.bib",
                mime="text/plain",
                use_container_width=True,
            )
        st.caption("You can also inspect or edit individual sections in the tabs below.")

    st.markdown("---")

# Main Navigation Tabs (Guided Flow)
st.markdown("### Interactive Exploration & Section-by-Section Editor")
tabs = st.tabs([
    ":material/upload_file: 1. Upload PDFs",
    ":material/description: 2. Summary Cards",
    ":material/table_chart: 3. Comparison & Taxonomy",
    ":material/travel_explore: 4. Research Gaps",
    ":material/edit_note: 5. Section Editor",
    ":material/policy: 6. Plagiarism Check",
    ":material/file_download: 7. Export IEEE .docx",
    ":material/chat: 8. Ask the Papers",
])

# ==================== TAB 1: UPLOAD ====================
with tabs[0]:
    st.header("Step 1: Upload & Clean Research PDFs")
    st.markdown(
        "Upload one or more academic research papers (PDF). PyMuPDF will extract pages, clean headers/footers, "
        "strip references, and detect metadata."
    )

    uploaded_files = st.file_uploader(
        "Upload Research Paper PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select PDF files from your device.",
    )

    if uploaded_files:
        if st.button(f"Process & Ingest {len(uploaded_files)} File(s)", icon=":material/upload_file:", type="primary"):
            has_keys = km.total_keys > 0 or groq_km.total_keys > 0
            with st.status(f"Processing & Cleaning {len(uploaded_files)} PDF(s)...", expanded=True) as upload_status:
                progress_bar = st.progress(0.0)
                for idx, file_obj in enumerate(uploaded_files, start=1):
                    upload_status.write(f"[{idx}/{len(uploaded_files)}] Extracting text & cleaning `{file_obj.name}`...")
                    file_bytes = file_obj.read()

                    try:
                        paper_data = process_pdf(
                            file_bytes_or_path=file_bytes,
                            filename=file_obj.name,
                            project_id=st.session_state.project_id,
                            use_llm_metadata=has_keys,
                            has_api_key=has_keys,
                        )

                        if paper_data.get("is_scanned"):
                            st.warning(f"Scanned or Image-Only Document: {paper_data['error_message']}")

                        paper_id = save_paper(paper_data)
                        upload_status.write(f"Stripped headers/footers, bibliography references, and indexed `{file_obj.name}`.")
                    except Exception as e:
                        st.error(f"Failed to process '{file_obj.name}': {e}")

                    progress_bar.progress(idx / len(uploaded_files))

                upload_status.update(label="All uploaded documents processed and indexed successfully", state="complete", expanded=False)
            st.success("Documents processed successfully!")
            time.sleep(1.0)
            st.rerun()

    st.markdown("---")
    st.subheader(f"Current Corpus ({len(papers)} Papers)")

    if papers:
        cols = st.columns([1, 4, 2, 2, 1])
        cols[0].markdown("**Cite ID**")
        cols[1].markdown("**Title & Authors**")
        cols[2].markdown("**Year / Venue**")
        cols[3].markdown("**DOI / Identifier**")
        cols[4].markdown("**Actions**")

        for p in papers:
            c1, c2, c3, c4, c5 = st.columns([1, 4, 2, 2, 1])
            c1.markdown(f"**[{p.get('citation_id')}]**")

            title_display = p.get("title") or p.get("filename")
            author_names = ", ".join(p.get("authors") or []) or "Author list not detected"
            c2.markdown(f"**{title_display}**\n\n<small>{author_names[:80]}</small>", unsafe_allow_html=True)

            venue_display = p.get("venue") or "Conference/Journal"
            year_display = p.get("year") or "N/A"
            c3.markdown(f"{year_display}\n\n*{venue_display}*")

            doi_display = p.get("doi") or "No DOI"
            c4.markdown(f"`{doi_display}`")

            if c5.button("Delete", icon=":material/delete:", key=f"del_paper_{p['id']}", help="Delete paper"):
                delete_paper(p["id"])
                st.rerun()
    else:
        st.info("No papers uploaded yet for this project. Upload your first PDF above!")


# ==================== TAB 2: PAPERS & CARDS ====================
with tabs[1]:
    st.header("Step 2 & 3: Structured Paper Cards & Citation Manager")
    st.markdown(
        "Synthesize each paper into a rich JSON **Paper Card** (problem, objective, methodology, "
        "dataset, metrics, key quantitative results, limitations, future work, and verbatim evidence snippets with page numbers)."
    )

    if not papers:
        st.info("Please upload research papers in Tab 1 first.")
    else:
        # Batch generation controls
        pending_papers = [
            p for p in papers
            if not ((p.get("paper_card_json") or {}).get("problem") and (p.get("paper_card_json") or {}).get("methodology"))
        ]

        b1, b2, b3 = st.columns([2, 2, 3])
        with b1:
            btn_pending_label = f"Extract Pending Cards ({len(pending_papers)})" if pending_papers else "All Cards Extracted"
            extract_pending = st.button(
                btn_pending_label,
                icon=":material/summarize:",
                type="primary",
                use_container_width=True,
                disabled=len(pending_papers) == 0,
                help="Extracts Paper Cards only for papers that don't have one yet. High-speed synthesis (~2-3s per paper).",
            )
        with b2:
            force_regen_all = st.button(
                "Regenerate All Cards",
                icon=":material/refresh:",
                type="secondary",
                use_container_width=True,
                help="Re-runs high-speed extraction across all uploaded papers, overwriting existing cards.",
            )
        with b3:
            import os
            active_p = os.getenv("AI_PROVIDER", "auto")
            st.caption(
                f"**High-Speed Extraction:** 1-pass synthesis (~2-3s/paper). "
                f"Provider: `{active_p}`."
            )

        target_papers = []
        if extract_pending:
            target_papers = pending_papers
        elif force_regen_all:
            target_papers = papers

        if target_papers:
            with st.status(f"Extracting Structured Paper Cards ({len(target_papers)} papers)...", expanded=True) as card_status:
                p_bar = st.progress(0.0)
                for i, p in enumerate(target_papers, start=1):
                    pages_data = get_paper_pages(p["id"])
                    card_status.write(f"[{i}/{len(target_papers)}] Extracting Methodology & Evidence for `{p.get('filename')}`...")

                    def progress_fn(msg: str, frac: float):
                        p_bar.progress((i - 1 + frac) / len(target_papers))

                    try:
                        card = generate_paper_card(
                            paper_data=p,
                            pages=pages_data,
                            lite_mode=lite_mode,
                            progress_callback=progress_fn,
                            key_manager=km,
                        )
                        update_paper_card(p["id"], card)
                        card_status.write(f"Card synthesized for `[{p.get('citation_id')}] {p.get('title', p.get('filename'))[:40]}`")
                    except Exception as e:
                        st.error(f"Error extracting card for '{p.get('filename')}': {e}")

                    p_bar.progress(i / len(target_papers))

                card_status.update(label="All Paper Cards Extracted Successfully", state="complete", expanded=False)
            st.success("Paper Cards extracted successfully!")
            time.sleep(0.5)
            st.rerun()

        st.markdown("---")

        # Citation Management & Deduplication
        with st.expander("Citation Reordering & Metadata Manager"):
            st.markdown("Reorder citation numbers or edit metadata:")
            paper_order = st.multiselect(
                "Citation Order (Drag or Select IDs)",
                options=[p["id"] for p in papers],
                default=[p["id"] for p in papers],
                format_func=lambda pid: next(
                    f"[{p['citation_id']}] {p.get('title', p.get('filename'))[:50]}..."
                    for p in papers
                    if p["id"] == pid
                ),
            )
            if st.button("Apply New Citation Order", icon=":material/swap_vert:"):
                if len(paper_order) == len(papers):
                    reorder_citations(st.session_state.project_id, paper_order)
                    st.success("Citations reordered!")
                    st.rerun()
                else:
                    st.warning("Please include all papers in the reordering list.")

        # Individual Paper Card Accordions
        for p in papers:
            card = p.get("paper_card_json") or {}
            has_card = bool(card.get("problem") or card.get("methodology"))

            badge = "Card Extracted" if has_card else "Card Pending"
            with st.expander(f"[{p.get('citation_id')}] {p.get('title', p.get('filename'))} — {badge}"):
                c_meta, c_btn = st.columns([4, 1])
                with c_meta:
                    st.markdown(f"**Authors:** {', '.join(p.get('authors') or ['Unknown'])}")
                    st.markdown(f"**Year:** {p.get('year') or 'N/A'} | **Venue:** {p.get('venue') or 'N/A'} | **DOI:** {p.get('doi') or 'N/A'}")

                with c_btn:
                    if st.button("Regenerate", icon=":material/refresh:", key=f"regen_card_{p['id']}"):
                        pages_data = get_paper_pages(p["id"])
                        with st.status(f"Regenerating Card for [{p.get('citation_id')}]...", expanded=True) as single_status:
                            single_status.write(f"Extracting text and structure from `{p.get('filename')}`...")
                            card = generate_paper_card(p, pages_data, lite_mode=lite_mode, key_manager=km)
                            update_paper_card(p["id"], card)
                            single_status.update(label="Card Extracted", state="complete", expanded=False)
                        st.success("Paper card updated!")
                        st.rerun()

                if has_card:
                    t1, t2, t3, t4 = st.tabs(["Core Synthesis", "Evidence Snippets", "Quality Notes", "Raw Text"])
                    with t1:
                        st.markdown(f"**Problem Tackled:**\n\n{card.get('problem', 'N/A')}")
                        st.markdown(f"**Objective:**\n\n{card.get('objective', 'N/A')}")
                        st.markdown(f"**Methodology:**\n\n{card.get('methodology', 'N/A')}")
                        st.markdown(f"**Datasets & Environments:**\n\n`{card.get('dataset', 'N/A')}`")
                        st.markdown(f"**Evaluation Metrics:** `{', '.join(card.get('metrics', []))}`")

                        st.markdown("**Key Quantitative Results:**")
                        for r in card.get("key_results", []):
                            st.markdown(f"- {r}")

                        st.markdown("**Key Contributions:**")
                        for c in card.get("contributions", []):
                            st.markdown(f"- {c}")

                        limitations = card.get("limitations", {})
                        if isinstance(limitations, dict):
                            st.markdown("**Stated Limitations:**")
                            for lim in limitations.get("stated", []):
                                st.markdown(f"- {lim}")
                            st.markdown("**Inferred Limitations:**")
                            for lim in limitations.get("inferred", []):
                                st.markdown(f"- {lim}")

                        st.markdown("**Future Work:**")
                        for f in card.get("future_work", []):
                            st.markdown(f"- {f}")

                    with t2:
                        st.markdown("**Verbatim Evidence Snippets (≤ 25 words with Page Citations):**")
                        snippets = card.get("evidence_snippets", [])
                        if snippets:
                            for s in snippets:
                                st.markdown(
                                    f'<div class="evidence-box">"{s.get("snippet")}" — <strong>Page {s.get("page_number")}</strong></div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("No evidence snippets recorded.")

                    with t3:
                        s_col, w_col = st.columns(2)
                        with s_col:
                            st.markdown("##### Strengths")
                            for st_item in card.get("strengths", []):
                                st.markdown(f"- {st_item}")
                        with w_col:
                            st.markdown("##### Weaknesses & Threats to Validity")
                            for wk_item in card.get("weaknesses", []):
                                st.markdown(f"- {wk_item}")

                    with t4:
                        st.text_area(
                            "Clean Body Text (References Stripped)",
                            p.get("clean_text", "")[:4000] + "...",
                            height=250,
                            key=f"clean_view_{p['id']}",
                        )
                else:
                    st.warning("Paper card not extracted yet. Click 'Regenerate' or run the batch extraction above.")


# ==================== TAB 3: DEEP ANALYSIS ====================
with tabs[2]:
    st.header("Step 4: Cross-Paper Deep Analysis")
    st.markdown("Comparative tables, taxonomy categorization, timeline evolution, and foundational key claims.")

    if not papers:
        st.info("Upload papers first.")
    else:
        # Comparison Table
        st.subheader("1. Multi-Paper Comparative Matrix")
        table_rows = build_comparison_table(papers)
        if table_rows:
            st.dataframe(table_rows, use_container_width=True)

        st.markdown("---")

        # Taxonomy and Timeline
        st.subheader("2. Taxonomy & Chronological Evolution")
        tax_data = get_analysis(st.session_state.project_id, "taxonomy")

        col_t1, col_t2 = st.columns([1, 4])
        with col_t1:
            if st.button("Generate Taxonomy & Trends", icon=":material/schema:", use_container_width=True):
                with st.status("Analyzing Corpus Taxonomy & Chronological Evolution...", expanded=True) as tax_status:
                    tax_status.write("Compiling methodological paradigms across uploaded papers...")
                    tax_status.write("Generating taxonomy categories, milestone timeline, and emerging trends...")
                    try:
                        tax_data = generate_taxonomy_and_timeline(papers, key_manager=km)
                        save_analysis(st.session_state.project_id, "taxonomy", tax_data)
                        tax_status.update(label="Taxonomy & Evolution Trends Generated", state="complete", expanded=False)
                        st.success("Corpus taxonomy and timeline evolution saved to database!")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        tax_status.update(label=f"Failed: {e}", state="error")
                        st.error(f"Taxonomy analysis failed: {e}")

        if tax_data:
            cat_col, time_col = st.columns(2)
            with cat_col:
                st.markdown("#### Methodological Categories")
                for cat in tax_data.get("taxonomy_categories", []):
                    st.markdown(f"**{cat.get('category_name')}**")
                    st.caption(cat.get("description", ""))
                    for cp in cat.get("papers", []):
                        st.markdown(f"- `[{cp.get('citation_id')}]` **{cp.get('title')}**: *{cp.get('distinctive_feature')}*")
                    st.markdown("")

            with time_col:
                st.markdown("#### Chronological Timeline & Milestones")
                for t in tax_data.get("timeline_evolution", []):
                    st.markdown(f"**{t.get('year')}: {t.get('milestone')}**")
                    st.caption(f"Papers involved: {t.get('papers_involved')} — {t.get('trend_summary')}")

                st.markdown("#### Emerging Trends")
                for tr in tax_data.get("emerging_trends", []):
                    st.markdown(f"- **Trend:** {tr}")
        else:
            st.info("Click 'Generate Taxonomy & Trends' to analyze the corpus paradigms.")

        st.markdown("---")

        # Key Claims for Introduction
        st.subheader("3. Key Claims Extractor (For Introduction)")
        claims_data = get_analysis(st.session_state.project_id, "key_claims")

        if st.button("Extract High-Level Key Claims", icon=":material/verified:"):
            with st.status("Extracting Foundational Key Claims across Papers...", expanded=True) as claims_status:
                claims_status.write("Scanning evidence across all papers for core scientific claims...")
                claims_status.write("Identifying theoretical consensus and empirical tensions...")
                try:
                    claims = extract_key_claims(papers, key_manager=km)
                    save_analysis(st.session_state.project_id, "key_claims", {"claims": claims})
                    claims_status.update(label="Key Claims Extracted Successfully", state="complete", expanded=False)
                    st.success("Key claims extracted with citation mappings!")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    claims_status.update(label=f"Extraction failed: {e}", state="error")
                    st.error(f"Failed to extract key claims: {e}")

        if claims_data and claims_data.get("claims"):
            for cl in claims_data["claims"]:
                cits = ", ".join(f"[{c}]" for c in cl.get("supporting_citations", []))
                st.markdown(f"- **{cl.get('claim')}** {cits}")
                st.caption(f"Tension / Nuance: {cl.get('tension_or_nuance')}")


# ==================== TAB 4: GAPS & RQ ====================
with tabs[3]:
    st.header("Step 4: Research Gap Finder & Research Question Generator")
    st.markdown(
        "Discovers contradictions, untested datasets/domains, missing comparisons, common limitations, "
        "and unaddressed future work across the corpus."
    )

    if not papers:
        st.info("Upload papers first.")
    else:
        gaps_data = get_analysis(st.session_state.project_id, "gaps")

        if st.button("Run Deep Research Gap Analysis", icon=":material/travel_explore:", type="primary"):
            with st.status("Running Deep Research Gap Analysis across Corpus...", expanded=True) as gap_status:
                gap_status.write("Checking cross-paper contradictions, unaddressed domains, and benchmarking gaps...")
                gap_status.write("Computing novelty, feasibility, and composite rank scores...")
                try:
                    gaps_data = find_research_gaps(papers, key_manager=km)
                    if gaps_data and gaps_data.get("gaps"):
                        save_analysis(st.session_state.project_id, "gaps", gaps_data)
                        num_gaps = len(gaps_data.get("gaps", []))
                        gap_status.update(label=f"Deep Research Gap Analysis Complete ({num_gaps} Gaps Identified)", state="complete", expanded=False)
                        st.success(f"Discovered {num_gaps} high-confidence research gaps with ranked evidence!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        gap_status.update(label="No gaps could be identified. Please verify paper cards in Tab 2.", state="error")
                        st.warning("No research gaps identified. Try regenerating paper cards in Tab 2 first.")
                except Exception as e:
                    gap_status.update(label=f"Gap analysis failed: {e}", state="error")
                    st.error(f"Deep Research Gap Analysis failed: {e}")

        if gaps_data and gaps_data.get("gaps"):
            corpus_syn = gaps_data.get("overall_corpus_synthesis", {})
            if corpus_syn:
                st.info(
                    f"**Single Greatest Roadblock in Field:** {corpus_syn.get('primary_bottleneck', 'N/A')}\n\n"
                    f"**Recommended Review Paper Framing:** {corpus_syn.get('recommended_review_focus', 'N/A')}"
                )

            st.markdown("### Ranked Research Gaps")
            gaps_list = gaps_data.get("gaps", [])

            selected_gap_idx = st.selectbox(
                "Select Gap to Inspect or Generate Research Questions",
                options=range(len(gaps_list)),
                format_func=lambda i: f"[{gaps_list[i].get('gap_id', f'GAP-{i+1}')}] {gaps_list[i].get('title')} (Rank: {gaps_list[i].get('composite_rank')})",
            )

            chosen_gap = gaps_list[selected_gap_idx]

            st.markdown(f"#### {chosen_gap.get('title')} (`{chosen_gap.get('type')}`)")
            st.markdown(chosen_gap.get("description", ""))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Confidence", f"{round(float(chosen_gap.get('confidence_score', 0)) * 100)}%")
            m2.metric("Novelty", f"{chosen_gap.get('novelty_score')}/5.0")
            m3.metric("Feasibility", f"{chosen_gap.get('feasibility_score')}/5.0")
            m4.metric("Composite Rank", f"{chosen_gap.get('composite_rank')}")

            st.markdown("**Ranked Evidence from Corpus:**")
            for ev in chosen_gap.get("evidence", []):
                st.markdown(
                    f"- `[{ev.get('citation_id')}]` **{ev.get('paper_title')}** (Page {ev.get('page_number')}): "
                    f"*{ev.get('snippet_or_note')}*"
                )

            st.markdown(f"**Suggested Research Question:**\n> {chosen_gap.get('suggested_research_question')}")
            st.markdown(f"**How to Frame Your Survey Paper Contribution:**\n*{chosen_gap.get('gap_to_contribution_advice')}*")

            st.markdown("---")
            st.markdown("#### Formulate Concrete Research Questions & Hypotheses")
            if st.button("Generate Formal Research Formulation for this Gap", icon=":material/psychology:"):
                with st.status(f"Formulating Research Questions for [{chosen_gap.get('gap_id', 'Gap')}]...", expanded=True) as rq_status:
                    rq_status.write("Deconstructing gap into primary and sub-research questions...")
                    rq_status.write("Defining testable hypotheses and empirical roadmap...")
                    try:
                        rq_data = generate_research_objectives(chosen_gap, key_manager=km)
                        st.session_state[f"rq_for_{chosen_gap.get('gap_id')}"] = rq_data
                        rq_status.update(label="Research Questions & Hypotheses Formulated", state="complete", expanded=False)
                        st.rerun()
                    except Exception as e:
                        rq_status.update(label=f"Failed: {e}", state="error")
                        st.error(f"Failed to formulate research questions: {e}")

            cached_rq = st.session_state.get(f"rq_for_{chosen_gap.get('gap_id')}")
            if cached_rq:
                st.markdown(f"##### Primary Research Question:\n**{cached_rq.get('primary_research_question')}**")
                st.markdown("##### Sub-Questions:")
                for sq in cached_rq.get("sub_questions", []):
                    st.markdown(f"- {sq}")
                st.markdown("##### Testable Hypotheses:")
                for hyp in cached_rq.get("hypotheses", []):
                    st.markdown(f"- {hyp}")
                st.markdown("##### Methodological Roadmap:")
                for step in cached_rq.get("methodological_roadmap", []):
                    st.markdown(f"1. {step}")
        else:
            st.info("Click 'Run Deep Research Gap Analysis' to evaluate contradictions and open directions.")


# ==================== TAB 5: SECTION WRITER ====================
with tabs[4]:
    st.header("Step 5: Review Paper Section Generator")
    st.markdown(
        "Generate your review paper section by section. Synthesis prompts enforce original academic prose, "
        "thematic organization, and strict citation mapping (`[1]`, `[2]`)."
    )

    if not papers:
        st.info("Upload papers first.")
    else:
        active_sec = st.selectbox("Select Section to Draft or Edit", REVIEW_SECTIONS, index=2)

        curr_sec_data = sections.get(active_sec, {})
        curr_content = curr_sec_data.get("content", "")

        col_w1, col_w2 = st.columns([3, 2])
        with col_w1:
            custom_inst = st.text_input(
                f"Custom Instructions for {active_sec} (Optional)",
                placeholder="e.g. Focus on computational complexity trade-offs between methods [1] and [3].",
            )
        with col_w2:
            auto_check_plag = st.checkbox("Auto-run Similarity Check after drafting", value=True)

        w_btn1, w_btn2, _ = st.columns([2, 2, 4])
        with w_btn1:
            if st.button(f"Draft '{active_sec}'", icon=":material/edit_note:", type="primary", use_container_width=True):
                with st.status(f"Drafting Section '{active_sec}' with IEEE Academic Synthesis...", expanded=True) as draft_status:
                    draft_status.write("Gathering corpus context, taxonomy, and research gap evidence...")
                    draft_status.write(f"Synthesizing original prose in '{selected_tone}' tone ({selected_length} length)...")
                    try:
                        analysis_tax = get_analysis(st.session_state.project_id, "taxonomy")
                        analysis_gaps = get_analysis(st.session_state.project_id, "gaps")
                        combined_analysis = {"taxonomy": analysis_tax, "gaps": analysis_gaps}

                        draft_text = generate_section_draft(
                            section_name=active_sec,
                            papers=papers,
                            analysis_data=combined_analysis,
                            previous_sections=sections,
                            custom_instructions=custom_inst,
                            tone=selected_tone,
                            target_length=selected_length,
                            key_manager=km,
                        )

                        # Initial similarity check
                        sim_rep = None
                        sim_score = 0.0
                        if auto_check_plag and corpus_index:
                            draft_status.write("Auditing drafted section against source corpus for verbatim overlap...")
                            sim_rep = check_section_similarity(
                                draft_text, corpus_index, threshold=sim_threshold, allow_direct_quotes=allow_direct_quotes
                            )
                            sim_score = sim_rep.get("overall_score", 0.0)

                        save_section(
                            project_id=st.session_state.project_id,
                            section_name=active_sec,
                            content=draft_text,
                            similarity_score=sim_score,
                            similarity_report=sim_rep,
                        )
                        draft_status.update(label=f"Section '{active_sec}' Drafted Successfully", state="complete", expanded=False)
                        st.success(f"'{active_sec}' drafted successfully!")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        draft_status.update(label=f"Drafting failed: {e}", state="error")
                        st.error(f"Failed to draft '{active_sec}': {e}")

        with w_btn2:
            if st.button("Batch Draft All Sections", icon=":material/library_books:", use_container_width=True):
                with st.status("Batch Drafting All 10 Review Paper Sections...", expanded=True) as batch_status:
                    prog = st.progress(0.0)
                    analysis_tax = get_analysis(st.session_state.project_id, "taxonomy")
                    analysis_gaps = get_analysis(st.session_state.project_id, "gaps")
                    combined_analysis = {"taxonomy": analysis_tax, "gaps": analysis_gaps}

                    for s_i, sec_name in enumerate(REVIEW_SECTIONS, start=1):
                        batch_status.write(f"[{s_i}/10] Drafting '{sec_name}'...")
                        try:
                            d_text = generate_section_draft(
                                section_name=sec_name,
                                papers=papers,
                                analysis_data=combined_analysis,
                                previous_sections=sections,
                                custom_instructions="",
                                tone=selected_tone,
                                target_length=selected_length,
                                key_manager=km,
                            )
                            s_rep = check_section_similarity(d_text, corpus_index, threshold=sim_threshold) if corpus_index else None
                            s_score = s_rep.get("overall_score", 0.0) if s_rep else 0.0

                            save_section(
                                project_id=st.session_state.project_id,
                                section_name=sec_name,
                                content=d_text,
                                similarity_score=s_score,
                                similarity_report=s_rep,
                            )
                        except Exception as e:
                            st.error(f"Failed to draft '{sec_name}': {e}")
                        prog.progress(s_i / len(REVIEW_SECTIONS))

                    batch_status.update(label="All 10 Review Sections Successfully Drafted", state="complete", expanded=False)
                st.success("Batch drafting completed!")
                time.sleep(1.0)
                st.rerun()

        st.markdown("---")

        # Current Section Editor and Preview
        if curr_content:
            rep = curr_sec_data.get("similarity_report_json", {})
            score = rep.get("overall_score", 0.0)
            band = rep.get("status_band", "green")
            badge_class = f"badge-{band}"

            st.markdown(
                f"### Section Draft: `{active_sec}` "
                f'<span class="{badge_class}">Similarity: {round(score * 100, 1)}% ({band.upper()})</span>',
                unsafe_allow_html=True,
            )

            edited_text = st.text_area(
                "Markdown Content (Edit freely; changes persist on save)",
                curr_content,
                height=400,
                key=f"text_edit_{active_sec}",
            )

            save_c1, save_c2 = st.columns([1, 4])
            with save_c1:
                if st.button("Save Manual Edits", icon=":material/save:"):
                    s_rep = check_section_similarity(edited_text, corpus_index, threshold=sim_threshold) if corpus_index else None
                    s_score = s_rep.get("overall_score", 0.0) if s_rep else 0.0
                    save_section(
                        project_id=st.session_state.project_id,
                        section_name=active_sec,
                        content=edited_text,
                        similarity_score=s_score,
                        similarity_report=s_rep,
                    )
                    st.success("Section saved!")
                    st.rerun()
                    s_rep = check_section_similarity(edited_text, corpus_index, threshold=sim_threshold) if corpus_index else None
                    s_score = s_rep.get("overall_score", 0.0) if s_rep else 0.0
                    save_section(
                        project_id=st.session_state.project_id,
                        section_name=active_sec,
                        content=edited_text,
                        similarity_score=s_score,
                        similarity_report=s_rep,
                    )
                    st.success("Section saved!")
                    st.rerun()

            with save_c2:
                # Copy section snippet
                st.caption("Tip: You can copy this Markdown text or export the full IEEE formatted paper in Tab 7.")
        else:
            st.info(f"'{active_sec}' has not been drafted yet. Click 'Draft' above.")


# ==================== TAB 6: PLAGIARISM CHECK ====================
with tabs[5]:
    st.header("Step 6: Plagiarism & Source-Overlap Verification")
    st.markdown(
        "> **Academic Integrity Notice:** ReviewForge uses local word n-gram (5-gram & 8-gram) matching "
        "and fuzzy sentence-level comparison (`rapidfuzz`) against your uploaded PDFs. "
        "No software can guarantee 0% similarity against the entire internet. "
        "Always run your final manuscript through your institution's licensed checker (e.g., Turnitin or iThenticate)."
    )

    if not sections:
        st.info("No sections drafted yet. Draft sections in Tab 5 first.")
    elif not corpus_index:
        st.info("No source papers available for similarity indexing.")
    else:
        sec_to_check = st.selectbox(
            "Select Section to Audit",
            [name for name in REVIEW_SECTIONS if name in sections and sections[name].get("content")],
        )

        sec_data = sections[sec_to_check]
        sec_content = sec_data.get("content", "")
        curr_report = sec_data.get("similarity_report_json", {})

        score = curr_report.get("overall_score", 0.0)
        band = curr_report.get("status_band", "green")
        flagged = curr_report.get("flagged_items", [])

        # Metric summary
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Overall Overlap", f"{round(score * 100, 1)}%", delta=f"{band.upper()}", delta_color="inverse")
        col_m2.metric("Total Sentences", f"{curr_report.get('total_sentences', 0)}")
        col_m3.metric("Flagged Sentences", f"{len(flagged)}")
        col_m4.metric("8-Gram Clashes", f"{curr_report.get('ngram_stats', {}).get('8gram_overlap_count', 0)}")

        st.markdown("---")

        # Action Buttons: Manual re-check or Auto-Rewrite Loop
        act_c1, act_c2 = st.columns(2)
        with act_c1:
            if st.button("Re-Run Plagiarism Check on this Section", icon=":material/policy:", use_container_width=True):
                with st.status(f"Auditing '{sec_to_check}' for Similarity & N-Gram Overlap...", expanded=True) as audit_status:
                    audit_status.write("Tokenizing sentences and matching against corpus 5-grams and 8-grams...")
                    audit_status.write("Computing Levenshtein fuzzy distance against source text...")
                    rep = check_section_similarity(
                        sec_content, corpus_index, threshold=sim_threshold, allow_direct_quotes=allow_direct_quotes
                    )
                    save_section(
                        project_id=st.session_state.project_id,
                        section_name=sec_to_check,
                        content=sec_content,
                        similarity_score=rep["overall_score"],
                        similarity_report=rep,
                    )
                    audit_status.update(label=f"Audit Complete: {round(rep['overall_score']*100, 1)}% Similarity ({rep['status_band'].upper()})", state="complete", expanded=False)
                st.rerun()

        with act_c2:
            if flagged:
                if st.button("Run Anti-Plagiarism Auto-Rewrite Loop", icon=":material/auto_fix_high:", type="primary", use_container_width=True):
                    with st.status(f"Running Anti-Plagiarism Auto-Rewrite Loop on '{sec_to_check}'...", expanded=True) as rewrite_status:
                        rewrite_status.write(f"Found {len(flagged)} flagged sentences exceeding {round(sim_threshold*100)}% threshold.")
                        rewrite_status.write("Synthesizing academically rigorous paraphrases without losing technical meaning...")

                        def rewrite_cb(msg: str, frac: float):
                            rewrite_status.write(f"{msg}")

                        final_text, final_rep, history = run_auto_rewrite_loop(
                            section_text=sec_content,
                            corpus_index=corpus_index,
                            threshold=sim_threshold,
                            progress_callback=rewrite_cb,
                            key_manager=km,
                        )

                        save_section(
                            project_id=st.session_state.project_id,
                            section_name=sec_to_check,
                            content=final_text,
                            similarity_score=final_rep["overall_score"],
                            similarity_report=final_rep,
                        )
                        st.session_state[f"rewrite_hist_{sec_to_check}"] = history
                        rewrite_status.update(label="Auto-Rewrite Loop Complete. Similarity reduced.", state="complete", expanded=False)
                    st.success("Auto-rewrite loop complete!")
                    st.rerun()

        # Display rewrite before/after if available
        rewrite_history = st.session_state.get(f"rewrite_hist_{sec_to_check}", [])
        if rewrite_history:
            st.markdown("#### Auto-Rewrite Before & After Log")
            for h in rewrite_history:
                with st.expander(f"Restructured Sentence (Attempt {h['attempt']})"):
                    st.markdown(f"**Original:** `{h['original']}`")
                    st.markdown(f"**Rewritten:** `{h['rewritten']}`")
                    st.caption(f"Matched source phrasing: *{h.get('source_match', '')}*")

        st.markdown("---")

        # Flagged sentences inspection
        st.subheader("Sentence-by-Sentence Audit")
        if not flagged:
            st.success("No sentences flagged! This section adheres to synthesis guidelines.")
        else:
            for item in flagged:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<strong>Flagged Sentence:</strong> "{item.get("sentence")}"<br>'
                    f'<span class="badge-red">{item.get("match_reason")}</span> • '
                    f'Matched Paper: <strong>[{item.get("matched_paper_citation")}] {item.get("matched_paper_title")}</strong><br>'
                    f'<small>Source text excerpt: <em>"{item.get("matched_source_snippet")}"</em></small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Self-overlap across sections
        st.subheader("Cross-Section Self-Overlap Check")
        self_dups = check_self_overlap(sec_to_check, sec_content, sections)
        if self_dups:
            st.warning(f"Detected {len(self_dups)} phrase duplication(s) with other sections in this paper:")
            for d in self_dups:
                st.markdown(f"- Duplicated with **{d['other_section']}**: *\"{d['overlapping_phrase']}\"*")
        else:
            st.caption("No redundant sentences detected across other review sections.")

        st.markdown("---")

        # Download report
        rep_md = generate_similarity_report_markdown(sections)
        st.download_button(
            "Download Full Plagiarism Verification Audit (.md)",
            icon=":material/article:",
            data=rep_md,
            file_name=f"{st.session_state.project_name.lower().replace(' ', '_')}_similarity_report.md",
            mime="text/markdown",
        )


# ==================== TAB 7: EXPORT ====================
with tabs[6]:
    st.header("Step 7: IEEE-Format Export & Deliverables")
    st.markdown("Export your drafted review paper as IEEE-format `.docx`, Markdown, and BibTeX.")

    if not sections:
        st.info("No drafted sections available to export.")
    else:
        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            doc_title = st.text_input(
                "Paper Title",
                value=f"A Comprehensive Survey of {st.session_state.project_name}",
            )
            authors_str = st.text_input(
                "Authors",
                value="Lead Author, Contributing Researcher, and Senior Investigator",
            )
        with exp_c2:
            affil_str = st.text_input(
                "Affiliations",
                value="Department of Computer Science & Engineering, Institute of Advanced Technology",
            )
            include_ai_disc = st.checkbox(
                "Include 'AI Usage & Sources' Appendix",
                value=True,
                help="Appends an AI methodology & verification disclosure adhering to IEEE/ACM publisher policies.",
            )

        st.markdown("---")

        # 1. IEEE DOCX Export
        try:
            docx_bytes = generate_ieee_docx(
                sections=sections,
                papers=papers,
                paper_title=doc_title,
                authors=authors_str,
                affiliations=affil_str,
                include_ai_appendix=include_ai_disc,
            )

            st.download_button(
                "Download IEEE-Format .docx (Two-Column, Formatted)",
                icon=":material/description:",
                data=docx_bytes,
                file_name=f"{st.session_state.project_name.lower().replace(' ', '_')}_ieee.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
            )
        except Exception as e:
            st.error(f"Error generating .docx: {e}")

        # 2. Markdown Export
        md_text = generate_markdown_export(sections, papers, paper_title=doc_title)
        st.download_button(
            "Download Markdown (.md)",
            icon=":material/article:",
            data=md_text,
            file_name=f"{st.session_state.project_name.lower().replace(' ', '_')}.md",
            mime="text/markdown",
        )

        # 3. BibTeX Export
        bib_text = generate_bibtex_export(papers)
        st.download_button(
            "Download BibTeX (.bib)",
            icon=":material/format_quote:",
            data=bib_text,
            file_name=f"{st.session_state.project_name.lower().replace(' ', '_')}_references.bib",
            mime="text/plain",
        )


# ==================== TAB 8: ASK THE PAPERS ====================
with tabs[7]:
    st.header("Step 8: Ask the Papers (Grounded Q&A)")
    st.markdown(
        "Chat with your uploaded paper corpus. All responses are **strictly grounded** in the uploaded PDFs "
        "and cite exact paper IDs and page numbers."
    )

    if not papers:
        st.info("Upload papers first.")
    else:
        chat_history = get_chat_history(st.session_state.project_id)

        # Clear Chat button
        if st.button("Clear Chat History", icon=":material/delete_sweep:"):
            clear_chat_history(st.session_state.project_id)
            st.rerun()

        # Display Chat History
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["message"])

        # Chat Input
        user_query = st.chat_input("Ask a question about the uploaded papers (e.g. Which paper achieved lowest latency?)...")
        if user_query:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            save_chat_message(st.session_state.project_id, role="user", message=user_query)

            # Build grounded corpus context
            corpus_chunks = []
            for p in papers:
                cit_id = p.get("citation_id")
                card = p.get("paper_card_json") or {}
                snippets = card.get("evidence_snippets", [])
                snip_str = " | ".join(f"p.{s.get('page_number')}: {s.get('snippet')}" for s in snippets)
                corpus_chunks.append(
                    f"[Paper {cit_id}] {p.get('title')} ({p.get('year')}):\n"
                    f"Method: {card.get('methodology')}\n"
                    f"Dataset: {card.get('dataset')}\n"
                    f"Results: {'; '.join(card.get('key_results', []))}\n"
                    f"Limitations: {json.dumps(card.get('limitations', {}))}\n"
                    f"Snippets: {snip_str}\n"
                )
            context_str = "\n".join(corpus_chunks)

            # Load ask prompt
            ask_prompt_tmpl = (PROMPTS_DIR / "ask_papers.txt").read_text(encoding="utf-8")
            prompt = safe_fill_template(
                ask_prompt_tmpl,
                corpus_context=context_str,
                user_query=user_query,
                chat_history="\n".join(f"{m['role']}: {m['message']}" for m in chat_history[-4:]),
            )

            with st.chat_message("assistant"):
                with st.status("Searching & Synthesizing Grounded Answer...", expanded=True) as ask_status:
                    ask_status.write("Scanning extracted evidence snippets across all corpus papers...")
                    ask_status.write("Synthesizing answer strictly grounded with [Paper ID] page citations...")
                    try:
                        ans = call_llm(prompt=prompt, temperature=0.1)
                        ask_status.update(label="Answer Generated", state="complete", expanded=False)
                    except Exception as e:
                        ask_status.update(label=f"Failed: {e}", state="error")
                        ans = f"Error generating answer: {e}"
                st.markdown(ans)
                save_chat_message(st.session_state.project_id, role="assistant", message=ans)

# ==================== WORKSTATION FOOTER ====================
st.markdown(
    """
    <div class="app-footer">
        <div class="footer-left">
            <span class="footer-brand">ReviewForge</span>
            <span class="footer-sep">•</span>
            <span class="footer-desc">Systematic Literature Review Studio</span>
        </div>
        <div class="footer-right">
            <span>IEEE Two-Column Standard</span>
            <span class="footer-sep">•</span>
            <span>Plagiarism-Audited</span>
            <span class="footer-sep">•</span>
            <span>© 2026 ReviewForge</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
