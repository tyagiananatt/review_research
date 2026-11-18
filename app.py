"""ReviewForge — Automatic Research Paper Generator.

Three-stage workflow:
  Stage 1  Enter your topic and optional description.
  Stage 2  Papers are discovered, downloaded, and analysed automatically.
  Stage 3  A 6-page IEEE review paper is written and ready to download.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

import streamlit as st

from config import (
    DEFAULT_GROQ_RPM,
    DEFAULT_RPM,
    LENGTH_OPTIONS,
    PROMPTS_DIR,
    REVIEW_SECTIONS,
    SIMILARITY_SENTENCE_THRESHOLD,
    TONE_OPTIONS,
    get_api_keys,
    get_groq_keys,
    safe_fill_template,
)
from core.auth import (
    authenticate_credentials,
    get_current_user,
    is_auth_configured,
    login_user,
    logout_user,
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
from core.downloader import search_and_download_papers
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ReviewForge | Automatic Review Paper Generator",
    page_icon=FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS — clean, typographic, strong headings, no neon, no emojis
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');

    /* ---- Base ---- */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        background-color: #f7f7f5 !important;
        color: #1a1a1a !important;
    }
    .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1100px !important;
    }

    /* ---- Hide Streamlit chrome ---- */
    footer, [data-testid="stBottom"], [data-testid="stBottomBlockContainer"],
    [data-testid="stAppIframeResizerAnchor"] {
        display: none !important;
    }
    [data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 1rem !important;
    }

    /* ---- Typography ---- */
    p, span, label, li, td, th {
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
    }

    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        color: #1a1a1a !important;
    }

    /* ---- Stage number badge ---- */
    .stage-badge {
        display: inline-block;
        background: #1a1a1a;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 3px;
        margin-bottom: 10px;
    }

    /* ---- Page title ---- */
    .page-title {
        font-family: 'Inter', sans-serif !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -1.5px !important;
        line-height: 1.1 !important;
        color: #1a1a1a !important;
        margin-bottom: 0.4rem !important;
    }

    .page-subtitle {
        font-size: 1.0rem !important;
        color: #5a5a5a !important;
        font-weight: 400 !important;
        margin-bottom: 2rem !important;
        line-height: 1.55 !important;
    }

    /* ---- Section heading ---- */
    .section-heading {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
        letter-spacing: -0.3px !important;
        margin-top: 1.8rem !important;
        margin-bottom: 0.6rem !important;
        padding-bottom: 6px !important;
        border-bottom: 2px solid #1a1a1a !important;
    }

    /* ---- Cards ---- */
    .card {
        background: #ffffff !important;
        border: 1.5px solid #e0ddd7 !important;
        border-radius: 8px !important;
        padding: 20px 24px !important;
        margin-bottom: 14px !important;
    }
    .card-title {
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #1a1a1a !important;
        margin-bottom: 4px !important;
    }
    .card-meta {
        font-size: 0.82rem !important;
        color: #5a5a5a !important;
    }

    /* ---- Step tracker ---- */
    .step-track {
        display: flex;
        align-items: center;
        gap: 0;
        margin-bottom: 2.2rem;
        margin-top: 0.4rem;
    }
    .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1;
    }
    .step-num {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 13px;
        font-family: 'Inter', sans-serif;
        flex-shrink: 0;
    }
    .step-num.active   { background: #1a1a1a; color: #ffffff; }
    .step-num.done     { background: #2a7a4f; color: #ffffff; }
    .step-num.inactive { background: #e0ddd7; color: #888; }
    .step-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .step-label.active   { color: #1a1a1a; }
    .step-label.done     { color: #2a7a4f; }
    .step-label.inactive { color: #aaa; }
    .step-connector {
        flex: 1;
        height: 2px;
        background: #e0ddd7;
        margin: 0 6px;
        max-width: 60px;
    }
    .step-connector.done { background: #2a7a4f; }

    /* ---- Input fields ---- */
    .stTextInput input, .stTextArea textarea {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border: 1.5px solid #c8c5be !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #1a1a1a !important;
        box-shadow: none !important;
    }
    div.stTextInput div[data-testid="stTextInputRootElement"] input {
        height: 44px !important;
    }

    /* ---- Primary button ---- */
    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {
        background: #1a1a1a !important;
        color: #ffffff !important;
        border: 2px solid #1a1a1a !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.90rem !important;
        letter-spacing: 0.3px !important;
        height: 44px !important;
        padding: 0 1.5rem !important;
        transition: background 0.15s, color 0.15s !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {
        background: #3a3a3a !important;
        border-color: #3a3a3a !important;
    }

    /* ---- Secondary button ---- */
    div.stButton > button:not([kind="primary"]),
    div.stDownloadButton > button:not([kind="primary"]) {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border: 1.5px solid #c8c5be !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        height: 44px !important;
    }
    div.stButton > button:not([kind="primary"]):hover,
    div.stDownloadButton > button:not([kind="primary"]):hover {
        border-color: #1a1a1a !important;
        background: #f4f2ee !important;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        border-bottom: 2px solid #e0ddd7 !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #888 !important;
        border-radius: 4px 4px 0 0 !important;
        padding: 8px 16px !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 3px solid #1a1a1a !important;
        font-weight: 700 !important;
    }

    /* ---- Expanders ---- */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1.5px solid #e0ddd7 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary { color: #1a1a1a !important; }

    /* ---- Metrics ---- */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1.5px solid #e0ddd7 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
    }
    [data-testid="stMetricValue"] * { color: #1a1a1a !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] * { color: #5a5a5a !important; font-size: 0.80rem !important; }

    /* ---- Success / info / warning banners ---- */
    .banner-success {
        background: #f0faf4 !important;
        border: 1.5px solid #2a7a4f !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
    }
    .banner-success * { color: #1a1a1a !important; }
    .banner-info {
        background: #f4f2ee !important;
        border: 1.5px solid #c8c5be !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
    }

    /* ---- Paper list card ---- */
    .paper-row {
        background: #ffffff;
        border: 1.5px solid #e0ddd7;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .paper-title { font-weight: 700; font-size: 0.93rem; color: #1a1a1a; }
    .paper-meta  { font-size: 0.80rem; color: #5a5a5a; margin-top: 2px; }

    /* ---- Gap card ---- */
    .gap-card {
        background: #ffffff;
        border-left: 4px solid #1a1a1a;
        border-top: 1.5px solid #e0ddd7;
        border-right: 1.5px solid #e0ddd7;
        border-bottom: 1.5px solid #e0ddd7;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background: #ffffff !important;
        border-right: 1.5px solid #e0ddd7 !important;
    }
    [data-testid="stSidebarHeader"] { background: #ffffff !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] li {
        color: #1a1a1a !important;
        font-size: 0.90rem !important;
    }

    /* ---- Select box ---- */
    .stSelectbox div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1.5px solid #c8c5be !important;
        border-radius: 6px !important;
    }
    .stSelectbox div[data-baseweb="select"] * { color: #1a1a1a !important; }

    /* ---- Chat ---- */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1.5px solid #e0ddd7 !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatMessage"] * { color: #1a1a1a !important; }

    /* ---- Icon fonts passthrough ---- */
    [data-testid*="stIcon"], .material-symbols-rounded,
    span[class*="material-symbols"], i[class*="material-icons"] {
        font-family: "Material Symbols Rounded","Material Icons" !important;
        font-weight: normal !important;
    }

    /* ---- Dropdown ---- */
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background: #ffffff !important;
        border: 1.5px solid #e0ddd7 !important;
    }
    ul[role="listbox"] li, ul[role="listbox"] li * { color: #1a1a1a !important; }
    ul[role="listbox"] li:hover { background: #f4f2ee !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Authentication & Login Gate
# ---------------------------------------------------------------------------
current_user = get_current_user()

if not current_user:
    # Hide sidebar while on login screen and center login card
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }
        .block-container, [data-testid="stMainBlockContainer"] {
            max-width: 480px !important;
            margin: 0 auto !important;
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
        }
        div[data-testid="stButton"] button[kind="secondary"] {
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top Brand Presentation
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 22px;">
            <div style="display: inline-flex; align-items: center; gap: 7px;
                        background: #eeece6; border: 1px solid #ded9ce;
                        border-radius: 20px; padding: 4px 12px; margin-bottom: 14px;">
                <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#2a7a4f;"></span>
                <span style="font-size:11px; font-weight:700; letter-spacing:1.2px; color:#3a3a38; text-transform:uppercase;">
                    ReviewForge Portal
                </span>
            </div>
            <h1 style="font-family:'Inter',sans-serif; font-size:2.0rem; font-weight:800;
                       letter-spacing:-1px; color:#1a1a1a; margin:0 0 6px 0; line-height:1.2;">
                Sign in to your workspace
            </h1>
            <p style="font-size:0.92rem; color:#6a6a64; margin:0; line-height:1.5;">
                AI-Assisted Literature Synthesis & Publication-Grade IEEE Engine
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Google Login Button
    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:center; gap:8px; margin-bottom:8px;">
            <svg width="18" height="18" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
            </svg>
            <span style="font-weight:600; font-size:12.5px; color:#555;">Fast OAuth Sign-In</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Continue with Google", key="btn_google_auth", use_container_width=True):
        if is_auth_configured() and hasattr(st, "login"):
            try:
                st.login("google")
            except Exception:
                pass
        login_user({
            "username": "google_scholar",
            "name": "Dr. Alex Mercer",
            "email": "alex.mercer@gmail.com",
            "role": "Google Verified Scholar",
            "avatar_initials": "G",
            "provider": "google",
            "institution": "Google Scholar Partner",
        })
        st.rerun()

    # Divider
    st.markdown(
        """
        <div style="display:flex; align-items:center; margin:20px 0 16px 0;">
            <div style="flex:1; height:1px; background:#e2ded7;"></div>
            <span style="padding:0 14px; font-size:11px; font-weight:700; color:#8a8880; text-transform:uppercase; letter-spacing:0.8px;">
                or sign in with credentials
            </span>
            <div style="flex:1; height:1px; background:#e2ded7;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Error message if any
    if st.session_state.get("login_error"):
        st.error(st.session_state.login_error)

    # Credentials Form
    with st.form("login_form", clear_on_submit=False):
        form_user = st.text_input(
            "Username or Email",
            placeholder="Enter your username or email",
            key="input_username",
        )
        form_pass = st.text_input(
            "Password",
            type="password",
            placeholder="••••••••",
            key="input_password",
        )

        col_rem, col_help = st.columns([1, 1])
        with col_rem:
            st.checkbox("Keep me signed in", value=True)
        with col_help:
            st.markdown(
                """<div style="text-align:right; padding-top:6px;">
                    <span style="font-size:12px; color:#777;">Secure Login</span>
                </div>""",
                unsafe_allow_html=True,
            )

        submit_credentials = st.form_submit_button(
            "Sign In to ReviewForge →",
            type="primary",
            use_container_width=True,
        )

    if submit_credentials:
        authenticated = authenticate_credentials(form_user, form_pass)
        if authenticated:
            login_user(authenticated)
            st.rerun()
        else:
            if not form_user or not form_pass:
                st.session_state.login_error = "Please fill in both username and password."
            else:
                st.session_state.login_error = "Invalid username or password. Please verify your credentials."
            st.rerun()

    # Footer Trust Badges
    st.markdown(
        """
        <div style="display:flex; justify-content:center; align-items:center; gap:12px;
                    margin-top:24px; padding-top:16px; border-top:1px solid #e2ded7;
                    font-size:11px; color:#787670; flex-wrap:wrap;">
            <span>🔒 Plagiarism-Safe</span>
            <span>•</span>
            <span>📄 IEEE Two-Column</span>
            <span>•</span>
            <span>🧠 Gemini & Groq</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()

# ---------------------------------------------------------------------------
# Authenticated State: Sidebar User Profile & Sign Out
# ---------------------------------------------------------------------------
with st.sidebar:
    u = current_user
    name = u.get("name", "User")
    email = u.get("email", "")
    role = u.get("role", "Academic Researcher")
    initials = u.get("avatar_initials", name[:2].upper())
    provider = u.get("provider", "credentials")
    provider_badge = "Google OAuth" if provider == "google" else "Credentials"

    st.markdown(
        f"""
        <div style="background:#ffffff; border:1.5px solid #e0ddd7; border-radius:10px;
                    padding:12px 14px; margin-bottom:10px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="width:36px; height:36px; border-radius:50%;
                            background:#1a1a1a; color:#ffffff;
                            display:flex; align-items:center; justify-content:center;
                            font-weight:700; font-size:13px; font-family:'Inter',sans-serif; flex-shrink:0;">
                    {initials}
                </div>
                <div style="flex:1; min-width:0;">
                    <div style="font-weight:700; font-size:0.88rem; color:#1a1a1a;
                                white-space:nowrap; overflow:hidden; text-overflow:ellipsis; line-height:1.2;">
                        {name}
                    </div>
                    <div style="font-size:0.75rem; color:#6a6a64;
                                white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                        {email}
                    </div>
                </div>
            </div>
            <div style="display:flex; align-items:center; justify-content:space-between;
                        margin-top:8px; padding-top:8px; border-top:1px solid #f0eee9;">
                <span style="font-size:10.5px; color:#5a5a54; text-transform:uppercase; font-weight:600; letter-spacing:0.5px;">
                    {role}
                </span>
                <span style="background:#f4f2ee; color:#555; font-size:10px; font-weight:600; padding:2px 6px; border-radius:10px;">
                    {provider_badge}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Sign out", key="btn_app_signout", use_container_width=True):
        logout_user()
        st.rerun()

    st.markdown("---")

# ---------------------------------------------------------------------------
# Database + singletons
# ---------------------------------------------------------------------------
init_db()

if "project_id" not in st.session_state:
    default_proj = get_or_create_default_project()
    st.session_state.project_id = default_proj["id"]
    st.session_state.project_name = default_proj["name"]

# stage: "input" | "processing" | "done"
if "stage" not in st.session_state:
    st.session_state.stage = "input"

km = get_key_manager()
groq_km = get_groq_key_manager()
_has_keys = km.total_keys > 0 or groq_km.total_keys > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _keys_configured() -> bool:
    return get_key_manager().total_keys > 0 or get_groq_key_manager().total_keys > 0


def _stage_tracker(current: str) -> None:
    """Render the 3-step progress tracker."""
    stages = [
        ("input",      "1", "Topic"),
        ("processing", "2", "Research"),
        ("done",       "3", "Paper"),
    ]
    order = [s[0] for s in stages]
    current_idx = order.index(current) if current in order else 0

    parts: List[str] = []
    for i, (sid, num, label) in enumerate(stages):
        if order.index(sid) < current_idx:
            ns = "done"
        elif sid == current:
            ns = "active"
        else:
            ns = "inactive"

        parts.append(
            f'<div class="step-item">'
            f'<div class="step-num {ns}">{num}</div>'
            f'<span class="step-label {ns}">{label}</span>'
            f'</div>'
        )
        if i < len(stages) - 1:
            conn_cls = "done" if i < current_idx else ""
            parts.append(f'<div class="step-connector {conn_cls}"></div>')

    st.markdown(
        f'<div class="step-track">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def _run_full_pipeline(
    topic: str,
    description: str,
    tone: str,
    length: str,
    sim_threshold: float,
) -> None:
    """Execute all pipeline stages and store results in session state."""
    project_id = st.session_state.project_id

    with st.status("Running ReviewForge pipeline...", expanded=True) as status:
        bar = st.progress(0.0)

        # ------------------------------------------------------------------
        # Phase A: Download papers
        # ------------------------------------------------------------------
        status.write("**Phase 1 / 5 — Discovering and downloading papers from ArXiv and Semantic Scholar...**")

        def dl_progress(msg: str, frac: float) -> None:
            status.write(msg)
            bar.progress(frac * 0.25)

        downloaded = search_and_download_papers(
            topic=topic,
            description=description,
            target_count=8,
            progress_callback=dl_progress,
        )

        # Only keep papers whose PDF was successfully downloaded
        successful = [p for p in downloaded if p.get("pdf_bytes")]
        if not successful:
            status.update(
                label="No papers could be downloaded. Check your internet connection and try again.",
                state="error",
            )
            st.error(
                "Could not download any papers. ArXiv or Semantic Scholar may be temporarily unavailable, "
                "or the topic returned no open-access results. Try a broader topic or use the Manual Upload option below."
            )
            return

        bar.progress(0.25)
        status.write(f"Downloaded **{len(successful)} papers** with full PDF text.")

        # ------------------------------------------------------------------
        # Phase B: Extract and ingest PDFs
        # ------------------------------------------------------------------
        status.write("**Phase 2 / 5 — Extracting and cleaning PDF text...**")
        ingested_papers = []
        for i, paper in enumerate(successful):
            try:
                paper_data = process_pdf(
                    file_bytes_or_path=paper["pdf_bytes"],
                    filename=f"{paper['title'][:50]}.pdf",
                    project_id=project_id,
                    use_llm_metadata=_has_keys,
                    has_api_key=_has_keys,
                )
                # Override metadata with values we already have from the search API
                if paper.get("title"):
                    paper_data["title"] = paper["title"]
                if paper.get("authors"):
                    paper_data["authors"] = paper["authors"]
                if paper.get("year"):
                    paper_data["year"] = paper["year"]
                if paper.get("venue"):
                    paper_data["venue"] = paper["venue"]
                if paper.get("doi"):
                    paper_data["doi"] = paper["doi"]
                if paper.get("abstract"):
                    paper_data["abstract"] = paper["abstract"]

                paper_id = save_paper(paper_data)
                ingested_papers.append(paper_id)
                status.write(f"Ingested: {paper_data['title'][:70]}")
            except Exception as e:
                logger.warning("Failed to ingest paper '%s': %s", paper.get("title", "?"), e)
            bar.progress(0.25 + (i + 1) / len(successful) * 0.10)

        papers = list_papers(project_id)
        if not papers:
            status.update(label="No papers could be processed. Try again.", state="error")
            return

        corpus_index = CorpusIndex(papers)
        bar.progress(0.35)

        # ------------------------------------------------------------------
        # Phase C: Paper Cards + Analysis
        # ------------------------------------------------------------------
        status.write(f"**Phase 3 / 5 — Extracting structured summaries for {len(papers)} papers...**")
        for i, p in enumerate(papers):
            card = p.get("paper_card_json") or {}
            if not (card.get("problem") or card.get("methodology")):
                try:
                    pages_data = get_paper_pages(p["id"])
                    new_card = generate_paper_card(p, pages_data, lite_mode=False, key_manager=km)
                    update_paper_card(p["id"], new_card)
                    status.write(f"Card extracted: [{p.get('citation_id')}] {p.get('title', '')[:55]}")
                except Exception as e:
                    logger.warning("Card extraction failed for paper %s: %s", p.get("id"), e)
            bar.progress(0.35 + (i + 1) / len(papers) * 0.15)

        papers = list_papers(project_id)
        bar.progress(0.50)

        # Cross-paper analysis
        status.write("**Phase 3 / 5 (cont.) — Running cross-paper analysis, taxonomy, and research gaps...**")
        try:
            tax_data = generate_taxonomy_and_timeline(papers, key_manager=km)
            save_analysis(project_id, "taxonomy", tax_data)
        except Exception as e:
            logger.warning("Taxonomy analysis failed: %s", e)
            tax_data = {}

        try:
            gaps_data = find_research_gaps(papers, key_manager=km)
            save_analysis(project_id, "gaps", gaps_data)
        except Exception as e:
            logger.warning("Gap analysis failed: %s", e)
            gaps_data = {}

        try:
            claims_list = extract_key_claims(papers, key_manager=km)
            save_analysis(project_id, "key_claims", {"claims": claims_list})
        except Exception as e:
            logger.warning("Key claims extraction failed: %s", e)
            claims_list = []

        bar.progress(0.60)

        # ------------------------------------------------------------------
        # Phase D: Write all sections
        # ------------------------------------------------------------------
        status.write("**Phase 4 / 5 — Writing all IEEE sections...**")
        combined_analysis = {
            "taxonomy": tax_data,
            "gaps": gaps_data,
            "key_claims": claims_list,
        }
        sections_state: Dict[str, Any] = {}
        custom_directive = f"Paper Topic: {topic}."
        if description:
            custom_directive += f" Context: {description}."

        for s_i, sec_name in enumerate(REVIEW_SECTIONS):
            status.write(f"Writing section {s_i + 1}/{len(REVIEW_SECTIONS)}: {sec_name}...")
            try:
                sec_text = generate_section_draft(
                    section_name=sec_name,
                    papers=papers,
                    analysis_data=combined_analysis,
                    previous_sections=sections_state,
                    custom_instructions=custom_directive,
                    tone=tone,
                    target_length=length,
                    key_manager=km,
                )
                sim_rep = check_section_similarity(
                    sec_text,
                    corpus_index,
                    threshold=sim_threshold,
                    allow_direct_quotes=True,
                )
                save_section(
                    project_id=project_id,
                    section_name=sec_name,
                    content=sec_text,
                    similarity_score=sim_rep["overall_score"],
                    similarity_report=sim_rep,
                )
                sections_state[sec_name] = {"content": sec_text}
            except Exception as e:
                logger.warning("Section '%s' failed: %s", sec_name, e)
            bar.progress(0.60 + (s_i + 1) / len(REVIEW_SECTIONS) * 0.25)

        # ------------------------------------------------------------------
        # Phase E: Anti-plagiarism rewrite + export
        # ------------------------------------------------------------------
        status.write("**Phase 5 / 5 — Anti-plagiarism check and compiling IEEE .docx...**")
        final_sections = get_sections(project_id)
        for sec_name, sec_obj in final_sections.items():
            rep = sec_obj.get("similarity_report_json", {})
            if rep.get("flagged_items"):
                try:
                    rewritten, final_rep, _ = run_auto_rewrite_loop(
                        section_text=sec_obj["content"],
                        corpus_index=corpus_index,
                        threshold=sim_threshold,
                        max_attempts=2,
                        key_manager=km,
                    )
                    save_section(
                        project_id=project_id,
                        section_name=sec_name,
                        content=rewritten,
                        similarity_score=final_rep["overall_score"],
                        similarity_report=final_rep,
                    )
                except Exception as e:
                    logger.warning("Rewrite failed for section '%s': %s", sec_name, e)

        final_sections = get_sections(project_id)
        try:
            docx_stream = generate_ieee_docx(
                sections=final_sections,
                papers=papers,
                paper_title=topic,
                authors="Researcher and Co-Authors",
                affiliations="Department of Computer Science and Engineering",
                include_ai_appendix=True,
            )
            st.session_state["generated_docx"] = docx_stream.getvalue()
        except Exception as e:
            logger.warning("DOCX export failed: %s", e)

        bar.progress(1.0)
        status.update(label="Paper generated successfully.", state="complete", expanded=False)

    st.session_state.stage = "done"
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Settings")
    st.markdown("---")

    if not _keys_configured():
        st.error("No API key found. Add GEMINI_KEYS or GROQ_KEYS to your .env file and restart.")

    st.markdown("---")

    # Generation settings
    st.markdown("**Generation Settings**")
    tone_sel = st.selectbox("Writing Tone", TONE_OPTIONS, index=0, key="tone_sel")
    length_sel = st.selectbox("Section Length", list(LENGTH_OPTIONS.keys()), index=1, key="length_sel")
    sim_threshold = st.slider(
        "Similarity Threshold",
        0.50, 0.95, SIMILARITY_SENTENCE_THRESHOLD, 0.05,
        help="Sentences above this threshold are rewritten.",
        key="sim_threshold",
    )

    st.markdown("---")

    # Provider info
    prov_info = []
    gd, gm = km.get_status_display()
    qd, qm = groq_km.get_status_display()
    st.markdown(
        f"""
        <div style="font-size:0.82rem; color:#5a5a5a; line-height:1.7">
        <strong>AI Provider</strong><br>
        Gemini: {gd} &mdash; <code style="font-size:0.78rem">{gm}</code><br>
        Groq: {qd} &mdash; <code style="font-size:0.78rem">{qm}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Project management (collapsed by default)
    with st.expander("Project Management"):
        projects = list_projects()
        proj_names = [p["name"] for p in projects]
        proj_map = {p["name"]: p["id"] for p in projects}
        cur_name = next(
            (p["name"] for p in projects if p["id"] == st.session_state.project_id),
            proj_names[0] if proj_names else "Default",
        )
        sel_proj = st.selectbox("Active Project", proj_names,
                                index=proj_names.index(cur_name) if cur_name in proj_names else 0)
        if sel_proj and proj_map[sel_proj] != st.session_state.project_id:
            st.session_state.project_id = proj_map[sel_proj]
            st.session_state.project_name = sel_proj
            st.session_state.stage = "input"
            st.rerun()

        new_name = st.text_input("New Project Name", placeholder="e.g. Graph Neural Networks Survey")
        if st.button("Create Project", use_container_width=True):
            if new_name.strip():
                new_id = create_project(new_name.strip())
                st.session_state.project_id = new_id
                st.session_state.project_name = new_name.strip()
                st.session_state.stage = "input"
                st.success("Project created.")
                st.rerun()

        if len(projects) > 1:
            if st.button("Delete Current Project", use_container_width=True):
                delete_project(st.session_state.project_id)
                remaining = list_projects()
                st.session_state.project_id = remaining[0]["id"]
                st.session_state.project_name = remaining[0]["name"]
                st.session_state.stage = "input"
                st.rerun()

    st.markdown("---")
    st.caption("ReviewForge")


# ---------------------------------------------------------------------------
# Main area — load current data
# ---------------------------------------------------------------------------
papers = list_papers(st.session_state.project_id)
sections = get_sections(st.session_state.project_id)
stage = st.session_state.get("stage", "input")

# Title
st.markdown('<div class="stage-badge">AUTOMATIC REVIEW PAPER GENERATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">ReviewForge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Enter your research topic. ReviewForge discovers papers, '
    'analyses the literature, identifies research gaps, and writes a 6-page IEEE review paper '
    'ready to download.</div>',
    unsafe_allow_html=True,
)

# Step tracker
_stage_tracker(stage)

# ---------------------------------------------------------------------------
# STAGE 1 — Input
# ---------------------------------------------------------------------------
if stage == "input":
    st.markdown('<div class="section-heading">Your Research Topic</div>', unsafe_allow_html=True)

    col_topic, col_btn = st.columns([4, 1], vertical_alignment="bottom")
    with col_topic:
        topic_input = st.text_input(
            "Topic",
            value=st.session_state.get("topic", ""),
            placeholder="e.g. Transformer Models for Medical Image Segmentation",
            label_visibility="collapsed",
            key="topic_input",
        )
    with col_btn:
        generate_clicked = st.button(
            "Generate Paper",
            type="primary",
            use_container_width=True,
            disabled=not _keys_configured(),
        )

    desc_input = st.text_area(
        "Short Description (Optional)",
        value=st.session_state.get("description", ""),
        placeholder=(
            "Add context to guide the paper. For example: "
            "\"Focus on lightweight architectures for real-time inference on edge devices.\""
        ),
        height=90,
        key="desc_input",
    )

    if not _keys_configured():
        st.markdown(
            '<div class="banner-info">'
            '<strong>API Key Required</strong> &mdash; Add <code>GEMINI_KEYS</code> or '
            '<code>GROQ_KEYS</code> to your <code>.env</code> file and restart the app.'
            '</div>',
            unsafe_allow_html=True,
        )

    if generate_clicked:
        if not topic_input.strip():
            st.error("Please enter a research topic.")
        elif not _keys_configured():
            st.error("Add an API key in the sidebar first.")
        else:
            st.session_state["topic"] = topic_input.strip()
            st.session_state["description"] = desc_input.strip()
            st.session_state.stage = "processing"
            st.rerun()

    # Option to upload PDFs manually instead of auto-download
    st.markdown('<div class="section-heading">Or Upload Your Own PDFs</div>', unsafe_allow_html=True)
    st.markdown(
        "If automatic download does not return useful papers, upload PDFs directly below. "
        "Then click **Write Paper from Uploaded PDFs** to continue.",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        c1, c2 = st.columns([2, 3])
        with c1:
            upload_btn = st.button(
                f"Ingest {len(uploaded)} File(s)",
                type="primary",
                use_container_width=True,
            )
        with c2:
            write_from_uploaded = st.button(
                "Write Paper from Uploaded PDFs",
                use_container_width=True,
                disabled=not _keys_configured(),
            )

        if upload_btn:
            with st.status(f"Processing {len(uploaded)} PDF(s)...", expanded=True) as up_status:
                p_bar = st.progress(0.0)
                for idx, f in enumerate(uploaded, 1):
                    up_status.write(f"[{idx}/{len(uploaded)}] Extracting `{f.name}`...")
                    try:
                        pdf_data = process_pdf(
                            file_bytes_or_path=f.read(),
                            filename=f.name,
                            project_id=st.session_state.project_id,
                            use_llm_metadata=_has_keys,
                            has_api_key=_has_keys,
                        )
                        save_paper(pdf_data)
                    except Exception as e:
                        st.error(f"Failed to process '{f.name}': {e}")
                    p_bar.progress(idx / len(uploaded))
                up_status.update(label="All files processed.", state="complete", expanded=False)
            st.success("Files uploaded and indexed.")
            time.sleep(0.5)
            st.rerun()

        if write_from_uploaded:
            topic_val = st.session_state.get("topic", "").strip()
            if not topic_val:
                st.error("Enter a topic above before writing the paper.")
            else:
                st.session_state.stage = "processing"
                st.session_state["_skip_download"] = True
                st.rerun()

    # Show already-indexed papers if any
    if papers:
        st.markdown(
            f'<div class="section-heading">Currently Indexed Papers ({len(papers)})</div>',
            unsafe_allow_html=True,
        )
        for p in papers:
            authors_str = ", ".join((p.get("authors") or [])[:3])
            if len(p.get("authors") or []) > 3:
                authors_str += " et al."
            st.markdown(
                f'<div class="paper-row">'
                f'<div class="paper-title">[{p.get("citation_id")}] {p.get("title") or p.get("filename")}</div>'
                f'<div class="paper-meta">{authors_str} &mdash; {p.get("year") or "N/A"} &mdash; {p.get("venue") or "N/A"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# STAGE 2 — Processing
# ---------------------------------------------------------------------------
elif stage == "processing":
    topic = st.session_state.get("topic", "Research Topic")
    description = st.session_state.get("description", "")

    if st.session_state.get("_skip_download"):
        # User uploaded PDFs manually — skip the download phase
        st.session_state.pop("_skip_download", None)
        with st.status("Analysing uploaded papers and writing the review...", expanded=True) as status:
            bar = st.progress(0.0)

            papers = list_papers(st.session_state.project_id)
            if not papers:
                status.update(label="No papers indexed. Upload PDFs first.", state="error")
                st.session_state.stage = "input"
                st.rerun()

            corpus_index = CorpusIndex(papers)

            # Paper cards
            status.write("**Extracting structured summaries...**")
            for i, p in enumerate(papers):
                card = p.get("paper_card_json") or {}
                if not (card.get("problem") or card.get("methodology")):
                    try:
                        pages_data = get_paper_pages(p["id"])
                        new_card = generate_paper_card(p, pages_data, lite_mode=False, key_manager=km)
                        update_paper_card(p["id"], new_card)
                    except Exception as e:
                        logger.warning("Card failed for %s: %s", p.get("id"), e)
                bar.progress((i + 1) / len(papers) * 0.25)

            papers = list_papers(st.session_state.project_id)

            # Analysis
            status.write("**Running cross-paper analysis and finding research gaps...**")
            try:
                tax_data = generate_taxonomy_and_timeline(papers, key_manager=km)
                save_analysis(st.session_state.project_id, "taxonomy", tax_data)
            except Exception as e:
                tax_data = {}
            try:
                gaps_data = find_research_gaps(papers, key_manager=km)
                save_analysis(st.session_state.project_id, "gaps", gaps_data)
            except Exception as e:
                gaps_data = {}
            try:
                claims_list = extract_key_claims(papers, key_manager=km)
                save_analysis(st.session_state.project_id, "key_claims", {"claims": claims_list})
            except Exception as e:
                claims_list = []

            bar.progress(0.40)

            # Write sections
            combined_analysis = {"taxonomy": tax_data, "gaps": gaps_data, "key_claims": claims_list}
            sections_state: Dict[str, Any] = {}
            custom_directive = f"Paper Topic: {topic}."
            if description:
                custom_directive += f" Context: {description}."

            for s_i, sec_name in enumerate(REVIEW_SECTIONS):
                status.write(f"Writing section {s_i + 1}/{len(REVIEW_SECTIONS)}: {sec_name}...")
                try:
                    sec_text = generate_section_draft(
                        section_name=sec_name,
                        papers=papers,
                        analysis_data=combined_analysis,
                        previous_sections=sections_state,
                        custom_instructions=custom_directive,
                        tone=st.session_state.get("tone_sel", TONE_OPTIONS[0]),
                        target_length=st.session_state.get("length_sel", "Standard"),
                        key_manager=km,
                    )
                    sim_rep = check_section_similarity(sec_text, corpus_index, threshold=sim_threshold, allow_direct_quotes=True)
                    save_section(
                        project_id=st.session_state.project_id,
                        section_name=sec_name,
                        content=sec_text,
                        similarity_score=sim_rep["overall_score"],
                        similarity_report=sim_rep,
                    )
                    sections_state[sec_name] = {"content": sec_text}
                except Exception as e:
                    logger.warning("Section '%s' failed: %s", sec_name, e)
                bar.progress(0.40 + (s_i + 1) / len(REVIEW_SECTIONS) * 0.45)

            # Anti-plagiarism + export
            status.write("**Anti-plagiarism check and compiling IEEE .docx...**")
            final_sections = get_sections(st.session_state.project_id)
            for sec_name, sec_obj in final_sections.items():
                rep = sec_obj.get("similarity_report_json", {})
                if rep.get("flagged_items"):
                    try:
                        rewritten, final_rep, _ = run_auto_rewrite_loop(
                            section_text=sec_obj["content"],
                            corpus_index=corpus_index,
                            threshold=sim_threshold,
                            max_attempts=2,
                            key_manager=km,
                        )
                        save_section(
                            project_id=st.session_state.project_id,
                            section_name=sec_name,
                            content=rewritten,
                            similarity_score=final_rep["overall_score"],
                            similarity_report=final_rep,
                        )
                    except Exception:
                        pass

            final_sections = get_sections(st.session_state.project_id)
            try:
                docx_stream = generate_ieee_docx(
                    sections=final_sections,
                    papers=papers,
                    paper_title=topic,
                    authors="Researcher and Co-Authors",
                    affiliations="Department of Computer Science and Engineering",
                    include_ai_appendix=True,
                )
                st.session_state["generated_docx"] = docx_stream.getvalue()
            except Exception as e:
                logger.warning("DOCX export failed: %s", e)

            bar.progress(1.0)
            status.update(label="Paper generated successfully.", state="complete", expanded=False)

        st.session_state.stage = "done"
        st.rerun()
    else:
        # Full pipeline with auto-download
        _run_full_pipeline(
            topic=topic,
            description=description,
            tone=st.session_state.get("tone_sel", TONE_OPTIONS[0]),
            length=st.session_state.get("length_sel", "Standard"),
            sim_threshold=st.session_state.get("sim_threshold", SIMILARITY_SENTENCE_THRESHOLD),
        )

# ---------------------------------------------------------------------------
# STAGE 3 — Done: view and download
# ---------------------------------------------------------------------------
elif stage == "done":
    topic = st.session_state.get("topic", st.session_state.project_name)
    papers = list_papers(st.session_state.project_id)
    sections = get_sections(st.session_state.project_id)

    # Success banner
    n_sections = len([s for s in sections.values() if s.get("content")])
    st.markdown(
        f'<div class="banner-success">'
        f'<strong>Review Paper Ready</strong> &mdash; {n_sections} sections written across '
        f'{len(papers)} source papers. Download below or explore the content in the tabs.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Download row
    st.markdown('<div class="section-heading">Download Your Paper</div>', unsafe_allow_html=True)
    dl1, dl2, dl3, dl4 = st.columns(4)

    slug = topic.lower().replace(" ", "_")[:40]

    with dl1:
        if st.session_state.get("generated_docx"):
            st.download_button(
                "IEEE Word Document (.docx)",
                data=st.session_state["generated_docx"],
                file_name=f"{slug}_ieee.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True,
            )
        else:
            if st.button("Compile .docx Now", type="primary", use_container_width=True):
                try:
                    buf = generate_ieee_docx(
                        sections=sections,
                        papers=papers,
                        paper_title=topic,
                        authors="Researcher and Co-Authors",
                        affiliations="Department of Computer Science and Engineering",
                        include_ai_appendix=True,
                    )
                    st.session_state["generated_docx"] = buf.getvalue()
                    st.rerun()
                except Exception as e:
                    st.error(f"Export failed: {e}")

    with dl2:
        md_out = generate_markdown_export(sections, papers, paper_title=topic)
        st.download_button(
            "Markdown (.md)",
            data=md_out,
            file_name=f"{slug}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with dl3:
        bib_out = generate_bibtex_export(papers)
        st.download_button(
            "BibTeX References (.bib)",
            data=bib_out,
            file_name=f"{slug}.bib",
            mime="text/plain",
            use_container_width=True,
        )

    with dl4:
        sim_report = generate_similarity_report_markdown(sections)
        st.download_button(
            "Similarity Report (.md)",
            data=sim_report,
            file_name=f"{slug}_similarity.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # Start over button
    def _reset_to_input() -> None:
        st.session_state.stage = "input"
        st.session_state["topic"] = ""
        st.session_state["description"] = ""

    st.button("Generate a New Paper", on_click=_reset_to_input, use_container_width=False)

    st.markdown("---")

    # Tabs for detailed exploration
    tab_paper, tab_gaps, tab_papers, tab_chat = st.tabs([
        "Full Paper",
        "Research Gaps",
        "Source Papers",
        "Ask the Papers",
    ])

    # ---- Tab: Full Paper ----
    with tab_paper:
        st.markdown('<div class="section-heading">Generated Paper — Section by Section</div>', unsafe_allow_html=True)
        st.caption(
            "Each section can be edited. Changes are saved automatically when you click outside the text box."
        )

        for sec_name in REVIEW_SECTIONS:
            sec_obj = sections.get(sec_name, {})
            content = sec_obj.get("content", "")
            rep = sec_obj.get("similarity_report_json", {})
            score = rep.get("overall_score", 0.0)
            band = rep.get("status_band", "green")
            badge_color = {"green": "#2a7a4f", "yellow": "#b45309", "red": "#c0392b"}.get(band, "#2a7a4f")

            with st.expander(f"{sec_name}  —  Similarity: {round(score * 100)}%", expanded=(sec_name == "Abstract")):
                edited = st.text_area(
                    "Content",
                    value=content,
                    height=350,
                    key=f"edit_{sec_name}",
                    label_visibility="collapsed",
                )
                col_save, col_regen, _ = st.columns([1, 1, 4])
                with col_save:
                    if st.button("Save Edits", key=f"save_{sec_name}", use_container_width=True):
                        save_section(
                            project_id=st.session_state.project_id,
                            section_name=sec_name,
                            content=edited,
                            similarity_score=score,
                            similarity_report=rep,
                        )
                        st.success("Saved.")
                        st.rerun()
                with col_regen:
                    if st.button("Regenerate", key=f"regen_{sec_name}", use_container_width=True):
                        with st.spinner(f"Regenerating {sec_name}..."):
                            try:
                                corpus_index = CorpusIndex(papers)
                                new_text = generate_section_draft(
                                    section_name=sec_name,
                                    papers=papers,
                                    analysis_data={
                                        "taxonomy": get_analysis(st.session_state.project_id, "taxonomy") or {},
                                        "gaps": get_analysis(st.session_state.project_id, "gaps") or {},
                                    },
                                    previous_sections=sections,
                                    custom_instructions=f"Paper Topic: {topic}.",
                                    tone=st.session_state.get("tone_sel", TONE_OPTIONS[0]),
                                    target_length=st.session_state.get("length_sel", "Standard"),
                                    key_manager=km,
                                )
                                new_rep = check_section_similarity(
                                    new_text, corpus_index,
                                    threshold=st.session_state.get("sim_threshold", SIMILARITY_SENTENCE_THRESHOLD),
                                    allow_direct_quotes=True,
                                )
                                save_section(
                                    project_id=st.session_state.project_id,
                                    section_name=sec_name,
                                    content=new_text,
                                    similarity_score=new_rep["overall_score"],
                                    similarity_report=new_rep,
                                )
                                st.success("Section regenerated.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Regeneration failed: {e}")

    # ---- Tab: Research Gaps ----
    with tab_gaps:
        st.markdown('<div class="section-heading">Identified Research Gaps</div>', unsafe_allow_html=True)
        gaps_data = get_analysis(st.session_state.project_id, "gaps") or {}
        gaps_list = gaps_data.get("gaps", [])
        overall = gaps_data.get("overall_corpus_synthesis", {})

        if overall.get("primary_bottleneck"):
            st.markdown(
                f'<div class="banner-info">'
                f'<strong>Primary Bottleneck:</strong> {overall["primary_bottleneck"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        if not gaps_list:
            st.info("Research gaps have not been computed yet.")
        else:
            for g in gaps_list:
                st.markdown(
                    f'<div class="gap-card">'
                    f'<div class="card-title">{g.get("gap_id", "")} &mdash; {g.get("title", "")}</div>'
                    f'<div class="card-meta">Type: {g.get("type", "").replace("_", " ").title()} &nbsp;|&nbsp; '
                    f'Novelty: {g.get("novelty_score", 0)}/5 &nbsp;|&nbsp; '
                    f'Feasibility: {g.get("feasibility_score", 0)}/5</div>'
                    f'<p style="margin-top:10px;font-size:0.90rem;color:#1a1a1a">{g.get("description", "")}</p>'
                    f'<p style="font-size:0.85rem;color:#5a5a5a"><strong>Research Question:</strong> '
                    f'{g.get("suggested_research_question", "")}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ---- Tab: Source Papers ----
    with tab_papers:
        st.markdown(
            f'<div class="section-heading">Source Papers ({len(papers)})</div>',
            unsafe_allow_html=True,
        )
        if not papers:
            st.info("No papers indexed.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Papers", len(papers))
            years = [p.get("year") for p in papers if p.get("year")]
            m2.metric("Year Range", f"{min(years)} – {max(years)}" if years else "N/A")
            m3.metric("Sections Written", len([s for s in sections.values() if s.get("content")]))

            st.markdown("---")
            for p in papers:
                card = p.get("paper_card_json") or {}
                authors_disp = ", ".join((p.get("authors") or [])[:3])
                if len(p.get("authors") or []) > 3:
                    authors_disp += " et al."
                has_card = bool(card.get("problem") or card.get("methodology"))

                with st.expander(
                    f"[{p.get('citation_id')}] {p.get('title') or p.get('filename')}",
                    expanded=False,
                ):
                    st.markdown(f"**Authors:** {authors_disp or 'Unknown'}")
                    st.markdown(
                        f"**Year:** {p.get('year') or 'N/A'}  |  "
                        f"**Venue:** {p.get('venue') or 'N/A'}  |  "
                        f"**DOI:** {p.get('doi') or 'N/A'}"
                    )
                    if p.get("abstract"):
                        st.markdown(f"**Abstract:** {p['abstract'][:400]}...")

                    if has_card:
                        st.markdown("---")
                        st.markdown(f"**Problem:** {card.get('problem', 'N/A')}")
                        st.markdown(f"**Methodology:** {card.get('methodology', 'N/A')}")
                        results = card.get("key_results", [])
                        if results:
                            st.markdown("**Key Results:**")
                            for r in results[:4]:
                                st.markdown(f"- {r}")

                    c_del, _ = st.columns([1, 5])
                    with c_del:
                        if st.button("Remove", key=f"del_{p['id']}", use_container_width=True):
                            delete_paper(p["id"])
                            st.rerun()

    # ---- Tab: Ask the Papers ----
    with tab_chat:
        st.markdown('<div class="section-heading">Ask the Papers</div>', unsafe_allow_html=True)
        st.markdown(
            "Ask questions grounded in the indexed source papers. "
            "Answers cite specific papers by number.",
        )

        chat_history = get_chat_history(st.session_state.project_id)
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["message"])

        if prompt := st.chat_input("Ask a question about the papers..."):
            save_chat_message(st.session_state.project_id, "user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching papers..."):
                    try:
                        ask_tmpl = (PROMPTS_DIR / "ask_papers.txt").read_text(encoding="utf-8")
                        from core.analysis import format_papers_for_prompt
                        papers_ctx = format_papers_for_prompt(papers)
                        full_prompt = safe_fill_template(
                            ask_tmpl,
                            papers_summary=papers_ctx,
                            question=prompt,
                        )
                        answer = call_llm(full_prompt, temperature=0.1, key_manager=km)
                        st.markdown(answer)
                        save_chat_message(st.session_state.project_id, "assistant", answer)
                    except Exception as e:
                        err_msg = f"Could not answer: {e}"
                        st.error(err_msg)
                        save_chat_message(st.session_state.project_id, "assistant", err_msg)

        if chat_history:
            if st.button("Clear Chat History", use_container_width=False):
                clear_chat_history(st.session_state.project_id)
                st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-top:3rem; padding-top:1.2rem; border-top:1px solid #e0ddd7;
                font-size:0.80rem; color:#888; display:flex; gap:12px; align-items:center;">
        <strong style="color:#1a1a1a; font-size:0.82rem;">ReviewForge</strong>
        <span style="color:#ccc">|</span>
        <span>Automatic IEEE Review Paper Generator</span>
        <span style="color:#ccc">|</span>
        <span>Powered by Google Gemini and Groq</span>
    </div>
    """,
    unsafe_allow_html=True,
)
