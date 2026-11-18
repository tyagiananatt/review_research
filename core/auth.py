"""Authentication and user session management for ReviewForge."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import streamlit as st


# User credentials repository
DEMO_USERS = {
    "demo": {
        "username": "demo",
        "password": "demo123",
        "name": "Dr. Alex Vance",
        "email": "researcher@reviewforge.ai",
        "role": "Academic Researcher",
        "avatar_initials": "AV",
        "institution": "Institute of Advanced Computing",
    }
}


def is_auth_configured() -> bool:
    """Return True when secrets.toml contains a real [auth] client_id."""
    try:
        cid = st.secrets.get("auth", {}).get("client_id", "")
        return bool(cid) and "REPLACE_WITH" not in cid
    except Exception:
        return False


def is_google_logged_in() -> bool:
    """Check if the user is authenticated via native Streamlit Google auth."""
    try:
        user_info = getattr(st, "user", None)
        if user_info is None:
            return False
        return bool(user_info.get("is_logged_in", False)) or bool(user_info.get("email", ""))
    except Exception:
        return False


def get_current_user() -> Optional[Dict[str, Any]]:
    """Return the currently authenticated user dictionary, or None."""
    # 1. Check session state credentials auth
    if "auth_user" in st.session_state and st.session_state.auth_user:
        return st.session_state.auth_user

    # 2. Check native Streamlit Google OAuth
    if is_google_logged_in():
        user_info = getattr(st, "user", {})
        email = user_info.get("email", "scholar@google.com")
        name = user_info.get("name", email.split("@")[0].capitalize())
        user_dict = {
            "username": email.split("@")[0],
            "name": name,
            "email": email,
            "role": "Google Verified Scholar",
            "avatar_initials": name[:2].upper() if name else "G",
            "provider": "google",
            "institution": "Google Scholar",
            "login_time": time.time(),
        }
        st.session_state.auth_user = user_dict
        return user_dict

    return None


def authenticate_credentials(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Validate username and password against registered users."""
    u = (username or "").strip().lower()
    p = (password or "").strip()

    if u in DEMO_USERS and DEMO_USERS[u]["password"] == p:
        user = DEMO_USERS[u].copy()
        user.pop("password", None)
        user["provider"] = "credentials"
        user["login_time"] = time.time()
        return user
    return None


def login_user(user_dict: Dict[str, Any]) -> None:
    """Save user in session state and record login."""
    st.session_state.auth_user = user_dict
    st.session_state.login_error = None


def logout_user() -> None:
    """Clear user session and native auth if present."""
    st.session_state.pop("auth_user", None)
    st.session_state.pop("login_error", None)

    if hasattr(st, "logout"):
        try:
            st.logout()
        except Exception:
            pass
