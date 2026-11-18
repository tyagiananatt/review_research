"""Unit tests for ReviewForge authentication module."""

from core.auth import authenticate_credentials, DEMO_USERS


def test_demo_credentials_valid():
    user = authenticate_credentials("demo", "demo123")
    assert user is not None
    assert user["username"] == "demo"
    assert user["name"] == "Dr. Alex Vance"
    assert user["email"] == "researcher@reviewforge.ai"
    assert user["role"] == "Academic Researcher"
    assert user["provider"] == "credentials"
    assert "password" not in user


def test_demo_credentials_case_insensitive_username():
    user = authenticate_credentials("DEMO", "demo123")
    assert user is not None
    assert user["username"] == "demo"


def test_invalid_password():
    user = authenticate_credentials("demo", "wrongpass")
    assert user is None


def test_invalid_username():
    user = authenticate_credentials("nonexistent_user", "demo123")
    assert user is None


def test_empty_credentials():
    assert authenticate_credentials("", "") is None
    assert authenticate_credentials("demo", "") is None
    assert authenticate_credentials("", "demo123") is None
