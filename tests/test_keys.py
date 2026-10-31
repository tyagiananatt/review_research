"""Unit tests for KeyManager rotation, cooldowns, mock error handling, and model fallback."""

from __future__ import annotations

import time
import pytest
from core.keys import KeyManager


def test_key_manager_initialization():
    keys = ["key-alpha", "key-beta", "key-gamma"]
    km = KeyManager(api_keys=keys, models=["gemini-2.5-flash", "gemini-2.5-flash-lite"], cooldown_seconds=2)
    assert km.total_keys == 3
    assert km.active_model == "gemini-2.5-flash"
    display, model = km.get_status_display()
    assert "Key #1 of 3" in display
    # Ensure key secret is never exposed in display
    assert "key-alpha" not in display
    assert "key-beta" not in display
    assert model == "gemini-2.5-flash"


def test_key_round_robin():
    keys = ["k1", "k2", "k3"]
    km = KeyManager(api_keys=keys)
    assert km.get_current_key() == "k1"
    assert km.rotate_next() == "k2"
    assert km.rotate_next() == "k3"
    assert km.rotate_next() == "k1"


def test_key_cooldown_on_failure():
    keys = ["k1", "k2"]
    km = KeyManager(api_keys=keys, cooldown_seconds=10)

    # Initially k1
    assert km.get_current_key() == "k1"

    # Simulate 429 quota error on k1
    next_idx, wait_sec = km.mark_key_failed("429 Resource Exhausted")
    assert next_idx == 1
    assert wait_sec == 10

    # k1 is on cooldown, get_current_key should return k2
    assert km.get_current_key() == "k2"

    display, _ = km.get_status_display()
    assert "1 on cooldown" in display


def test_all_keys_on_cooldown():
    keys = ["k1", "k2"]
    km = KeyManager(api_keys=keys, cooldown_seconds=2)

    # Fail both keys
    km.mark_key_failed("429 Quota Exceeded")
    km.mark_key_failed("503 Service Unavailable")

    # Should still return a key (the one that cools down earliest) rather than crashing
    key = km.get_current_key()
    assert key in ["k1", "k2"]


def test_model_fallback_chain():
    models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    km = KeyManager(api_keys=["k1"], models=models)

    assert km.active_model == "gemini-2.5-flash"

    # Trigger fallback
    next_model = km.fallback_to_next_model()
    assert next_model == "gemini-2.5-flash-lite"
    assert km.active_model == "gemini-2.5-flash-lite"

    # Attempting past end should stay on last model
    assert km.fallback_to_next_model() == "gemini-2.5-flash-lite"

    # Reset
    assert km.reset_model() == "gemini-2.5-flash"
    assert km.active_model == "gemini-2.5-flash"


def test_mock_llm_failover_loop():
    """Simulate a mock API caller using KeyManager to verify failover loop under simulated 429 errors."""
    keys = ["k1-bad", "k2-bad", "k3-good"]
    km = KeyManager(api_keys=keys, models=["gemini-2.5-flash", "gemini-2.5-flash-lite"], cooldown_seconds=5)

    def mock_gemini_call(key: str, model: str) -> str:
        if "bad" in key:
            raise RuntimeError("429 Resource Exhausted: Quota exceeded for project")
        return f"SUCCESS from {key} using {model}"

    # Execution simulation with failover
    success = False
    result = None
    retries = 0

    while retries < 4 and not success:
        curr_key = km.get_current_key()
        curr_model = km.active_model
        try:
            result = mock_gemini_call(curr_key, curr_model)
            km.mark_key_success()
            success = True
        except Exception as e:
            km.mark_key_failed(str(e))
            if retries >= 1:
                km.fallback_to_next_model()
            retries += 1

    assert success is True
    assert "SUCCESS from k3-good" in result
    assert km.get_current_key() == "k3-good"
