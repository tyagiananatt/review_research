"""Key management, rotation, cooldown, and model fallback for Google Gemini API."""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class KeyManager:
    """Manages multiple Gemini API keys with round-robin rotation,

    automatic cooldown upon rate limits (429/quota/5xx/auth), and model fallback.
    """

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        cooldown_seconds: int = 60,
    ) -> None:
        """Initialize the KeyManager.

        Args:
            api_keys: List of Gemini API keys.
            models: Ordered model fallback chain (e.g. ['gemini-2.5-flash', 'gemini-2.5-flash-lite']).
            cooldown_seconds: Cooldown duration in seconds when a key encounters an error.
        """
        self._lock = threading.Lock()
        self._keys: List[str] = [k.strip() for k in (api_keys or []) if k.strip()]
        self._cooldown_seconds = cooldown_seconds
        # Cooldown expiration timestamps per key index
        self._cooldowns: dict[int, float] = {}
        # Key failure counts
        self._failure_counts: dict[int, int] = {}
        self._current_key_idx: int = 0

        # Model fallback chain
        self._models: List[str] = models or ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
        self._current_model_idx: int = 0

    def set_keys(self, keys: List[str]) -> None:
        """Dynamically update the available keys."""
        with self._lock:
            self._keys = [k.strip() for k in keys if k.strip()]
            self._cooldowns.clear()
            self._failure_counts.clear()
            self._current_key_idx = 0

    @property
    def total_keys(self) -> int:
        """Return the total number of configured keys."""
        with self._lock:
            return len(self._keys)

    @property
    def active_model(self) -> str:
        """Return the current active model name."""
        with self._lock:
            return self._models[self._current_model_idx]

    def get_status_display(self) -> Tuple[str, str]:
        """Return human-readable active key label and model name.

        NEVER reveals the API key itself.
        Returns:
            (key_display_string, model_name)
            e.g. ("Key #1 of 3 (Active)", "gemini-2.5-flash")
        """
        with self._lock:
            total = len(self._keys)
            if total == 0:
                return ("No Keys Configured", self._models[self._current_model_idx])

            idx = self._current_key_idx
            cooling_count = sum(
                1 for i in range(total) if self._cooldowns.get(i, 0) > time.time()
            )

            status = f"Key #{idx + 1} of {total}"
            if cooling_count > 0:
                status += f" ({cooling_count} on cooldown)"
            return (status, self._models[self._current_model_idx])

    def get_current_key(self) -> Optional[str]:
        """Get the current available API key, rotating past any keys currently on cooldown."""
        with self._lock:
            total = len(self._keys)
            if total == 0:
                return None

            now = time.time()
            # Try to find an available key starting from current index
            for offset in range(total):
                candidate_idx = (self._current_key_idx + offset) % total
                if self._cooldowns.get(candidate_idx, 0) <= now:
                    self._current_key_idx = candidate_idx
                    return self._keys[candidate_idx]

            # If ALL keys are on cooldown, pick the one with the earliest expiration and log
            earliest_idx = min(self._cooldowns.keys(), key=lambda i: self._cooldowns[i])
            self._current_key_idx = earliest_idx
            logger.warning(
                "All %d keys are currently cooling down. Earliest key is #%d (cooling for %.1fs).",
                total,
                earliest_idx + 1,
                max(0.0, self._cooldowns[earliest_idx] - now),
            )
            return self._keys[earliest_idx]

    def rotate_next(self) -> Optional[str]:
        """Manually or sequentially advance to the next round-robin key."""
        with self._lock:
            total = len(self._keys)
            if total == 0:
                return None
            self._current_key_idx = (self._current_key_idx + 1) % total
            return self._keys[self._current_key_idx]

    def mark_key_failed(self, error_message: str = "") -> Tuple[int, float]:
        """Mark current key as failed, put it on cooldown, and rotate to next key.

        Returns:
            (new_key_index, wait_seconds)
        """
        with self._lock:
            total = len(self._keys)
            now = time.time()
            failed_idx = self._current_key_idx

            self._failure_counts[failed_idx] = self._failure_counts.get(failed_idx, 0) + 1
            # Progressive cooldown based on repeated failures
            cooldown = self._cooldown_seconds * min(3, self._failure_counts[failed_idx])
            self._cooldowns[failed_idx] = now + cooldown

            logger.warning(
                "Key #%d failed (error: %s). Cooldown set to %ds. Rotating...",
                failed_idx + 1,
                error_message[:100],
                cooldown,
            )

            # Advance to next key
            if total > 1:
                self._current_key_idx = (self._current_key_idx + 1) % total

            return (self._current_key_idx, cooldown)

    def mark_key_success(self) -> None:
        """Reset consecutive failure count for the current key after a successful request."""
        with self._lock:
            self._failure_counts[self._current_key_idx] = 0

    def fallback_to_next_model(self) -> str:
        """Switch to the next model in the fallback chain."""
        with self._lock:
            if self._current_model_idx < len(self._models) - 1:
                self._current_model_idx += 1
                logger.warning("Switched fallback model to: %s", self._models[self._current_model_idx])
            return self._models[self._current_model_idx]

    def reset_model(self) -> str:
        """Reset back to the primary model."""
        with self._lock:
            self._current_model_idx = 0
            return self._models[0]

    def reset_cooldowns(self) -> None:
        """Clear all cooldowns and reset model index to primary."""
        with self._lock:
            self._cooldowns.clear()
            self._failure_counts.clear()
            self._current_model_idx = 0
            self._current_key_idx = 0
