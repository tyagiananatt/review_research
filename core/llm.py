"""AI Provider client supporting Google Gemini and Groq with key rotation, caching, rate limiting, and failover."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore

from config import (
    AI_PROVIDER,
    DEFAULT_GROQ_RPM,
    DEFAULT_RPM,
    GROQ_MODEL_CHAIN,
    GROQ_PRIMARY_MODEL,
    INITIAL_BACKOFF_SECONDS,
    MAX_RETRIES_PER_CALL,
    MODEL_CHAIN,
    PRIMARY_MODEL,
    get_api_keys,
    get_groq_keys,
)
from core.db import get_cached_llm, set_cached_llm
from core.keys import KeyManager

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe rate limiter ensuring we do not exceed requests per minute (RPM)."""

    def __init__(self, rpm: int = DEFAULT_RPM) -> None:
        self.rpm = max(1, rpm)
        self._lock = threading.Lock()
        self._timestamps: List[float] = []

    def set_rpm(self, rpm: int) -> None:
        with self._lock:
            self.rpm = max(1, rpm)

    def wait(self) -> None:
        """Wait until a request can be safely dispatched under the RPM limit."""
        with self._lock:
            now = time.time()
            # Clean timestamps older than 60 seconds
            self._timestamps = [t for t in self._timestamps if now - t < 60.0]

            if len(self._timestamps) >= self.rpm:
                oldest = self._timestamps[0]
                sleep_time = max(0.1, 60.0 - (now - oldest) + 0.05)
                logger.info("Rate limiter holding for %.2f seconds (RPM limit: %d)...", sleep_time, self.rpm)
                time.sleep(sleep_time)
                now = time.time()
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]

            self._timestamps.append(time.time())


# Singleton instances
_GLOBAL_RATE_LIMITER = RateLimiter(rpm=DEFAULT_RPM)
_GLOBAL_GROQ_RATE_LIMITER = RateLimiter(rpm=DEFAULT_GROQ_RPM)
_GLOBAL_KEY_MANAGER: Optional[KeyManager] = None
_GLOBAL_GROQ_KEY_MANAGER: Optional[KeyManager] = None


def get_key_manager() -> KeyManager:
    """Get or initialize the global Gemini KeyManager singleton."""
    global _GLOBAL_KEY_MANAGER
    if _GLOBAL_KEY_MANAGER is None:
        keys = get_api_keys()
        _GLOBAL_KEY_MANAGER = KeyManager(api_keys=keys, models=MODEL_CHAIN)
    return _GLOBAL_KEY_MANAGER


def get_groq_key_manager() -> KeyManager:
    """Get or initialize the global Groq KeyManager singleton."""
    global _GLOBAL_GROQ_KEY_MANAGER
    if _GLOBAL_GROQ_KEY_MANAGER is None:
        keys = get_groq_keys()
        _GLOBAL_GROQ_KEY_MANAGER = KeyManager(api_keys=keys, models=GROQ_MODEL_CHAIN)
    return _GLOBAL_GROQ_KEY_MANAGER


def is_groq_key_manager(km: Optional[KeyManager]) -> bool:
    """Check if key manager is configured with Groq models."""
    if km is None or not hasattr(km, "_models") or not km._models:
        return False
    return any(
        any(tag in m.lower() for tag in ["llama", "mixtral", "gemma", "whisper", "deepseek"])
        for m in km._models
    )


def is_gemini_key_manager(km: Optional[KeyManager]) -> bool:
    """Check if key manager is configured with Gemini models."""
    if km is None or not hasattr(km, "_models") or not km._models:
        return False
    return any("gemini" in m.lower() for m in km._models)


def update_llm_keys(
    gemini_keys: Optional[List[str]] = None,
    groq_keys: Optional[List[str]] = None,
    provider: Optional[str] = None,
) -> None:
    """Update in-memory KeyManagers and save them persistently to .env."""
    from config import save_keys_to_env
    save_keys_to_env(gemini_keys=gemini_keys, groq_keys=groq_keys, provider=provider)
    if gemini_keys is not None:
        get_key_manager().set_keys(gemini_keys)
    if groq_keys is not None:
        get_groq_key_manager().set_keys(groq_keys)


def clean_json_text(text: str) -> str:
    """Strip markdown code fences and extraneous text around JSON."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def parse_llm_json(text: str) -> Any:
    """Safely parse JSON from LLM output, handling markdown fences, trailing commas, and quirks."""
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Strip trailing commas
        cleaned_no_trailing = re.sub(r",\s*([\]}])", r"\1", cleaned)
        try:
            return json.loads(cleaned_no_trailing)
        except Exception:
            pass

        start_obj = cleaned.find("{")
        end_obj = cleaned.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            candidate = cleaned[start_obj : end_obj + 1]
            try:
                return json.loads(candidate)
            except Exception:
                try:
                    candidate_fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
                    return json.loads(candidate_fixed)
                except Exception:
                    pass

        start_arr = cleaned.find("[")
        end_arr = cleaned.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            candidate = cleaned[start_arr : end_arr + 1]
            try:
                return json.loads(candidate)
            except Exception:
                try:
                    candidate_fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
                    return json.loads(candidate_fixed)
                except Exception:
                    pass
        raise


def call_groq(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    as_json: bool = False,
    use_cache: bool = True,
    key_manager: Optional[KeyManager] = None,
    rate_limiter: Optional[RateLimiter] = None,
    custom_model: Optional[str] = None,
) -> str:
    """Execute a prompt against Groq API with ultra-fast inference and key rotation."""
    if Groq is None:
        raise RuntimeError("groq package is not installed. Run: pip install groq")

    km = key_manager if (key_manager and is_groq_key_manager(key_manager)) else get_groq_key_manager()
    rl = rate_limiter or _GLOBAL_GROQ_RATE_LIMITER

    cache_prompt = prompt
    if system_instruction:
        cache_prompt = f"SYSTEM: {system_instruction}\n\nUSER:\n{prompt}"

    model_name = custom_model or km.active_model

    if use_cache:
        cached = get_cached_llm(prompt=cache_prompt, model=f"groq:{model_name}")
        if cached is not None:
            logger.info("Retrieved response from cache for Groq model %s", model_name)
            return cached

    retries = 0
    backoff = 1.0
    last_error: Optional[Exception] = None

    # Guard against exceeding Groq TPM limits (8,000 TPM on free tier ~ 12,000 chars)
    trimmed_prompt = prompt
    if len(trimmed_prompt) > 12000:
        logger.warning("Prompt length (%d chars) exceeds safe Groq TPM threshold; trimming middle...", len(trimmed_prompt))
        trimmed_prompt = prompt[:5500] + "\n\n... [middle context truncated to observe TPM limits] ...\n\n" + prompt[-5500:]

    while retries < MAX_RETRIES_PER_CALL:
        key = km.get_current_key()
        if not key:
            km.reset_cooldowns()
            key = km.get_current_key()
            if not key:
                raise ValueError(
                    "No Groq API key configured. Please add a free Groq API key in the sidebar under 'API Key Configuration' (get one at https://console.groq.com/keys) or in .env."
                )

        model_to_use = custom_model or km.active_model
        try:
            rl.wait()
            client = Groq(api_key=key)

            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": trimmed_prompt})

            kwargs: Dict[str, Any] = {
                "model": model_to_use,
                "messages": messages,
                "temperature": temperature,
            }
            if as_json:
                kwargs["response_format"] = {"type": "json_object"}

            completion = client.chat.completions.create(**kwargs)
            res_text = completion.choices[0].message.content or ""

            km.mark_key_success()
            if use_cache and res_text:
                set_cached_llm(prompt=cache_prompt, model=f"groq:{model_to_use}", response_text=res_text)

            return res_text

        except Exception as e:
            last_error = e
            err_str = str(e)
            logger.warning("Groq call failed on key: %s", err_str)
            is_tpm_error = any(term in err_str.lower() for term in ["413", "rate_limit_exceeded", "tokens per minute", "tpm"])
            if is_tpm_error:
                logger.warning("Groq TPM limit triggered; trimming prompt further without punitive key cooldown...")
                if len(trimmed_prompt) > 6000:
                    trimmed_prompt = trimmed_prompt[:2800] + "\n\n... [context truncated to fit token limits] ...\n\n" + trimmed_prompt[-2800:]
                # Rotate key without long cooldown
                km.rotate_next()
            else:
                km.mark_key_failed(err_str)

            if retries >= 1 and not custom_model:
                km.fallback_to_next_model()
            time.sleep(backoff)
            backoff *= 1.5
            retries += 1

    raise RuntimeError(f"Groq generation failed after retries: {last_error}")


def call_gemini(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    as_json: bool = False,
    use_cache: bool = True,
    key_manager: Optional[KeyManager] = None,
    rate_limiter: Optional[RateLimiter] = None,
    custom_model: Optional[str] = None,
) -> str:
    """Execute a prompt against Gemini with failover to Groq if quota is exhausted."""
    km = key_manager if (key_manager and is_gemini_key_manager(key_manager)) else get_key_manager()
    rl = rate_limiter or _GLOBAL_RATE_LIMITER
    groq_km = get_groq_key_manager()

    import os
    from config import AI_PROVIDER
    active_prov = os.getenv("AI_PROVIDER", AI_PROVIDER)

    # If provider is explicitly set to groq, route to Groq
    if active_prov == "groq" and groq_km.total_keys > 0:
        return call_groq(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            as_json=as_json,
            use_cache=use_cache,
            custom_model=custom_model,
        )

    # If no Gemini keys configured but Groq keys exist, route directly to Groq
    if km.total_keys == 0 and groq_km.total_keys > 0:
        logger.info("No Gemini keys found; routing request directly to Groq...")
        return call_groq(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            as_json=as_json,
            use_cache=use_cache,
            custom_model=custom_model,
        )

    if genai is None:
        if groq_km.total_keys > 0:
            return call_groq(prompt=prompt, system_instruction=system_instruction, temperature=temperature, as_json=as_json)
        raise RuntimeError("google-genai SDK is not installed or available.")

    cache_prompt = prompt
    if system_instruction:
        cache_prompt = f"SYSTEM: {system_instruction}\n\nUSER:\n{prompt}"

    model_name = custom_model or km.active_model

    if use_cache:
        cached = get_cached_llm(prompt=cache_prompt, model=model_name)
        if cached is not None:
            logger.info("Retrieved LLM response from SQLite cache for model %s", model_name)
            return cached

    retries = 0
    backoff = INITIAL_BACKOFF_SECONDS
    last_error: Optional[Exception] = None

    while retries < MAX_RETRIES_PER_CALL:
        key = km.get_current_key()
        if not key:
            if groq_km.total_keys > 0:
                logger.info("No Gemini keys available; falling over to Groq...")
                return call_groq(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    as_json=as_json,
                    use_cache=use_cache,
                )
            raise ValueError(
                "No Gemini or Groq API key available. Please configure GEMINI_KEYS or GROQ_KEYS in .env or the sidebar."
            )

        model_to_use = custom_model or km.active_model

        try:
            rl.wait()
            client = genai.Client(api_key=key)

            config_args: Dict[str, Any] = {"temperature": temperature}
            if system_instruction:
                config_args["system_instruction"] = system_instruction
            if as_json:
                config_args["response_mime_type"] = "application/json"

            config = types.GenerateContentConfig(**config_args)

            response = client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=config,
            )

            response_text = response.text or ""
            km.mark_key_success()

            if use_cache and response_text:
                set_cached_llm(prompt=cache_prompt, model=model_to_use, response_text=response_text)

            return response_text

        except Exception as e:
            last_error = e
            err_str = str(e)
            err_lower = err_str.lower()
            logger.warning("Gemini API call failed on model %s: %s", model_to_use, err_str)

            # Check if this is a quota exhaustion (429 / RESOURCE_EXHAUSTED / per day limit)
            is_quota_exhausted = any(
                term in err_lower for term in [
                    "429", "resource_exhausted", "quota exceeded", "per day limit exceeded",
                    "free_tier_requests", "quota failure", "generate content requests per day limit"
                ]
            )
            # Check if model is deprecated or unavailable (e.g. 404)
            is_model_unavailable = any(
                term in err_lower for term in ["404", "no longer available", "not found for api version", "model_not_found"]
            )

            if (is_quota_exhausted or is_model_unavailable) and groq_km.total_keys > 0:
                logger.warning(
                    "Gemini issue (%s). Seamlessly falling over to Groq (%s)...",
                    err_str[:60],
                    groq_km.active_model,
                )
                try:
                    return call_groq(
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temperature,
                        as_json=as_json,
                        use_cache=use_cache,
                    )
                except Exception as groq_err:
                    logger.error("Groq fallback also failed: %s", groq_err)

            if is_quota_exhausted and groq_km.total_keys == 0:
                if any(term in err_lower for term in ["per day limit exceeded", "retry in", "resource_exhausted"]):
                    raise RuntimeError(
                        "⚠️ Gemini Free-Tier Daily Quota Limit (20 requests/day) Exceeded! "
                        "Google has paused requests for this key for today.\n\n"
                        "👉 Quick Fix (Instant): Add a free Groq API key in the sidebar under 'API Key Configuration' (get one in 30 seconds at https://console.groq.com/keys) or add more Gemini keys in .env."
                    )

            # Check if this is a model availability or temporary high demand issue
            is_model_demand_or_404 = any(
                term in err_lower
                for term in [
                    "503", "unavailable", "high demand", "overloaded", "spikes in demand",
                    "404", "not_found", "no longer available", "not found"
                ]
            )
            if is_model_demand_or_404 and not custom_model:
                next_m = km.fallback_to_next_model()
                logger.warning("Model issue (%s). Switched to model: %s", err_str[:70], next_m)

            is_quota_or_rate = any(
                term in err_lower
                for term in ["429", "quota", "resource_exhausted", "rate limit"]
            )
            is_auth_error = any(
                term in err_lower for term in ["401", "403", "api_key_invalid", "unauthenticated"]
            )

            if is_quota_or_rate or is_auth_error:
                km.mark_key_failed(error_message=err_str)
                if retries >= 1 and not custom_model:
                    km.fallback_to_next_model()

            time.sleep(backoff)
            backoff *= 2.0
            retries += 1

    # Final attempt: try Groq if configured before throwing error
    if groq_km.total_keys > 0:
        logger.info("All Gemini attempts failed. Executing final attempt via Groq...")
        return call_groq(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            as_json=as_json,
            use_cache=use_cache,
        )

    raise RuntimeError(
        f"Failed to generate content after {MAX_RETRIES_PER_CALL} retries. Last error: {last_error}"
    )


def call_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    as_json: bool = False,
    use_cache: bool = True,
    provider: Optional[str] = None,
    key_manager: Optional[KeyManager] = None,
    custom_model: Optional[str] = None,
) -> str:
    """Unified entry point for AI generation with provider selection."""
    import os
    from config import AI_PROVIDER
    active_prov = provider or os.getenv("AI_PROVIDER", AI_PROVIDER)
    if active_prov == "groq":
        return call_groq(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            as_json=as_json,
            use_cache=use_cache,
            key_manager=key_manager,
            custom_model=custom_model,
        )
    return call_gemini(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        as_json=as_json,
        use_cache=use_cache,
        key_manager=key_manager,
        custom_model=custom_model,
    )


def call_gemini_json(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.1,
    use_cache: bool = True,
    key_manager: Optional[KeyManager] = None,
) -> Any:
    """Execute LLM prompt requesting JSON and parse the response into Python dict/list."""
    raw_text = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        as_json=True,
        use_cache=use_cache,
        key_manager=key_manager,
    )
    return parse_llm_json(raw_text)
