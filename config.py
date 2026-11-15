"""Configuration settings for ReviewForge."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "reviewforge.db"

# LLM & Model Chain
PRIMARY_MODEL = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
MODEL_CHAIN: List[str] = [
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# Groq Configuration (Ultra-fast fallback and primary option)
GROQ_PRIMARY_MODEL = os.getenv("GROQ_PRIMARY_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
GROQ_MODEL_CHAIN: List[str] = [
    GROQ_PRIMARY_MODEL,
    GROQ_FALLBACK_MODEL,
    "qwen/qwen3.8-27b",
]
DEFAULT_GROQ_RPM = int(os.getenv("GROQ_RPM", "30"))
AI_PROVIDER = os.getenv("AI_PROVIDER", "auto")

def is_valid_api_key(key: str) -> bool:
    """Validate that an API key is non-empty and not a placeholder."""
    k = key.strip()
    if not k or len(k) < 8:
        return False
    lower = k.lower()
    return not any(p in lower for p in ["your_", "placeholder", "key_here", "example", "none"])


def get_api_keys() -> List[str]:
    """Retrieve all valid Gemini API keys from environment or config."""
    raw_keys = os.getenv("GEMINI_KEYS", "").strip()
    if raw_keys:
        keys = [k.strip() for k in raw_keys.split(",") if is_valid_api_key(k)]
        if keys:
            return keys
    single_key = os.getenv("GEMINI_API_KEY", "").strip()
    if is_valid_api_key(single_key):
        return [single_key]
    return []


def get_groq_keys() -> List[str]:
    """Retrieve all valid Groq API keys from environment or config."""
    raw_keys = os.getenv("GROQ_KEYS", "").strip()
    if raw_keys:
        keys = [k.strip() for k in raw_keys.split(",") if is_valid_api_key(k)]
        if keys:
            return keys
    single_key = os.getenv("GROQ_API_KEY", "").strip()
    if is_valid_api_key(single_key):
        return [single_key]
    return []


def save_keys_to_env(
    gemini_keys: Optional[List[str]] = None,
    groq_keys: Optional[List[str]] = None,
    provider: Optional[str] = None,
) -> None:
    """Save updated keys and provider settings into .env persistently."""
    env_file = BASE_DIR / ".env"
    lines: List[str] = []
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()

    updated = {
        "GEMINI_KEYS": False,
        "GROQ_KEYS": False,
        "AI_PROVIDER": False,
    }

    new_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("GEMINI_KEYS=") and gemini_keys is not None:
            val = ",".join(gemini_keys)
            new_lines.append(f"GEMINI_KEYS={val}")
            updated["GEMINI_KEYS"] = True
            os.environ["GEMINI_KEYS"] = val
        elif stripped.startswith("GROQ_KEYS=") and groq_keys is not None:
            val = ",".join(groq_keys)
            new_lines.append(f"GROQ_KEYS={val}")
            updated["GROQ_KEYS"] = True
            os.environ["GROQ_KEYS"] = val
        elif stripped.startswith("AI_PROVIDER=") and provider is not None:
            new_lines.append(f"AI_PROVIDER={provider}")
            updated["AI_PROVIDER"] = True
            os.environ["AI_PROVIDER"] = provider
        else:
            new_lines.append(line)

    if gemini_keys is not None and not updated["GEMINI_KEYS"]:
        val = ",".join(gemini_keys)
        new_lines.append(f"GEMINI_KEYS={val}")
        os.environ["GEMINI_KEYS"] = val
    if groq_keys is not None and not updated["GROQ_KEYS"]:
        val = ",".join(groq_keys)
        new_lines.append(f"GROQ_KEYS={val}")
        os.environ["GROQ_KEYS"] = val
    if provider is not None and not updated["AI_PROVIDER"]:
        new_lines.append(f"AI_PROVIDER={provider}")
        os.environ["AI_PROVIDER"] = provider

    env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

# Rate Limiting & Resilience
DEFAULT_RPM = int(os.getenv("GEMINI_RPM", "15"))
KEY_COOLDOWN_SECONDS = int(os.getenv("KEY_COOLDOWN_SECONDS", "60"))
MAX_RETRIES_PER_CALL = 4
INITIAL_BACKOFF_SECONDS = 1.5

# Chunking Parameters (Section-aware map-reduce)
CHUNK_TARGET_TOKENS = 3500
CHUNK_OVERLAP_TOKENS = 350
CHARS_PER_TOKEN = 4  # Standard English heuristic: ~4 characters per token

# Similarity & Plagiarism Parameters
SIMILARITY_SENTENCE_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
MAX_REWRITE_ATTEMPTS = 3
NGRAM_SIZES = (5, 8)
SIMILARITY_BANDS = {
    "green": 0.15,   # <= 15% overall overlap is considered safe
    "yellow": 0.25,  # 15% - 25% is moderate
    "red": 1.00      # > 25% requires review
}

# Standard Review Paper Sections
REVIEW_SECTIONS = [
    "Abstract",
    "Index Terms",
    "Introduction",
    "Background & Taxonomy",
    "Thematic Literature Review",
    "Comparative Analysis",
    "Research Gaps & Challenges",
    "Future Directions",
    "Conclusion",
    "References",
]

TONE_OPTIONS = [
    "Academic & Formal",
    "Critical & Analytical",
    "Executive & Accessible",
]

LENGTH_OPTIONS = {
    "Concise": "~300 - 500 words per section",
    "Standard": "~600 - 900 words per section",
    "Comprehensive": "~1,000 - 1,500 words per section",
}


def safe_fill_template(template: str, **kwargs: Any) -> str:
    """Safely replace {key} placeholders in templates without failing on JSON curly braces."""
    text = template
    for k, v in kwargs.items():
        text = text.replace(f"{{{k}}}", str(v))
    return text
