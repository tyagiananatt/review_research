"""Section-aware document chunking for map-reduce summarization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import CHARS_PER_TOKEN, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS

# Regex to detect standard academic section headings
HEADING_REGEX = re.compile(
    r"(?m)^(?:"
    r"(?:[IVXLCDM]+\.?\s+[A-Z][A-Za-z\s]{2,40})|"  # Roman numerals: I. INTRODUCTION
    r"(?:\d+\.?\d*\.?\s+[A-Z][A-Za-z\s]{2,40})|"  # Arabic numerals: 1. Introduction, 2.1 Model
    r"(?:#{1,3}\s+[A-Za-z\s]{2,40})|"              # Markdown headings: ## Methods
    r"(?:[A-Z\s]{4,35})$"                          # ALL CAPS: METHODOLOGY
    r")"
)


@dataclass
class PaperChunk:
    """Represents a section-aware text chunk with provenance information."""

    index: int
    text: str
    start_page: int
    end_page: int
    approx_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "approx_tokens": self.approx_tokens,
        }


def estimate_tokens(text: str) -> int:
    """Estimate token count based on character length (~4 chars/token)."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def is_heading(line: str) -> bool:
    """Check if a line matches academic heading heuristics."""
    line = line.strip()
    if not line or len(line) > 60:
        return False
    return bool(HEADING_REGEX.match(line))


def chunk_paper_pages(
    pages: List[Dict[str, Any]],
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[PaperChunk]:
    """Split page-segmented text into section-aware chunks respecting heading boundaries.

    Args:
        pages: List of dicts with 'page_number' and 'clean_text'.
        target_tokens: Desired token budget per chunk (default ~3500).
        overlap_tokens: Overlap token budget between chunks (default ~350).

    Returns:
        List of PaperChunk instances.
    """
    if not pages:
        return []

    target_chars = target_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    # Flatten pages into paragraph blocks with page tags
    blocks: List[Dict[str, Any]] = []
    for p in pages:
        page_num = p.get("page_number", 1)
        raw_text = p.get("clean_text") or p.get("raw_text") or ""
        # Split on double newlines to isolate paragraphs/headings
        paragraphs = [para.strip() for para in re.split(r"\n\s*\n", raw_text) if para.strip()]
        for para in paragraphs:
            blocks.append({
                "page": page_num,
                "text": para,
                "is_heading": is_heading(para),
                "chars": len(para),
            })

    if not blocks:
        return []

    # If the total text is smaller than target_chars + 10%, return as a single chunk
    total_chars = sum(b["chars"] for b in blocks)
    if total_chars <= target_chars * 1.1:
        full_text = "\n\n".join(b["text"] for b in blocks)
        return [
            PaperChunk(
                index=1,
                text=full_text,
                start_page=blocks[0]["page"],
                end_page=blocks[-1]["page"],
                approx_tokens=estimate_tokens(full_text),
            )
        ]

    chunks: List[PaperChunk] = []
    current_blocks: List[Dict[str, Any]] = []
    current_chars = 0
    chunk_idx = 1
    i = 0

    while i < len(blocks):
        b = blocks[i]
        current_blocks.append(b)
        current_chars += b["chars"]

        # Check if we reached the target threshold
        if current_chars >= target_chars:
            # Look ahead for a nearby heading boundary within the next 2 blocks
            lookahead = 0
            while lookahead < 2 and (i + lookahead + 1) < len(blocks):
                next_b = blocks[i + lookahead + 1]
                if next_b["is_heading"]:
                    # Clean break before the upcoming heading
                    break
                current_blocks.append(next_b)
                current_chars += next_b["chars"]
                lookahead += 1
            i += lookahead

            # Assemble current chunk
            chunk_text = "\n\n".join(cb["text"] for cb in current_blocks)
            chunk = PaperChunk(
                index=chunk_idx,
                text=chunk_text,
                start_page=current_blocks[0]["page"],
                end_page=current_blocks[-1]["page"],
                approx_tokens=estimate_tokens(chunk_text),
            )
            chunks.append(chunk)
            chunk_idx += 1

            # Compute overlap blocks for next chunk
            overlap_acc = 0
            overlap_blocks: List[Dict[str, Any]] = []
            for cb in reversed(current_blocks):
                overlap_acc += cb["chars"]
                overlap_blocks.insert(0, cb)
                if overlap_acc >= overlap_chars:
                    break

            current_blocks = overlap_blocks
            current_chars = sum(cb["chars"] for cb in current_blocks)

        i += 1

    # Add remaining trailing blocks
    if current_blocks:
        chunk_text = "\n\n".join(cb["text"] for cb in current_blocks)
        # Avoid duplicate chunk if it's identical to the previous chunk
        if not chunks or chunk_text != chunks[-1].text:
            chunks.append(
                PaperChunk(
                    index=chunk_idx,
                    text=chunk_text,
                    start_page=current_blocks[0]["page"],
                    end_page=current_blocks[-1]["page"],
                    approx_tokens=estimate_tokens(chunk_text),
                )
            )

    return chunks
