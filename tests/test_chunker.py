"""Unit tests for section-aware document chunking."""

from __future__ import annotations

import pytest
from core.chunker import (
    chunk_paper_pages,
    estimate_tokens,
    is_heading,
)


def test_estimate_tokens():
    text = "Hello world from the academic test."
    tokens = estimate_tokens(text)
    assert tokens > 0
    assert tokens == len(text) // 4


def test_is_heading():
    assert is_heading("1. Introduction") is True
    assert is_heading("I. RELATED WORK") is True
    assert is_heading("## Methodology") is True
    assert is_heading("EXPERIMENTS AND RESULTS") is True
    assert is_heading("This is a normal paragraph sentence that happens to be long and is not a section heading.") is False
    assert is_heading("") is False


def test_chunk_paper_pages_short():
    pages = [
        {"page_number": 1, "clean_text": "Short intro.\n\nSome body text."},
        {"page_number": 2, "clean_text": "Conclusion paragraph."},
    ]
    chunks = chunk_paper_pages(pages, target_tokens=1000)
    assert len(chunks) == 1
    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 2
    assert "Short intro." in chunks[0].text
    assert "Conclusion paragraph." in chunks[0].text


def test_chunk_paper_pages_large_with_headings():
    # Construct a multi-paragraph text that exceeds target tokens
    para1 = "1. Introduction\n" + ("This is detailed text introducing the problem. " * 30)
    para2 = "2. Methodology\n" + ("Here we describe the algorithmic architecture and loss formulation. " * 40)
    para3 = "3. Experiments\n" + ("Extensive benchmarking across standard datasets demonstrates superior F1 scores. " * 50)
    para4 = "4. Conclusion\n" + ("In this survey we summarized key theoretical contributions. " * 25)

    pages = [
        {"page_number": 1, "clean_text": para1 + "\n\n" + para2},
        {"page_number": 2, "clean_text": para3 + "\n\n" + para4},
    ]

    # Use a small target token size to force multiple chunks
    chunks = chunk_paper_pages(pages, target_tokens=150, overlap_tokens=30)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.approx_tokens > 0
        assert 1 <= c.start_page <= 2
        assert 1 <= c.end_page <= 2
