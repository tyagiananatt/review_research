"""Unit tests for local similarity and plagiarism checking (n-grams and RapidFuzz)."""

from __future__ import annotations

import pytest
from core.similarity import (
    CorpusIndex,
    check_section_similarity,
    check_self_overlap,
    extract_words,
    generate_ngrams,
    is_direct_quote,
    split_into_sentences,
)


def test_split_into_sentences():
    text = "Smith et al. conducted an experiment in 2023. The results showed 95% accuracy! Did it work? Yes."
    sents = split_into_sentences(text)
    assert len(sents) == 4
    # Check that "Smith et al." was not prematurely split
    assert "Smith et al. conducted an experiment in 2023." in sents[0]


def test_generate_ngrams():
    words = ["deep", "learning", "models", "excel", "in", "pattern", "recognition"]
    ngrams_5 = generate_ngrams(words, 5)
    assert len(ngrams_5) == 3
    assert ("deep", "learning", "models", "excel", "in") in ngrams_5


def test_is_direct_quote():
    assert is_direct_quote('"The model outperforms previous baselines by 4.2%" [1]') is True
    assert is_direct_quote('“Our findings indicate significant latency reduction.” [2]') is True
    assert is_direct_quote("The model outperforms previous baselines by 4.2% [1]") is False


def test_similarity_check_exact_and_fuzzy():
    # Source paper corpus
    corpus = [
        {
            "citation_id": 1,
            "title": "Attention Mechanisms in Deep Vision",
            "clean_text": (
                "Transformers have revolutionized modern artificial intelligence. "
                "Our empirical evaluations demonstrate a substantial reduction in computational latency "
                "when using sparse attention matrices across multimodal benchmarks."
            ),
        }
    ]

    index = CorpusIndex(corpus)

    # 1. Section with exact copy: 8-gram or higher
    copied_section = (
        "Our empirical evaluations demonstrate a substantial reduction in computational latency "
        "when using sparse attention matrices across multimodal benchmarks [1]."
    )
    rep_copied = check_section_similarity(copied_section, index, threshold=0.75)
    assert rep_copied["flagged_sentences_count"] >= 1
    assert rep_copied["overall_score"] > 0.5

    # 2. Section with completely synthesized original prose
    original_section = (
        "In contrast to dense projection paradigms, architectural pruning yields notable "
        "efficiency gains across various vision tasks [1]. However, memory bandwidth remains a bottleneck."
    )
    rep_orig = check_section_similarity(original_section, index, threshold=0.75)
    assert rep_orig["status_band"] == "green"
    assert rep_orig["overall_score"] < 0.20


def test_similarity_quote_exclusion():
    corpus = [
        {
            "citation_id": 1,
            "title": "Quantum Neural Computing",
            "clean_text": "Qubits provide super-polynomial speedups for combinatorial optimization tasks.",
        }
    ]
    index = CorpusIndex(corpus)

    quoted_section = (
        '"Qubits provide super-polynomial speedups for combinatorial optimization tasks." [1]\n\n'
        'This foundational conjecture inspires current variational hybrid algorithms.'
    )

    # When allow_direct_quotes is True, the quoted sentence is excluded from penalty
    rep_quoted = check_section_similarity(quoted_section, index, threshold=0.75, allow_direct_quotes=True)
    assert rep_quoted["flagged_sentences_count"] == 0
    assert rep_quoted["status_band"] == "green"


def test_check_self_overlap():
    all_sections = {
        "Introduction": {
            "content": "Deep reinforcement learning struggles with sample efficiency when exploration rewards are sparse and delayed."
        },
        "Literature Review": {
            "content": "Deep reinforcement learning struggles with sample efficiency when exploration rewards are sparse and delayed. In this paper we survey solutions."
        },
    }

    dups = check_self_overlap(
        current_section_name="Literature Review",
        current_section_text=all_sections["Literature Review"]["content"],
        all_sections=all_sections,
    )
    assert len(dups) >= 1
    assert dups[0]["other_section"] == "Introduction"
