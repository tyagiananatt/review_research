"""Local, lightweight similarity checker and plagiarism detector (no GPU, no ML models)."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None  # type: ignore

from config import NGRAM_SIZES, SIMILARITY_BANDS, SIMILARITY_SENTENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Abbreviations commonly found in academic papers to avoid splitting sentences prematurely
ABBREVIATIONS = {
    "al.", "e.g.", "i.e.", "etc.", "fig.", "figs.", "tab.", "ref.", "refs.",
    "eq.", "eqs.", "sec.", "secs.", "dr.", "prof.", "dept.", "vol.", "no.",
    "vs.", "approx.", "pp.", "p."
}


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting common academic abbreviations and citations."""
    text = text.strip()
    if not text:
        return []

    # Replace newlines within paragraphs with spaces
    text = re.sub(r"\n+", " ", text)

    # Heuristic sentence boundary split
    raw_splits = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[])", text)

    sentences: List[str] = []
    buffer = ""

    for chunk in raw_splits:
        chunk = chunk.strip()
        if not chunk:
            continue
        if buffer:
            chunk = buffer + " " + chunk
            buffer = ""

        # Check if the split happened on an abbreviation like "et al."
        last_word = chunk.split()[-1].lower() if chunk.split() else ""
        if last_word in ABBREVIATIONS and not chunk.endswith(".."):
            buffer = chunk
            continue

        sentences.append(chunk)

    if buffer:
        sentences.append(buffer)

    return sentences


def extract_words(text: str) -> List[str]:
    """Tokenize text into lowercase alphabetic/numeric words."""
    return re.findall(r"\b[A-Za-z0-9_-]+\b", text.lower())


def generate_ngrams(words: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Generate a set of tuple n-grams from a list of words."""
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def is_direct_quote(sentence: str) -> bool:
    """Check if the sentence is formatted as an explicit direct quote."""
    # E.g. "..." [1] or “...” [2]
    stripped = sentence.strip()
    if (stripped.startswith('"') and '"' in stripped[1:]) or (stripped.startswith('“') and '”' in stripped[1:]):
        return True
    return False


class CorpusIndex:
    """Fast in-memory index of all uploaded source papers for similarity checking."""

    def __init__(self, papers: List[Dict[str, Any]]) -> None:
        self.papers = papers
        # Map: citation_id -> list of sentences
        self.paper_sentences: Dict[int, List[str]] = {}
        # Precomputed n-grams for each paper
        self.paper_5grams: Dict[int, Set[Tuple[str, ...]]] = {}
        self.paper_8grams: Dict[int, Set[Tuple[str, ...]]] = {}
        self.all_5grams: Set[Tuple[str, ...]] = set()
        self.all_8grams: Set[Tuple[str, ...]] = set()

        self._build_index()

    def _build_index(self) -> None:
        for p in self.papers:
            cit_id = p.get("citation_id", 1)
            raw_text = p.get("clean_text") or p.get("full_text") or ""
            if not raw_text:
                continue

            sentences = split_into_sentences(raw_text)
            self.paper_sentences[cit_id] = sentences

            words = extract_words(raw_text)
            g5 = generate_ngrams(words, 5)
            g8 = generate_ngrams(words, 8)

            self.paper_5grams[cit_id] = g5
            self.paper_8grams[cit_id] = g8
            self.all_5grams.update(g5)
            self.all_8grams.update(g8)


def mask_direct_quotes(text: str) -> str:
    """Mask text inside double quotation marks to exclude allowed direct quotes from overlap counts."""
    return re.sub(r'("[^"]+"|\u201c[^\u201d]+\u201d)', ' ', text)


def check_section_similarity(
    section_text: str,
    corpus_index: CorpusIndex,
    threshold: float = SIMILARITY_SENTENCE_THRESHOLD,
    allow_direct_quotes: bool = True,
) -> Dict[str, Any]:
    """Check a generated review section against all source papers for verbatim and fuzzy similarity.

    Args:
        section_text: Markdown or plain text of the generated section.
        corpus_index: Pre-indexed corpus of uploaded papers.
        threshold: Sentence similarity cutoff (0.0 - 1.0, default 0.75).
        allow_direct_quotes: If True, quoted text is excluded from penalties.

    Returns:
        Structured similarity report dict.
    """
    sentences = split_into_sentences(section_text)
    if not sentences:
        return {
            "overall_score": 0.0,
            "status_band": "green",
            "total_sentences": 0,
            "flagged_sentences_count": 0,
            "flagged_items": [],
            "ngram_stats": {"5gram_overlap_count": 0, "8gram_overlap_count": 0},
        }

    # If quotes are permitted, mask them out before calculating word/ngram overlap ratios
    text_for_ngrams = mask_direct_quotes(section_text) if allow_direct_quotes else section_text
    sec_words = extract_words(text_for_ngrams)
    sec_5grams = generate_ngrams(sec_words, 5)
    sec_8grams = generate_ngrams(sec_words, 8)

    overlap_5grams = sec_5grams.intersection(corpus_index.all_5grams)
    overlap_8grams = sec_8grams.intersection(corpus_index.all_8grams)

    flagged_items: List[Dict[str, Any]] = []
    flagged_word_count = 0
    total_words = max(1, len(sec_words))

    for s_idx, sent in enumerate(sentences):
        sent_words = extract_words(sent)
        if len(sent_words) < 5:
            # Very short fragments (headings, etc.) are skipped
            continue

        # Check if direct quote
        quote_present = is_direct_quote(sent)
        if quote_present and allow_direct_quotes:
            continue

        sent_5grams = generate_ngrams(sent_words, 5)
        sent_8grams = generate_ngrams(sent_words, 8)

        has_8gram_match = bool(sent_8grams.intersection(corpus_index.all_8grams))
        has_5gram_match = bool(sent_5grams.intersection(corpus_index.all_5grams))

        max_sim = 0.0
        best_cit = 0
        best_title = ""
        best_match_snippet = ""

        # Rapidfuzz sentence comparison against corpus
        if fuzz is not None:
            # Fast candidate search: only compare with corpus sentences sharing words
            sent_word_set = set(sent_words)
            for cit_id, corpus_sents in corpus_index.paper_sentences.items():
                for c_sent in corpus_sents:
                    c_words = extract_words(c_sent)
                    if not c_words:
                        continue
                    # Overlap heuristic to avoid calculating fuzz on unrelated sentences
                    shared_words = len(sent_word_set.intersection(c_words))
                    if shared_words < 3:
                        continue

                    # partial_ratio detects if a sentence is copied inside a longer sentence
                    score = fuzz.partial_ratio(sent.lower(), c_sent.lower()) / 100.0
                    if score > max_sim:
                        max_sim = score
                        best_cit = cit_id
                        best_match_snippet = c_sent

        # Determine match reason
        match_reason = None
        if max_sim >= threshold:
            match_reason = f"Fuzzy sentence match ({round(max_sim * 100)}%)"
        elif has_8gram_match:
            match_reason = "Verbatim 8-word sequence match"
            max_sim = max(max_sim, 0.85)
        elif len(sent_5grams.intersection(corpus_index.all_5grams)) >= 2:
            match_reason = "Multiple 5-word sequence overlaps"
            max_sim = max(max_sim, 0.78)

        if match_reason:
            flagged_word_count += len(sent_words)
            # Find paper title
            for p in corpus_index.papers:
                if p.get("citation_id") == best_cit:
                    best_title = p.get("title") or p.get("filename", "")
                    break

            flagged_items.append({
                "sentence_index": s_idx,
                "sentence": sent,
                "similarity_score": round(max_sim, 3),
                "matched_paper_citation": best_cit,
                "matched_paper_title": best_title,
                "matched_source_snippet": best_match_snippet[:250],
                "match_reason": match_reason,
                "is_quote": quote_present,
            })

    # Overall similarity percentage based on proportion of words in flagged sentences and 8-gram overlaps
    word_overlap_ratio = min(1.0, flagged_word_count / total_words)
    ngram_overlap_ratio = min(1.0, (len(overlap_8grams) * 8 + len(overlap_5grams) * 5) / total_words)
    overall_score = round(max(word_overlap_ratio, ngram_overlap_ratio), 3)

    # Determine status band
    if overall_score <= SIMILARITY_BANDS["green"]:
        status_band = "green"
    elif overall_score <= SIMILARITY_BANDS["yellow"]:
        status_band = "yellow"
    else:
        status_band = "red"

    return {
        "overall_score": overall_score,
        "status_band": status_band,
        "total_sentences": len(sentences),
        "flagged_sentences_count": len(flagged_items),
        "flagged_items": flagged_items,
        "ngram_stats": {
            "5gram_overlap_count": len(overlap_5grams),
            "8gram_overlap_count": len(overlap_8grams),
        },
    }


def check_self_overlap(
    current_section_name: str,
    current_section_text: str,
    all_sections: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Check for phrase/sentence duplication across different sections of the review paper."""
    sentences = split_into_sentences(current_section_text)
    duplications: List[Dict[str, Any]] = []

    for s_idx, sent in enumerate(sentences):
        words = extract_words(sent)
        if len(words) < 8:
            continue

        sent_ngrams = generate_ngrams(words, 7)

        for other_name, other_sec in all_sections.items():
            if other_name == current_section_name:
                continue
            other_text = other_sec.get("content", "")
            other_words = extract_words(other_text)
            other_ngrams = generate_ngrams(other_words, 7)

            overlap = sent_ngrams.intersection(other_ngrams)
            if overlap:
                duplications.append({
                    "sentence_index": s_idx,
                    "sentence": sent,
                    "other_section": other_name,
                    "overlapping_phrase": " ".join(next(iter(overlap))),
                })
                break

    return duplications
