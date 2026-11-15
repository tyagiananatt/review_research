"""Section-by-section review paper generation with anti-plagiarism auto-rewrite loop."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (
    LENGTH_OPTIONS,
    MAX_REWRITE_ATTEMPTS,
    PROMPTS_DIR,
    SIMILARITY_SENTENCE_THRESHOLD,
    TONE_OPTIONS,
    safe_fill_template,
)
from core.analysis import format_papers_for_prompt
from core.keys import KeyManager
from core.llm import call_gemini
from core.similarity import CorpusIndex, check_section_similarity

logger = logging.getLogger(__name__)


def build_citation_mapping(papers: List[Dict[str, Any]]) -> str:
    """Format available papers into an authoritative numbered citation list."""
    lines = []
    for p in papers:
        cit_id = p.get("citation_id", 1)
        authors = p.get("authors") or []
        first_author = authors[0] if authors else "Unknown"
        author_str = f"{first_author} et al." if len(authors) > 1 else first_author
        title = p.get("title", p.get("filename", "Untitled"))
        year = p.get("year", "n.d.")
        lines.append(f"[{cit_id}] {author_str}, \"{title}\", {year}.")
    return "\n".join(lines)


def format_ieee_references_section(papers: List[Dict[str, Any]]) -> str:
    """Deterministically generate the standard IEEE References section from uploaded papers."""
    lines = ["## References\n"]
    for p in papers:
        cit_id = p.get("citation_id", 1)
        authors = p.get("authors") or []
        if not authors:
            author_fmt = "Anon."
        elif len(authors) == 1:
            author_fmt = authors[0]
        elif len(authors) == 2:
            author_fmt = f"{authors[0]} and {authors[1]}"
        else:
            author_fmt = f"{authors[0]} et al."

        title = p.get("title", "Untitled")
        venue = p.get("venue") or "IEEE Transactions / Conference"
        year = p.get("year") or "2024"
        doi = p.get("doi")
        doi_part = f", doi: {doi}" if doi else "."

        ref_entry = f"[{cit_id}] {author_fmt}, \"{title},\" *{venue}*, {year}{doi_part}"
        lines.append(ref_entry)

    return "\n\n".join(lines)


def generate_section_draft(
    section_name: str,
    papers: List[Dict[str, Any]],
    analysis_data: Optional[Dict[str, Any]] = None,
    previous_sections: Optional[Dict[str, Dict[str, Any]]] = None,
    custom_instructions: str = "",
    tone: str = "Academic & Formal",
    target_length: str = "Standard",
    key_manager: Optional[KeyManager] = None,
) -> str:
    """Draft a single section of the review paper adhering to strict synthesis and citation rules."""
    # Special handling for References: format deterministically without LLM hallucination
    if section_name.strip().lower() == "references":
        return format_ieee_references_section(papers)

    prompt_tmpl = (PROMPTS_DIR / "section_writer.txt").read_text(encoding="utf-8")

    citation_map = build_citation_mapping(papers)
    paper_cards_summary = format_papers_for_prompt(papers)

    # Previous sections summary to maintain cohesive continuity
    prev_summary_parts = []
    if previous_sections:
        for name, sec_obj in previous_sections.items():
            if name != section_name and sec_obj.get("content"):
                prev_summary_parts.append(f"- Section '{name}': {sec_obj['content'][:200]}...")
    prev_summary = "\n".join(prev_summary_parts) if prev_summary_parts else "None drafted yet."

    # Compact analysis context to guarantee staying within TPM rate limits
    compact_analysis: Dict[str, Any] = {}
    if analysis_data:
        tax = analysis_data.get("taxonomy") or {}
        if isinstance(tax, dict) and tax.get("taxonomy_categories"):
            compact_analysis["taxonomy"] = [
                f"{c.get('category_name')}: {c.get('description', '')[:120]}"
                for c in tax.get("taxonomy_categories", [])[:3]
            ]
            if tax.get("emerging_trends"):
                compact_analysis["trends"] = tax.get("emerging_trends", [])[:3]

        gaps = analysis_data.get("gaps") or {}
        if isinstance(gaps, dict):
            bottleneck = gaps.get("overall_corpus_synthesis", {}).get("primary_bottleneck")
            if bottleneck:
                compact_analysis["primary_bottleneck"] = str(bottleneck)[:200]
            if gaps.get("gaps"):
                compact_analysis["key_gaps"] = [
                    f"[{g.get('gap_id')}]: {g.get('title')} - Question: {g.get('suggested_research_question')}"
                    for g in gaps.get("gaps", [])[:3]
                ]

    analysis_ctx_str = json.dumps(compact_analysis, indent=1) if compact_analysis else "None provided."
    length_desc = LENGTH_OPTIONS.get(target_length, LENGTH_OPTIONS["Standard"])

    prompt = safe_fill_template(
        prompt_tmpl,
        section_name=section_name,
        tone=tone,
        target_length=f"{target_length} ({length_desc})",
        custom_instructions=custom_instructions or "None provided.",
        citation_mapping=citation_map,
        paper_cards_json=paper_cards_summary,
        analysis_context=analysis_ctx_str,
        previous_sections_summary=prev_summary,
        max_citation_id=max([p.get("citation_id", 1) for p in papers], default=1),
    )

    from core.llm import call_llm
    draft = call_llm(
        prompt=prompt,
        temperature=0.25,
        use_cache=True,
        key_manager=key_manager,
    )

    return draft.strip()


def rewrite_sentence(
    flagged_sentence: str,
    context: str,
    source_match: str,
    key_manager: Optional[KeyManager] = None,
) -> str:
    """Call Gemini to rewrite a flagged sentence to eliminate similarity while preserving citations."""
    prompt_tmpl = (PROMPTS_DIR / "sentence_rewrite.txt").read_text(encoding="utf-8")
    prompt = safe_fill_template(
        prompt_tmpl,
        flagged_sentence=flagged_sentence,
        context=context[:400],
        source_match=source_match[:400],
    )

    rewritten = call_gemini(
        prompt=prompt,
        temperature=0.3,
        use_cache=False,  # Re-writes should be fresh
        key_manager=key_manager,
    )

    # Clean quotes or commentary if LLM added them
    clean = rewritten.strip().strip('"').strip("'")
    return clean


def run_auto_rewrite_loop(
    section_text: str,
    corpus_index: CorpusIndex,
    threshold: float = SIMILARITY_SENTENCE_THRESHOLD,
    max_attempts: int = MAX_REWRITE_ATTEMPTS,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    key_manager: Optional[KeyManager] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Run the anti-plagiarism auto-rewrite loop on a generated section until under threshold.

    Returns:
        (final_text, final_report, rewrite_history)
    """
    current_text = section_text
    rewrite_history: List[Dict[str, Any]] = []

    # Initial check
    report = check_section_similarity(current_text, corpus_index, threshold=threshold)

    attempt = 0
    while attempt < max_attempts and report["flagged_items"]:
        attempt += 1
        flagged = report["flagged_items"]
        if progress_callback:
            progress_callback(
                f"Auto-rewrite iteration {attempt}/{max_attempts}: restructuring {len(flagged)} flagged sentences...",
                attempt / (max_attempts + 1),
            )

        # Rewrite each flagged sentence
        for item in flagged:
            orig_sent = item["sentence"]
            source_match = item.get("matched_source_snippet", "")
            try:
                new_sent = rewrite_sentence(
                    flagged_sentence=orig_sent,
                    context=current_text,
                    source_match=source_match,
                    key_manager=key_manager,
                )
                if new_sent and new_sent != orig_sent:
                    current_text = current_text.replace(orig_sent, new_sent, 1)
                    rewrite_history.append({
                        "attempt": attempt,
                        "original": orig_sent,
                        "rewritten": new_sent,
                        "old_similarity": item["similarity_score"],
                        "source_match": source_match[:150],
                    })
            except Exception as e:
                logger.warning("Failed to rewrite sentence: %s", e)

        # Re-check similarity
        report = check_section_similarity(current_text, corpus_index, threshold=threshold)
        if report["status_band"] == "green" or not report["flagged_items"]:
            logger.info("Section reached green similarity threshold on attempt %d", attempt)
            break

    if progress_callback:
        progress_callback("Plagiarism check and auto-rewrite complete.", 1.0)

    return current_text, report, rewrite_history
