"""Research gap identification across 5 dimensions and gap-to-contribution generation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from config import PROMPTS_DIR, safe_fill_template
from core.analysis import format_papers_for_prompt
from core.keys import KeyManager
from core.llm import call_gemini_json

logger = logging.getLogger(__name__)


def find_research_gaps(
    papers: List[Dict[str, Any]], key_manager: Optional[KeyManager] = None
) -> Dict[str, Any]:
    """Find research gaps across contradictions, untested domains, missing comparisons,

    common limitations, and unaddressed future work.
    """
    if not papers:
        return {"gaps": [], "overall_corpus_synthesis": {}}

    prompt_tmpl = (PROMPTS_DIR / "gaps.txt").read_text(encoding="utf-8")
    papers_summary = format_papers_for_prompt(papers)
    prompt = safe_fill_template(prompt_tmpl, papers_summary=papers_summary)

    try:
        result = call_gemini_json(prompt=prompt, temperature=0.2, key_manager=key_manager)
        if isinstance(result, dict):
            gaps = result.get("gaps", [])

            # Compute composite rank score and sort descending
            for g in gaps:
                nov = float(g.get("novelty_score", 3.0))
                feas = float(g.get("feasibility_score", 3.0))
                g["composite_rank"] = round(nov * 0.5 + feas * 0.5, 2)

            gaps.sort(key=lambda x: x.get("composite_rank", 0.0), reverse=True)
            result["gaps"] = gaps
            return result
    except Exception as e:
        logger.error("Failed to discover research gaps: %s", e)
        raise e


def generate_research_objectives(
    gap: Dict[str, Any], key_manager: Optional[KeyManager] = None
) -> Dict[str, Any]:
    """Generate detailed research questions, hypotheses, and objectives from an identified gap."""
    prompt_tmpl = (PROMPTS_DIR / "gap_rq.txt").read_text(encoding="utf-8")

    evidence_items = gap.get("evidence", [])
    evidence_str = "\n".join(
        f"- [Paper {e.get('citation_id', '?')}] {e.get('paper_title', '')}: {e.get('snippet_or_note', '')} (p. {e.get('page_number', '?')})"
        for e in evidence_items
    )

    prompt = safe_fill_template(
        prompt_tmpl,
        gap_title=gap.get("title", ""),
        gap_type=gap.get("type", ""),
        gap_description=gap.get("description", ""),
        evidence_text=evidence_str or "No explicit snippet recorded.",
    )

    try:
        result = call_gemini_json(prompt=prompt, temperature=0.2, key_manager=key_manager)
        if isinstance(result, dict):
            return result
    except Exception as e:
        logger.error("Failed to generate research objectives for gap: %s", e)

    return {
        "primary_research_question": f"How can we resolve the gap in {gap.get('title', '')}?",
        "sub_questions": [],
        "hypotheses": [],
        "methodological_roadmap": [],
        "review_paper_framing": gap.get("gap_to_contribution_advice", ""),
    }
