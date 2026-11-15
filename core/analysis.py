"""Cross-paper deep analysis: comparison tables, taxonomies, timelines, and key claims."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from config import PROMPTS_DIR, safe_fill_template
from core.keys import KeyManager
from core.llm import call_gemini_json

logger = logging.getLogger(__name__)


def build_comparison_table(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a tabular comparison across all papers based on their Paper Cards."""
    table_rows: List[Dict[str, Any]] = []

    for p in papers:
        card = p.get("paper_card_json") or {}
        limitations = card.get("limitations", {})
        if isinstance(limitations, dict):
            lim_list = limitations.get("stated", []) + limitations.get("inferred", [])
        elif isinstance(limitations, list):
            lim_list = limitations
        else:
            lim_list = [str(limitations)]

        metrics_list = card.get("metrics", [])
        if isinstance(metrics_list, list):
            metrics_str = ", ".join(metrics_list[:4])
        else:
            metrics_str = str(metrics_list)

        results_list = card.get("key_results", [])
        results_str = "; ".join(results_list[:3]) if isinstance(results_list, list) else str(results_list)

        table_rows.append({
            "Citation": f"[{p.get('citation_id', '?')}]",
            "Paper Title": p.get("title", card.get("title", p.get("filename", "Untitled"))),
            "Year": p.get("year") or "N/A",
            "Methodology": card.get("methodology") or "N/A",
            "Datasets": card.get("dataset") or "N/A",
            "Key Metrics": metrics_str or "N/A",
            "Key Results": results_str or "N/A",
            "Limitations": "; ".join(lim_list[:2]) if lim_list else "None noted",
        })

    return table_rows


def format_papers_for_prompt(papers: List[Dict[str, Any]], max_total_chars: int = 7500) -> str:
    """Format papers and their cards into clean, high-density textual context strictly under TPM limits."""
    if not papers:
        return ""
    chars_per_paper = max(250, min(800, max_total_chars // max(1, len(papers))))
    entries = []
    for p in papers:
        cit_id = p.get("citation_id", 1)
        card = p.get("paper_card_json") or {}

        results = card.get("key_results", [])
        res_str = "; ".join(str(r) for r in results[:2]) if isinstance(results, list) else str(results)[:150]

        contribs = card.get("contributions", [])
        contrib_str = "; ".join(str(c) for c in contribs[:2]) if isinstance(contribs, list) else str(contribs)[:150]

        limits = card.get("limitations", {})
        if isinstance(limits, dict):
            limits_str = "; ".join(limits.get("stated", [])[:2] + limits.get("inferred", [])[:1])
        else:
            limits_str = str(limits)[:120]

        prob = str(card.get("problem", "N/A"))[:180]
        meth = str(card.get("methodology", "N/A"))[:200]
        dset = str(card.get("dataset", "N/A"))[:100]

        entry = (
            f"--- [Paper {cit_id}] ---\n"
            f"Title: {p.get('title', 'Untitled')}\n"
            f"Year: {p.get('year', 'N/A')}\n"
            f"Problem: {prob}\n"
            f"Methodology: {meth}\n"
            f"Dataset: {dset}\n"
            f"Key Results: {res_str}\n"
            f"Contributions: {contrib_str}\n"
            f"Limitations: {limits_str}\n"
        )
        entries.append(entry[:chars_per_paper])
    return "\n\n".join(entries)


def generate_taxonomy_and_timeline(
    papers: List[Dict[str, Any]], key_manager: Optional[KeyManager] = None
) -> Dict[str, Any]:
    """Generate categorization taxonomy and timeline evolution across papers."""
    if not papers:
        return {"taxonomy_categories": [], "timeline_evolution": [], "emerging_trends": []}

    prompt_tmpl = (PROMPTS_DIR / "taxonomy.txt").read_text(encoding="utf-8")
    papers_summary = format_papers_for_prompt(papers)
    prompt = safe_fill_template(prompt_tmpl, papers_summary=papers_summary)

    try:
        result = call_gemini_json(prompt=prompt, temperature=0.2, key_manager=key_manager)
        if isinstance(result, dict):
            return {
                "taxonomy_categories": result.get("taxonomy_categories", []),
                "timeline_evolution": result.get("timeline_evolution", []),
                "emerging_trends": result.get("emerging_trends", []),
            }
    except Exception as e:
        logger.error("Failed to generate taxonomy/timeline: %s", e)
        raise e


def extract_key_claims(
    papers: List[Dict[str, Any]], key_manager: Optional[KeyManager] = None
) -> List[Dict[str, Any]]:
    """Extract foundational key claims across papers for the Introduction section."""
    if not papers:
        return []

    prompt_tmpl = (PROMPTS_DIR / "key_claims.txt").read_text(encoding="utf-8")
    papers_summary = format_papers_for_prompt(papers)
    prompt = safe_fill_template(prompt_tmpl, paper_cards_json=papers_summary)

    try:
        result = call_gemini_json(prompt=prompt, temperature=0.2, key_manager=key_manager)
        if isinstance(result, dict):
            return result.get("key_claims", [])
        if isinstance(result, list):
            return result
    except Exception as e:
        logger.error("Failed to extract key claims: %s", e)
        raise e

    return []
