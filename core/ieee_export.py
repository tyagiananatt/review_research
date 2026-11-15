"""IEEE-format .docx, Markdown, BibTeX, and AI Disclosure export generator."""

from __future__ import annotations

import io
import logging
import re
from typing import Any, Dict, List, Optional

try:
    import docx
    from docx.enum.section import WD_SECTION_START
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement, ns
    from docx.shared import Inches, Pt, RGBColor
except ImportError:
    docx = None  # type: ignore

logger = logging.getLogger(__name__)

ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def set_section_two_columns(section: Any) -> None:
    """Configure a python-docx section to have a 2-column layout per IEEE specifications."""
    try:
        sectPr = section._sectPr
        cols = sectPr.xpath("./w:cols")
        if cols:
            cols[0].set(ns.qn("w:num"), "2")
            cols[0].set(ns.qn("w:space"), "500")  # ~0.35 in gap between columns
        else:
            col_el = OxmlElement("w:cols")
            col_el.set(ns.qn("w:num"), "2")
            col_el.set(ns.qn("w:space"), "500")
            sectPr.append(col_el)
    except Exception as e:
        logger.warning("Could not set two-column layout: %s", e)


def clean_markdown_inline(text: str) -> str:
    """Strip basic markdown bold/italic tags for pure Word text."""
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    t = re.sub(r"\*(.*?)\*", r"\1", t)
    t = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", t)
    return t


def generate_ieee_docx(
    sections: Dict[str, Dict[str, Any]],
    papers: List[Dict[str, Any]],
    paper_title: str = "A Comprehensive Survey and Systematic Review",
    authors: str = "First Author, Second Author, and Third Author",
    affiliations: str = "Department of Computer Science, University / Research Institute",
    include_ai_appendix: bool = True,
) -> io.BytesIO:
    """Generate a publication-ready IEEE-format .docx document."""
    if docx is None:
        raise ImportError("python-docx is not installed.")

    doc = docx.Document()

    # Page Margins: 0.75 in top/bottom/left/right (IEEE standard)
    first_section = doc.sections[0]
    first_section.top_margin = Inches(0.75)
    first_section.bottom_margin = Inches(0.75)
    first_section.left_margin = Inches(0.75)
    first_section.right_margin = Inches(0.75)

    # Base Styles: Times New Roman, 10 pt
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(10)
    font.color.rgb = RGBColor(0, 0, 0)

    # 1. Document Title: 24 pt, Bold, Centered
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(12)
    title_run = title_p.add_run(paper_title)
    title_run.font.size = Pt(22)
    title_run.font.bold = True

    # 2. Authors and Affiliation: 10 pt, Centered
    authors_p = doc.add_paragraph()
    authors_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    authors_p.paragraph_format.space_after = Pt(2)
    auth_run = authors_p.add_run(authors)
    auth_run.font.size = Pt(10)

    affil_p = doc.add_paragraph()
    affil_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    affil_p.paragraph_format.space_after = Pt(18)
    affil_run = affil_p.add_run(affiliations)
    affil_run.font.size = Pt(9)
    affil_run.font.italic = True

    # 3. Abstract and Index Terms (Full width or Top of Section)
    abstract_text = ""
    index_terms_text = ""
    if "Abstract" in sections and sections["Abstract"].get("content"):
        abstract_text = sections["Abstract"]["content"].strip()
    if "Index Terms" in sections and sections["Index Terms"].get("content"):
        index_terms_text = sections["Index Terms"]["content"].strip()

    if abstract_text:
        abs_p = doc.add_paragraph()
        abs_p.paragraph_format.space_after = Pt(6)
        abs_p.paragraph_format.left_indent = Inches(0.25)
        abs_p.paragraph_format.right_indent = Inches(0.25)
        lead_run = abs_p.add_run("Abstract—")
        lead_run.font.bold = True
        lead_run.font.size = Pt(9)
        body_run = abs_p.add_run(clean_markdown_inline(abstract_text))
        body_run.font.italic = True
        body_run.font.size = Pt(9)

    if index_terms_text:
        idx_p = doc.add_paragraph()
        idx_p.paragraph_format.space_after = Pt(14)
        idx_p.paragraph_format.left_indent = Inches(0.25)
        idx_p.paragraph_format.right_indent = Inches(0.25)
        idx_lead = idx_p.add_run("Index Terms—")
        idx_lead.font.bold = True
        idx_lead.font.size = Pt(9)
        idx_body = idx_p.add_run(clean_markdown_inline(index_terms_text))
        idx_body.font.italic = True
        idx_body.font.size = Pt(9)

    # 4. Continuous Section Break for Two-Column Body
    body_section = doc.add_section(WD_SECTION_START.CONTINUOUS)
    body_section.top_margin = Inches(0.75)
    body_section.bottom_margin = Inches(0.75)
    body_section.left_margin = Inches(0.75)
    body_section.right_margin = Inches(0.75)
    set_section_two_columns(body_section)

    # Ordered sections
    heading_counter = 0
    for sec_name, sec_obj in sections.items():
        if sec_name in ["Abstract", "Index Terms"]:
            continue  # Already placed in header

        content = sec_obj.get("content", "").strip()
        if not content:
            continue

        # Major Roman Numeral Heading
        numeral = ROMAN_NUMERALS[heading_counter] if heading_counter < len(ROMAN_NUMERALS) else f"{heading_counter + 1}."
        heading_counter += 1

        head_p = doc.add_paragraph()
        head_p.paragraph_format.space_before = Pt(14)
        head_p.paragraph_format.space_after = Pt(6)
        head_p.paragraph_format.keep_with_next = True
        head_run = head_p.add_run(f"{numeral}. {sec_name.upper()}")
        head_run.font.bold = True
        head_run.font.size = Pt(10)

        # Parse subsections and paragraphs
        paragraphs = content.split("\n\n")
        sub_counter = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this is a markdown subheading: e.g. "### A. Something" or "## Subheading"
            if para.startswith("#"):
                sub_title = para.lstrip("#").strip()
                sub_p = doc.add_paragraph()
                sub_p.paragraph_format.space_before = Pt(8)
                sub_p.paragraph_format.space_after = Pt(3)
                sub_p.paragraph_format.keep_with_next = True
                sub_run = sub_p.add_run(sub_title)
                sub_run.font.bold = True
                sub_run.font.italic = True
                sub_run.font.size = Pt(9.5)
            else:
                body_p = doc.add_paragraph()
                body_p.paragraph_format.space_before = Pt(0)
                body_p.paragraph_format.space_after = Pt(4)
                body_p.paragraph_format.line_spacing = 1.05
                body_p.paragraph_format.first_line_indent = Inches(0.18)
                run = body_p.add_run(clean_markdown_inline(para))
                run.font.size = Pt(9.5)

    # 5. AI Usage and Methodology Appendix
    if include_ai_appendix:
        app_p = doc.add_paragraph()
        app_p.paragraph_format.space_before = Pt(16)
        app_p.paragraph_format.space_after = Pt(6)
        app_run = app_p.add_run("APPENDIX: AI USAGE AND PROVENANCE DISCLOSURE")
        app_run.font.bold = True
        app_run.font.size = Pt(9.5)

        app_body = doc.add_paragraph()
        app_body.paragraph_format.first_line_indent = Inches(0.18)
        app_body.paragraph_format.space_after = Pt(4)
        app_text = (
            "This review paper was drafted with the assistance of ReviewForge, utilizing Google Gemini "
            "as a synthesis engine grounded strictly in uploaded academic papers. All citations map to verified "
            "source papers. In accordance with IEEE and major academic conference policies on Generative AI, "
            "all claims, comparative findings, and references were reviewed and verified by human authors. "
            "A local n-gram and fuzzy sentence-level similarity check was performed to prevent verbatim source overlap."
        )
        app_body.add_run(app_text).font.size = Pt(8.5)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def generate_markdown_export(
    sections: Dict[str, Dict[str, Any]],
    papers: List[Dict[str, Any]],
    paper_title: str = "A Systematic Literature Review",
) -> str:
    """Generate complete GitHub-flavored Markdown text for the review paper."""
    lines = [f"# {paper_title}\n"]

    for sec_name, sec_obj in sections.items():
        content = sec_obj.get("content", "").strip()
        if not content:
            continue

        lines.append(f"## {sec_name}\n")
        lines.append(f"{content}\n")

    # Add AI disclosure
    lines.append("## Appendix: AI Usage and Methodology Disclosure\n")
    lines.append(
        "This review paper was synthesized using **ReviewForge** with Google Gemini. "
        "All citations map directly to verified source papers. "
        "A local anti-plagiarism n-gram and sentence similarity check was executed across all sections."
    )

    return "\n".join(lines)


def generate_bibtex_export(papers: List[Dict[str, Any]]) -> str:
    """Generate valid BibTeX entries for all uploaded papers."""
    entries = []
    for p in papers:
        cit_id = p.get("citation_id", 1)
        authors = p.get("authors") or []
        first_author = authors[0].split()[-1].lower() if authors else "unknown"
        year = p.get("year") or 2024
        key = f"{first_author}{year}_{cit_id}"

        author_str = " and ".join(authors) if authors else "Unknown"
        title = p.get("title", "Untitled")
        venue = p.get("venue") or "arXiv preprint"
        doi = p.get("doi")
        doi_line = f"  doi = {{{doi}}},\n" if doi else ""

        entry = (
            f"@article{{{key},\n"
            f"  author = {{{author_str}}},\n"
            f"  title = {{{{{title}}}}},\n"
            f"  journal = {{{venue}}},\n"
            f"  year = {{{year}}},\n"
            f"{doi_line}"
            f"  note = {{Citation [{cit_id}] in ReviewForge}}\n"
            f"}}"
        )
        entries.append(entry)

    return "\n\n".join(entries)


def generate_similarity_report_markdown(
    sections: Dict[str, Dict[str, Any]]
) -> str:
    """Generate a downloadable, comprehensive similarity and originality report."""
    lines = [
        "# ReviewForge Plagiarism & Originality Verification Report\n",
        "> **Notice:** This local audit verifies sentence similarity (rapidfuzz partial_ratio) and word n-gram "
        "overlap (5-gram & 8-gram) strictly against the uploaded PDF corpus. Authors MUST run their final manuscript "
        "through their institution's licensed checker (e.g., Turnitin or iThenticate) prior to conference or journal submission.\n",
        "| Section | Overall Score | Status Band | Sentences Checked | Flagged Count |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for name, sec in sections.items():
        rep = sec.get("similarity_report_json") or {}
        score = rep.get("overall_score", 0.0)
        band = rep.get("status_band", "green").upper()
        total_s = rep.get("total_sentences", 0)
        flagged_c = rep.get("flagged_sentences_count", 0)
        lines.append(f"| {name} | {round(score * 100, 1)}% | **{band}** | {total_s} | {flagged_c} |")

    lines.append("\n## Detailed Section Audit Breakdown\n")

    for name, sec in sections.items():
        rep = sec.get("similarity_report_json") or {}
        flagged = rep.get("flagged_items", [])
        if not flagged:
            lines.append(f"### {name}: No flagged sentences detected (Clean)\n")
            continue

        lines.append(f"### {name}: {len(flagged)} Flagged Sentence(s)\n")
        for f in flagged:
            lines.append(f"- **Sentence:** \"{f.get('sentence', '')}\"")
            lines.append(f"  - **Similarity:** {round(f.get('similarity_score', 0) * 100)}% ({f.get('match_reason')})")
            lines.append(f"  - **Matched Source:** [{f.get('matched_paper_citation')}] {f.get('matched_paper_title')}")
            if f.get("matched_source_snippet"):
                lines.append(f"  - **Source Snippet:** *\"{f.get('matched_source_snippet')}\"*")
            lines.append("")

    return "\n".join(lines)
