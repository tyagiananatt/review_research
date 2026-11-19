"""Unit tests for the IEEE-format PDF exporter (core/pdf_export.py).

The exporter must render the reference document's two-column IEEE layout using
embedded Latin Modern fonts and land on an exact page count. The content below
is calibrated to the reference paper's measured volume (~4,000 words over six
pages, including a table, an equation, a chart and a diagram).
"""

from __future__ import annotations

import pytest

from core.pdf_export import _register_fonts, generate_ieee_pdf


_LOREM = (
    "Early detection of health abnormalities in livestock depends on continuous behavioural "
    "observation, yet commercial herds routinely exceed several hundred animals per stockperson. "
    "Under intensive stocking densities, manual inspection fails to surface subclinical indicators "
    "such as lameness or ketosis before tissue damage has already occurred [1], [2]. Machine vision "
    "offers a non-invasive substitute, although most pipelines evaluate activity at pen scale and "
    "therefore average away the individual-level deviations that matter clinically [3]. "
)


def _p(n: int) -> str:
    return (_LOREM * n).strip()


def _sections() -> dict:
    return {
        "Abstract": {"content": _p(2)},
        "Index Terms": {
            "content": "Animal health, anomaly detection, computer vision, explainable AI, temporal modelling"
        },
        "Introduction": {"content": _p(6) + "\n\n### A. Scope and Contributions\n\n" + _p(2)},
        "Thematic Literature Review": {
            "content": _p(5)
            + "\n\n```table:Representative vision benchmarks for livestock monitoring\n"
            "Study | Species | Architecture | Reported metric\n"
            "Tu [1] | Pigs | YOLOv5 + ByteTrack | mAP 94.2%\n"
            "Chen [9] | Cattle | YOLOv5 + ArcFace | Accuracy 96.8%\n"
            "Ramesh [19] | Cattle | YOLOv8 + DINOv2 | F1 92.1%\n```\n\n"
            + _p(2)
        },
        "Comparative Analysis": {
            "content": _p(5)
            + "\n\n```equation\n"
            "P(y \\mid x) = \\frac{\\exp(f_y(x))}{\\sum_{j} \\exp(f_j(x))}\n```\n\n"
            + _p(2)
            + "\n\n```chart:bar:Reported mean average precision by backbone\n"
            "YOLOv5: 92\nYOLOv7-tiny: 94.7\nYOLOv8: 95.3\n```\n\n"
            + _p(2)
            + "\n\n```diagram:End-to-end review pipeline\n"
            "PDF corpus -> Text extraction -> Paper cards -> Gap synthesis -> Section drafts\n```\n\n"
            + _p(1)
        },
        "Research Gaps & Challenges": {"content": _p(5)},
        "Future Directions": {"content": _p(5)},
        "Conclusion": {"content": _p(3)},
        "References": {
            "content": "\n\n".join(
                f"[{i}] A. Author et al., \"A representative study of livestock monitoring,\" "
                f"*IEEE Trans. Example*, vol. {i}, no. 2, pp. 1-9, {2020 + i}, doi: 10.1109/EX.{2020 + i}."
                for i in range(1, 22)
            )
        },
    }


def _papers() -> list:
    return [
        {"citation_id": i, "authors": [f"Author {i}"], "title": f"Study {i}", "year": 2020 + i}
        for i in range(1, 22)
    ]


_TITLE = (
    "Explainable Artificial Intelligence for Individualised Early Health Anomaly Detection in "
    "Animals: A Review of Computer Vision, Behavioural Monitoring and Temporal Analysis"
)
_AUTHORS = "Yashwant, Vansh Tyagi, Mohit Sharma, Anant Tyagi, Anil, Gautam Kumar"
_AFFIL = (
    "School of Computer Science and Engineering, Lovely Professional University, Phagwara, Punjab, India"
)


def _page_count(pdf_bytes: bytes) -> int:
    fitz = pytest.importorskip("fitz")
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


def test_fonts_available():
    assert _register_fonts() is True


def test_generate_ieee_pdf_exactly_six_pages():
    out = generate_ieee_pdf(
        sections=_sections(),
        papers=_papers(),
        paper_title=_TITLE,
        authors=_AUTHORS,
        affiliations=_AFFIL,
        target_pages=6,
    )
    data = out.getvalue()
    assert data.startswith(b"%PDF")
    assert len(data) > 10_000
    assert _page_count(data) == 6


def test_generate_ieee_pdf_embeds_latin_modern_fonts():
    out = generate_ieee_pdf(
        sections=_sections(),
        papers=_papers(),
        paper_title=_TITLE,
        authors=_AUTHORS,
        affiliations=_AFFIL,
        target_pages=6,
    )
    data = out.getvalue()
    # Fonts are embedded as subsets, so the base name appears with a 6-letter tag.
    assert b"LMRoman10-Regular" in data
    assert b"LMRoman17-Regular" in data


def test_generate_ieee_pdf_handles_empty_sections_gracefully():
    out = generate_ieee_pdf(
        sections={},
        papers=[],
        paper_title="Minimal Review",
        authors="A. Author",
        affiliations="Example University",
        target_pages=None,
    )
    data = out.getvalue()
    assert data.startswith(b"%PDF")
    assert out.tell() == 0
