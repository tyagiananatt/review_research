"""Integration and pipeline tests for PDF extraction, DOCX generation, and exports."""

from __future__ import annotations

import io
import fitz
import pytest

from core.db import init_db, save_paper, list_papers, create_project
from core.pdf_extract import process_pdf
from core.ieee_export import (
    generate_ieee_docx,
    generate_markdown_export,
    generate_bibtex_export,
    generate_similarity_report_markdown,
)


def create_sample_pdf_bytes() -> bytes:
    """Create a multi-page academic PDF in-memory using fitz."""
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        fitz.Point(50, 72),
        "Attention Is All You Need in Vision Transformers\n"
        "Alice Smith, Bob Jones, and Charlie Brown\n"
        "Published in IEEE Trans. AI, 2024. DOI: 10.1109/TAI.2024.123456\n\n"
        "Abstract— Vision transformers have reshaped computer vision by replacing "
        "convolutions with self-attention mechanisms. In this paper we investigate "
        "sparse attention topologies that reduce computational complexity while maintaining accuracy.\n\n"
        "1. Introduction\n"
        "Deep convolutional networks have dominated vision pipelines for over a decade. "
        "However, self-attention architectures capture long-range dependencies more effectively.\n"
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        fitz.Point(50, 72),
        "2. Methodology and Architecture\n"
        "We propose SparseVision, an attention formulation that restricts token interactions "
        "to local spatio-temporal neighborhoods. The attention matrix is sparsified via top-k masking.\n\n"
        "3. Experimental Results\n"
        "Evaluations on ImageNet-1k demonstrate an 84.5% top-1 accuracy with a 32% reduction in FLOPs.\n\n"
        "References\n"
        "[1] A. Vaswani et al., Attention is all you need, NeurIPS, 2017.\n"
        "[2] A. Dosovitskiy et al., An image is worth 16x16 words, ICLR, 2021.\n"
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pdf_extraction_and_cleaning(tmp_path):
    pdf_bytes = create_sample_pdf_bytes()
    test_db = tmp_path / "test.db"
    init_db(test_db)
    proj_id = create_project("Test Project", db_path=test_db)

    paper_data = process_pdf(
        file_bytes_or_path=pdf_bytes,
        filename="sample_paper.pdf",
        project_id=proj_id,
        use_llm_metadata=False,
        has_api_key=False,
    )

    assert paper_data["is_scanned"] is False
    assert len(paper_data["pages"]) == 2
    assert "Attention Is All You Need" in paper_data["title"]
    assert paper_data["year"] == 2024
    assert paper_data["doi"] == "10.1109/TAI.2024.123456"
    assert "Vision transformers have reshaped" in paper_data["abstract"]
    assert "Vaswani" in paper_data["references_text"]
    # References should be stripped from clean_text
    assert "Vaswani" not in paper_data["clean_text"]


def test_ieee_docx_and_exports(tmp_path):
    test_db = tmp_path / "test_export.db"
    init_db(test_db)
    proj_id = create_project("Export Project", db_path=test_db)

    sample_papers = [
        {
            "id": 1,
            "project_id": proj_id,
            "citation_id": 1,
            "filename": "vision_trans.pdf",
            "title": "Sparse Attention in Vision",
            "authors": ["Alice Smith", "Bob Jones"],
            "year": 2024,
            "venue": "IEEE TAI",
            "doi": "10.1109/TAI.2024.123456",
            "paper_card_json": {
                "problem": "High FLOP overhead of full attention",
                "methodology": "Sparse neighborhood masking",
                "dataset": "ImageNet-1k",
                "metrics": ["Top-1 Accuracy", "FLOPs"],
                "key_results": ["84.5% top-1 accuracy", "32% FLOP reduction"],
            },
        }
    ]

    sample_sections = {
        "Abstract": {"content": "This paper reviews attention mechanisms across multimodal systems."},
        "Index Terms": {"content": "Deep Learning, Transformers, Survey."},
        "Introduction": {
            "content": "Attention mechanisms have revolutionized the field of visual reasoning [1]."
        },
        "Literature Review": {
            "content": "Various sparse attention variants have emerged to alleviate memory constraints [1]."
        },
        "References": {
            "content": "[1] A. Smith and B. Jones, \"Sparse Attention in Vision,\" *IEEE TAI*, 2024."
        },
    }

    # 1. Test DOCX generation
    docx_stream = generate_ieee_docx(
        sections=sample_sections,
        papers=sample_papers,
        paper_title="Survey of Sparse Transformers",
        authors="Test Researcher",
        affiliations="Research Institute",
    )
    assert docx_stream is not None
    assert docx_stream.getbuffer().nbytes > 1000

    # 2. Test Markdown export
    md_out = generate_markdown_export(sample_sections, sample_papers, "Survey Title")
    assert "# Survey Title" in md_out
    assert "## Introduction" in md_out
    assert "Appendix: AI Usage" in md_out

    # 3. Test BibTeX export
    bib_out = generate_bibtex_export(sample_papers)
    assert "@article{" in bib_out
    assert "Alice Smith and Bob Jones" in bib_out
    assert "10.1109/TAI.2024.123456" in bib_out

    # 4. Test Plagiarism report markdown
    rep_out = generate_similarity_report_markdown(sample_sections)
    assert "# ReviewForge Plagiarism & Originality" in rep_out
