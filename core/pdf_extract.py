"""PDF text extraction, multi-stage cleaning, and metadata parsing."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

try:
    import pypdf
except ImportError:
    pypdf = None  # type: ignore

from config import PROMPTS_DIR
from core.llm import call_gemini_json

logger = logging.getLogger(__name__)


def extract_raw_pages_fitz(stream_or_path: Union[bytes, str, Path]) -> List[str]:
    """Extract text page by page using PyMuPDF (fitz)."""
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is not installed.")

    pages_text: List[str] = []
    if isinstance(stream_or_path, bytes):
        doc = fitz.open(stream=stream_or_path, filetype="pdf")
    else:
        doc = fitz.open(str(stream_or_path))

    try:
        for page in doc:
            pages_text.append(page.get_text("text") or "")
    finally:
        doc.close()

    return pages_text


def extract_raw_pages_pypdf(stream_or_path: Union[bytes, str, Path]) -> List[str]:
    """Fallback: Extract text page by page using pypdf."""
    if pypdf is None:
        raise ImportError("pypdf is not installed.")

    pages_text: List[str] = []
    if isinstance(stream_or_path, bytes):
        reader = pypdf.PdfReader(io.BytesIO(stream_or_path))
    else:
        reader = pypdf.PdfReader(str(stream_or_path))

    for page in reader.pages:
        pages_text.append(page.extract_text() or "")

    return pages_text


def extract_pdf_pages(
    stream_or_path: Union[bytes, str, Path]
) -> Tuple[List[str], bool]:
    """Extract raw text from PDF pages with fitz and pypdf fallback.

    Returns:
        (pages_list, is_scanned_or_empty)
    """
    raw_pages: List[str] = []
    try:
        raw_pages = extract_raw_pages_fitz(stream_or_path)
    except Exception as e:
        logger.warning("PyMuPDF extraction failed (%s). Falling back to pypdf...", e)
        try:
            raw_pages = extract_raw_pages_pypdf(stream_or_path)
        except Exception as pypdf_err:
            logger.error("Both fitz and pypdf failed to extract text: %s", pypdf_err)
            raise RuntimeError(f"Failed to read PDF: {pypdf_err}") from pypdf_err

    # Check for empty or scanned PDF
    total_words = sum(len(p.split()) for p in raw_pages)
    is_scanned = total_words < 50

    return raw_pages, is_scanned


def clean_hyphenated_linebreaks(text: str) -> str:
    """Fix words broken by end-of-line hyphens (e.g. 'algo-\nrithm' -> 'algorithm')."""
    # Replace hyphen followed by newline and optional whitespace
    return re.sub(r"(\b[A-Za-z]+)-\s*\n\s*([A-Za-z]+\b)", r"\1\2", text)


def remove_repeated_headers_footers(pages: List[str]) -> List[str]:
    """Detect and remove repeated lines (running headers and footers) appearing across pages."""
    if len(pages) <= 2:
        return pages

    first_lines: Dict[str, int] = {}
    last_lines: Dict[str, int] = {}

    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        if lines:
            first_lines[lines[0]] = first_lines.get(lines[0], 0) + 1
            last_lines[lines[-1]] = last_lines.get(lines[-1], 0) + 1

    num_pages = len(pages)
    threshold = max(2, int(num_pages * 0.45))

    common_headers = {line for line, cnt in first_lines.items() if cnt >= threshold and len(line) > 3}
    common_footers = {line for line, cnt in last_lines.items() if cnt >= threshold and len(line) > 3}

    cleaned_pages: List[str] = []
    for page in pages:
        lines = page.splitlines()
        filtered_lines = []
        for line in lines:
            s_line = line.strip()
            # Strip if recognized as running header/footer
            if s_line in common_headers or s_line in common_footers:
                continue
            # Strip standalone page numbers: e.g. "1", "12", "Page 4 of 12"
            if re.match(r"^(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?$", s_line, re.IGNORECASE):
                continue
            filtered_lines.append(line)
        cleaned_pages.append("\n".join(filtered_lines))

    return cleaned_pages


def split_references_section(full_text: str) -> Tuple[str, str]:
    """Split the full document text into body clean text and references section."""
    # Look for References / Bibliography heading in the latter half of the text
    min_split_pos = int(len(full_text) * 0.4)
    search_sub = full_text[min_split_pos:]

    ref_pattern = re.compile(
        r"(?:\n|\A)\s*(?:[IVX\d\.\s]+)?\b(references|bibliography|literature cited)\b\s*(?:\n|\Z)",
        re.IGNORECASE,
    )

    match = ref_pattern.search(search_sub)
    if match:
        split_idx = min_split_pos + match.start()
        body_text = full_text[:split_idx].strip()
        ref_text = full_text[split_idx:].strip()
        return body_text, ref_text

    return full_text, ""


def extract_regex_metadata(first_page_text: str) -> Dict[str, Any]:
    """Extract baseline metadata using fast regex heuristics from page 1."""
    meta: Dict[str, Any] = {
        "title": "",
        "authors": [],
        "year": None,
        "venue": "",
        "doi": "",
        "abstract": "",
    }

    # DOI search
    doi_match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", first_page_text)
    if doi_match:
        meta["doi"] = doi_match.group(1).rstrip(".")

    # Year search (e.g. copyright 2023, published in 2024, arXiv:2401...)
    arxiv_match = re.search(r"arXiv:(\d{2})\d{2}\.", first_page_text)
    if arxiv_match:
        meta["year"] = 2000 + int(arxiv_match.group(1))
    else:
        year_match = re.search(r"\b(19\d{2}|20[0-2]\d)\b", first_page_text)
        if year_match:
            meta["year"] = int(year_match.group(1))

    # Abstract search
    abstract_match = re.search(
        r"(?i)\babstract\b[\s:—\.\-]*([\s\S]*?)(?=(?:\n\s*(?:[IVX\d\.]+\s*)?(?:introduction|index terms|keywords|categories)|\Z))",
        first_page_text,
    )
    if abstract_match:
        meta["abstract"] = re.sub(r"\s+", " ", abstract_match.group(1)).strip()

    # Heuristic title from top non-empty lines
    lines = [l.strip() for l in first_page_text.splitlines() if l.strip()]
    candidate_title_lines = []
    for line in lines[:8]:
        # Skip preprint stamps or arXiv lines
        if any(w in line.lower() for w in ["arxiv:", "doi:", "http", "vol.", "issue"]):
            continue
        if re.search(r"(?i)\b(abstract|contents)\b", line):
            break
        candidate_title_lines.append(line)
        if len(" ".join(candidate_title_lines)) > 140:
            break

    if candidate_title_lines:
        meta["title"] = " ".join(candidate_title_lines[:3])

    return meta


def refine_metadata_with_llm(
    first_page_text: str, fallback_meta: Dict[str, Any], has_key: bool = True
) -> Dict[str, Any]:
    """Refine metadata using Gemini extraction prompt on the first page, falling back to regex."""
    if not has_key:
        return fallback_meta

    prompt_path = PROMPTS_DIR / "extract_metadata.txt"
    if not prompt_path.exists():
        return fallback_meta

    prompt_template = prompt_path.read_text(encoding="utf-8")
    user_prompt = f"{prompt_template}\n\nFIRST PAGE TEXT:\n{first_page_text[:4000]}"

    try:
        result = call_gemini_json(prompt=user_prompt, temperature=0.0, use_cache=True)
        if isinstance(result, dict):
            return {
                "title": result.get("title") or fallback_meta.get("title", ""),
                "authors": result.get("authors") or fallback_meta.get("authors", []),
                "year": result.get("year") or fallback_meta.get("year"),
                "venue": result.get("venue") or fallback_meta.get("venue", ""),
                "doi": result.get("doi") or fallback_meta.get("doi", ""),
                "abstract": result.get("abstract") or fallback_meta.get("abstract", ""),
            }
    except Exception as e:
        logger.warning("LLM metadata extraction failed (%s). Using regex metadata.", e)

    return fallback_meta


def process_pdf(
    file_bytes_or_path: Union[bytes, str, Path],
    filename: str,
    project_id: int,
    citation_id: Optional[int] = None,
    use_llm_metadata: bool = True,
    has_api_key: bool = True,
) -> Dict[str, Any]:
    """Complete extraction and cleaning pipeline for an uploaded PDF."""
    raw_pages, is_scanned = extract_pdf_pages(file_bytes_or_path)

    if is_scanned:
        return {
            "project_id": project_id,
            "citation_id": citation_id,
            "filename": filename,
            "title": f"Scanned or Image-only PDF ({filename})",
            "authors": [],
            "year": None,
            "venue": "",
            "doi": "",
            "abstract": "",
            "full_text": "",
            "clean_text": "",
            "references_text": "",
            "is_scanned": True,
            "metadata_json": {},
            "pages": [],
            "error_message": (
                f"'{filename}' appears to be a scanned or image-based PDF with no selectable text. "
                "Optical Character Recognition (OCR) is required before processing."
            ),
        }

    # Step 1: Clean repeated headers and footers across pages
    cleaned_page_texts = remove_repeated_headers_footers(raw_pages)

    # Step 2: Clean hyphenated linebreaks on each page
    pages_data: List[Dict[str, Any]] = []
    final_cleaned_pages: List[str] = []

    for idx, page_raw in enumerate(raw_pages, start=1):
        cleaned_text = clean_hyphenated_linebreaks(cleaned_page_texts[idx - 1])
        final_cleaned_pages.append(cleaned_text)
        pages_data.append({
            "page_number": idx,
            "raw_text": page_raw,
            "clean_text": cleaned_text,
        })

    # Step 3: Combine full text and split references
    full_text = "\n\n".join(final_cleaned_pages)
    clean_body_text, references_text = split_references_section(full_text)

    # Step 4: Extract metadata from first page
    first_page_text = final_cleaned_pages[0] if final_cleaned_pages else ""
    regex_meta = extract_regex_metadata(first_page_text)

    if use_llm_metadata and has_api_key:
        final_meta = refine_metadata_with_llm(first_page_text, regex_meta, has_key=True)
    else:
        final_meta = regex_meta

    return {
        "project_id": project_id,
        "citation_id": citation_id,
        "filename": filename,
        "title": final_meta.get("title") or filename.rsplit(".", 1)[0],
        "authors": final_meta.get("authors") or [],
        "year": final_meta.get("year"),
        "venue": final_meta.get("venue") or "",
        "doi": final_meta.get("doi") or "",
        "abstract": final_meta.get("abstract") or "",
        "full_text": full_text,
        "clean_text": clean_body_text,
        "references_text": references_text,
        "is_scanned": False,
        "metadata_json": final_meta,
        "pages": pages_data,
        "error_message": None,
    }
