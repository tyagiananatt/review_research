"""Automatic paper discovery and download from ArXiv and Semantic Scholar."""

from __future__ import annotations

import io
import logging
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_PDF_BASE = "https://arxiv.org/pdf/"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2/"

# Timeout for HTTP requests (seconds)
REQUEST_TIMEOUT = 30
PDF_DOWNLOAD_TIMEOUT = 60


def _http_get(url: str, timeout: int = REQUEST_TIMEOUT) -> bytes:
    """Simple HTTP GET with a custom User-Agent to avoid 403s."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "ReviewForge/1.0 (academic paper downloader; "
                "contact: reviewforge@example.com)"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def search_arxiv(
    query: str,
    max_results: int = 8,
    description: str = "",
) -> List[Dict[str, Any]]:
    """Search ArXiv for papers matching the query.

    Returns a list of dicts with keys: arxiv_id, title, authors, year,
    abstract, pdf_url, source.
    """
    # Build a focused query using title and abstract fields
    encoded_q = urllib.parse.quote(query)
    full_query = f"ti:{encoded_q}+OR+abs:{encoded_q}"
    if description:
        desc_q = urllib.parse.quote(description)
        full_query = f"({full_query})+AND+(ti:{desc_q}+OR+abs:{desc_q})"

    url = (
        f"{ARXIV_API}?search_query={full_query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )

    try:
        raw = _http_get(url).decode("utf-8")
    except Exception as e:
        logger.warning("ArXiv search failed: %s", e)
        return []

    # Parse Atom XML without external dependencies
    results: List[Dict[str, Any]] = []
    entries = re.findall(r"<entry>(.*?)</entry>", raw, re.DOTALL)

    for entry in entries:
        try:
            id_match = re.search(r"<id>https?://arxiv\.org/abs/([^<]+)</id>", entry)
            title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            abstract_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            year_match = re.search(r"<published>(\d{4})", entry)
            author_matches = re.findall(r"<name>(.*?)</name>", entry)

            if not id_match or not title_match:
                continue

            arxiv_id = id_match.group(1).strip()
            # Strip version suffix e.g. v1, v2
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

            title = re.sub(r"\s+", " ", title_match.group(1)).strip()
            abstract = re.sub(r"\s+", " ", abstract_match.group(1)).strip() if abstract_match else ""
            year = int(year_match.group(1)) if year_match else None
            authors = [a.strip() for a in author_matches[:6]]
            pdf_url = f"{ARXIV_PDF_BASE}{arxiv_id}.pdf"

            results.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "doi": f"10.48550/arXiv.{arxiv_id}",
                "venue": "arXiv preprint",
                "source": "arxiv",
            })
        except Exception as parse_err:
            logger.debug("Failed to parse ArXiv entry: %s", parse_err)
            continue

    return results


def search_semantic_scholar(
    query: str,
    max_results: int = 8,
) -> List[Dict[str, Any]]:
    """Search Semantic Scholar for papers (no API key required for basic queries).

    Returns a list of dicts with keys: title, authors, year, abstract,
    pdf_url, doi, venue, source.
    """
    encoded_q = urllib.parse.quote(query)
    fields = "title,authors,year,abstract,externalIds,openAccessPdf,venue"
    url = (
        f"{SEMANTIC_SCHOLAR_SEARCH}"
        f"?query={encoded_q}&fields={fields}&limit={max_results}"
    )

    # Try up to 2 times with a backoff to handle transient 429 rate limits
    for attempt in range(2):
        try:
            raw = _http_get(url).decode("utf-8")
            break
        except Exception as e:
            err = str(e)
            if "429" in err and attempt == 0:
                logger.info("Semantic Scholar 429 rate limit; waiting 3 s before retry...")
                time.sleep(3)
                continue
            logger.warning("Semantic Scholar search failed: %s", e)
            return []
    else:
        return []

    import json
    try:
        data = json.loads(raw)
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for paper in data.get("data", []):
        try:
            title = paper.get("title", "")
            if not title:
                continue

            authors = [a.get("name", "") for a in paper.get("authors", [])[:6]]
            year = paper.get("year")
            abstract = paper.get("abstract") or ""
            venue = paper.get("venue") or ""
            external_ids = paper.get("externalIds") or {}
            doi = external_ids.get("DOI") or ""
            arxiv_id = external_ids.get("ArXiv") or ""

            pdf_url = ""
            open_access = paper.get("openAccessPdf") or {}
            if open_access.get("url"):
                pdf_url = open_access["url"]
            elif arxiv_id:
                pdf_url = f"{ARXIV_PDF_BASE}{arxiv_id}.pdf"

            results.append({
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "doi": doi,
                "venue": venue,
                "arxiv_id": arxiv_id,
                "source": "semantic_scholar",
            })
        except Exception as pe:
            logger.debug("Failed to parse Semantic Scholar record: %s", pe)
            continue

    return results


def deduplicate_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove duplicates by normalising and comparing titles."""
    seen: set = set()
    unique: List[Dict[str, Any]] = []

    def normalise(t: str) -> str:
        return re.sub(r"[^a-z0-9]", "", t.lower())

    for r in results:
        key = normalise(r.get("title", ""))
        if key and key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def download_pdf(url: str, timeout: int = PDF_DOWNLOAD_TIMEOUT) -> Optional[bytes]:
    """Download a PDF from the given URL and return the raw bytes, or None on failure."""
    if not url:
        return None
    try:
        data = _http_get(url, timeout=timeout)
        # Validate it's actually a PDF
        if data[:4] == b"%PDF":
            return data
        # ArXiv sometimes redirects: check for HTML response with a PDF link
        if b"<html" in data[:200].lower():
            logger.debug("URL %s returned HTML, not PDF.", url)
            return None
        return data
    except Exception as e:
        logger.warning("Failed to download PDF from %s: %s", url, e)
        return None


def search_and_download_papers(
    topic: str,
    description: str = "",
    target_count: int = 8,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> List[Dict[str, Any]]:
    """Search ArXiv and Semantic Scholar, deduplicate, and download PDFs.

    Returns a list of dicts containing paper metadata plus 'pdf_bytes' (bytes | None).
    Progress is reported via the optional callback: fn(message, fraction_0_to_1).
    """
    if progress_callback:
        progress_callback("Searching ArXiv for relevant papers...", 0.05)

    arxiv_results = search_arxiv(query=topic, max_results=target_count, description=description)
    logger.info("ArXiv returned %d results for query: %s", len(arxiv_results), topic)

    if progress_callback:
        progress_callback(
            f"Found {len(arxiv_results)} papers on ArXiv. Searching Semantic Scholar...", 0.15
        )

    ss_results = search_semantic_scholar(query=topic, max_results=target_count)
    logger.info("Semantic Scholar returned %d results.", len(ss_results))

    combined = arxiv_results + ss_results
    unique = deduplicate_results(combined)

    # Prefer entries that have a PDF URL and sort so PDFs come first
    unique.sort(key=lambda x: (0 if x.get("pdf_url") else 1, -(x.get("year") or 0)))

    # Limit to target count
    candidates = unique[:target_count]

    if progress_callback:
        progress_callback(
            f"Found {len(candidates)} unique papers. Downloading PDFs...", 0.25
        )

    downloaded: List[Dict[str, Any]] = []
    for i, paper in enumerate(candidates):
        pdf_url = paper.get("pdf_url", "")
        if progress_callback:
            msg = f"Downloading PDF {i + 1}/{len(candidates)}: {paper['title'][:60]}..."
            progress_callback(msg, 0.25 + (i / len(candidates)) * 0.65)

        pdf_bytes = download_pdf(pdf_url) if pdf_url else None
        paper["pdf_bytes"] = pdf_bytes
        paper["pdf_url"] = pdf_url

        if pdf_bytes:
            logger.info("Downloaded PDF for: %s", paper["title"])
        else:
            logger.warning("No PDF available for: %s", paper["title"])

        downloaded.append(paper)
        # Be polite to APIs
        time.sleep(0.3)

    successful = [p for p in downloaded if p.get("pdf_bytes")]
    logger.info(
        "Successfully downloaded %d/%d PDFs.", len(successful), len(candidates)
    )

    if progress_callback:
        progress_callback(
            f"Downloaded {len(successful)} papers successfully.", 0.90
        )

    return downloaded
