"""SQLite database storage and caching for ReviewForge."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config import DB_PATH


def _get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Obtain a SQLite connection with row factory configured."""
    path = str(db_path or DB_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initialize database schema if it doesn't already exist."""
    with _get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                settings_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                citation_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                title TEXT,
                authors TEXT,
                year INTEGER,
                venue TEXT,
                doi TEXT,
                abstract TEXT,
                full_text TEXT,
                clean_text TEXT,
                references_text TEXT,
                is_scanned INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}',
                paper_card_json TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS paper_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                raw_text TEXT,
                clean_text TEXT,
                FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS llm_cache (
                hash_key TEXT PRIMARY KEY,
                prompt_hash TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response_text TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                analysis_type TEXT NOT NULL,
                data_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(project_id, analysis_type),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS review_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                section_name TEXT NOT NULL,
                content TEXT NOT NULL,
                similarity_score REAL DEFAULT 0.0,
                similarity_report_json TEXT DEFAULT '{}',
                updated_at REAL NOT NULL,
                UNIQUE(project_id, section_name),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                citations_json TEXT DEFAULT '[]',
                timestamp REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_papers_project ON papers(project_id, citation_id);
            CREATE INDEX IF NOT EXISTS idx_paper_pages ON paper_pages(paper_id, page_number);
            CREATE INDEX IF NOT EXISTS idx_llm_cache ON llm_cache(prompt_hash);
            CREATE INDEX IF NOT EXISTS idx_review_sections ON review_sections(project_id);
            """
        )
        conn.commit()


# ==================== LLM CACHING ====================


def compute_cache_key(prompt: str, model: str) -> str:
    """Compute a deterministic SHA-256 hash for a prompt and model pair."""
    payload = f"{model}:::{prompt.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_cached_llm(
    prompt: str, model: str, db_path: Optional[Union[str, Path]] = None
) -> Optional[str]:
    """Retrieve cached LLM response if available."""
    cache_key = compute_cache_key(prompt, model)
    with _get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT response_text FROM llm_cache WHERE hash_key = ?", (cache_key,)
        ).fetchone()
        if row:
            return str(row["response_text"])
    return None


def set_cached_llm(
    prompt: str, model: str, response_text: str, db_path: Optional[Union[str, Path]] = None
) -> None:
    """Store LLM response in cache."""
    cache_key = compute_cache_key(prompt, model)
    prompt_hash = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache (hash_key, prompt_hash, model, prompt, response_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (cache_key, prompt_hash, model, prompt, response_text, time.time()),
        )
        conn.commit()


# ==================== PROJECT MANAGEMENT ====================


def get_or_create_default_project(
    db_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Ensure a default project exists and return its record."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM projects ORDER BY id ASC LIMIT 1").fetchone()
        if row:
            return dict(row)
        now = time.time()
        cur = conn.execute(
            "INSERT INTO projects (name, created_at, updated_at, settings_json) VALUES (?, ?, ?, ?)",
            ("Default Review Project", now, now, json.dumps({"tone": "Academic & Formal"})),
        )
        conn.commit()
        new_row = conn.execute("SELECT * FROM projects WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(new_row)


def list_projects(db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """List all user projects."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]


def create_project(
    name: str, settings: Optional[Dict[str, Any]] = None, db_path: Optional[Union[str, Path]] = None
) -> int:
    """Create a new project."""
    init_db(db_path)
    now = time.time()
    with _get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO projects (name, created_at, updated_at, settings_json) VALUES (?, ?, ?, ?)",
            (name, now, now, json.dumps(settings or {})),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_project(
    project_id: int,
    name: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Update project name or settings."""
    with _get_connection(db_path) as conn:
        fields = ["updated_at = ?"]
        params: List[Any] = [time.time()]
        if name is not None:
            fields.append("name = ?")
            params.append(name)
        if settings is not None:
            fields.append("settings_json = ?")
            params.append(json.dumps(settings))
        params.append(project_id)
        conn.execute(f"UPDATE projects SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()


def delete_project(project_id: int, db_path: Optional[Union[str, Path]] = None) -> None:
    """Delete a project and all associated papers, analysis, and sections."""
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()


# ==================== PAPER MANAGEMENT ====================


def save_paper(
    paper_data: Dict[str, Any], db_path: Optional[Union[str, Path]] = None
) -> int:
    """Insert a new paper record with its extracted pages."""
    init_db(db_path)
    project_id = paper_data["project_id"]
    with _get_connection(db_path) as conn:
        # Determine next citation ID if not provided
        if "citation_id" not in paper_data or paper_data["citation_id"] is None:
            max_cit = conn.execute(
                "SELECT COALESCE(MAX(citation_id), 0) as max_cit FROM papers WHERE project_id = ?",
                (project_id,),
            ).fetchone()["max_cit"]
            citation_id = max_cit + 1
        else:
            citation_id = paper_data["citation_id"]

        cur = conn.execute(
            """
            INSERT INTO papers (
                project_id, citation_id, filename, title, authors, year, venue,
                doi, abstract, full_text, clean_text, references_text,
                is_scanned, metadata_json, paper_card_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                citation_id,
                paper_data.get("filename", "untitled.pdf"),
                paper_data.get("title", ""),
                json.dumps(paper_data.get("authors", [])),
                paper_data.get("year"),
                paper_data.get("venue", ""),
                paper_data.get("doi", ""),
                paper_data.get("abstract", ""),
                paper_data.get("full_text", ""),
                paper_data.get("clean_text", ""),
                paper_data.get("references_text", ""),
                1 if paper_data.get("is_scanned", False) else 0,
                json.dumps(paper_data.get("metadata_json", {})),
                json.dumps(paper_data.get("paper_card_json", {})),
                time.time(),
            ),
        )
        paper_id = int(cur.lastrowid)

        # Insert individual pages if provided
        pages = paper_data.get("pages", [])
        for page in pages:
            conn.execute(
                """
                INSERT INTO paper_pages (paper_id, page_number, raw_text, clean_text)
                VALUES (?, ?, ?, ?)
                """,
                (
                    paper_id,
                    page.get("page_number", 1),
                    page.get("raw_text", ""),
                    page.get("clean_text", ""),
                ),
            )
        conn.commit()
        return paper_id


def list_papers(
    project_id: int, db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """Retrieve all papers for a project, sorted by citation_id."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM papers WHERE project_id = ? ORDER BY citation_id ASC", (project_id,)
        ).fetchall()
        papers = []
        for r in rows:
            p = dict(r)
            try:
                p["authors"] = json.loads(p["authors"] or "[]")
            except Exception:
                p["authors"] = []
            try:
                p["metadata_json"] = json.loads(p["metadata_json"] or "{}")
            except Exception:
                p["metadata_json"] = {}
            try:
                p["paper_card_json"] = json.loads(p["paper_card_json"] or "{}")
            except Exception:
                p["paper_card_json"] = {}
            papers.append(p)
        return papers


def get_paper_pages(
    paper_id: int, db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """Retrieve pages of a specific paper."""
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM paper_pages WHERE paper_id = ? ORDER BY page_number ASC",
            (paper_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_paper_metadata(
    paper_id: int, updates: Dict[str, Any], db_path: Optional[Union[str, Path]] = None
) -> None:
    """Update metadata for an existing paper."""
    allowed_keys = [
        "title", "authors", "year", "venue", "doi", "abstract",
        "citation_id", "metadata_json", "paper_card_json"
    ]
    fields = []
    params: List[Any] = []
    for k, v in updates.items():
        if k in allowed_keys:
            fields.append(f"{k} = ?")
            if k in ("authors", "metadata_json", "paper_card_json") and not isinstance(v, str):
                params.append(json.dumps(v))
            else:
                params.append(v)
    if not fields:
        return
    params.append(paper_id)
    with _get_connection(db_path) as conn:
        conn.execute(f"UPDATE papers SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()


def update_paper_card(
    paper_id: int, card_dict: Dict[str, Any], db_path: Optional[Union[str, Path]] = None
) -> None:
    """Store or update the structured Paper Card JSON for a paper."""
    update_paper_metadata(paper_id, {"paper_card_json": card_dict}, db_path=db_path)


def delete_paper(paper_id: int, db_path: Optional[Union[str, Path]] = None) -> None:
    """Delete a paper record."""
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        conn.commit()


def reorder_citations(
    project_id: int, paper_ids_in_order: List[int], db_path: Optional[Union[str, Path]] = None
) -> None:
    """Reorder citation IDs for a project to [1, 2, ...]."""
    with _get_connection(db_path) as conn:
        for idx, pid in enumerate(paper_ids_in_order, start=1):
            conn.execute(
                "UPDATE papers SET citation_id = ? WHERE id = ? AND project_id = ?",
                (idx, pid, project_id),
            )
        conn.commit()


# ==================== CROSS-PAPER ANALYSIS ====================


def save_analysis(
    project_id: int,
    analysis_type: str,
    data: Dict[str, Any],
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Save or update cross-paper analysis (e.g. taxonomy, gaps, key_claims)."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO analysis (project_id, analysis_type, data_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id, analysis_type) DO UPDATE SET
                data_json = excluded.data_json,
                updated_at = excluded.updated_at
            """,
            (project_id, analysis_type, json.dumps(data), time.time()),
        )
        conn.commit()


def get_analysis(
    project_id: int, analysis_type: str, db_path: Optional[Union[str, Path]] = None
) -> Optional[Dict[str, Any]]:
    """Retrieve cross-paper analysis by type."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT data_json FROM analysis WHERE project_id = ? AND analysis_type = ?",
            (project_id, analysis_type),
        ).fetchone()
        if row:
            try:
                return json.loads(row["data_json"])
            except Exception:
                return None
    return None


# ==================== REVIEW SECTIONS ====================


def save_section(
    project_id: int,
    section_name: str,
    content: str,
    similarity_score: float = 0.0,
    similarity_report: Optional[Dict[str, Any]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Save or update a drafted review section."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO review_sections (project_id, section_name, content, similarity_score, similarity_report_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, section_name) DO UPDATE SET
                content = excluded.content,
                similarity_score = excluded.similarity_score,
                similarity_report_json = excluded.similarity_report_json,
                updated_at = excluded.updated_at
            """,
            (
                project_id,
                section_name,
                content,
                similarity_score,
                json.dumps(similarity_report or {}),
                time.time(),
            ),
        )
        conn.commit()


def get_sections(
    project_id: int, db_path: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, Any]]:
    """Retrieve all drafted review sections for a project as a dict keyed by section_name."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM review_sections WHERE project_id = ? ORDER BY id ASC",
            (project_id,),
        ).fetchall()
        result = {}
        for r in rows:
            data = dict(r)
            try:
                data["similarity_report_json"] = json.loads(data["similarity_report_json"] or "{}")
            except Exception:
                data["similarity_report_json"] = {}
            result[r["section_name"]] = data
        return result


# ==================== ASK THE PAPERS CHAT ====================


def save_chat_message(
    project_id: int,
    role: str,
    message: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Record a chat message in the grounded Q&A thread."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chat_history (project_id, role, message, citations_json, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (project_id, role, message, json.dumps(citations or []), time.time()),
        )
        conn.commit()


def get_chat_history(
    project_id: int, db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """Retrieve chat history for the project."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM chat_history WHERE project_id = ? ORDER BY timestamp ASC",
            (project_id,),
        ).fetchall()
        messages = []
        for r in rows:
            item = dict(r)
            try:
                item["citations_json"] = json.loads(item["citations_json"] or "[]")
            except Exception:
                item["citations_json"] = []
            messages.append(item)
        return messages


def clear_chat_history(project_id: int, db_path: Optional[Union[str, Path]] = None) -> None:
    """Clear chat messages for a project."""
    with _get_connection(db_path) as conn:
        conn.execute("DELETE FROM chat_history WHERE project_id = ?", (project_id,))
        conn.commit()
