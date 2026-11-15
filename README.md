# ReviewForge 🔬
**Free, Lightweight, AI-Assisted Academic Review Paper Engine with Plagiarism-Safe Verification**

ReviewForge is a complete, publication-grade web application built in Python and Streamlit that assists researchers in synthesizing comprehensive literature review papers from uploaded PDF corpora. It operates with **zero heavy ML models, no GPU requirements, and minimal RAM**, using Google Gemini as its sole AI provider alongside local n-gram and fuzzy similarity algorithms.

---

## 🌟 Key Features

### 1. Robust PDF Ingestion & Multi-Stage Cleaning
- **Dual-Engine Extraction**: Fast page-by-page extraction via **PyMuPDF (`fitz`)** with automatic fallback to **`pypdf`**.
- **Automated Text Cleaning**:
  - Removes recurring headers, footers, and page numbers across pages.
  - Reconstructs hyphenated line-breaks (e.g. `algo-\nrithm` $\rightarrow$ `algorithm`).
  - Separates reference lists into a distinct field to keep body text clean.
- **Scanned Document Detection**: Detects image-only or scanned PDFs without selectable text and warns the user that OCR is required.
- **Metadata Parser**: Combines regex heuristics with a first-page Gemini call to identify Title, Authors, Year, Venue, DOI, and Abstract.

### 2. Section-Aware Map-Reduce Chunking
- Splits lengthy papers into section-aware chunks (~3,000–4,000 tokens, 10% overlap) along heading boundaries (`1. Introduction`, `II. RELATED WORK`, etc.).
- **Map Step**: Extracts structured notes per chunk (problems, methods, benchmarks, metrics, findings, limitations).
- **Reduce Step**: Consolidates chunk notes into a unified **Paper Card**.
- **Lite Mode**: Summarizes short papers in a single step to conserve API quota.

### 3. Structured Paper Cards (JSON Enforced)
Each paper is distilled into an authoritative schema:
- **Problem & Objective**: Specific research questions and hypotheses.
- **Methodology & Architecture**: Algorithms, models, and mathematical formulations.
- **Datasets & Metrics**: Benchmark environments and quantitative evaluation metrics.
- **Key Quantitative Results**: Empirical findings with concrete numbers.
- **Contributions & Limitations**: Stated limitations and inferred theoretical risks.
- **Future Directions & Keywords**: Open challenges and domain tags.
- **Quality Notes**: Strengths and weaknesses / threats to validity.
- **Evidence Snippets**: 3 to 5 verbatim snippets ($\le$ 25 words) tagged with page numbers.

### 4. Cross-Paper Deep Analysis & Research Gap Discovery
- **Comparative Matrix**: Tabular comparison of method, dataset, metrics, results, and limitations.
- **Taxonomy & Evolution**: Categorizes paradigms into architectural families and generates a chronological milestone timeline.
- **5-Dimension Research Gap Finder**:
  1. *Contradictions & Conflicting Findings* across papers.
  2. *Untested Datasets / Cross-Domain Generalization* gaps.
  3. *Missing Comparisons / Baseline Evaluations*.
  4. *Common Recurring Limitations*.
  5. *Unaddressed Future-Work Items*.
- **Gap Ranking**: Ranks gaps by composite Novelty and Feasibility scores.
- **Research Question Formulation**: Automatically generates testable hypotheses, sub-questions, and methodological roadmaps from any chosen gap.

### 5. Review Paper Generation (Section-by-Section)
- Drafts all standard review sections: *Abstract, Index Terms, Introduction, Background & Taxonomy, Thematic Literature Review, Comparative Analysis, Research Gaps & Challenges, Future Directions, Conclusion, and References*.
- **Strict Synthesis Rules**: Gemini is instructed to synthesize ideas across papers, never copy sentences, never quote more than 10 words verbatim, and maintain an academic voice.
- **Authoritative Citation Mapping**: Every claim is attributed to a bracketed citation marker (`[1]`, `[2]`) mapped strictly to the uploaded corpus. References are formatted deterministically from uploaded papers to eliminate phantom citations.

### 6. Plagiarism-Safe Architecture (Top Priority)
- **Local Overlap Audit (Free, No GPU)**:
  - Computes word 5-gram and 8-gram exact overlaps against the full text of all uploaded PDFs.
  - Computes fuzzy sentence similarity (`rapidfuzz.fuzz.partial_ratio`) against corpus sentences.
  - Categorizes similarity into status bands: **Green ($\le$15%)**, **Yellow (15%–25%)**, and **Red (>25%)**.
- **Quote Control**: Quoted text in double quotes (`"..." [1]`) is excluded from similarity penalties.
- **Anti-Plagiarism Auto-Rewrite Loop**: Flagged sentences are automatically rewritten by Gemini (inverting clauses and restructuring syntax while preserving scientific meaning and citations) and re-audited up to 3 times.
- **Self-Overlap Checker**: Detects accidental phrase duplication across different sections of the review paper.
- **Honest Academic Disclaimer**: Reminds researchers to run final manuscripts through institutional checkers (Turnitin/iThenticate) and adhere to publisher AI disclosure policies.

### 7. Multi-Format Publication Export
- **IEEE-Format `.docx`**: Two-column layout, 22pt title, author block, italicized abstract/index terms, Roman numeral headings (`I. INTRODUCTION`), single-spaced Times New Roman font, and numbered references.
- **Markdown (`.md`)**: Full structured paper with section hierarchy.
- **BibTeX (`.bib`)**: Standard citations for all papers in the corpus.
- **AI Methodology Appendix**: Disclosure statement on AI synthesis and verification tools.
- **Audit Report (`.md`)**: Comprehensive similarity breakdown for peer review records.

### 8. "Ask the Papers" Grounded Q&A Chat
- Interactive Q&A chat grounded **exclusively** in the uploaded PDFs.
- Attributes all answers with specific paper IDs and page numbers (e.g. `[1, p. 4]`).
- Explicitly flags when information cannot be verified from the corpus.

---

## 🏗️ Project Architecture

```
paper/
├── app.py                     # Streamlit multi-tab user interface
├── config.py                  # Global settings, model chains, rate limits
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variables template
├── README.md                  # Complete documentation and setup guide
├── core/
│   ├── __init__.py
│   ├── keys.py                # KeyManager: round-robin, 60s cooldowns, model fallback
│   ├── llm.py                 # Gemini client: SQLite caching, rate limiting, retry backoff
│   ├── pdf_extract.py         # PyMuPDF/pypdf extraction, header/footer cleaning, metadata regex
│   ├── chunker.py             # Section-aware heading chunker (3k-4k tokens, 10% overlap)
│   ├── paper_card.py          # Map-reduce chunk summarization into structured JSON cards
│   ├── analysis.py            # Comparison table, taxonomy categorization, timeline trends
│   ├── gaps.py                # 5-dimension gap finder & research question generator
│   ├── writer.py              # Section drafting & anti-plagiarism auto-rewrite loop
│   ├── similarity.py          # Local 5/8-gram overlap and RapidFuzz fuzzy sentence matching
│   ├── ieee_export.py         # IEEE two-column .docx, Markdown, BibTeX, & audit exports
│   └── db.py                  # SQLite storage for projects, papers, sections, cache, chat
├── prompts/                   # Externalized, versioned LLM prompt templates
│   ├── extract_metadata.txt   # First-page paper metadata parsing
│   ├── chunk_summary.txt      # Map step chunk note extraction
│   ├── paper_card_merge.txt   # Reduce step Paper Card consolidation
│   ├── taxonomy.txt           # Methodological grouping & trend timeline
│   ├── gaps.txt               # Research gap discovery across 5 dimensions
│   ├── gap_rq.txt             # Formal research question & hypothesis formulation
│   ├── key_claims.txt         # Foundational claims for Introduction
│   ├── section_writer.txt     # Section synthesizer with strict anti-copying rules
│   ├── sentence_rewrite.txt   # Sentence-level syntax restructuring
│   └── ask_papers.txt         # Grounded corpus Q&A prompt
└── tests/                     # Comprehensive test suite
    ├── __init__.py
    ├── test_chunker.py        # Token estimation & heading boundary chunking tests
    ├── test_keys.py           # KeyManager failover, cooldowns, & model fallback tests
    ├── test_similarity.py     # Sentence splitting, n-gram overlap, & quote exclusion tests
    └── test_pipeline.py       # End-to-end PDF extraction and IEEE .docx export tests
```

---

## 🔑 KeyManager & Resilience Strategy

1. **Multi-Key Round Robin**:
   - Provide 1 to 5 keys in `.env` as `GEMINI_KEYS=key1,key2,key3`.
   - Distributes API requests across keys sequentially.
2. **Automatic Cooldown on Errors**:
   - On `429 (Resource Exhausted)`, `Quota Exceeded`, `5xx`, or `Auth` errors, the failing key is immediately placed on a **60-second cooldown**.
   - Consecutive failures double the cooldown time.
   - Automatically switches to the next healthy key in the pool.
3. **Model Fallback Chain**:
   - Primary: `gemini-2.5-flash`
   - Fallback: `gemini-2.5-flash-lite` (configurable in `config.py`).
4. **Persistent SQLite Caching**:
   - Every LLM prompt and response is hashed (`SHA-256(model + prompt)`) and stored in SQLite.
   - Re-running analysis or loading previously drafted sections incurs **0 API calls** and **0 latency**.
5. **Rate Limiting**:
   - Global thread-safe sliding-window rate limiter prevents bursts and keeps usage safely within the free-tier quota (default: 15 RPM).

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11 or 3.12
- Google Gemini API key (at least 1 key; 2–3 recommended for higher free throughput)

### 2. Installation

Clone or enter the directory:
```bash
cd d:/test/paper
```

Create and activate a virtual environment:
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key(s):
```env
GEMINI_KEYS=your_first_gemini_key,your_second_gemini_key
GEMINI_RPM=15
GEMINI_PRIMARY_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-2.5-flash-lite
SIMILARITY_THRESHOLD=0.75
```
*(You can also input or override keys directly within the Streamlit sidebar at runtime).*

### 4. Running the Application

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

The web application will open in your browser at `http://localhost:8501`.

---

## 🧪 Running Unit & Integration Tests

The project includes an automated test suite verifying chunking logic, KeyManager failover under simulated 429 errors, RapidFuzz similarity checking, PDF ingestion, and IEEE `.docx` generation.

Run pytest:
```bash
pytest -v tests
```

All 18 unit and integration tests execute in under 5 seconds with zero external network dependencies.

---

## 🌐 Deployment Guidelines

### Deploying to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, set branch to `main`, and main file path to `app.py`.
4. Under **Advanced Settings $\rightarrow$ Secrets**, enter your Gemini keys:
   ```toml
   GEMINI_KEYS = "AIzaSyKey1,AIzaSyKey2"
   GEMINI_RPM = "15"
   ```
5. Click **Deploy**.

### Deploying via Docker
A minimal `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and run:
```bash
docker build -t reviewforge .
docker run -p 8501:8501 -e GEMINI_KEYS="your_key_here" reviewforge
```

---

## 📜 Ethical AI & Academic Integrity Notice

ReviewForge is designed strictly as an assistive synthesis tool for researchers.
- Authors retain sole responsibility for reviewing, fact-checking, and validating all synthesized claims, citations, and data figures.
- ReviewForge deterministic citation mapping guarantees that references map only to uploaded files and never invent phantom citations.
- The built-in local plagiarism audit checks against the uploaded corpus; authors must submit final manuscripts to their target institution/venue's licensed plagiarism detection system (Turnitin, iThenticate) and append the generated AI usage disclosure per venue policies.

---

## 📄 License
This project is licensed under the Apache 2.0 License.
