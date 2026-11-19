"""IEEE-format PDF export.

Produces a two-column, Latin-Modern-typeset PDF that mirrors the layout of the
project's reference paper (US-Letter, ~0.67 in margins, 10 pt / 12 pt body,
small-caps section headings, 7 pt tables, numbered display/inline equations,
figures and simple diagrams), and targets an exact page count.

Public entry point: :func:`generate_ieee_pdf`.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

try:  # optional hyphenation (improves justified narrow columns)
    import pyphen  # type: ignore

    _HYPHEN_DICT = pyphen.Pyphen(lang="en_US")
except Exception:  # pragma: no cover - hyphenation is a nicety, not a requirement
    _HYPHEN_DICT = None

logger = logging.getLogger(__name__)

# matplotlib needs a writable cache/config directory; on managed hosts (e.g.
# Render) the default may be read-only, so point it at a temp dir up front.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="rf_mpl_"))

# ---------------------------------------------------------------------------
# Geometry (points) — measured from the reference document
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = 612.0, 792.0          # US Letter
MARGIN_LEFT = 48.6
MARGIN_RIGHT = 47.4
MARGIN_TOP = 51.0
BOTTOM_Y = 73.5                         # distance of text bottom from page bottom
COL_WIDTH = 253.0
COL_GAP = 10.0
FULL_WIDTH = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT
COL1_X = MARGIN_LEFT
COL2_X = MARGIN_LEFT + COL_WIDTH + COL_GAP

HEADER_BOTTOM = 454.0                   # where columns begin on page 1 (top y in pt)

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts")

ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

# Vertical-stretch search bounds for the exact-page-count fitter. 1.0 reproduces
# the reference leading exactly (10 pt / 12 pt); the band around it lets the
# fitter trade a little vertical density for page count without changing fonts.
MIN_STRETCH = 0.86
MAX_STRETCH = 1.15


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
_FONTS_REGISTERED = False


def _register_fonts() -> bool:
    """Register the bundled Latin Modern TTFs with ReportLab. Returns success."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return True

    family = {
        "LM": "lmroman10-regular.ttf",
        "LM-Bold": "lmroman10-bold.ttf",
        "LM-Italic": "lmroman10-italic.ttf",
        "LM-BoldItalic": "lmroman10-bolditalic.ttf",
        "LM-Caps": "lmromancaps10-regular.ttf",
        "LM-Title": "lmroman17-regular.ttf",
    }
    for name, fname in family.items():
        path = os.path.join(FONT_DIR, fname)
        if not os.path.exists(path):
            logger.warning("Missing font file: %s", path)
            return False
        pdfmetrics.registerFont(TTFont(name, path))

    pdfmetrics.registerFontFamily(
        "LM", normal="LM", bold="LM-Bold", italic="LM-Italic", boldItalic="LM-BoldItalic"
    )
    _FONTS_REGISTERED = True
    return True


# ---------------------------------------------------------------------------
# Math / chart rendering (matplotlib, Agg backend)
# ---------------------------------------------------------------------------
_MATH_DPI = 600
_RENDER_DIR = tempfile.mkdtemp(prefix="rf_pdf_")


def _reset_render_cache() -> None:
    """Drop cached renders and clear the scratch image directory."""
    _render_mathtext.cache_clear()
    _render_math_safe.cache_clear()
    _render_chart.cache_clear()
    _render_diagram.cache_clear()
    try:
        for name in os.listdir(_RENDER_DIR):
            p = os.path.join(_RENDER_DIR, name)
            if os.path.isfile(p):
                os.remove(p)
    except OSError:
        pass


def _new_image_path(suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, dir=_RENDER_DIR)
    os.close(fd)
    return path


@lru_cache(maxsize=256)
def _render_mathtext(latex: str, fontsize: float) -> Optional[Tuple[str, float, float]]:
    """Render a LaTeX math snippet to a transparent PNG. Returns (path, w_pt, h_pt)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["mathtext.default"] = "it"
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0.0, 0.0, f"${latex}$", fontsize=fontsize)
        tmp = _new_image_path(".png")
        fig.savefig(
            tmp, dpi=_MATH_DPI, transparent=True, bbox_inches="tight", pad_inches=0.01
        )
        plt.close(fig)

        from PIL import Image as PILImage  # matplotlib depends on Pillow

        with PILImage.open(tmp) as im:
            px_w, px_h = im.size
        w_pt = px_w * 72.0 / _MATH_DPI
        h_pt = px_h * 72.0 / _MATH_DPI
        return tmp, w_pt, h_pt
    except Exception as e:  # pragma: no cover
        logger.warning("Math render failed for %r: %s", latex, e)
        return None


@lru_cache(maxsize=64)
def _render_chart(kind: str, rows: Tuple[Tuple[str, float], ...], caption: str = "") -> Optional[str]:
    """Render a simple chart (bar/line/pie) to a PNG. Returns the file path."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r[0] for r in rows]
        values = [r[1] for r in rows]

        fig, ax = plt.subplots(figsize=(3.6, 2.3), dpi=300)
        if kind == "pie":
            ax.pie(values, labels=labels, autopct="%1.0f%%", textprops={"fontsize": 6})
            ax.axis("equal")
        elif kind == "line":
            ax.plot(labels, values, marker="o", linewidth=1.0, markersize=3)
            ax.tick_params(labelsize=6)
            ax.grid(True, linewidth=0.3, alpha=0.5)
        else:  # bar (default)
            ax.bar(labels, values, color="#4a4a4a")
            ax.tick_params(labelsize=6)
            ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
        if kind in ("bar", "line") and len(labels) > 4:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_ha("right")
        fig.tight_layout(pad=0.2)

        tmp = _new_image_path(".png")
        fig.savefig(tmp, dpi=300, bbox_inches="tight", transparent=False)
        plt.close(fig)
        return tmp
    except Exception as e:  # pragma: no cover
        logger.warning("Chart render failed: %s", e)
        return None


@lru_cache(maxsize=64)
def _render_diagram(nodes: Tuple[str, ...]) -> Optional[str]:
    """Render a boxed process diagram; wraps onto multiple rows for many stages."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        clean = [n.strip() for n in nodes if n.strip()]
        n = len(clean)
        if n == 0:
            return None
        per_row = 4 if n > 4 else n
        rows = (n + per_row - 1) // per_row
        fig_w = min(3.7, 0.95 * per_row + 0.45)
        fig_h = 0.8 * rows + 0.2
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        ax.set_xlim(0, per_row)
        ax.set_ylim(0, rows)
        ax.axis("off")

        for i, label in enumerate(clean):
            r, c = divmod(i, per_row)
            y0 = rows - 1 - r
            ax.add_patch(
                FancyBboxPatch(
                    (c + 0.10, y0 + 0.25), 0.80, 0.50,
                    boxstyle="round,pad=0.01,rounding_size=0.06",
                    linewidth=0.7, edgecolor="#333333", facecolor="#f2f4f7",
                )
            )
            ax.text(c + 0.5, y0 + 0.5, label, ha="center", va="center", fontsize=6, wrap=True)
            if i < n - 1:
                if c < per_row - 1:
                    ax.add_patch(
                        FancyArrowPatch(
                            (c + 0.92, y0 + 0.5), (c + 1.08, y0 + 0.5),
                            arrowstyle="-|>", mutation_scale=8, linewidth=0.7, color="#333333",
                        )
                    )
                else:  # wrap to the next row
                    ax.add_patch(
                        FancyArrowPatch(
                            (c + 0.5, y0 + 0.24), (0.5, y0 - 0.24),
                            arrowstyle="-|>", mutation_scale=8, linewidth=0.7, color="#333333",
                            connectionstyle="arc3,rad=-0.35",
                        )
                    )
        tmp = _new_image_path(".png")
        fig.savefig(tmp, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        return tmp
    except Exception as e:  # pragma: no cover
        logger.warning("Diagram render failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# LaTeX normalisation (mathtext supports only a subset of LaTeX)
# ---------------------------------------------------------------------------
_ENV_RE = re.compile(
    r"\\begin\{(?P<env>[a-zA-Z*]+)\}(?:\{[^{}]*\})?(?P<body>.*?)\\end\{(?P=env)\}",
    re.DOTALL,
)
_DELIMITED_ENVS = {
    "matrix": (None, None),
    "smallmatrix": (None, None),
    "array": (None, None),
    "bmatrix": (r"\left[", r"\right]"),
    "pmatrix": (r"\left(", r"\right)"),
    "Bmatrix": (r"\left\{", r"\right\}"),
    "vmatrix": (r"\left|", r"\right|"),
    "Vmatrix": (r"\left\|", r"\right\|"),
}
_ROW_SEP = "\x00"  # sentinel for stack-row separators we insert ourselves


def _strip_dollars(tex: str) -> str:
    """Remove any surrounding ``$`` / ``$$`` math delimiters the model added."""
    t = tex.strip()
    while True:
        if t.startswith("$$") and t.endswith("$$") and len(t) >= 4:
            t = t[2:-2].strip()
        elif t.startswith("$") and t.endswith("$") and len(t) >= 2:
            t = t[1:-1].strip()
        else:
            break
    return t


def _rows_to_substack(body: str) -> str:
    """Flatten a LaTeX environment body into a single mathtext expression."""
    rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
    if not rows:
        return ""
    joined = [r" \quad ".join(c.strip() for c in row.split("&")) for row in rows]
    if len(joined) == 1:
        return joined[0]
    return r"\substack{" + (f" {_ROW_SEP} ").join(joined) + "}"


def _convert_environments(tex: str) -> str:
    """Rewrite LaTeX environments mathtext cannot parse into stackable forms."""

    def repl(m: "re.Match") -> str:
        env = m.group("env")
        inner = _rows_to_substack(m.group("body"))
        if env in _DELIMITED_ENVS:
            left, right = _DELIMITED_ENVS[env]
            return f"{left} {inner} {right}" if left else inner
        if env == "cases":
            return r"\left\{ " + inner + r" \right."
        return inner

    return _ENV_RE.sub(repl, tex)


def _sanitize_latex(tex: str) -> str:
    """Best-effort conversion of arbitrary LaTeX into mathtext-safe input."""
    t = tex
    # Drop control words mathtext rejects outright.
    t = re.sub(
        r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle|limits|nolimits"
        r"|nonumber|notag|hfill|middle|mathstrut)(?![a-zA-Z])",
        "",
        t,
    )
    t = re.sub(r"\\(?:tag|label)\{[^{}]*\}", "", t)
    t = re.sub(r"\\(?:hspace|vspace|phantom|hphantom|vphantom)\{[^{}]*\}", "", t)
    # Strip surrounding $ / $$ delimiters *after* removing \tag/\label so the
    # closing delimiter is still adjacent to the end of the string at this point.
    t = _strip_dollars(t)
    t = re.sub(r"\\big[lrm]?(?![a-zA-Z])|\\Big[lrm]?(?![a-zA-Z])|\\bigg[lrm]?(?![a-zA-Z])|\\Bigg[lrm]?(?![a-zA-Z])", "", t)
    t = t.replace("\\!", "")
    # Alias shorthands mathtext does not know to the ones it does.
    t = re.sub(r"\\le(?![a-zA-Z])", r"\\leq", t)
    t = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq", t)
    t = re.sub(r"\\ne(?![a-zA-Z])", r"\\neq", t)
    t = re.sub(r"\\lt(?![a-zA-Z])", "<", t)
    t = re.sub(r"\\gt(?![a-zA-Z])", ">", t)
    t = re.sub(r"\\to(?![a-zA-Z])", r"\\rightarrow", t)
    t = re.sub(r"\\bm(?![a-zA-Z])", r"\\boldsymbol", t)
    # Turn unsupported environments into stackable expressions.
    t = _convert_environments(t)
    # Any remaining top-level line break becomes a stack as well.
    if "\\\\" in t:
        parts = [p.strip() for p in re.split(r"\\\\", t) if p.strip()]
        if len(parts) > 1:
            t = r"\substack{" + (f" {_ROW_SEP} ").join(parts) + "}"
    t = t.replace(_ROW_SEP, r" \\ ")
    return t.strip()


def _looks_like_math(inner: str) -> bool:
    """Guard against treating currency or ordinary prose as inline math."""
    s = inner.strip()
    if not s:
        return False
    if "\\" in s or re.search(r"[=_^{}]", s):
        return True
    if " " in s:
        return False
    if re.fullmatch(r"[\d.,%]+", s):  # bare numbers/amounts are not math
        return False
    return bool(re.search(r"[A-Za-z0-9]", s))


@lru_cache(maxsize=512)
def _render_math_safe(latex: str, fontsize: float) -> Optional[Tuple[str, float, float]]:
    """Sanitise then render a LaTeX snippet; ``None`` if it still cannot render."""
    sanitized = _sanitize_latex(latex)
    if not sanitized:
        return None
    return _render_mathtext(sanitized, fontsize)


# ---------------------------------------------------------------------------
# Inline markdown / math -> ReportLab markup
# ---------------------------------------------------------------------------
_TYPOGRAPHIC_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2032": "'", "\u2033": '"', "\u02bc": "'",
}


def _normalize_typography(text: str) -> str:
    """Replace typographic quotes/apostrophes with plain ASCII equivalents."""
    for src, dst in _TYPOGRAPHIC_MAP.items():
        if src in text:
            text = text.replace(src, dst)
    return text


def _md_inline(text: str, inline_math: bool = True) -> str:
    """Convert a line of markdown to ReportLab paragraph markup (XML-escaped)."""
    # Escape first so user/model text cannot inject markup...
    t = escape(_normalize_typography(text))
    # ...then re-introduce the tags we control. Bold is never used in body text,
    # so drop the markers but keep the words; italic is kept (journal titles).
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', t)

    if inline_math:
        def _math_sub(m: "re.Match") -> str:
            inner = m.group(1)
            if not _looks_like_math(inner):
                return m.group(0)
            rendered = _render_math_safe(inner, 10.0)
            if not rendered:
                plain = re.sub(r"\\[a-zA-Z]+", "", inner).replace("{", "").replace("}", "").strip()
                return f"<i>{plain}</i>" if plain else ""
            path, w, h = rendered
            src = path.replace("\\", "/")
            return f'<img src="{src}" width="{w:.1f}" height="{h:.1f}" valign="-1.5"/>'

        # Word-boundary guards keep a currency ``$5`` from pairing with the
        # next real delimiter and swallowing the following inline math.
        t = re.sub(r"(?<![\w$])\$([^$\n]+?)\$(?!\w)", _math_sub, t)
    return t


# ---------------------------------------------------------------------------
# Content parsing
# ---------------------------------------------------------------------------
FENCE_RE = re.compile(r"```(\w+)(?:\s*:\s*([^\n]*))?\n(.*?)```", re.DOTALL)


def _parse_blocks(content: str) -> List[Dict[str, Any]]:
    """Split a section's content into typed blocks (fenced elements + prose)."""
    blocks: List[Dict[str, Any]] = []
    pos = 0
    for m in FENCE_RE.finditer(content):
        if m.start() > pos:
            _parse_prose(content[pos:m.start()], blocks)
        kind = m.group(1).strip().lower()
        caption = (m.group(2) or "").strip()
        body = m.group(3).strip()
        blocks.append({"type": kind, "caption": caption, "body": body})
        pos = m.end()
    if pos < len(content):
        _parse_prose(content[pos:], blocks)
    return blocks


def _parse_prose(text: str, blocks: List[Dict[str, Any]]) -> None:
    """Split free text into heading and paragraph blocks."""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        first = para.split("\n", 1)[0].strip()
        if first.startswith("#"):
            level = len(first) - len(first.lstrip("#"))
            blocks.append({"type": "heading", "level": level, "text": first.lstrip("#").strip()})
            rest = para.split("\n", 1)[1].strip() if "\n" in para else ""
            if rest:
                blocks.append({"type": "paragraph", "text": rest})
        else:
            blocks.append({"type": "paragraph", "text": " ".join(para.split())})


# ---------------------------------------------------------------------------
# Custom flowable: numbered equation
# ---------------------------------------------------------------------------
class EquationFlowable(Flowable):
    """A centred equation image with a right-aligned equation number."""

    def __init__(self, path: str, w: float, h: float, number: int, max_width: float):
        super().__init__()
        self.path = path
        self.w = w
        self.h = h
        self.number = number
        self.max_width = max_width

    def wrap(self, availWidth, availHeight):  # noqa: N802
        self._avail = availWidth
        scale = min(1.0, (availWidth - 24) / self.w) if self.w else 1.0
        self._dw = self.w * scale
        self._dh = self.h * scale
        return availWidth, self._dh + 6

    def draw(self):
        from reportlab.pdfbase.pdfmetrics import stringWidth

        self.canv.saveState()
        x = (self._avail - self._dw) / 2.0
        self.canv.drawImage(
            self.path, x, 3, width=self._dw, height=self._dh, mask="auto"
        )
        label = f"({self.number})"
        self.canv.setFont("LM", 10)
        lw = stringWidth(label, "LM", 10)
        self.canv.drawString(self._avail - lw, self._dh / 2.0, label)
        self.canv.restoreState()


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _make_styles(stretch: float = 1.0) -> Dict[str, ParagraphStyle]:
    body_leading = 12.0 * stretch
    return {
        "title": ParagraphStyle(
            "title", fontName="LM-Title", fontSize=23.9, leading=28 * stretch,
            alignment=TA_CENTER, spaceAfter=10,
        ),
        "author": ParagraphStyle(
            "author", fontName="LM-Bold", fontSize=11, leading=13 * stretch, alignment=TA_CENTER,
        ),
        "affil": ParagraphStyle(
            "affil", fontName="LM-Italic", fontSize=8, leading=10 * stretch, alignment=TA_CENTER,
        ),
        "abstract": ParagraphStyle(
            "abstract", fontName="LM", fontSize=9, leading=10.5 * stretch, alignment=TA_JUSTIFY,
            firstLineIndent=0, spaceAfter=4 * stretch,
        ),
        "h1": ParagraphStyle(
            "h1", fontName="LM-Caps", fontSize=10, leading=12 * stretch, alignment=TA_CENTER,
            spaceBefore=10 * stretch, spaceAfter=4 * stretch, keepWithNext=1,
        ),
        "h2": ParagraphStyle(
            "h2", fontName="LM-Italic", fontSize=10, leading=12 * stretch, alignment=TA_LEFT,
            spaceBefore=8 * stretch, spaceAfter=3 * stretch, keepWithNext=1,
        ),
        "body": ParagraphStyle(
            "body", fontName="LM", fontSize=10, leading=body_leading, alignment=TA_JUSTIFY,
            firstLineIndent=12, spaceAfter=1.5 * stretch,
            hyphenationLang="en_US" if _HYPHEN_DICT else None,
            embeddedHyphenation=1,
        ),
        "body_first": ParagraphStyle(
            "body_first", fontName="LM", fontSize=10, leading=body_leading, alignment=TA_JUSTIFY,
            firstLineIndent=0, spaceAfter=1.5 * stretch,
        ),
        "ref": ParagraphStyle(
            "ref", fontName="LM", fontSize=8, leading=9.5 * stretch, alignment=TA_JUSTIFY,
            firstLineIndent=-10, leftIndent=10, spaceAfter=2 * stretch,
        ),
        "tbl_caption": ParagraphStyle(
            "tbl_caption", fontName="LM-Caps", fontSize=8, leading=10 * stretch, alignment=TA_CENTER,
            spaceBefore=4 * stretch, spaceAfter=2,
        ),
        "tbl_label": ParagraphStyle(
            "tbl_label", fontName="LM", fontSize=8, leading=10, alignment=TA_CENTER,
        ),
        "tbl_cell": ParagraphStyle(
            "tbl_cell", fontName="LM", fontSize=7, leading=8.5, alignment=TA_CENTER,
        ),
        "tbl_head": ParagraphStyle(
            "tbl_head", fontName="LM-Bold", fontSize=7, leading=8.5, alignment=TA_CENTER,
        ),
        "fig_caption": ParagraphStyle(
            "fig_caption", fontName="LM", fontSize=8, leading=10 * stretch, alignment=TA_JUSTIFY,
            spaceBefore=2, spaceAfter=4 * stretch,
        ),
        "eq_placeholder": ParagraphStyle(
            "eq_placeholder", fontName="LM-Italic", fontSize=8, leading=10 * stretch,
            alignment=TA_CENTER, spaceBefore=3 * stretch, spaceAfter=3 * stretch,
        ),
    }


# ---------------------------------------------------------------------------
# Table / figure builders
# ---------------------------------------------------------------------------
def _build_table(block: Dict[str, Any], styles: Dict[str, ParagraphStyle], index: int) -> List[Any]:
    rows_raw = [r for r in block["body"].splitlines() if r.strip()]
    grid: List[List[str]] = []
    for r in rows_raw:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue  # markdown separator row
        grid.append(cells)
    if not grid:
        return []

    ncol = max(len(r) for r in grid)
    grid = [r + [""] * (ncol - len(r)) for r in grid]

    header = [Paragraph(_md_inline(c, inline_math=False), styles["tbl_head"]) for c in grid[0]]
    data = [header]
    for r in grid[1:]:
        data.append([Paragraph(_md_inline(c, inline_math=False), styles["tbl_cell"]) for c in r])

    col_w = COL_WIDTH / ncol
    table = Table(data, colWidths=[col_w] * ncol, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    caption_text = block.get("caption") or "Summary of compared studies"
    label = Paragraph(f"TABLE {ROMAN_NUMERALS[index] if index < len(ROMAN_NUMERALS) else index + 1}", styles["tbl_label"])
    cap = Paragraph(_md_inline(caption_text, inline_math=False), styles["tbl_caption"])
    return [KeepTogether([label, cap, Spacer(1, 3), table])]


def _chart_rows(body: str, caption: str) -> Tuple[str, List[Tuple[str, float]], str]:
    """Parse a chart block into (kind, rows, caption)."""
    chart_kind = "bar"
    caption_text = caption
    if caption and ":" in caption:
        head, tail = caption.split(":", 1)
        if head.strip().lower() in ("bar", "line", "pie"):
            chart_kind = head.strip().lower()
            caption_text = tail.strip()
    rows: List[Tuple[str, float]] = []
    for line in body.splitlines():
        if ":" in line:
            lab, val = line.rsplit(":", 1)
            try:
                rows.append((lab.strip(), float(val.strip())))
            except ValueError:
                continue
    return chart_kind, rows, caption_text


def _diagram_nodes(body: str) -> List[str]:
    """Split diagram stages on arrows, pipes or newlines."""
    return [n.strip() for n in re.split(r"->|=>|\u2192|\||\n", body) if n.strip()]


def _looks_like_chart_body(body: str) -> bool:
    """True when every line looks like ``Label: number`` (i.e. chart data)."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    for line in lines:
        if ":" not in line:
            return False
        try:
            float(line.rsplit(":", 1)[1].strip())
        except ValueError:
            return False
    return True


def _build_figure(block: Dict[str, Any], styles: Dict[str, ParagraphStyle], index: int) -> List[Any]:
    kind = block["type"]
    caption_text = block.get("caption") or ""
    path: Optional[str] = None

    if kind == "figure":  # generic figure: infer chart vs diagram
        kind = "chart" if _looks_like_chart_body(block["body"]) else "diagram"

    if kind == "chart":
        chart_kind, rows, caption_text = _chart_rows(block["body"], caption_text)
        if rows:
            path = _render_chart(chart_kind, tuple(rows))
    elif kind == "diagram":
        nodes = _diagram_nodes(block["body"])
        if nodes:
            path = _render_diagram(tuple(nodes))

    if not path:
        note = (caption_text or block.get("body", "")).strip()
        note = note.splitlines()[0][:70] if note else "visual element"
        return [Paragraph(
            f"[Fig. {index} could not be rendered: {escape(note)}]",
            styles["eq_placeholder"],
        )]

    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        px_w, px_h = im.size
    w = min(COL_WIDTH, 240.0)
    h = w * px_h / px_w

    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    cap = Paragraph(f"<b>Fig. {index}.</b> {_md_inline(caption_text, inline_math=False)}", styles["fig_caption"])
    return [KeepTogether([Spacer(1, 3), img, cap])]


# ---------------------------------------------------------------------------
# Story construction
# ---------------------------------------------------------------------------
def _author_block(authors: str, affiliations: str, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    names = [a.strip() for a in re.split(r",| and ", authors) if a.strip()] or ["Author"]
    affils = [a.strip() for a in affiliations.split(";") if a.strip()]
    if not affils:
        affils = [affiliations.strip()]
    if len(affils) < len(names):
        affils = (affils * ((len(names) // len(affils)) + 1))[: len(names)]

    def cell(i: int) -> Any:
        inner = [
            Paragraph(escape(names[i]), styles["author"]),
            Spacer(1, 3),
            Paragraph(escape(affils[i]), styles["affil"]),
        ]
        return inner

    rows = []
    for start in range(0, len(names), 3):
        chunk = names[start:start + 3]
        while len(chunk) < 3:
            chunk.append("")
        cells = []
        for j in range(3):
            gi = start + j
            if gi < len(names):
                cells.append(cell(gi))
            else:
                cells.append(Paragraph("", styles["affil"]))
        rows.append(cells)

    col_w = FULL_WIDTH / 3.0
    tbl = Table(rows, colWidths=[col_w] * 3, hAlign="CENTER")
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return [tbl, Spacer(1, 10)]


def _build_story(
    sections: Dict[str, Dict[str, Any]],
    papers: List[Dict[str, Any]],
    paper_title: str,
    authors: str,
    affiliations: str,
    include_ai_appendix: bool,
    stretch: float,
) -> List[Any]:
    styles = _make_styles(stretch)
    story: List[Any] = [NextPageTemplate("Later")]

    # ---- Header (page 1, full width) ----
    story.append(Paragraph(_md_inline(paper_title, inline_math=False), styles["title"]))
    story.extend(_author_block(authors, affiliations, styles))
    story.append(FrameBreak())

    # ---- Abstract / Index Terms (column 1) ----
    abstract = (sections.get("Abstract", {}) or {}).get("content", "").strip()
    index_terms = (sections.get("Index Terms", {}) or {}).get("content", "").strip()
    if abstract:
        story.append(
            Paragraph(f"<b><i>Abstract&#8212;</i></b>{_md_inline(abstract)}", styles["abstract"])
        )
    if index_terms:
        story.append(
            Paragraph(f"<b><i>Index Terms&#8212;</i></b>{_md_inline(index_terms)}", styles["abstract"])
        )
    if abstract or index_terms:
        story.append(Spacer(1, 6))

    # ---- Body sections ----
    heading_counter = 0
    table_idx = 0
    fig_idx = 0
    eq_idx = 0

    for sec_name, sec_obj in sections.items():
        if sec_name in ("Abstract", "Index Terms"):
            continue
        content = (sec_obj or {}).get("content", "").strip()
        if not content:
            continue

        is_refs = sec_name.strip().lower() == "references"
        if not is_refs:
            numeral = ROMAN_NUMERALS[heading_counter] if heading_counter < len(ROMAN_NUMERALS) else str(heading_counter + 1)
            heading_counter += 1
            story.append(Paragraph(f"{numeral}. {escape(sec_name)}", styles["h1"]))

        blocks = _parse_blocks(content)
        first_para = True
        for block in blocks:
            btype = block["type"]
            if btype == "heading":
                story.append(Paragraph(_md_inline(block["text"], inline_math=False), styles["h2"]))
                first_para = True
            elif btype == "paragraph":
                if is_refs:
                    story.append(Paragraph(_md_inline(block["text"], inline_math=False), styles["ref"]))
                else:
                    style = styles["body_first"] if first_para else styles["body"]
                    story.append(Paragraph(_md_inline(block["text"]), style))
                    first_para = False
            elif btype == "table":
                els = _build_table(block, styles, table_idx)
                if els:
                    story.append(CondPageBreak(60))
                    story.extend(els)
                    table_idx += 1
                    first_para = True
            elif btype in ("equation", "eq"):
                eq_idx += 1
                rendered = _render_math_safe(block["body"], 10.0)
                if rendered:
                    path, w, h = rendered
                    story.append(Spacer(1, 3))
                    story.append(EquationFlowable(path, w, h, eq_idx, COL_WIDTH))
                    story.append(Spacer(1, 3))
                else:
                    story.append(Paragraph(
                        f"[Equation ({eq_idx}) could not be typeset]",
                        styles["eq_placeholder"],
                    ))
                first_para = True
            elif btype in ("chart", "diagram", "figure"):
                els = _build_figure(block, styles, fig_idx + 1)
                if els:
                    story.append(CondPageBreak(70))
                    story.extend(els)
                    fig_idx += 1
                    first_para = True

    # ---- AI appendix ----
    if include_ai_appendix:
        story.append(Spacer(1, 8))
        story.append(Paragraph("APPENDIX: AI USAGE AND PROVENANCE DISCLOSURE", styles["h1"]))
        story.append(Paragraph(
            "This review was drafted with the assistance of ReviewForge, using large language models as a "
            "synthesis engine grounded strictly in the retrieved source papers. Every citation maps to a "
            "verified source document. In line with IEEE guidance on generative AI, all claims and references "
            "were reviewed by the human authors. A local n-gram and fuzzy sentence-level similarity audit was "
            "run across sections to prevent verbatim source overlap.",
            styles["body_first"],
        ))

    return story


# ---------------------------------------------------------------------------
# Document template
# ---------------------------------------------------------------------------
class _ReviewDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=(PAGE_W, PAGE_H),
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=PAGE_H - BOTTOM_Y,
            title="",
            author="",
        )
        header_h = (PAGE_H - MARGIN_TOP) - HEADER_BOTTOM
        first_frames = [
            Frame(MARGIN_LEFT, HEADER_BOTTOM, FULL_WIDTH, header_h, id="header",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(COL1_X, BOTTOM_Y, COL_WIDTH, HEADER_BOTTOM - BOTTOM_Y, id="c1",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(COL2_X, BOTTOM_Y, COL_WIDTH, HEADER_BOTTOM - BOTTOM_Y, id="c2",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        ]
        later_frames = [
            Frame(COL1_X, BOTTOM_Y, COL_WIDTH, (PAGE_H - MARGIN_TOP) - BOTTOM_Y, id="c1",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(COL2_X, BOTTOM_Y, COL_WIDTH, (PAGE_H - MARGIN_TOP) - BOTTOM_Y, id="c2",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        ]
        self.addPageTemplates([
            PageTemplate(id="First", frames=first_frames),
            PageTemplate(id="Later", frames=later_frames),
        ])


def _render_to_pages(story: List[Any]) -> Tuple[bytes, int]:
    buf = io.BytesIO()
    doc = _ReviewDoc(buf)
    doc.build(story)
    data = buf.getvalue()
    pages = data.count(b"/Type /Page\n") + data.count(b"/Type /Page/")
    # Robust page count via PyMuPDF if available
    try:
        import fitz

        with fitz.open(stream=data, filetype="pdf") as d:
            pages = d.page_count
    except Exception:
        pass
    return data, pages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_ieee_pdf(
    sections: Dict[str, Dict[str, Any]],
    papers: List[Dict[str, Any]],
    paper_title: str = "A Comprehensive Survey and Systematic Review",
    authors: str = "First Author, Second Author, Third Author",
    affiliations: str = "Department of Computer Science and Engineering",
    include_ai_appendix: bool = True,
    target_pages: Optional[int] = 6,
) -> io.BytesIO:
    """Render the review paper as an IEEE-format PDF tuned to ``target_pages``."""
    if not _register_fonts():
        raise RuntimeError(
            "Latin Modern fonts are missing from assets/fonts; cannot render PDF."
        )

    _reset_render_cache()

    def build(stretch: float) -> Tuple[bytes, int]:
        story = _build_story(
            sections, papers, paper_title, authors, affiliations, include_ai_appendix, stretch
        )
        return _render_to_pages(story)

    data, pages = build(1.0)
    if not target_pages or pages == target_pages:
        out = io.BytesIO(data)
        out.seek(0)
        return out

    # Page count grows monotonically with the vertical-stretch factor. Expand a
    # bracket [lo, hi] outward from 1.0 until it encloses the target page count,
    # then bisect for the boundary value that lands exactly on the target.
    lo = hi = 1.0
    dlo, plo = data, pages
    dhi, phi = data, pages

    while plo > target_pages and lo > MIN_STRETCH:
        hi, phi, dhi = lo, plo, dlo
        lo = max(MIN_STRETCH, round(lo - 0.06, 4))
        dlo, plo = build(lo)
    while phi < target_pages and hi < MAX_STRETCH:
        lo, plo, dlo = hi, phi, dhi
        hi = min(MAX_STRETCH, round(hi + 0.06, 4))
        dhi, phi = build(hi)

    best = min(((abs(plo - target_pages), dlo), (abs(phi - target_pages), dhi)), key=lambda c: c[0])

    if plo < target_pages < phi:
        for _ in range(16):
            mid = (lo + hi) / 2.0
            dm, pm = build(mid)
            if abs(pm - target_pages) < best[0]:
                best = (abs(pm - target_pages), dm)
            if pm == target_pages:
                break
            if pm < target_pages:
                lo = mid
            else:
                hi = mid

    out = io.BytesIO(best[1])
    out.seek(0)
    return out
