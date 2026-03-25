"""
MYCONEX Document Ingestion Pipeline
=====================================
Inspired by OpenDataLoader PDF.

Parses PDF, HTML, and plain-text documents into structured data for agent
consumption.  Uses only Python stdlib + optional lightweight deps; never
requires a heavy ML stack just to read a file.

Supported formats:
  PDF   — via pdfminer.six (pip install pdfminer.six) or pdftotext CLI fallback
  HTML  — stdlib html.parser
  Text  — direct read
  Markdown — direct read (treated as text with header detection)

Output:
  DocumentResult — normalized dataclass with sections, tables, metadata, and
                   to_markdown() / to_json() / to_agent_payload() helpers.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class TableRow:
    cells: list[str]


@dataclass
class Table:
    headers: list[str]
    rows: list[TableRow]
    caption: str = ""

    def to_markdown(self) -> str:
        if not self.headers and not self.rows:
            return ""
        header_row = " | ".join(self.headers) if self.headers else ""
        sep = " | ".join(["---"] * (len(self.headers) or (len(self.rows[0].cells) if self.rows else 1)))
        data_rows = [" | ".join(r.cells) for r in self.rows]
        lines = []
        if self.caption:
            lines.append(f"*{self.caption}*\n")
        if header_row:
            lines += [header_row, sep]
        lines += data_rows
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "caption": self.caption,
            "headers": self.headers,
            "rows": [r.cells for r in self.rows],
        }


@dataclass
class DocumentSection:
    title: str
    level: int          # 1 = h1/top, 2 = h2, etc.
    content: str
    tables: list[Table] = field(default_factory=list)


@dataclass
class DocumentResult:
    """Normalised representation of an ingested document."""
    source: str                            # file path or URL
    format: str                            # "pdf" | "html" | "text" | "markdown"
    title: str = ""
    abstract: str = ""                     # populated for scientific papers
    sections: list[DocumentSection] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    raw_text: str = ""
    success: bool = True
    error: Optional[str] = None
    processed_at: float = field(default_factory=time.time)

    # ── Export helpers ────────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        """Reconstruct document as Markdown string."""
        parts: list[str] = []
        if self.title:
            parts.append(f"# {self.title}\n")
        if self.abstract:
            parts.append(f"**Abstract:** {self.abstract}\n")
        for sec in self.sections:
            prefix = "#" * min(sec.level + 1, 6)
            if sec.title:
                parts.append(f"\n{prefix} {sec.title}\n")
            if sec.content:
                parts.append(sec.content)
            for tbl in sec.tables:
                parts.append("\n" + tbl.to_markdown())
        for tbl in self.tables:
            parts.append("\n" + tbl.to_markdown())
        return "\n".join(parts).strip()

    def to_json(self) -> dict:
        """Export as a serialisable dict."""
        return {
            "source": self.source,
            "format": self.format,
            "title": self.title,
            "abstract": self.abstract,
            "sections": [
                {
                    "title": s.title,
                    "level": s.level,
                    "content": s.content[:2000],
                    "tables": [t.to_dict() for t in s.tables],
                }
                for s in self.sections
            ],
            "tables": [t.to_dict() for t in self.tables],
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
        }

    def to_agent_payload(self, max_chars: int = 8000) -> dict:
        """
        Compact payload for agent consumption — fits within context budget.

        Truncates raw_text and includes structured summary.
        """
        text = self.to_markdown()
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[...truncated, {len(text) - max_chars} chars omitted]"
        return {
            "title": self.title,
            "format": self.format,
            "sections": len(self.sections),
            "tables": len(self.tables),
            "content": text,
            "metadata": self.metadata,
        }

    @property
    def word_count(self) -> int:
        return len(self.raw_text.split())

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.raw_text.encode()).hexdigest()[:12]


# ─── PDF Parser ───────────────────────────────────────────────────────────────

class PDFParser:
    """
    PDF → text/structure pipeline.

    Strategy (first available wins):
      1. pdfminer.six   — python library, best quality
      2. pdftotext CLI  — poppler-utils, good quality
      3. Binary scan    — regex over raw PDF bytes, last resort
    """

    def parse(self, path: Path) -> tuple[str, dict]:
        """
        Parse a PDF file.

        Returns:
            (text, metadata) tuple.
        """
        # Try pdfminer first
        try:
            return self._parse_pdfminer(path)
        except ImportError:
            logger.debug("[pdf] pdfminer not available, trying pdftotext")
        except Exception as exc:
            logger.debug("[pdf] pdfminer failed: %s, trying pdftotext", exc)

        # Try pdftotext CLI
        try:
            return self._parse_pdftotext(path)
        except FileNotFoundError:
            logger.debug("[pdf] pdftotext not found, falling back to binary scan")
        except Exception as exc:
            logger.debug("[pdf] pdftotext failed: %s, falling back to binary scan", exc)

        # Binary scan fallback
        return self._parse_binary(path)

    def _parse_pdfminer(self, path: Path) -> tuple[str, dict]:
        from pdfminer.high_level import extract_text, extract_pages  # type: ignore[import]
        from pdfminer.layout import LTTextBox, LTPage  # type: ignore[import]

        text = extract_text(str(path))
        metadata: dict = {}
        try:
            from pdfminer.pdfpage import PDFPage  # type: ignore[import]
            from pdfminer.pdfparser import PDFParser as _P  # type: ignore[import]
            from pdfminer.pdfdocument import PDFDocument  # type: ignore[import]
            with open(path, "rb") as f:
                parser = _P(f)
                doc = PDFDocument(parser)
                if doc.info:
                    raw_meta = doc.info[0]
                    for k, v in raw_meta.items():
                        try:
                            metadata[k] = v.decode() if isinstance(v, bytes) else str(v)
                        except Exception:
                            pass
        except Exception:
            pass
        return text or "", metadata

    def _parse_pdftotext(self, path: Path) -> tuple[str, dict]:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftotext exit {result.returncode}: {result.stderr[:200]}")
        return result.stdout, {}

    def _parse_binary(self, path: Path) -> tuple[str, dict]:
        """
        Extract readable text from PDF bytes using regex.
        Handles simple PDFs; may miss text in compressed streams.
        """
        raw = path.read_bytes()
        # Extract text from BT...ET blocks (uncompressed PDF text streams)
        parts: list[str] = []
        for match in re.finditer(rb"BT(.*?)ET", raw, re.DOTALL):
            block = match.group(1)
            # Extract string literals: (text) or <hex>
            for s in re.finditer(rb"\(([^)]*)\)", block):
                try:
                    text = s.group(1).decode("latin-1").strip()
                    if text and len(text) > 1:
                        parts.append(text)
                except Exception:
                    pass
        return " ".join(parts), {}


# ─── HTML Parser ──────────────────────────────────────────────────────────────

class HTMLDocParser:
    """Parse HTML into a DocumentResult using stdlib html.parser."""

    def parse(self, html: str, source: str = "") -> DocumentResult:
        from html.parser import HTMLParser

        class _P(HTMLParser):
            def __init__(self):
                super().__init__()
                self.title = ""
                self.sections: list[tuple[int, str, list[str]]] = []  # (level, heading, paras)
                self.tables: list[list[list[str]]] = []
                self._current_heading = ""
                self._current_level = 0
                self._current_paras: list[str] = []
                self._in_table = False
                self._current_row: list[str] = []
                self._current_table: list[list[str]] = []
                self._buf = ""
                self._skip_tags = {"script", "style", "nav", "footer"}
                self._in_skip = False
                self._tag = ""

            def handle_starttag(self, tag, attrs):
                self._tag = tag
                if tag in self._skip_tags:
                    self._in_skip = True
                if tag == "table":
                    self._in_table = True
                    self._current_table = []
                if tag in ("tr",):
                    self._current_row = []
                if tag in ("h1", "h2", "h3", "h4"):
                    if self._current_heading or self._current_paras:
                        self.sections.append((self._current_level, self._current_heading, self._current_paras[:]))
                    self._current_level = int(tag[1])
                    self._current_heading = ""
                    self._current_paras = []
                    self._buf = ""

            def handle_endtag(self, tag):
                if tag in self._skip_tags:
                    self._in_skip = False
                if tag == "table":
                    self.tables.append(self._current_table[:])
                    self._in_table = False
                if tag == "tr" and self._in_table:
                    if self._current_row:
                        self._current_table.append(self._current_row[:])
                if tag in ("td", "th"):
                    self._current_row.append(self._buf.strip())
                    self._buf = ""
                if tag in ("h1", "h2", "h3", "h4"):
                    if self._current_level == 0:
                        self.title = self._buf.strip()
                    self._current_heading = self._buf.strip()
                    self._buf = ""
                if tag == "p":
                    if self._buf.strip():
                        self._current_paras.append(self._buf.strip())
                    self._buf = ""

            def handle_data(self, data):
                if self._in_skip:
                    return
                self._buf += data
                if self._tag == "title":
                    self.title = data.strip()

            def finish(self):
                if self._current_heading or self._current_paras:
                    self.sections.append((self._current_level, self._current_heading, self._current_paras[:]))

        p = _P()
        p.feed(html)
        p.finish()

        sections = []
        all_text_parts = []
        for level, heading, paras in p.sections:
            content = "\n\n".join(paras)
            all_text_parts.append(f"{heading}\n{content}")
            sections.append(DocumentSection(title=heading, level=level, content=content))

        tables = []
        for raw_table in p.tables:
            if not raw_table:
                continue
            headers = raw_table[0] if raw_table else []
            rows = [TableRow(cells=r) for r in raw_table[1:]]
            tables.append(Table(headers=headers, rows=rows))

        raw_text = re.sub(r"\s+", " ", " ".join(all_text_parts)).strip()

        return DocumentResult(
            source=source,
            format="html",
            title=p.title,
            sections=sections,
            tables=tables,
            raw_text=raw_text,
        )


# ─── Scientific Paper Extractor ───────────────────────────────────────────────

class ScientificPaperExtractor:
    """
    Extract structured sections from scientific paper text.

    Handles common paper conventions: Abstract, Introduction, Methods,
    Results, Discussion, Conclusion, References.
    """

    _SECTION_PATTERNS = [
        r"^(abstract|introduction|background|related work|literature review|"
        r"methodology|methods|materials and methods|experimental|experiments|"
        r"results|findings|discussion|conclusion|conclusions|future work|"
        r"acknowledgements?|references|bibliography)\s*$",
        r"^\d+\.?\s+(introduction|background|methods|results|discussion|conclusion)",
    ]

    def extract(self, text: str) -> dict:
        """
        Extract structured sections from paper text.

        Returns:
            Dict with keys: abstract, sections (list of dicts), references (list).
        """
        lines = text.splitlines()
        abstract = self._extract_abstract(lines)
        sections = self._split_sections(lines)
        references = self._extract_references(text)

        return {
            "abstract": abstract,
            "sections": sections,
            "references": references[:50],  # cap at 50
        }

    def _extract_abstract(self, lines: list[str]) -> str:
        in_abstract = False
        parts: list[str] = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^abstract\s*$", stripped, re.IGNORECASE):
                in_abstract = True
                continue
            if in_abstract:
                if self._is_section_header(stripped) and stripped.lower() != "abstract":
                    break
                parts.append(stripped)
        return " ".join(parts).strip()[:2000]

    def _split_sections(self, lines: list[str]) -> list[dict]:
        sections: list[dict] = []
        current_title = ""
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if self._is_section_header(stripped):
                if current_title or current_lines:
                    sections.append({
                        "title": current_title,
                        "content": " ".join(current_lines).strip()[:3000],
                    })
                current_title = stripped
                current_lines = []
            else:
                if stripped:
                    current_lines.append(stripped)

        if current_title or current_lines:
            sections.append({
                "title": current_title,
                "content": " ".join(current_lines).strip()[:3000],
            })
        return sections

    def _is_section_header(self, text: str) -> bool:
        if not text or len(text) > 80:
            return False
        for pattern in self._SECTION_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        # Numbered section headers like "1. Introduction" or "2.1 Methods"
        if re.match(r"^\d+(\.\d+)*\.?\s+\w", text):
            return True
        return False

    def _extract_references(self, text: str) -> list[str]:
        refs: list[str] = []
        # Find references section
        ref_match = re.search(
            r"(?:references|bibliography)\s*\n(.*?)(?:\n\n\n|\Z)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if ref_match:
            ref_text = ref_match.group(1)
            # Split on numbered entries or author-year patterns
            entries = re.split(r"\n(?=\[\d+\]|\d+\.\s+[A-Z])", ref_text)
            refs = [e.strip().replace("\n", " ") for e in entries if e.strip()]
        return refs


# ─── Table Extractor ─────────────────────────────────────────────────────────

class TextTableExtractor:
    """
    Detect and parse ASCII/text tables from plain text.

    Handles:
      - Pipe-separated tables: | col1 | col2 |
      - Space-aligned tables:  col1    col2    col3
      - CSV-like tables:       val1,val2,val3
    """

    def extract(self, text: str) -> list[Table]:
        tables: list[Table] = []
        tables.extend(self._extract_pipe_tables(text))
        tables.extend(self._extract_csv_tables(text))
        return tables

    def _extract_pipe_tables(self, text: str) -> list[Table]:
        tables: list[Table] = []
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if "|" in line and line.count("|") >= 2:
                table_lines = []
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                if len(table_lines) >= 2:
                    rows = [
                        [c.strip() for c in l.split("|") if c.strip()]
                        for l in table_lines
                        if not re.match(r"^\s*\|[-:|\s]+\|\s*$", l)  # skip separator rows
                    ]
                    if rows:
                        headers = rows[0]
                        data_rows = [TableRow(cells=r) for r in rows[1:]]
                        tables.append(Table(headers=headers, rows=data_rows))
            else:
                i += 1
        return tables

    def _extract_csv_tables(self, text: str) -> list[Table]:
        """Detect CSV-like blocks (≥3 consistent commas per line, ≥2 lines)."""
        tables: list[Table] = []
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.count(",") >= 2:
                block = []
                while i < len(lines) and lines[i].count(",") >= 2:
                    block.append(lines[i])
                    i += 1
                if len(block) >= 3:
                    rows = [r.split(",") for r in block]
                    headers = [c.strip() for c in rows[0]]
                    data_rows = [TableRow(cells=[c.strip() for c in r]) for r in rows[1:]]
                    tables.append(Table(headers=headers, rows=data_rows))
            else:
                i += 1
        return tables


# ─── Document Processor ───────────────────────────────────────────────────────

class DocumentProcessor:
    """
    Main document ingestion pipeline.

    Dispatches to format-specific parsers and enriches output with
    scientific paper structure detection and table extraction.

    Usage:
        proc = DocumentProcessor()
        result = await proc.process("/path/to/paper.pdf")
        payload = result.to_agent_payload()
    """

    def __init__(self) -> None:
        self._pdf_parser = PDFParser()
        self._html_parser = HTMLDocParser()
        self._paper_extractor = ScientificPaperExtractor()
        self._table_extractor = TextTableExtractor()

    async def process(
        self,
        source: str,
        format_hint: Optional[str] = None,
        is_scientific: bool = False,
    ) -> DocumentResult:
        """
        Process a document from a file path or URL.

        Args:
            source:       File path (str/Path) or HTTP/HTTPS URL.
            format_hint:  Override format detection ("pdf"|"html"|"text").
            is_scientific: Enable scientific paper section extraction.

        Returns:
            DocumentResult with normalised structure.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, self._process_sync, source, format_hint, is_scientific
            )
        except Exception as exc:
            logger.error("[doc_processor] failed for %s: %s", source, exc)
            return DocumentResult(
                source=str(source), format=format_hint or "unknown",
                success=False, error=str(exc),
            )

    def _process_sync(
        self, source: str, format_hint: Optional[str], is_scientific: bool
    ) -> DocumentResult:
        source_str = str(source)
        fmt = format_hint or self._detect_format(source_str)

        if fmt == "pdf":
            return self._process_pdf(source_str, is_scientific)
        if fmt == "html":
            return self._process_html(source_str)
        return self._process_text(source_str, fmt, is_scientific)

    def _detect_format(self, source: str) -> str:
        lower = source.lower()
        if lower.endswith(".pdf"):
            return "pdf"
        if lower.endswith((".html", ".htm")):
            return "html"
        if lower.endswith(".md"):
            return "markdown"
        if lower.startswith("http"):
            return "html"   # assume HTML for URLs
        return "text"

    def _process_pdf(self, source: str, is_scientific: bool) -> DocumentResult:
        path = Path(source)
        if not path.is_file():
            return DocumentResult(source=source, format="pdf",
                                  success=False, error=f"File not found: {source}")

        raw_text, metadata = self._pdf_parser.parse(path)
        if not raw_text.strip():
            return DocumentResult(source=source, format="pdf",
                                  success=False, error="PDF parsed as empty",
                                  metadata=metadata)

        tables = self._table_extractor.extract(raw_text)
        result = DocumentResult(
            source=source, format="pdf",
            raw_text=raw_text,
            tables=tables,
            metadata={**metadata, "pages": metadata.get("Pages", "unknown")},
        )

        if is_scientific or self._looks_scientific(raw_text):
            paper = self._paper_extractor.extract(raw_text)
            result.abstract = paper["abstract"]
            result.sections = [
                DocumentSection(
                    title=s["title"], level=1,
                    content=s["content"],
                )
                for s in paper["sections"]
            ]
            result.metadata["references_count"] = len(paper["references"])
        else:
            # Split into paragraphs as sections
            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            result.sections = [
                DocumentSection(title="", level=1, content=p[:2000])
                for p in paragraphs[:50]
            ]

        # Try to extract title from first meaningful line
        first_lines = [l.strip() for l in raw_text.splitlines() if l.strip()][:5]
        if first_lines:
            result.title = first_lines[0][:200]

        return result

    def _process_html(self, source: str) -> DocumentResult:
        if source.startswith("http"):
            import urllib.request
            try:
                with urllib.request.urlopen(source, timeout=20) as resp:
                    html = resp.read().decode("utf-8", errors="replace")
            except Exception as exc:
                return DocumentResult(source=source, format="html",
                                      success=False, error=str(exc))
        else:
            path = Path(source)
            if not path.is_file():
                return DocumentResult(source=source, format="html",
                                      success=False, error=f"File not found: {source}")
            html = path.read_text(errors="replace")

        return self._html_parser.parse(html, source=source)

    def _process_text(self, source: str, fmt: str, is_scientific: bool) -> DocumentResult:
        if Path(source).is_file():
            raw_text = Path(source).read_text(errors="replace")
        else:
            raw_text = source   # treat source itself as text content

        tables = self._table_extractor.extract(raw_text)
        result = DocumentResult(
            source=source, format=fmt,
            raw_text=raw_text,
            tables=tables,
        )

        if is_scientific or (fmt == "text" and self._looks_scientific(raw_text)):
            paper = self._paper_extractor.extract(raw_text)
            result.abstract = paper["abstract"]
            result.sections = [
                DocumentSection(title=s["title"], level=1, content=s["content"])
                for s in paper["sections"]
            ]
        else:
            # Split on double newlines
            paras = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            result.sections = [
                DocumentSection(title="", level=1, content=p[:2000])
                for p in paras[:100]
            ]

        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        result.title = lines[0][:200] if lines else ""
        return result

    @staticmethod
    def _looks_scientific(text: str) -> bool:
        indicators = [
            "abstract", "introduction", "methodology", "results",
            "conclusion", "references", "doi:", "arxiv", "et al.",
        ]
        text_lower = text[:3000].lower()
        return sum(1 for ind in indicators if ind in text_lower) >= 3
