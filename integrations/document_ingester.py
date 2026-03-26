"""
MYCONEX Document Ingester
--------------------------
Extracts text from PDFs, EPUBs, Markdown, and plain text files,
runs them through the Fabric/Hermes pipeline, and stores results
in the knowledge base exactly like the other ingesters.

Supported formats:
  .pdf   — via pypdf (pip install pypdf)
  .epub  — via ebooklib + html2text (pip install ebooklib html2text)
  .md .txt .rst — plain UTF-8 read

Env vars:
  DOC_INGEST_MAX_CHUNK   — max chars per chunk sent to LLM (default: 8000)
  DOC_INGEST_OVERLAP     — overlap between chunks in chars (default: 200)

API (called from dashboard /api/submit/doc):
    result = await ingest_document(path_or_bytes, filename, title="optional")
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_BASE      = Path.home() / ".myconex"
_DOC_FILE  = _BASE / "doc_insights.json"
_CHUNK_SZ  = int(os.getenv("DOC_INGEST_MAX_CHUNK", "8000"))
_OVERLAP   = int(os.getenv("DOC_INGEST_OVERLAP",  "200"))


# ── Text extraction ────────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        raise RuntimeError("pypdf not installed — run: pip install pypdf")


def _extract_epub(path: str) -> str:
    try:
        import ebooklib
        from ebooklib import epub
        import html2text
        book = epub.read_epub(path)
        h2t  = html2text.HTML2Text()
        h2t.ignore_links = True
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            parts.append(h2t.handle(item.get_content().decode("utf-8", errors="replace")))
        return "\n".join(parts)
    except ImportError:
        raise RuntimeError("ebooklib/html2text not installed — run: pip install ebooklib html2text")


def _extract_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def extract_text(path: str) -> str:
    """Extract plain text from a document file based on its extension."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext == ".epub":
        return _extract_epub(path)
    elif ext in (".md", ".txt", ".rst", ".text", ".log", ".csv"):
        return _extract_text(path)
    else:
        # Try plain text as fallback
        try:
            return _extract_text(path)
        except Exception:
            raise RuntimeError(f"Unsupported document format: {ext}")


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk(text: str, size: int = _CHUNK_SZ, overlap: int = _OVERLAP) -> list[str]:
    chunks = []
    start  = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


# ── LLM processing ────────────────────────────────────────────────────────────

async def _process_chunk(chunk: str, title: str, chunk_idx: int) -> dict:
    """Run a single chunk through the Fabric/Ollama summarise + topic extract pipeline."""
    try:
        from integrations.fabric_helper import run_fabric_pattern  # type: ignore[import]
        summary = await run_fabric_pattern("summarize", chunk)
        topics_raw = await run_fabric_pattern("extract_wisdom", chunk)
    except Exception:
        # Fallback: direct Ollama call
        try:
            import httpx
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            prompt = (
                f"Summarise the following text in 3-5 sentences and list 3-5 key topics.\n\n"
                f"Text:\n{chunk[:4000]}\n\n"
                f"Reply as JSON: {{\"summary\": \"...\", \"topics\": [\"...\", ...]}}"
            )
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                          "prompt": prompt, "stream": False},
                )
            raw = r.json().get("response", "{}")
            parsed = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            summary     = parsed.get("summary", chunk[:300])
            topics_raw  = "\n".join(parsed.get("topics", []))
        except Exception as exc:
            logger.warning("[doc-ingester] chunk %d LLM failed: %s", chunk_idx, exc)
            summary    = chunk[:300]
            topics_raw = ""

    topics = [
        t.strip().lstrip("-*• ").strip()
        for t in topics_raw.splitlines()
        if t.strip() and len(t.strip()) > 2
    ][:8]

    return {
        "chunk_idx":    chunk_idx,
        "summary":      summary[:600],
        "topics":       topics,
        "raw":          {"summarize": summary},
    }


# ── Main entry point ──────────────────────────────────────────────────────────

async def ingest_document(
    source: Union[str, bytes],
    filename: str,
    title: str = "",
) -> dict:
    """
    Ingest a document into the MYCONEX knowledge base.

    source  — file path string, or raw bytes
    filename — original filename (used for format detection)
    title    — optional human-readable title

    Returns a result dict with keys: ok, title, chunks, topics, stored_at
    """
    title = title or Path(filename).stem

    # Write bytes to a temp file if needed
    tmp_path = None
    if isinstance(source, bytes):
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(source)
            tmp_path = tf.name
        file_path = tmp_path
    else:
        file_path = source

    try:
        # 1. Extract text
        logger.info("[doc-ingester] extracting text from %s", filename)
        text = await asyncio.get_event_loop().run_in_executor(
            None, lambda: extract_text(file_path)
        )
        if not text.strip():
            return {"ok": False, "error": "No text extracted from document"}

        # 2. Deduplicate by content hash
        doc_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        existing = _load_insights()
        if any(e.get("doc_hash") == doc_hash for e in existing):
            return {"ok": False, "error": f"Document already ingested (hash {doc_hash})"}

        # 3. Chunk + process
        chunks = _chunk(text)
        logger.info("[doc-ingester] processing %d chunk(s) from '%s'", len(chunks), title)

        chunk_results = []
        for i, chunk in enumerate(chunks[:20]):  # cap at 20 chunks per doc
            result = await _process_chunk(chunk, title, i)
            chunk_results.append(result)
            await asyncio.sleep(0.1)  # don't hammer Ollama

        # 4. Aggregate topics across chunks
        all_topics: dict[str, int] = {}
        for cr in chunk_results:
            for t in cr.get("topics", []):
                all_topics[t] = all_topics.get(t, 0) + 1
        top_topics = [t for t, _ in sorted(all_topics.items(), key=lambda x: -x[1])[:10]]

        full_summary = " ".join(cr["summary"] for cr in chunk_results[:3])

        # 5. Store insight record
        record = {
            "title":        title,
            "filename":     filename,
            "doc_hash":     doc_hash,
            "chunk_count":  len(chunk_results),
            "topics":       top_topics,
            "summary":      full_summary[:1000],
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "stored_at":    datetime.now(timezone.utc).isoformat(),
            "source":       "document",
        }
        existing.append(record)
        _BASE.mkdir(parents=True, exist_ok=True)
        _DOC_FILE.write_text(json.dumps(existing[-500:], indent=2, ensure_ascii=False))

        # 6. Embed into vector store
        try:
            from integrations.knowledge_store import embed_and_store_batch
            embed_items = [
                {
                    "text":   f"{title}\n\n{cr['summary']}",
                    "source": "document",
                    "title":  title,
                    "topics": cr.get("topics", []),
                    "ts":     record["stored_at"],
                }
                for cr in chunk_results
            ]
            await embed_and_store_batch(embed_items)
        except Exception as exc:
            logger.warning("[doc-ingester] embedding failed: %s", exc)

        # 7. Knowledge graph ingestion
        try:
            from core.knowledge_graph import get_graph
            await get_graph().ingest_text(full_summary[:2000], source="document", title=title)
        except Exception:
            pass

        logger.info("[doc-ingester] ingested '%s' — %d chunks, %d topics",
                    title, len(chunk_results), len(top_topics))
        return {
            "ok":        True,
            "title":     title,
            "chunks":    len(chunk_results),
            "topics":    top_topics,
            "stored_at": record["stored_at"],
        }

    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def _load_insights() -> list[dict]:
    try:
        if _DOC_FILE.exists():
            return json.loads(_DOC_FILE.read_text())
    except Exception:
        pass
    return []
