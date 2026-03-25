"""
MYCONEX Novelty Scanner
========================
Actively scans frontier sources for ideas, ranks them by relevance to
MYCONEX capabilities, generates implementation proposals, and feeds them
into the AutonomousOptimizationLoop's proposal queue.

Sources scanned:
  - HuggingFace trending models
  - GitHub trending repositories (Python/AI)
  - arXiv recent AI/ML papers (RSS)
  - Papers With Code (latest papers)
  - Hacker News (AI-related stories)
  - Reddit r/MachineLearning (top weekly)

Pipeline:
  1. Fetch  — parallel HTTP/RSS gather across all sources
  2. Parse  — extract structured IdeaSignals (title, summary, url, tags)
  3. Score  — rank by relevance to MYCONEX capability keywords
  4. Filter — keep top-N above relevance threshold
  5. Propose — LLM generates an ImprovementOpportunity per top idea
  6. Enqueue — write proposals to ~/.myconex/novelty_queue.json
  7. Loop   — repeat every `scan_interval_hours` (default: 6)

Usage:
    scanner = NoveltyScanner(agent=rlm_agent)
    await scanner.run_once()                    # single scan
    await scanner.run(scan_interval_hours=6)    # scheduled loop
    scanner.inject_into_loop(autonomous_loop)   # register with loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

_MYCONEX_DIR   = Path.home() / ".myconex"
_NOVELTY_QUEUE = _MYCONEX_DIR / "novelty_queue.json"
_NOVELTY_LOG   = _MYCONEX_DIR / "novelty_history.jsonl"
_LESSONS_FILE  = Path(__file__).parent.parent / "lessons.md"

# ── MYCONEX Relevance Keywords ────────────────────────────────────────────────
# Ideas matching these topics score higher during ranking.

_RELEVANCE_KEYWORDS: dict[str, float] = {
    # Core AI / LLM
    "llm": 1.0, "language model": 1.0, "transformer": 0.9, "inference": 0.9,
    "fine-tuning": 0.8, "gguf": 1.0, "quantization": 0.9, "rlhf": 0.8,
    "mixture of experts": 1.0, "moe": 1.0, "speculative decoding": 0.9,
    # Agent systems
    "agent": 1.0, "multi-agent": 1.0, "orchestration": 1.0, "autonomous": 0.9,
    "tool use": 0.9, "function calling": 0.9, "chain of thought": 0.8,
    "planning": 0.8, "reasoning": 0.8, "reflection": 0.8,
    # Context / Memory
    "context window": 0.9, "long context": 0.9, "memory": 0.8,
    "retrieval augmented": 0.9, "rag": 0.9, "vector store": 0.8,
    "embedding": 0.7, "knowledge graph": 0.7,
    # Distributed / Mesh
    "distributed": 0.8, "mesh": 0.8, "nats": 0.9, "redis": 0.7,
    "pub/sub": 0.7, "peer to peer": 0.7, "edge inference": 0.9,
    # Efficiency
    "bitnet": 1.0, "1-bit": 0.9, "flash attention": 0.9, "kv cache": 0.8,
    "throughput": 0.6, "latency": 0.6, "batching": 0.6,
    # Self-improvement
    "self-improvement": 1.0, "self-play": 0.8, "meta-learning": 0.8,
    "continual learning": 0.7, "online learning": 0.7,
    # Misc frontier
    "vision language": 0.7, "multimodal": 0.7, "code generation": 0.8,
    "open source": 0.5, "benchmark": 0.5,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IdeaSignal:
    """Raw idea extracted from a frontier source."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    summary: str = ""
    url: str = ""
    source: str = ""            # "huggingface" | "github" | "arxiv" | "pwc" | "hn" | "reddit"
    tags: list[str] = field(default_factory=list)
    raw_score: float = 0.0      # raw signal strength (stars, votes, citations)
    relevance_score: float = 0.0
    fetched_at: float = field(default_factory=time.time)

    def combined_score(self) -> float:
        """Weighted combination for final ranking."""
        return 0.7 * self.relevance_score + 0.3 * min(1.0, self.raw_score / 100.0)


@dataclass
class NoveltyProposal:
    """An IdeaSignal transformed into an actionable MYCONEX improvement."""
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    signal: IdeaSignal = field(default_factory=IdeaSignal)
    title: str = ""
    description: str = ""
    target_file: str = ""
    impact: str = "medium"
    category: str = "feature"
    proposed_change: str = ""
    priority_score: float = 0.5
    generated_at: float = field(default_factory=time.time)
    consumed: bool = False      # True once the autonomous loop picks it up

    def to_opportunity_dict(self) -> dict:
        """Serialise as ImprovementOpportunity-compatible dict."""
        return {
            "title": self.title,
            "description": self.description,
            "target_file": self.target_file,
            "impact": self.impact,
            "category": self.category,
            "proposed_change": self.proposed_change,
            "priority_score": self.priority_score,
        }


@dataclass
class ScanReport:
    """Summary of a single novelty scan run."""
    scan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    sources_tried: int = 0
    sources_succeeded: int = 0
    signals_found: int = 0
    proposals_generated: int = 0
    proposals_queued: int = 0
    error: Optional[str] = None

    @property
    def duration_s(self) -> float:
        end = self.completed_at or time.time()
        return end - self.started_at


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Parsers (no external deps)
# ═══════════════════════════════════════════════════════════════════════════════

class _SimpleHTMLExtractor(HTMLParser):
    """Extract text and links from HTML without external dependencies."""

    def __init__(self):
        super().__init__()
        self._text_parts: list[str] = []
        self._links: list[dict] = []
        self._current_href: Optional[str] = None
        self._in_script = False
        self._in_style = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag in ("script", "style"):
            self._in_script = (tag == "script")
            self._in_style = (tag == "style")
        if tag == "a" and "href" in attrs_dict:
            self._current_href = attrs_dict["href"]

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._in_script = False
            self._in_style = False
        if tag == "a":
            self._current_href = None

    def handle_data(self, data):
        if self._in_script or self._in_style:
            return
        stripped = data.strip()
        if stripped:
            self._text_parts.append(stripped)
            if self._current_href:
                self._links.append({"text": stripped, "href": self._current_href})

    def get_text(self) -> str:
        return " ".join(self._text_parts)

    def get_links(self) -> list[dict]:
        return self._links


def _parse_html(html: str) -> tuple[str, list[dict]]:
    """Return (plain_text, links) from HTML string."""
    parser = _SimpleHTMLExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.get_text(), parser.get_links()


def _parse_rss(xml: str) -> list[dict]:
    """
    Minimal RSS/Atom parser — returns list of {title, link, summary, pubdate}.
    No external deps; handles both RSS 2.0 and Atom.
    """
    items = []
    # Match <item> or <entry> blocks
    for block_match in re.finditer(
        r"<(?:item|entry)>(.*?)</(?:item|entry)>", xml, re.DOTALL
    ):
        block = block_match.group(1)

        def _tag(name: str) -> str:
            m = re.search(rf"<{name}[^>]*>(?:<!\[CDATA\[)?(.*?)(?:]]>)?</{name}>",
                          block, re.DOTALL)
            return m.group(1).strip() if m else ""

        title   = re.sub(r"<[^>]+>", "", _tag("title"))
        link    = _tag("link") or _tag("id")
        summary = re.sub(r"<[^>]+>", " ", _tag("description") or _tag("summary"))[:500]
        pubdate = _tag("pubDate") or _tag("published") or _tag("updated")

        if title:
            items.append({
                "title": title.strip(),
                "link": link.strip(),
                "summary": summary.strip(),
                "pubdate": pubdate.strip(),
            })
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# Source Fetchers
# ═══════════════════════════════════════════════════════════════════════════════

async def _http_get(url: str, timeout: float = 20.0) -> Optional[str]:
    """Async HTTP GET; returns text body or None on error."""
    try:
        import httpx
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "MYCONEX-NoveltyScanner/1.0 (+https://github.com/myconex)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception as exc:
        logger.debug("[novelty_scanner] GET %s failed: %s", url, exc)
        return None


async def fetch_huggingface_trending(limit: int = 20) -> list[IdeaSignal]:
    """Scrape HuggingFace trending models."""
    signals: list[IdeaSignal] = []
    # HF models sorted by trending score
    html = await _http_get(
        "https://huggingface.co/models?sort=trending&limit=25", timeout=25
    )
    if not html:
        return signals

    text, links = _parse_html(html)

    # Extract model cards: look for links like /org/model-name
    seen: set[str] = set()
    for link in links:
        href = link.get("href", "")
        label = link.get("text", "").strip()
        if not href or not label:
            continue
        # Model links: /username/model-name (two path segments)
        parts = href.strip("/").split("/")
        if len(parts) == 2 and parts[0] and parts[1] and href not in seen:
            seen.add(href)
            signals.append(IdeaSignal(
                title=f"HF model: {parts[1].replace('-', ' ')}",
                summary=f"Trending HuggingFace model: {href}. Label: {label}",
                url=f"https://huggingface.co{href}",
                source="huggingface",
                tags=_extract_tags_from_text(label + " " + href),
            ))
            if len(signals) >= limit:
                break

    logger.debug("[novelty_scanner] HuggingFace: %d signals", len(signals))
    return signals


async def fetch_github_trending(language: str = "python", period: str = "weekly",
                                 limit: int = 20) -> list[IdeaSignal]:
    """Scrape GitHub trending repositories."""
    signals: list[IdeaSignal] = []
    url = f"https://github.com/trending/{language}?since={period}"
    html = await _http_get(url, timeout=25)
    if not html:
        return signals

    text, links = _parse_html(html)

    # Repo links: /owner/repo pattern; avoid noise
    seen: set[str] = set()
    for link in links:
        href = link.get("href", "").strip()
        parts = href.strip("/").split("/")
        if (len(parts) == 2
                and parts[0] and parts[1]
                and not parts[0].startswith(("site:", "login", "signup", "about"))
                and href not in seen):
            seen.add(href)
            repo_name = parts[1].replace("-", " ").replace("_", " ")
            signals.append(IdeaSignal(
                title=f"GitHub: {parts[0]}/{parts[1]}",
                summary=f"Trending Python repo: {href}",
                url=f"https://github.com{href}",
                source="github",
                tags=_extract_tags_from_text(repo_name),
            ))
            if len(signals) >= limit:
                break

    logger.debug("[novelty_scanner] GitHub trending: %d signals", len(signals))
    return signals


async def fetch_arxiv_rss(category: str = "cs.AI", limit: int = 20) -> list[IdeaSignal]:
    """Fetch recent arXiv papers via RSS."""
    signals: list[IdeaSignal] = []
    rss_url = f"https://rss.arxiv.org/rss/{category}"
    xml = await _http_get(rss_url, timeout=20)
    if not xml:
        return signals

    items = _parse_rss(xml)
    for item in items[:limit]:
        title   = item.get("title", "")
        summary = item.get("summary", "")
        link    = item.get("link", "")
        if not title:
            continue
        signals.append(IdeaSignal(
            title=f"arXiv: {title}",
            summary=summary[:400],
            url=link,
            source="arxiv",
            tags=_extract_tags_from_text(title + " " + summary),
        ))

    logger.debug("[novelty_scanner] arXiv %s: %d signals", category, len(signals))
    return signals


async def fetch_papers_with_code(limit: int = 15) -> list[IdeaSignal]:
    """Fetch latest papers from Papers With Code JSON API."""
    signals: list[IdeaSignal] = []
    url = "https://paperswithcode.com/api/v1/papers/?ordering=-published&format=json&page_size=25"
    body = await _http_get(url, timeout=20)
    if not body:
        return signals

    try:
        data = json.loads(body)
        results = data.get("results", [])
        for paper in results[:limit]:
            title    = paper.get("title", "")
            abstract = paper.get("abstract", "")[:400]
            url_slug = paper.get("url_pdf", "") or paper.get("paper_url", "")
            if not title:
                continue
            signals.append(IdeaSignal(
                title=f"PwC: {title}",
                summary=abstract,
                url=url_slug,
                source="pwc",
                raw_score=float(paper.get("stars", 0) or 0),
                tags=_extract_tags_from_text(title + " " + abstract),
            ))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.debug("[novelty_scanner] PwC parse error: %s", exc)

    logger.debug("[novelty_scanner] Papers With Code: %d signals", len(signals))
    return signals


async def fetch_hacker_news_rss(limit: int = 20) -> list[IdeaSignal]:
    """Fetch AI-related stories from Hacker News RSS."""
    signals: list[IdeaSignal] = []
    xml = await _http_get("https://news.ycombinator.com/rss", timeout=15)
    if not xml:
        return signals

    items = _parse_rss(xml)
    for item in items[:limit]:
        title   = item.get("title", "")
        link    = item.get("link", "")
        summary = item.get("summary", "")
        if not title:
            continue
        signals.append(IdeaSignal(
            title=f"HN: {title}",
            summary=summary[:300],
            url=link,
            source="hn",
            tags=_extract_tags_from_text(title),
        ))

    logger.debug("[novelty_scanner] Hacker News: %d signals", len(signals))
    return signals


async def fetch_reddit_ml(limit: int = 15) -> list[IdeaSignal]:
    """Fetch top posts from r/MachineLearning via JSON API."""
    signals: list[IdeaSignal] = []
    url = "https://www.reddit.com/r/MachineLearning/top.json?limit=25&t=week"
    body = await _http_get(url, timeout=15)
    if not body:
        return signals

    try:
        data = json.loads(body)
        posts = data.get("data", {}).get("children", [])
        for post in posts[:limit]:
            pd = post.get("data", {})
            title  = pd.get("title", "")
            score  = float(pd.get("score", 0))
            url_p  = pd.get("url", "")
            text   = pd.get("selftext", "")[:300]
            if not title:
                continue
            signals.append(IdeaSignal(
                title=f"Reddit ML: {title}",
                summary=text or title,
                url=url_p,
                source="reddit",
                raw_score=score,
                tags=_extract_tags_from_text(title + " " + text),
            ))
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.debug("[novelty_scanner] Reddit parse error: %s", exc)

    logger.debug("[novelty_scanner] Reddit ML: %d signals", len(signals))
    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# Relevance Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_tags_from_text(text: str) -> list[str]:
    """Extract keyword tags from freeform text."""
    lower = text.lower()
    return [kw for kw in _RELEVANCE_KEYWORDS if kw in lower]


def score_relevance(signal: IdeaSignal) -> float:
    """
    Score a signal's relevance to MYCONEX by keyword matching.
    Returns 0.0–1.0.
    """
    combined = (signal.title + " " + signal.summary).lower()
    total_weight = 0.0
    for keyword, weight in _RELEVANCE_KEYWORDS.items():
        if keyword in combined:
            total_weight += weight
    # Normalise: 5 perfect matches → score 1.0
    return min(1.0, total_weight / 5.0)


def _deduplicate(signals: list[IdeaSignal]) -> list[IdeaSignal]:
    """Remove near-duplicate signals (same URL or very similar title)."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[IdeaSignal] = []
    for sig in signals:
        # Normalise title for fuzzy comparison
        norm_title = re.sub(r"\W+", " ", sig.title.lower()).strip()
        if sig.url in seen_urls or norm_title in seen_titles:
            continue
        if sig.url:
            seen_urls.add(sig.url)
        seen_titles.add(norm_title)
        unique.append(sig)
    return unique


# ═══════════════════════════════════════════════════════════════════════════════
# Proposal Generation (LLM-backed)
# ═══════════════════════════════════════════════════════════════════════════════

_PROPOSAL_PROMPT = """You are the MYCONEX novelty scanner — an autonomous agent that converts
frontier AI research signals into concrete implementation proposals for the MYCONEX codebase.

MYCONEX is a distributed AI mesh system with these capabilities:
- RLMAgent: recursive LLM orchestration, task decomposition, delegation
- MoE expert chain: flash-moe → Nous 8B → Nous 70B → OpenRouter → Ollama
- PersistentPythonREPL, CodebaseIndex, WebPageReader tools
- Mesh networking (NATS pub/sub), Redis state, Qdrant vector store
- Discord gateway, REST API, autonomous optimization loop
- BitNet 1-bit inference, llama-cpp-python GGUF backends
- Document ingestion, multi-source intelligence aggregation

Frontier signal:
  Title: {title}
  Source: {source}
  Summary: {summary}
  URL: {url}
  Tags: {tags}
  Relevance score: {relevance_score:.2f}

Based on this signal, generate a concrete implementation proposal for MYCONEX.
Focus on: what specific code could be added or improved, which file it would go in,
and what the practical benefit would be.

Respond with JSON only:
{{
  "title": "short action-oriented title (max 60 chars)",
  "description": "2-3 sentences: what the signal suggests and why it's valuable for MYCONEX",
  "target_file": "relative/path/to/best/target/file.py (use existing files when possible)",
  "impact": "high|medium|low",
  "category": "feature|performance|quality|bugfix",
  "proposed_change": "concrete description of the change in plain English (3-5 sentences)",
  "priority_score": 0.0-1.0
}}"""


async def _generate_proposal(signal: IdeaSignal, agent) -> Optional[NoveltyProposal]:
    """Use the LLM agent to generate an implementation proposal from a signal."""
    prompt = _PROPOSAL_PROMPT.format(
        title=signal.title,
        source=signal.source,
        summary=signal.summary[:500],
        url=signal.url,
        tags=", ".join(signal.tags[:10]),
        relevance_score=signal.relevance_score,
    )
    try:
        raw = await agent.chat(
            [
                {"role": "system", "content": "You are a senior software architect. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            logger.debug("[novelty_scanner] no JSON in proposal for: %s", signal.title[:50])
            return None
        data = json.loads(json_match.group())
        return NoveltyProposal(
            signal=signal,
            title=data.get("title", signal.title)[:80],
            description=data.get("description", ""),
            target_file=data.get("target_file", ""),
            impact=data.get("impact", "medium"),
            category=data.get("category", "feature"),
            proposed_change=data.get("proposed_change", ""),
            priority_score=float(data.get("priority_score", signal.relevance_score)),
        )
    except Exception as exc:
        logger.warning("[novelty_scanner] proposal generation error: %s", exc)
        return None


def _proposal_from_signal_no_llm(signal: IdeaSignal) -> NoveltyProposal:
    """Fallback: create a proposal directly from the signal without LLM."""
    # Guess target file from tags
    tag_to_file = {
        "gguf": "core/gateway/python_repl.py",
        "bitnet": "core/gateway/python_repl.py",
        "1-bit": "core/gateway/python_repl.py",
        "inference": "orchestration/agents/rlm_agent.py",
        "quantization": "core/gateway/python_repl.py",
        "agent": "orchestration/agents/rlm_agent.py",
        "multi-agent": "orchestration/agent_roster.py",
        "orchestration": "orchestration/agents/rlm_agent.py",
        "memory": "orchestration/agents/context_manager.py",
        "rag": "orchestration/agents/context_manager.py",
        "retrieval augmented": "orchestration/agents/context_manager.py",
        "vector store": "core/memory/vector_store.py",
        "mesh": "core/gateway/mesh_gateway.py",
        "moe": "integrations/moe_hermes_integration.py",
        "self-improvement": "core/autonomous_loop.py",
    }
    target = "orchestration/agents/rlm_agent.py"  # sensible default
    for tag in signal.tags:
        if tag in tag_to_file:
            target = tag_to_file[tag]
            break

    return NoveltyProposal(
        signal=signal,
        title=f"Integrate: {signal.title[:50]}",
        description=(
            f"Frontier signal from {signal.source}: {signal.title}. "
            f"{signal.summary[:200]}"
        ),
        target_file=target,
        impact="medium",
        category="feature",
        proposed_change=(
            f"Investigate and integrate insights from: {signal.url}. "
            f"Relevant to MYCONEX via tags: {', '.join(signal.tags[:5])}. "
            f"Explore adding this capability to {target}."
        ),
        priority_score=signal.relevance_score,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Queue Management
# ═══════════════════════════════════════════════════════════════════════════════

class NoveltyQueue:
    """
    File-backed queue of NoveltyProposals.
    Consumed one at a time by the AutonomousOptimizationLoop.
    """

    def __init__(self, path: Path = _NOVELTY_QUEUE) -> None:
        self._path = path
        _MYCONEX_DIR.mkdir(parents=True, exist_ok=True)

    def enqueue(self, proposals: list[NoveltyProposal]) -> int:
        """Append proposals to the queue. Returns count added."""
        existing = self._load()
        # Avoid duplicates by proposal_id
        existing_ids = {p["proposal_id"] for p in existing}
        new_proposals = [
            {**asdict(p), "signal": asdict(p.signal)}
            for p in proposals
            if p.proposal_id not in existing_ids and not p.consumed
        ]
        existing.extend(new_proposals)
        self._save(existing)
        logger.info("[novelty_queue] enqueued %d new proposals (total=%d)",
                    len(new_proposals), len(existing))
        return len(new_proposals)

    def dequeue(self) -> Optional[dict]:
        """
        Pop the highest-priority unconsumed proposal.
        Returns the proposal dict (ImprovementOpportunity-compatible) or None.
        """
        proposals = self._load()
        unconsumed = [p for p in proposals if not p.get("consumed", False)]
        if not unconsumed:
            return None

        # Sort by priority_score descending
        unconsumed.sort(key=lambda p: p.get("priority_score", 0.0), reverse=True)
        top = unconsumed[0]
        top["consumed"] = True

        # Write back
        for p in proposals:
            if p["proposal_id"] == top["proposal_id"]:
                p["consumed"] = True
        self._save(proposals)

        logger.info("[novelty_queue] dequeued: %s (score=%.2f)",
                    top.get("title", "?")[:60], top.get("priority_score", 0))
        return top

    def peek(self) -> Optional[dict]:
        """Return the top proposal without consuming it."""
        proposals = self._load()
        unconsumed = [p for p in proposals if not p.get("consumed", False)]
        if not unconsumed:
            return None
        unconsumed.sort(key=lambda p: p.get("priority_score", 0.0), reverse=True)
        return unconsumed[0]

    def depth(self) -> int:
        """Number of unconsumed proposals in the queue."""
        return sum(1 for p in self._load() if not p.get("consumed", False))

    def clear_consumed(self) -> int:
        """Remove consumed proposals, keeping only unconsumed. Returns removed count."""
        proposals = self._load()
        active = [p for p in proposals if not p.get("consumed", False)]
        removed = len(proposals) - len(active)
        self._save(active)
        return removed

    def _load(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            return json.loads(self._path.read_text()) or []
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, proposals: list[dict]) -> None:
        try:
            self._path.write_text(json.dumps(proposals, indent=2))
        except OSError as exc:
            logger.error("[novelty_queue] save failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Scanner
# ═══════════════════════════════════════════════════════════════════════════════

class NoveltyScanner:
    """
    Autonomous frontier scanner for MYCONEX.

    Fetches ideas from multiple sources in parallel, ranks them by relevance,
    generates LLM-backed implementation proposals, and enqueues them for the
    AutonomousOptimizationLoop to pick up.

    Usage:
        scanner = NoveltyScanner(agent=rlm_agent)
        await scanner.run_once()
        scanner.inject_into_loop(autonomous_loop)
        await autonomous_loop.run()          # loop will consume proposals
    """

    # Minimum relevance score to include in proposals
    RELEVANCE_THRESHOLD: float = 0.20

    # How many top signals to turn into proposals per scan
    MAX_PROPOSALS_PER_SCAN: int = 5

    # Default scan interval in hours
    DEFAULT_INTERVAL_HOURS: float = 6.0

    def __init__(
        self,
        agent=None,                             # RLMAgent (optional; uses LLM for proposals)
        queue: Optional[NoveltyQueue] = None,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        max_proposals: int = MAX_PROPOSALS_PER_SCAN,
        scan_interval_hours: float = DEFAULT_INTERVAL_HOURS,
        enabled_sources: Optional[list[str]] = None,  # None = all sources
    ) -> None:
        self.agent = agent
        self.queue = queue or NoveltyQueue()
        self.relevance_threshold = relevance_threshold
        self.max_proposals = max_proposals
        self.scan_interval_hours = scan_interval_hours
        self.enabled_sources = set(enabled_sources) if enabled_sources else {
            "huggingface", "github", "arxiv", "pwc", "hn", "reddit"
        }
        self._stop = asyncio.Event()
        self._scan_history: list[ScanReport] = []
        self._last_scan_at: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────────────────

    async def run_once(self) -> ScanReport:
        """Execute a single scan cycle. Returns the ScanReport."""
        report = ScanReport()
        try:
            # 1. Fetch
            all_signals = await self._fetch_all(report)

            # 2. Score + deduplicate
            for sig in all_signals:
                sig.relevance_score = score_relevance(sig)
            all_signals = _deduplicate(all_signals)
            report.signals_found = len(all_signals)

            # 3. Filter by threshold and rank
            relevant = [
                s for s in all_signals
                if s.relevance_score >= self.relevance_threshold
            ]
            relevant.sort(key=lambda s: s.combined_score(), reverse=True)
            top_signals = relevant[:self.max_proposals]

            logger.info(
                "[novelty_scanner] scan %s: %d signals → %d relevant → %d selected",
                report.scan_id, len(all_signals), len(relevant), len(top_signals),
            )

            # 4. Generate proposals
            proposals = await self._generate_proposals(top_signals)
            report.proposals_generated = len(proposals)

            # 5. Enqueue
            if proposals:
                added = self.queue.enqueue(proposals)
                report.proposals_queued = added

            # 6. Log scan
            self._last_scan_at = time.time()
            report.completed_at = self._last_scan_at
            self._scan_history.append(report)
            self._write_novelty_log(report, proposals)

            logger.info(
                "[novelty_scanner] scan complete: %d proposals queued (queue depth=%d)",
                report.proposals_queued, self.queue.depth(),
            )

        except Exception as exc:
            import traceback
            report.error = traceback.format_exc(limit=5)
            report.completed_at = time.time()
            logger.error("[novelty_scanner] scan error: %s", exc)

        return report

    async def run(
        self,
        scan_interval_hours: Optional[float] = None,
        max_scans: Optional[int] = None,
    ) -> None:
        """
        Run the novelty scanner on a schedule.

        Args:
            scan_interval_hours: Override default interval.
            max_scans:           Stop after N scans (None = forever).
        """
        if scan_interval_hours is not None:
            self.scan_interval_hours = scan_interval_hours
        interval_s = self.scan_interval_hours * 3600

        scan_count = 0
        logger.info(
            "[novelty_scanner] starting (interval=%.1fh, max_scans=%s, sources=%s)",
            self.scan_interval_hours, max_scans, sorted(self.enabled_sources),
        )

        while not self._stop.is_set():
            await self.run_once()
            scan_count += 1
            if max_scans is not None and scan_count >= max_scans:
                logger.info("[novelty_scanner] reached max_scans=%d, stopping", max_scans)
                break
            if not self._stop.is_set():
                logger.debug("[novelty_scanner] sleeping %.1fh until next scan", self.scan_interval_hours)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=interval_s)
                except asyncio.TimeoutError:
                    pass  # expected — time to scan again

    def stop(self) -> None:
        """Signal the scanner loop to stop."""
        self._stop.set()

    def inject_into_loop(self, loop) -> None:
        """
        Register this scanner with an AutonomousOptimizationLoop.

        The loop will call `dequeue_proposal()` at the start of each cycle;
        if a proposal is available, it takes priority over the LLM-generated one.
        """
        if hasattr(loop, "register_novelty_scanner"):
            loop.register_novelty_scanner(self)
            logger.info("[novelty_scanner] injected into autonomous loop")
        else:
            logger.warning(
                "[novelty_scanner] loop does not support register_novelty_scanner() — "
                "update autonomous_loop.py to enable proposal injection"
            )

    def dequeue_proposal(self) -> Optional[dict]:
        """Pop the top proposal (for use by AutonomousOptimizationLoop)."""
        return self.queue.dequeue()

    def status(self) -> dict:
        """Return current scanner status."""
        return {
            "enabled_sources": sorted(self.enabled_sources),
            "scan_interval_hours": self.scan_interval_hours,
            "scans_run": len(self._scan_history),
            "last_scan_at": (
                datetime.fromtimestamp(self._last_scan_at, tz=timezone.utc).isoformat()
                if self._last_scan_at else None
            ),
            "queue_depth": self.queue.depth(),
            "relevance_threshold": self.relevance_threshold,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _fetch_all(self, report: ScanReport) -> list[IdeaSignal]:
        """Run all enabled source fetchers in parallel."""
        fetchers: dict[str, Any] = {
            "huggingface": fetch_huggingface_trending,
            "github": fetch_github_trending,
            "arxiv": fetch_arxiv_rss,
            "pwc": fetch_papers_with_code,
            "hn": fetch_hacker_news_rss,
            "reddit": fetch_reddit_ml,
        }

        tasks = {}
        for name, fn in fetchers.items():
            if name in self.enabled_sources:
                tasks[name] = asyncio.create_task(fn())

        report.sources_tried = len(tasks)
        all_signals: list[IdeaSignal] = []

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.warning("[novelty_scanner] source %s failed: %s", name, result)
            elif isinstance(result, list):
                all_signals.extend(result)
                report.sources_succeeded += 1

        return all_signals

    async def _generate_proposals(
        self, signals: list[IdeaSignal]
    ) -> list[NoveltyProposal]:
        """Generate implementation proposals for the top signals."""
        proposals: list[NoveltyProposal] = []

        if self.agent is not None:
            # LLM-backed proposals (in parallel, up to 3 at a time)
            sem = asyncio.Semaphore(3)

            async def _safe_generate(sig: IdeaSignal) -> Optional[NoveltyProposal]:
                async with sem:
                    return await _generate_proposal(sig, self.agent)

            raw = await asyncio.gather(
                *[_safe_generate(s) for s in signals], return_exceptions=True
            )
            for item in raw:
                if isinstance(item, NoveltyProposal):
                    proposals.append(item)
                elif isinstance(item, Exception):
                    logger.debug("[novelty_scanner] proposal error: %s", item)
        else:
            # No LLM — use rule-based fallback
            for sig in signals:
                proposals.append(_proposal_from_signal_no_llm(sig))

        # Sort by priority score
        proposals.sort(key=lambda p: p.priority_score, reverse=True)
        return proposals

    def _write_novelty_log(
        self, report: ScanReport, proposals: list[NoveltyProposal]
    ) -> None:
        """Append a scan summary to the JSONL history log."""
        try:
            _MYCONEX_DIR.mkdir(parents=True, exist_ok=True)
            entry = {
                "scan_id": report.scan_id,
                "timestamp": datetime.utcnow().isoformat(),
                "duration_s": report.duration_s,
                "signals_found": report.signals_found,
                "proposals_queued": report.proposals_queued,
                "sources": {"tried": report.sources_tried, "ok": report.sources_succeeded},
                "top_proposals": [
                    {"title": p.title, "score": p.priority_score, "source": p.signal.source}
                    for p in proposals[:5]
                ],
            }
            with open(_NOVELTY_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.debug("[novelty_scanner] history log write failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_novelty_scanner(
    agent=None,
    scan_interval_hours: float = 6.0,
    relevance_threshold: float = 0.20,
    max_proposals: int = 5,
    enabled_sources: Optional[list[str]] = None,
) -> NoveltyScanner:
    """Create a NoveltyScanner with sensible defaults."""
    return NoveltyScanner(
        agent=agent,
        scan_interval_hours=scan_interval_hours,
        relevance_threshold=relevance_threshold,
        max_proposals=max_proposals,
        enabled_sources=enabled_sources,
    )
