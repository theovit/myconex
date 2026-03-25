"""
MYCONEX RLM Context Manager
============================
Phase 3: Hierarchical context trees, token budgeting, and persistent memory.

Components:
  ContextFrame          — one frame in the RLM call stack
  RLMContextManager     — manages the nested frame tree, tracks token budgets
  SessionMemory         — in-process memory that watches patterns per session
                          (inspired by letta-ai/claude-subconscious)
  PersistentMemoryStore — cross-session JSON storage with context summarization
                          (inspired by Obsidian+Claude workflow)

Token estimation: uses a simple word-count heuristic (~1.3 tokens/word).
Swap in tiktoken/tokenizer for exact counts if needed.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Rough token estimate: 1 word ≈ 1.3 tokens (conservative, works without tiktoken)
_TOKENS_PER_WORD: float = 1.3
_MEMORY_DIR = Path.home() / ".myconex" / "memory"


# ─── Token Estimation ─────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Estimate token count without a tokenizer (good enough for budget tracking)."""
    return max(1, int(len(text.split()) * _TOKENS_PER_WORD))


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate token cost of an OpenAI-style messages list."""
    total = 0
    for m in messages:
        # 4 overhead tokens per message (role + delimiters)
        total += 4 + estimate_tokens(m.get("content", ""))
    return total + 2  # reply priming


# ─── Priority Levels ─────────────────────────────────────────────────────────

class Priority:
    """Token retention priority for context pruning."""
    CRITICAL = 4    # system prompts, task definitions — never pruned
    HIGH = 3        # recent user messages, active task context
    MEDIUM = 2      # intermediate results, working notes
    LOW = 1         # old turns, completed sub-task outputs


# ─── Context Frame ────────────────────────────────────────────────────────────

@dataclass
class ContextFrame:
    """
    One frame in the RLM agent call stack.

    A Manager agent creates a root frame; each delegate() call pushes a child
    frame.  Frames form a tree, not a linear stack, because parallel delegation
    spawns sibling frames.
    """
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_name: str = ""
    task_type: str = ""
    task_id: str = ""
    depth: int = 0
    tokens_budget: int = 4096         # total tokens this frame may consume
    tokens_used: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Parent/child relationships
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)   # child frame_ids

    # Per-frame messages with priority tags
    messages: list[dict] = field(default_factory=list)
    # messages format: {"role": ..., "content": ..., "priority": int, "tokens": int}

    # Result summary (written when frame completes)
    result_summary: Optional[str] = None

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.tokens_budget - self.tokens_used)

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None

    def add_message(self, role: str, content: str, priority: int = Priority.MEDIUM) -> None:
        """Add a message to this frame and track token usage."""
        tokens = estimate_tokens(content) + 4
        self.messages.append({
            "role": role,
            "content": content,
            "priority": priority,
            "tokens": tokens,
        })
        self.tokens_used += tokens

    def to_messages(self) -> list[dict]:
        """Return messages in OpenAI format (strip internal fields)."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def prune(self, target_tokens: int) -> int:
        """
        Prune low-priority messages until token usage is at or below target.

        Args:
            target_tokens: Desired token budget ceiling.

        Returns:
            Number of messages pruned.
        """
        pruned = 0
        for priority in (Priority.LOW, Priority.MEDIUM):
            if self.tokens_used <= target_tokens:
                break
            removable = [
                (i, m) for i, m in enumerate(self.messages)
                if m.get("priority", Priority.MEDIUM) == priority
                and m.get("role") != "system"
            ]
            # Remove oldest first
            for i, m in removable:
                if self.tokens_used <= target_tokens:
                    break
                self.tokens_used -= m.get("tokens", 0)
                self.messages[i] = None   # mark for removal
                pruned += 1
            self.messages = [m for m in self.messages if m is not None]
        return pruned

    def complete(self, summary: Optional[str] = None) -> None:
        self.completed_at = time.time()
        self.result_summary = summary


# ─── RLM Context Manager ──────────────────────────────────────────────────────

class RLMContextManager:
    """
    Manages a tree of ContextFrames for an RLM agent session.

    The root frame holds the user's original task.  Each delegate() call creates
    a child frame.  Token budgets flow top-down: a parent's remaining budget is
    split among its children.

    Usage:
        cm = RLMContextManager(total_budget=16384)
        root = cm.push_frame("inference-primary", "chat", "task-001")
        root.add_message("system", "You are a helpful agent.", Priority.CRITICAL)
        root.add_message("user", "Analyse this 500-line file.", Priority.HIGH)

        child = cm.push_frame("code-specialist", "code", "task-002",
                              parent_id=root.frame_id, budget_fraction=0.5)
        ...
        cm.pop_frame(child.frame_id, summary="Found 3 issues.")
        cm.pop_frame(root.frame_id, summary="Analysis complete.")
    """

    def __init__(self, total_budget: int = 16384, max_depth: int = 4) -> None:
        self.total_budget = total_budget
        self.max_depth = max_depth
        self._frames: dict[str, ContextFrame] = {}
        self._root_id: Optional[str] = None

    # ── Frame Management ──────────────────────────────────────────────────────

    def push_frame(
        self,
        agent_name: str,
        task_type: str,
        task_id: str,
        parent_id: Optional[str] = None,
        budget_fraction: float = 1.0,
    ) -> ContextFrame:
        """
        Push a new context frame onto the tree.

        Args:
            agent_name:      Name of the agent owning this frame.
            task_type:       Task type being executed.
            task_id:         Task identifier.
            parent_id:       Parent frame ID (None for root).
            budget_fraction: Fraction of parent's remaining budget to allocate
                             (ignored for root; uses total_budget).

        Returns:
            The newly created ContextFrame.
        """
        if parent_id:
            parent = self._frames.get(parent_id)
            if parent is None:
                raise ValueError(f"Parent frame not found: {parent_id}")
            if parent.depth >= self.max_depth:
                raise ValueError(
                    f"Max delegation depth ({self.max_depth}) reached "
                    f"at frame {parent_id}"
                )
            depth = parent.depth + 1
            budget = int(parent.tokens_remaining * budget_fraction)
            budget = max(512, budget)   # minimum viable budget
        else:
            depth = 0
            budget = self.total_budget

        frame = ContextFrame(
            agent_name=agent_name,
            task_type=task_type,
            task_id=task_id,
            depth=depth,
            tokens_budget=budget,
            parent_id=parent_id,
        )
        self._frames[frame.frame_id] = frame

        if parent_id and parent_id in self._frames:
            self._frames[parent_id].children.append(frame.frame_id)

        if self._root_id is None:
            self._root_id = frame.frame_id

        logger.debug(
            "[ctx] pushed frame %s (agent=%s, depth=%d, budget=%d)",
            frame.frame_id, agent_name, depth, budget,
        )
        return frame

    def pop_frame(self, frame_id: str, summary: Optional[str] = None) -> Optional[ContextFrame]:
        """
        Mark a frame as complete and record its result summary.

        Args:
            frame_id: Frame to complete.
            summary:  Short natural-language summary of the frame's output.

        Returns:
            The completed frame, or None if not found.
        """
        frame = self._frames.get(frame_id)
        if frame:
            frame.complete(summary)
            logger.debug(
                "[ctx] popped frame %s (tokens_used=%d/%d)",
                frame_id, frame.tokens_used, frame.tokens_budget,
            )
        return frame

    def get_frame(self, frame_id: str) -> Optional[ContextFrame]:
        return self._frames.get(frame_id)

    @property
    def root(self) -> Optional[ContextFrame]:
        return self._frames.get(self._root_id) if self._root_id else None

    def total_tokens_used(self) -> int:
        """Sum of tokens used across all frames."""
        return sum(f.tokens_used for f in self._frames.values())

    def prune_all(self, target_fraction: float = 0.75) -> int:
        """
        Prune low-priority messages across all frames to reach target_fraction
        of each frame's budget.

        Returns:
            Total messages pruned.
        """
        total_pruned = 0
        for frame in self._frames.values():
            if frame.tokens_used > frame.tokens_budget * target_fraction:
                pruned = frame.prune(int(frame.tokens_budget * target_fraction))
                if pruned:
                    logger.debug(
                        "[ctx] pruned %d messages from frame %s", pruned, frame.frame_id
                    )
                total_pruned += pruned
        return total_pruned

    def flatten_context(self, frame_id: Optional[str] = None) -> list[dict]:
        """
        Flatten the frame tree into a single messages list for the given frame
        (includes parent chain summaries for context).

        Args:
            frame_id: Frame to flatten from (default: root).

        Returns:
            OpenAI-format messages list.
        """
        fid = frame_id or self._root_id
        if fid is None:
            return []

        messages: list[dict] = []
        # Walk ancestors and inject their result summaries as system context
        frame = self._frames.get(fid)
        ancestor_summaries: list[str] = []
        pid = frame.parent_id if frame else None
        while pid:
            parent = self._frames.get(pid)
            if parent and parent.result_summary:
                ancestor_summaries.insert(
                    0, f"[Context from {parent.agent_name}]: {parent.result_summary}"
                )
            pid = parent.parent_id if parent else None

        if ancestor_summaries:
            messages.append({
                "role": "system",
                "content": "\n".join(ancestor_summaries),
            })

        # Add this frame's messages
        if frame:
            messages.extend(frame.to_messages())

        return messages

    def status(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "total_tokens_used": self.total_tokens_used(),
            "frames": len(self._frames),
            "max_depth": self.max_depth,
            "frame_summary": [
                {
                    "id": f.frame_id,
                    "agent": f.agent_name,
                    "task": f.task_type,
                    "depth": f.depth,
                    "tokens": f"{f.tokens_used}/{f.tokens_budget}",
                    "complete": f.is_complete,
                }
                for f in self._frames.values()
            ],
        }


# ─── Session Memory ───────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single persistent memory item."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    key: str = ""
    content: str = ""
    category: str = "general"   # "fact", "preference", "pattern", "summary"
    importance: float = 0.5     # 0.0 (ephemeral) → 1.0 (critical)
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    source: str = ""            # agent or session that created this entry


class SessionMemory:
    """
    In-process session memory: watches interaction patterns and builds a
    cumulative memory within a single agent session.

    Inspired by letta-ai/claude-subconscious: the agent maintains a "subconscious"
    layer that observes its own outputs and extracts reusable patterns.

    Entries are scored by recency and access frequency.  Low-importance entries
    are pruned when the memory grows beyond max_entries.
    """

    def __init__(self, session_id: str, max_entries: int = 200) -> None:
        self.session_id = session_id
        self.max_entries = max_entries
        self._entries: dict[str, MemoryEntry] = {}
        self._interaction_log: list[dict] = []  # lightweight task log for pattern detection

    # ── Core API ──────────────────────────────────────────────────────────────

    def store(
        self,
        key: str,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        source: str = "",
    ) -> MemoryEntry:
        """Store or update a memory entry."""
        entry = self._entries.get(key)
        if entry:
            entry.content = content
            entry.category = category
            entry.importance = max(entry.importance, importance)
            if tags:
                entry.tags = list(set(entry.tags) | set(tags))
        else:
            entry = MemoryEntry(
                key=key,
                content=content,
                category=category,
                importance=importance,
                tags=tags or [],
                source=source or self.session_id,
            )
            self._entries[key] = entry
            self._evict_if_needed()
        return entry

    def retrieve(self, key: str) -> Optional[str]:
        """Retrieve a memory entry by exact key."""
        entry = self._entries.get(key)
        if entry:
            entry.accessed_at = time.time()
            entry.access_count += 1
            return entry.content
        return None

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Keyword search over memory entry content and keys."""
        query_words = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries.values():
            text = f"{entry.key} {entry.content} {' '.join(entry.tags)}".lower()
            text_words = set(re.findall(r"\w+", text))
            overlap = len(query_words & text_words)
            if overlap:
                # Combine keyword overlap with importance and recency
                recency = 1.0 / (1.0 + (time.time() - entry.accessed_at) / 3600)
                score = overlap * (1 + entry.importance) * (1 + recency * 0.3)
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:top_k]]
        for e in results:
            e.accessed_at = time.time()
            e.access_count += 1
        return results

    def log_interaction(self, task_type: str, success: bool, duration_ms: float, model: str = "") -> None:
        """Record a task interaction for pattern analysis."""
        self._interaction_log.append({
            "task_type": task_type,
            "success": success,
            "duration_ms": duration_ms,
            "model": model,
            "ts": time.time(),
        })
        # Keep only last 500 interactions
        if len(self._interaction_log) > 500:
            self._interaction_log = self._interaction_log[-500:]

    def extract_patterns(self) -> dict:
        """
        Analyse the interaction log for patterns.

        Returns:
            Dict of pattern metrics: success_rate, avg_duration, slow_tasks,
            most_common_types, error_rate_by_type.
        """
        if not self._interaction_log:
            return {}

        by_type: dict[str, list[dict]] = {}
        for entry in self._interaction_log:
            by_type.setdefault(entry["task_type"], []).append(entry)

        patterns: dict[str, Any] = {}
        for task_type, entries in by_type.items():
            total = len(entries)
            successes = sum(1 for e in entries if e["success"])
            avg_ms = sum(e["duration_ms"] for e in entries) / total
            patterns[task_type] = {
                "count": total,
                "success_rate": round(successes / total, 3),
                "avg_duration_ms": round(avg_ms, 1),
                "slow": avg_ms > 5000,
            }
        return patterns

    def format_for_context(self, query: str = "", max_entries: int = 10) -> str:
        """Format relevant memory entries as a compact context block."""
        entries = self.search(query, top_k=max_entries) if query else list(self._entries.values())[:max_entries]
        if not entries:
            return ""
        lines = ["[Memory Context]"]
        for e in entries:
            lines.append(f"• [{e.category}] {e.key}: {e.content[:200]}")
        return "\n".join(lines)

    def status(self) -> dict:
        return {
            "session_id": self.session_id,
            "entries": len(self._entries),
            "interactions_logged": len(self._interaction_log),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        # Evict lowest-importance, least-recently-accessed entries
        evict_count = len(self._entries) - self.max_entries + 10  # headroom
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: (e.importance, e.accessed_at),
        )
        for entry in sorted_entries[:evict_count]:
            del self._entries[entry.key]


# ─── Persistent Memory Store ──────────────────────────────────────────────────

class PersistentMemoryStore:
    """
    Cross-session persistent memory with context summarization.

    Inspired by the Obsidian+Claude workflow: memories are stored as structured
    JSON files in ~/.myconex/memory/.  On load, older memories are summarized
    to stay within context budget.

    Each store is keyed by a namespace (e.g., "global", "agent-xyz", "discord").
    """

    def __init__(
        self,
        namespace: str = "global",
        memory_dir: Path = _MEMORY_DIR,
        max_entries: int = 500,
        summarize_threshold: int = 400,  # summarize when entries exceed this
    ) -> None:
        self.namespace = namespace
        self.memory_dir = memory_dir
        self.max_entries = max_entries
        self.summarize_threshold = summarize_threshold
        self._file = memory_dir / f"{namespace}.json"
        self._entries: dict[str, MemoryEntry] = {}
        self._loaded = False

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load entries from disk."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self._file.exists():
            self._loaded = True
            return
        try:
            data = json.loads(self._file.read_text())
            for raw in data.get("entries", []):
                entry = MemoryEntry(
                    entry_id=raw.get("entry_id", str(uuid.uuid4())[:8]),
                    key=raw["key"],
                    content=raw["content"],
                    category=raw.get("category", "general"),
                    importance=raw.get("importance", 0.5),
                    created_at=raw.get("created_at", time.time()),
                    accessed_at=raw.get("accessed_at", time.time()),
                    access_count=raw.get("access_count", 0),
                    tags=raw.get("tags", []),
                    source=raw.get("source", ""),
                )
                self._entries[entry.key] = entry
            logger.info(
                "[persistent_memory:%s] loaded %d entries", self.namespace, len(self._entries)
            )
        except Exception as exc:
            logger.warning("[persistent_memory:%s] load failed: %s", self.namespace, exc)
        self._loaded = True

    def save(self) -> None:
        """Persist entries to disk."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "namespace": self.namespace,
                "saved_at": time.time(),
                "entries": [
                    {
                        "entry_id": e.entry_id,
                        "key": e.key,
                        "content": e.content,
                        "category": e.category,
                        "importance": e.importance,
                        "created_at": e.created_at,
                        "accessed_at": e.accessed_at,
                        "access_count": e.access_count,
                        "tags": e.tags,
                        "source": e.source,
                    }
                    for e in self._entries.values()
                ],
            }
            self._file.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.error("[persistent_memory:%s] save failed: %s", self.namespace, exc)

    # ── API ───────────────────────────────────────────────────────────────────

    def store(
        self,
        key: str,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        source: str = "",
        autosave: bool = True,
    ) -> MemoryEntry:
        """Store or update a memory entry, optionally auto-saving to disk."""
        if not self._loaded:
            self.load()
        existing = self._entries.get(key)
        if existing:
            existing.content = content
            existing.importance = max(existing.importance, importance)
            if tags:
                existing.tags = list(set(existing.tags) | set(tags))
            entry = existing
        else:
            entry = MemoryEntry(
                key=key, content=content, category=category,
                importance=importance, tags=tags or [], source=source,
            )
            self._entries[key] = entry

        if len(self._entries) > self.max_entries:
            self._evict()
        if autosave:
            self.save()
        return entry

    def retrieve(self, key: str) -> Optional[str]:
        if not self._loaded:
            self.load()
        entry = self._entries.get(key)
        if entry:
            entry.accessed_at = time.time()
            entry.access_count += 1
            return entry.content
        return None

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        if not self._loaded:
            self.load()
        query_words = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries.values():
            text = f"{entry.key} {entry.content} {' '.join(entry.tags)}".lower()
            text_words = set(re.findall(r"\w+", text))
            overlap = len(query_words & text_words)
            if overlap:
                recency = 1.0 / (1.0 + (time.time() - entry.accessed_at) / 86400)
                score = overlap * (1 + entry.importance) * (1 + recency * 0.2)
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def delete(self, key: str, autosave: bool = True) -> bool:
        if not self._loaded:
            self.load()
        removed = self._entries.pop(key, None) is not None
        if removed and autosave:
            self.save()
        return removed

    def summarize_old_entries(
        self,
        llm_summarize_fn=None,
        max_age_days: float = 30.0,
    ) -> str:
        """
        Summarize entries older than max_age_days into a compact digest.

        If llm_summarize_fn is provided (async fn(text) -> str), it's called to
        generate the summary; otherwise a simple concatenation is used.

        Returns:
            The summary text.
        """
        if not self._loaded:
            self.load()

        cutoff = time.time() - max_age_days * 86400
        old = [e for e in self._entries.values() if e.created_at < cutoff]
        if not old:
            return ""

        raw = "\n".join(f"- [{e.category}] {e.key}: {e.content[:200]}" for e in old)
        summary = f"[Summary of {len(old)} older memories from {self.namespace}]\n{raw[:2000]}"

        # Remove old entries and replace with the summary
        for e in old:
            del self._entries[e.key]
        self.store(
            key=f"_summary_{int(time.time())}",
            content=summary,
            category="summary",
            importance=0.8,
            source="auto-summarize",
            autosave=False,
        )
        self.save()
        return summary

    def format_for_context(self, query: str = "", max_entries: int = 10) -> str:
        """Format relevant entries as a compact context block for LLM injection."""
        if not self._loaded:
            self.load()
        entries = self.search(query, top_k=max_entries) if query else sorted(
            self._entries.values(), key=lambda e: e.importance, reverse=True
        )[:max_entries]
        if not entries:
            return ""
        lines = [f"[Persistent Memory: {self.namespace}]"]
        for e in entries:
            lines.append(f"• [{e.category}] {e.key}: {e.content[:300]}")
        return "\n".join(lines)

    def status(self) -> dict:
        if not self._loaded:
            self.load()
        return {
            "namespace": self.namespace,
            "entries": len(self._entries),
            "file": str(self._file),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evict(self) -> None:
        evict_count = len(self._entries) - self.max_entries + 20
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: (e.importance, e.accessed_at),
        )
        for e in sorted_entries[:evict_count]:
            del self._entries[e.key]


# ─── Shared singletons ────────────────────────────────────────────────────────

_PERSISTENT_STORES: dict[str, PersistentMemoryStore] = {}


def get_persistent_store(namespace: str = "global") -> PersistentMemoryStore:
    """Return (creating if needed) a PersistentMemoryStore for the given namespace."""
    if namespace not in _PERSISTENT_STORES:
        store = PersistentMemoryStore(namespace=namespace)
        store.load()
        _PERSISTENT_STORES[namespace] = store
    return _PERSISTENT_STORES[namespace]
