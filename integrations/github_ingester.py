"""
MYCONEX GitHub Ingester
------------------------
Polls GitHub repositories for new commits, pull requests, and issues,
extracts topics and summaries, and stores them in the knowledge base.

Watches:
  1. Repos listed in GITHUB_REPOS (comma-separated owner/repo)
  2. Repos starred by GITHUB_USERNAME (if set)

Env vars:
  GITHUB_TOKEN          — Personal access token (recommended for rate limits)
  GITHUB_REPOS          — comma-separated list: "owner/repo,owner2/repo2"
  GITHUB_USERNAME       — GitHub username to watch starred repos
  GITHUB_INGEST_INTERVAL — polling interval in minutes (default: 60)
  GITHUB_INGEST_TYPES   — comma-separated: "commits,prs,issues" (default: all)
  GITHUB_MAX_PER_REPO   — max items per repo per poll (default: 10)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE         = Path.home() / ".myconex"
_GH_FILE      = _BASE / "github_insights.json"
_GH_STAMP     = _BASE / "github_last_poll.json"

_TOKEN        = os.getenv("GITHUB_TOKEN", "")
_REPOS_ENV    = os.getenv("GITHUB_REPOS", "")
_USERNAME     = os.getenv("GITHUB_USERNAME", "")
_INTERVAL     = int(os.getenv("GITHUB_INGEST_INTERVAL", "60"))
_TYPES_ENV    = os.getenv("GITHUB_INGEST_TYPES", "commits,prs,issues")
_MAX_PER_REPO = int(os.getenv("GITHUB_MAX_PER_REPO", "10"))

_INGEST_COMMITS = "commits" in _TYPES_ENV
_INGEST_PRS     = "prs"     in _TYPES_ENV
_INGEST_ISSUES  = "issues"  in _TYPES_ENV

_GH_API = "https://api.github.com"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _gh_request(path: str, params: dict | None = None) -> Any:
    url = f"{_GH_API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if _TOKEN:
        headers["Authorization"] = f"Bearer {_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def _load(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def _save(path: Path, data: Any) -> None:
    _BASE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Repo discovery ────────────────────────────────────────────────────────────

def _get_repos() -> list[str]:
    repos = [r.strip() for r in _REPOS_ENV.split(",") if r.strip()]
    if _USERNAME and not repos:
        try:
            starred = _gh_request(f"/users/{_USERNAME}/starred", {"per_page": "30"})
            repos = [r["full_name"] for r in starred if isinstance(r, dict)][:20]
        except Exception as exc:
            logger.warning("[github] could not fetch starred repos: %s", exc)
    return repos


# ── Item fetching ─────────────────────────────────────────────────────────────

def _fetch_commits(repo: str, since: str | None) -> list[dict]:
    params: dict = {"per_page": str(_MAX_PER_REPO)}
    if since:
        params["since"] = since
    try:
        items = _gh_request(f"/repos/{repo}/commits", params)
        out = []
        for c in (items or []):
            commit = c.get("commit", {})
            msg    = commit.get("message", "").split("\n")[0][:200]
            out.append({
                "type":    "commit",
                "repo":    repo,
                "sha":     c.get("sha", "")[:12],
                "message": msg,
                "author":  commit.get("author", {}).get("name", ""),
                "url":     c.get("html_url", ""),
                "ts":      commit.get("author", {}).get("date", ""),
                "text":    f"[{repo}] commit: {msg}",
                "topics":  _topics_from(msg),
            })
        return out
    except Exception as exc:
        logger.debug("[github] commits fetch failed for %s: %s", repo, exc)
        return []


def _fetch_prs(repo: str, since: str | None) -> list[dict]:
    try:
        items = _gh_request(f"/repos/{repo}/pulls",
                            {"state": "all", "sort": "updated",
                             "direction": "desc", "per_page": str(_MAX_PER_REPO)})
        cutoff = since or ""
        out = []
        for pr in (items or []):
            updated = pr.get("updated_at", "")
            if cutoff and updated < cutoff:
                continue
            title = pr.get("title", "")[:200]
            body  = (pr.get("body") or "")[:400]
            out.append({
                "type":    "pr",
                "repo":    repo,
                "number":  pr.get("number"),
                "title":   title,
                "state":   pr.get("state", ""),
                "url":     pr.get("html_url", ""),
                "ts":      updated,
                "text":    f"[{repo}] PR #{pr.get('number')}: {title}\n{body}",
                "topics":  _topics_from(f"{title} {body}"),
            })
        return out
    except Exception as exc:
        logger.debug("[github] PRs fetch failed for %s: %s", repo, exc)
        return []


def _fetch_issues(repo: str, since: str | None) -> list[dict]:
    try:
        params: dict = {"state": "all", "sort": "updated",
                        "direction": "desc", "per_page": str(_MAX_PER_REPO)}
        if since:
            params["since"] = since
        items = _gh_request(f"/repos/{repo}/issues", params)
        out = []
        for iss in (items or []):
            if iss.get("pull_request"):
                continue  # skip PRs listed as issues
            title = iss.get("title", "")[:200]
            body  = (iss.get("body") or "")[:400]
            labels = [lb["name"] for lb in (iss.get("labels") or [])]
            out.append({
                "type":    "issue",
                "repo":    repo,
                "number":  iss.get("number"),
                "title":   title,
                "state":   iss.get("state", ""),
                "labels":  labels,
                "url":     iss.get("html_url", ""),
                "ts":      iss.get("updated_at", ""),
                "text":    f"[{repo}] Issue #{iss.get('number')}: {title}\n{body}",
                "topics":  _topics_from(f"{title} {body} {' '.join(labels)}"),
            })
        return out
    except Exception as exc:
        logger.debug("[github] issues fetch failed for %s: %s", repo, exc)
        return []


def _topics_from(text: str) -> list[str]:
    stop = {"the","a","an","is","in","on","at","to","for","of","and","or","but",
            "this","that","it","with","from","by","as","be","was","were","are"}
    words = [w.strip(".,!?:;()[]\"'").lower() for w in text.split()]
    return list(dict.fromkeys(w for w in words if len(w) > 3 and w not in stop))[:6]


# ── Main ingest loop ──────────────────────────────────────────────────────────

class GitHubIngester:
    def __init__(self) -> None:
        self._stamps: dict[str, str] = _load(_GH_STAMP, {})

    async def run_forever(self) -> None:
        repos = _get_repos()
        if not repos:
            logger.info("[github] no repos configured — set GITHUB_REPOS or GITHUB_USERNAME")
            return
        logger.info("[github] watching %d repo(s), interval=%dm", len(repos), _INTERVAL)
        while True:
            await self._poll(repos)
            await asyncio.sleep(_INTERVAL * 60)

    async def _poll(self, repos: list[str]) -> None:
        all_items: list[dict] = []
        for repo in repos:
            since = self._stamps.get(repo)
            items: list[dict] = []
            if _INGEST_COMMITS:
                items.extend(await asyncio.get_event_loop().run_in_executor(
                    None, lambda r=repo, s=since: _fetch_commits(r, s)))
            if _INGEST_PRS:
                items.extend(await asyncio.get_event_loop().run_in_executor(
                    None, lambda r=repo, s=since: _fetch_prs(r, s)))
            if _INGEST_ISSUES:
                items.extend(await asyncio.get_event_loop().run_in_executor(
                    None, lambda r=repo, s=since: _fetch_issues(r, s)))

            new_items = [it for it in items if it["ts"] > (since or "")]
            if new_items:
                all_items.extend(new_items)
                latest = max(it["ts"] for it in new_items)
                self._stamps[repo] = latest
                logger.info("[github] %s — %d new item(s)", repo, len(new_items))

        if all_items:
            insights = _load(_GH_FILE, [])
            insights.extend(all_items)
            _save(_GH_FILE, insights[-1000:])
            _save(_GH_STAMP, self._stamps)

            # Embed into vector store
            try:
                from integrations.knowledge_store import embed_and_store_batch
                embed_queue = [
                    {"text": it["text"], "source": "github",
                     "title": it.get("title") or it.get("message", ""),
                     "topics": it.get("topics", []),
                     "ts": it["ts"]}
                    for it in all_items
                ]
                await embed_and_store_batch(embed_queue)
            except Exception as exc:
                logger.warning("[github] embedding failed: %s", exc)

            # Knowledge graph
            try:
                from core.knowledge_graph import get_graph
                for it in all_items[:10]:
                    await get_graph().ingest_text(it["text"], source="github",
                                                  title=it.get("title",""))
            except Exception:
                pass
