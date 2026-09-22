from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.importance import lookup

load_dotenv(PROJECT_ROOT / ".env")

# How each status moves a team, before importance scaling.
STATUS_WEIGHT = {
    "out": -1.0,
    "suspended": -1.0,
    "doubtful": -0.5,
    "returning": 0.8,
    "available": 0.0,
}

MAX_TEAM_SCORE = 1.0
DEFAULT_MAX_RESULTS = 8


@dataclass
class NewsFact:
    player: str
    side: str
    status: str
    source: str = ""
    importance: float = 0.0
    impact: float = 0.0


@dataclass
class NewsResult:
    home_score: float = 0.0
    away_score: float = 0.0
    facts: list[NewsFact] = field(default_factory=list)
    articles: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def delta(self) -> float:
        return self.home_score - self.away_score

    def explain(self) -> str:
        if self.error:
            return f"News branch unavailable: {self.error}"
        if not self.facts:
            return "No relevant team news found. Prediction is the network alone."
        lines = []
        for f in sorted(self.facts, key=lambda x: -abs(x.impact)):
            lines.append(
                f"  {f.player} ({f.side}, {f.status}): "
                f"importance {f.importance:.2f}, impact {f.impact:+.2f}"
            )
        return "\n".join(lines)


def search_news(home_team: str, away_team: str,
                max_results: int = DEFAULT_MAX_RESULTS) -> list[dict]:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        return []

    queries = [
        f"{home_team} vs {away_team} team news injuries",
        f"{home_team} injury news",
        f"{away_team} injury news",
    ]

    seen, articles = set(), []
    try:
        with DDGS() as ddgs:
            for query in queries:
                for r in ddgs.text(query, timelimit="w",
                                   max_results=max_results):
                    href = r.get("href") or r.get("url", "")
                    if href in seen:
                        continue
                    seen.add(href)
                    articles.append({
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "href": href,
                    })
    except Exception:
        return articles

    return articles[: max_results * 2]


def _llm_client():
    from openai import OpenAI

    key = os.getenv("LLM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("LLM_API_KEY is not set in .env")
    return OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=key,
    )


def extract_facts(articles: list[dict], home_team: str,
                  away_team: str) -> list[dict]:
    if not articles:
        return []

    snippets = "\n\n".join(
        f"[{i + 1}] {a['title']}\n{a['body'][:600]}"
        for i, a in enumerate(articles[:10])
    )
    user = (
        f"Home team: {home_team}\nAway team: {away_team}\n\n"
        f"Snippets:\n\n{snippets}"
    )

    client = _llm_client()
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        max_tokens=800,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": user},
        ],
    )

    raw = resp.choices[0].message.content or "{}"
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    items = data.get("items", [])
    return [
        i for i in items
        if isinstance(i, dict)
        and i.get("player")
        and i.get("side") in ("home", "away")
        and i.get("status") in STATUS_WEIGHT
    ]


def score_facts(items: list[dict], importance_table, home_team: str,
                away_team: str) -> tuple[float, float, list[NewsFact]]:
    facts: list[NewsFact] = []
    totals = {"home": 0.0, "away": 0.0}

    for item in items:
        side = item["side"]
        team = home_team if side == "home" else away_team
        imp = lookup(importance_table, item["player"], team=team)
        impact = STATUS_WEIGHT[item["status"]] * imp

        totals[side] += impact
        facts.append(NewsFact(
            player=item["player"],
            side=side,
            status=item["status"],
            importance=imp,
            impact=impact,
        ))

    home = max(-MAX_TEAM_SCORE, min(MAX_TEAM_SCORE, totals["home"]))
    away = max(-MAX_TEAM_SCORE, min(MAX_TEAM_SCORE, totals["away"]))
    return home, away, facts


def analyse_fixture(home_team: str, away_team: str,
                    importance_table) -> NewsResult:
    articles = search_news(home_team, away_team)
    if not articles:
        return NewsResult(articles=[], error="no articles found")

    try:
        items = extract_facts(articles, home_team, away_team)
    except Exception as exc:
        return NewsResult(articles=articles, error=str(exc))

    home, away, facts = score_facts(items, importance_table,
                                    home_team, away_team)
    return NewsResult(home_score=home, away_score=away,
                      facts=facts, articles=articles)
