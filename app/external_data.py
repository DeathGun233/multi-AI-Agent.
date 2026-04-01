from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.config import Settings


class ExternalDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExternalTicketBatch:
    provider: str
    records: list[dict[str, Any]]
    summary: dict[str, Any]


class ExternalDataService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_support_tickets(self, source: dict[str, Any]) -> ExternalTicketBatch:
        provider = str(source.get("provider", "")).strip().lower()
        if provider == "github_issues":
            return self._load_github_issues(source)
        if provider == "nyc_311":
            return self._load_nyc_311(source)
        if provider == "stack_overflow":
            return self._load_stack_overflow(source)
        if provider == "hacker_news":
            return self._load_hacker_news(source)
        raise ExternalDataError(f"unsupported provider: {provider}")

    def _load_github_issues(self, source: dict[str, Any]) -> ExternalTicketBatch:
        repo = str(source.get("repo", "")).strip() or "fastapi/fastapi"
        if "/" not in repo:
            raise ExternalDataError("GitHub 仓库格式必须是 owner/repo")
        state = str(source.get("state", "open")).strip() or "open"
        per_page = max(1, min(int(source.get("per_page", 5)), 20))
        params = urlencode({"state": state, "per_page": per_page})
        url = f"https://api.github.com/repos/{repo}/issues?{params}"
        payload = self._fetch_json(url)
        if not isinstance(payload, list):
            raise ExternalDataError("github issues response is not a list")
        tickets = []
        for item in payload:
            if "pull_request" in item:
                continue
            tickets.append(
                {
                    "customer": repo,
                    "message": item.get("title", ""),
                    "body": item.get("body") or "",
                    "source_id": f"issue#{item.get('number', '')}",
                    "source_url": item.get("html_url", ""),
                    "labels": [label.get("name", "") for label in item.get("labels", []) if isinstance(label, dict)],
                }
            )
        return ExternalTicketBatch(
            provider="github_issues",
            records=tickets,
            summary={"repo": repo, "state": state, "ticket_count": len(tickets)},
        )

    def _load_nyc_311(self, source: dict[str, Any]) -> ExternalTicketBatch:
        limit = max(1, min(int(source.get("limit", 6)), 20))
        complaint_type = str(source.get("complaint_type", "")).strip()
        borough = str(source.get("borough", "")).strip().upper()
        filters = ["agency is not null", "descriptor is not null"]
        if complaint_type:
            safe_type = complaint_type.replace("'", "''")
            filters.append(f"complaint_type = '{safe_type}'")
        if borough:
            safe_borough = borough.replace("'", "''")
            filters.append(f"borough = '{safe_borough}'")
        params = urlencode(
            {
                "$select": "unique_key,agency,complaint_type,descriptor,borough,incident_address,created_date,status",
                "$where": " AND ".join(filters),
                "$order": "created_date DESC",
                "$limit": limit,
            }
        )
        url = f"https://data.cityofnewyork.us/resource/erm2-nwe9.json?{params}"
        payload = self._fetch_json(url)
        if not isinstance(payload, list):
            raise ExternalDataError("nyc 311 response is not a list")
        tickets = []
        for item in payload:
            descriptor = item.get("descriptor", "")
            complaint = item.get("complaint_type", "")
            tickets.append(
                {
                    "customer": item.get("agency", "NYC 311"),
                    "message": f"{complaint} - {descriptor}".strip(" -"),
                    "body": f"{item.get('borough', '')} {item.get('incident_address', '')}".strip(),
                    "source_id": item.get("unique_key", ""),
                    "source_url": "https://data.cityofnewyork.us/resource/erm2-nwe9",
                    "status": item.get("status", ""),
                }
            )
        return ExternalTicketBatch(
            provider="nyc_311",
            records=tickets,
            summary={"borough": borough, "complaint_type": complaint_type, "ticket_count": len(tickets)},
        )

    def _load_stack_overflow(self, source: dict[str, Any]) -> ExternalTicketBatch:
        tagged = str(source.get("tagged", "")).strip() or "fastapi"
        sort = str(source.get("sort", "votes")).strip() or "votes"
        limit = max(1, min(int(source.get("limit", 5)), 20))
        params = urlencode(
            {
                "order": "desc",
                "sort": sort,
                "site": "stackoverflow",
                "tagged": tagged,
                "pagesize": limit,
                "filter": "default",
            }
        )
        url = f"https://api.stackexchange.com/2.3/questions?{params}"
        payload = self._fetch_json(url)
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ExternalDataError("stack overflow response is invalid")
        tickets = []
        for item in payload["items"]:
            tickets.append(
                {
                    "customer": "Stack Overflow",
                    "message": item.get("title", ""),
                    "body": f"标签：{', '.join(item.get('tags', []))}",
                    "source_id": f"question#{item.get('question_id', '')}",
                    "source_url": item.get("link", ""),
                    "status": "answered" if item.get("is_answered") else "unanswered",
                }
            )
        return ExternalTicketBatch(
            provider="stack_overflow",
            records=tickets,
            summary={"tagged": tagged, "sort": sort, "ticket_count": len(tickets)},
        )

    def _load_hacker_news(self, source: dict[str, Any]) -> ExternalTicketBatch:
        query = str(source.get("query", "")).strip() or "fastapi"
        limit = max(1, min(int(source.get("limit", 5)), 20))
        tags = str(source.get("tags", "story")).strip() or "story"
        params = urlencode({"query": query, "tags": tags, "hitsPerPage": limit})
        url = f"https://hn.algolia.com/api/v1/search_by_date?{params}"
        payload = self._fetch_json(url)
        if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list):
            raise ExternalDataError("hacker news response is invalid")
        tickets = []
        for item in payload["hits"]:
            tickets.append(
                {
                    "customer": "Hacker News",
                    "message": item.get("title") or item.get("story_title") or "",
                    "body": item.get("comment_text") or "",
                    "source_id": item.get("objectID", ""),
                    "source_url": item.get("url") or f"https://news.ycombinator.com/item?id={item.get('objectID', '')}",
                    "status": tags,
                }
            )
        return ExternalTicketBatch(
            provider="hacker_news",
            records=tickets,
            summary={"query": query, "tags": tags, "ticket_count": len(tickets)},
        )

    def _fetch_json(self, url: str) -> Any:
        headers = {
            "Accept": "application/json",
            "User-Agent": "FlowPilot/1.0",
        }
        if self.settings.github_token and "api.github.com" in url:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=self.settings.http_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ExternalDataError(str(exc)) from exc
