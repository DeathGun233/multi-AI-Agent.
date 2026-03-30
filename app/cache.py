from __future__ import annotations

import json
from typing import Any

from redis import Redis
from redis.exceptions import RedisError


class CacheStore:
    def __init__(self, redis_url: str | None) -> None:
        self.redis_url = redis_url
        self._client: Redis | None = None
        self._memory: dict[str, str] = {}
        if redis_url:
            try:
                self._client = Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
            except RedisError:
                self._client = None

    @property
    def enabled(self) -> bool:
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except RedisError:
            return False

    def set_json(self, key: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        if self.enabled and self._client:
            try:
                self._client.set(key, payload, ex=3600)
                return
            except RedisError:
                pass
        self._memory[key] = payload

    def get_json(self, key: str) -> dict[str, Any] | None:
        raw: str | None = None
        if self.enabled and self._client:
            try:
                raw = self._client.get(key)
            except RedisError:
                raw = None
        if raw is None:
            raw = self._memory.get(key)
        if not raw:
            return None
        return json.loads(raw)
