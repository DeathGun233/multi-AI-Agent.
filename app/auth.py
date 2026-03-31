from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, status
from sqlalchemy import select

from app.config import Settings
from app.db import Database, UserAccountRecord


ROLE_VIEWER = "viewer"
ROLE_OPERATOR = "operator"
ROLE_REVIEWER = "reviewer"
ROLE_ADMIN = "admin"

PBKDF2_PREFIX = "pbkdf2_sha256"
PBKDF2_ROUNDS = 120_000


@dataclass(frozen=True)
class AuthUser:
    username: str
    display_name: str
    role: str


@dataclass(frozen=True)
class RoleCapabilities:
    can_view: bool
    can_run: bool
    can_review: bool
    can_admin: bool


@dataclass(frozen=True)
class UserSeed:
    username: str
    password: str
    display_name: str
    role: str


class AuthService:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self.database.initialize()
        self.ensure_seeded_users()

    def ensure_seeded_users(self) -> None:
        seeds = self._load_seed_users(self.settings.users_json)
        with self.database.session() as session:
            existing = {
                record.username: record
                for record in session.scalars(select(UserAccountRecord)).all()
            }
            for seed in seeds:
                if seed.username in existing:
                    continue
                session.add(
                    UserAccountRecord(
                        username=seed.username,
                        password_hash=self.hash_password(seed.password),
                        display_name=seed.display_name,
                        role=seed.role,
                        is_active=True,
                    )
                )

    def authenticate(self, username: str, password: str) -> AuthUser | None:
        record = self._get_user_record(username.strip())
        if record is None or not record.is_active:
            return None
        if not self.verify_password(password, record.password_hash):
            return None
        return AuthUser(username=record.username, display_name=record.display_name, role=record.role)

    def build_session_cookie(self, user: AuthUser) -> str:
        payload = {
            "username": user.username,
        }
        encoded = base64.urlsafe_b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
        signature = hmac.new(
            self.settings.secret_key.encode("utf-8"),
            encoded.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{encoded}.{signature}"

    def read_session_cookie(self, token: str | None) -> AuthUser | None:
        if not token or "." not in token:
            return None
        encoded, signature = token.rsplit(".", 1)
        expected = hmac.new(
            self.settings.secret_key.encode("utf-8"),
            encoded.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            return None
        username = payload.get("username")
        record = self._get_user_record(username)
        if record is None or not record.is_active:
            return None
        return AuthUser(username=record.username, display_name=record.display_name, role=record.role)

    def get_user_from_request(self, request: Request) -> AuthUser | None:
        return self.read_session_cookie(request.cookies.get(self.settings.session_cookie_name))

    def require_user(self, request: Request) -> AuthUser:
        user = self.get_user_from_request(request)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
        return user

    def require_roles(self, request: Request, *roles: str) -> AuthUser:
        user = self.require_user(request)
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="permission denied")
        return user

    @staticmethod
    def capabilities_for(user: AuthUser | None) -> RoleCapabilities:
        if user is None:
            return RoleCapabilities(can_view=False, can_run=False, can_review=False, can_admin=False)
        return RoleCapabilities(
            can_view=True,
            can_run=user.role in {ROLE_OPERATOR, ROLE_ADMIN},
            can_review=user.role in {ROLE_REVIEWER, ROLE_ADMIN},
            can_admin=user.role == ROLE_ADMIN,
        )

    @staticmethod
    def hash_password(password: str, rounds: int = PBKDF2_ROUNDS) -> str:
        salt = secrets.token_hex(16)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), rounds)
        return f"{PBKDF2_PREFIX}${rounds}${salt}${derived.hex()}"

    @staticmethod
    def verify_password(password: str, stored_password: str) -> bool:
        if stored_password.startswith(f"{PBKDF2_PREFIX}$"):
            try:
                _, rounds_text, salt, stored_digest = stored_password.split("$", 3)
                rounds = int(rounds_text)
            except ValueError:
                return False
            candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), rounds).hex()
            return hmac.compare_digest(candidate, stored_digest)
        return hmac.compare_digest(password, stored_password)

    def _get_user_record(self, username: str | None) -> UserAccountRecord | None:
        if not username:
            return None
        with self.database.session() as session:
            return session.get(UserAccountRecord, username)

    @staticmethod
    def _load_seed_users(raw_users: str | None) -> list[UserSeed]:
        if raw_users:
            payload = json.loads(raw_users)
            return [
                UserSeed(
                    username=item["username"],
                    password=item["password"],
                    display_name=item.get("display_name", item["username"]),
                    role=item.get("role", ROLE_VIEWER),
                )
                for item in payload
            ]
        return [
            UserSeed(username="admin", password="admin123", display_name="系统管理员", role=ROLE_ADMIN),
            UserSeed(username="reviewer", password="reviewer123", display_name="审核负责人", role=ROLE_REVIEWER),
            UserSeed(username="operator", password="operator123", display_name="流程运营", role=ROLE_OPERATOR),
            UserSeed(username="viewer", password="viewer123", display_name="只读观察员", role=ROLE_VIEWER),
        ]
