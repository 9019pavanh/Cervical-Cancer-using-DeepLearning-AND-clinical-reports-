"""PostgreSQL authentication with salted password hashes and expiring sessions."""

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import psycopg
from fastapi import HTTPException, Request, Response, status
from psycopg import errors
from database import connect, initialize_database
SESSION_COOKIE = "cervical_ai_session"
SESSION_DAYS = 7
PBKDF2_ITERATIONS = 600_000
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

def validate_registration(name: str, email: str, password: str, role: str) -> None:
    if len(name.strip()) < 2 or len(name.strip()) > 80:
        raise HTTPException(400, "Name must contain 2–80 characters")
    if not EMAIL_PATTERN.match(email.strip()) or len(email) > 254:
        raise HTTPException(400, "Enter a valid email address")
    if len(password) < 10 or len(password) > 128:
        raise HTTPException(400, "Password must contain 10–128 characters")
    if not any(character.isalpha() for character in password) or not any(
        character.isdigit() for character in password
    ):
        raise HTTPException(400, "Password must include at least one letter and one number")
    if role not in {"Researcher", "Clinician"}:
        raise HTTPException(400, "Role must be Researcher or Clinician")


def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)


def register_user(name: str, email: str, password: str, role: str = "Researcher") -> Dict:
    validate_registration(name, email, password, role)
    initialize_database()
    normalized_email = email.strip().lower()
    salt = secrets.token_bytes(16)
    password_hash = hash_password(password, salt)
    now = datetime.now(timezone.utc)
    try:
        with connect() as connection:
            row = connection.execute(
                """
                INSERT INTO users(name,email,role,password_hash,salt,created_at,updated_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (name.strip(), normalized_email, role, password_hash, salt, now, now),
            ).fetchone()
    except errors.UniqueViolation as exc:
        raise HTTPException(409, "An account already exists for this email") from exc
    return {"id": row["id"], "name": name.strip(), "email": normalized_email, "role": role}


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    initialize_database()
    with connect() as connection:
        row = connection.execute(
            "SELECT id,name,email,role,password_hash,salt FROM users WHERE email=%s",
            (email.strip().lower(),),
        ).fetchone()
    if row is None:
        hash_password(password, secrets.token_bytes(16))
        return None
    supplied = hash_password(password, bytes(row["salt"]))
    if not hmac.compare_digest(supplied, bytes(row["password_hash"])):
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"]}


def create_session(response: Response, user_id: int, secure: bool = False) -> None:
    initialize_database()
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=SESSION_DAYS)
    with connect() as connection:
        connection.execute("DELETE FROM sessions WHERE expires_at <= %s", (now,))
        connection.execute(
            """
            INSERT INTO sessions(token_hash,user_id,expires_at,created_at)
            VALUES(%s,%s,%s,%s)
            """,
            (token_hash, user_id, expires, now),
        )
    response.set_cookie(
        SESSION_COOKIE,
        raw_token,
        max_age=SESSION_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )


def delete_session(request: Request, response: Response) -> None:
    token = request.cookies.get(SESSION_COOKIE)
    response.delete_cookie(SESSION_COOKIE, path="/")
    if not token:
        return
    try:
        initialize_database()
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash=%s", (token_hash,))
    except HTTPException:
        return


def current_user(request: Request) -> Optional[Dict]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    initialize_database()
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    with connect() as connection:
        row = connection.execute(
            """
            SELECT users.id,users.name,users.email,users.role
            FROM sessions JOIN users ON users.id=sessions.user_id
            WHERE sessions.token_hash=%s AND sessions.expires_at>%s
            """,
            (token_hash, now),
        ).fetchone()
    return dict(row) if row else None


def require_user(request: Request) -> Dict:
    user = current_user(request)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    return user
