# app/accounts.py
"""User accounts and password verification. S3 owns auth.

Replaces the open-access placeholder, which accepted any email with any
non-empty password whenever ``APP_EMAIL``/``APP_PASSWORD`` were unset — i.e.
by default. That is not acceptable for a study that collects identifiable
responses from student teachers under IRB, and the failure was silent.

Design notes, since this is the part that is worth getting right:

**Hashing.** ``hashlib.scrypt`` at RFC 7914's interactive parameters
(n=2^14, r=8, p=1) — memory-hard, ~190ms on the deployment laptop, and in the
standard library, so no dependency to install offline. The parameters are
stored alongside each hash, so they can be raised later without invalidating
existing accounts.

**Fail closed.** With no accounts in the database, nobody signs in. The old
behaviour let everybody in; the safe direction for "not configured yet" is
locked, not open.

**No account enumeration.** An unknown email and a wrong password give the
same message *and* cost the same time — an unknown email is verified against a
dummy hash rather than returning early, so response time does not reveal which
addresses are registered.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import unicodedata
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import delete, func, insert, select, update

from app.database.db import get_engine, users

#: RFC 7914 interactive parameters. 16 MB of memory per verification.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
KEY_BYTES = 32

#: Rejected outright. Not a policy engine — just a floor that stops "1234".
MIN_PASSWORD_LENGTH = 10


class AccountError(Exception):
    """Something the caller asked for cannot be done. Safe to show a human."""


@dataclass(frozen=True)
class Account:
    email: str
    name: str
    role: str = "student_teacher"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _normalise(password: str) -> bytes:
    """NFKC-normalise before hashing.

    Without this, a password typed with a composed accent on one keyboard and a
    combining accent on another is two different byte strings and the second
    login fails for no visible reason.
    """
    return unicodedata.normalize("NFKC", password).encode("utf-8")


def hash_password(password: str) -> str:
    """Hash a password into ``scrypt$n$r$p$salt$key``."""
    salt = os.urandom(SALT_BYTES)
    key = hashlib.scrypt(_normalise(password), salt=salt, n=SCRYPT_N,
                         r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_BYTES)
    return "$".join((
        "scrypt", str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P),
        base64.b64encode(salt).decode(), base64.b64encode(key).decode(),
    ))


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of a password against a stored hash."""
    try:
        scheme, n, r, p, salt_b64, key_b64 = encoded.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(key_b64)
        actual = hashlib.scrypt(_normalise(password), salt=salt, n=int(n),
                                r=int(r), p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


#: Verified against when the email is unknown, so a missing account costs the
#: same wall-clock time as a wrong password. Computed once at import.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(32))


def normalise_email(email: str) -> str:
    return email.strip().lower()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _require_engine():
    engine = get_engine()
    if engine is None:
        raise AccountError(
            "No database is reachable, so accounts cannot be read or written.")
    return engine


def count_accounts() -> int:
    """How many active accounts exist. Zero means nobody can sign in."""
    engine = get_engine()
    if engine is None:
        return 0
    with engine.connect() as connection:
        return int(connection.execute(
            select(func.count())
            .select_from(users)
            .where(users.c.is_active.is_(True))).scalar_one())


def list_accounts() -> list[Account]:
    engine = get_engine()
    if engine is None:
        return []
    with engine.connect() as connection:
        rows = connection.execute(
            select(users.c.email, users.c.name, users.c.role)
            .where(users.c.is_active.is_(True))
            .order_by(users.c.email)).mappings().all()
    return [Account(**dict(row)) for row in rows]


def create_account(email: str, password: str, name: str = "",
                   role: str = "student_teacher") -> Account:
    """Create one account. Raises AccountError if it already exists or is weak."""
    engine = _require_engine()
    email = normalise_email(email)
    if "@" not in email:
        raise AccountError(f"{email!r} is not an email address.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AccountError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")

    name = name or default_name(email)
    with engine.begin() as connection:
        exists = connection.execute(
            select(users.c.email).where(users.c.email == email)).first()
        if exists:
            raise AccountError(f"An account for {email} already exists.")
        connection.execute(insert(users).values(
            email=email, name=name, role=role,
            password_hash=hash_password(password)))
    return Account(email=email, name=name, role=role)


def set_password(email: str, password: str) -> None:
    engine = _require_engine()
    email = normalise_email(email)
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AccountError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    with engine.begin() as connection:
        result = connection.execute(
            update(users).where(users.c.email == email)
            .values(password_hash=hash_password(password)))
        if result.rowcount == 0:
            raise AccountError(f"No account for {email}.")


def delete_account(email: str) -> None:
    engine = _require_engine()
    with engine.begin() as connection:
        result = connection.execute(
            delete(users).where(users.c.email == normalise_email(email)))
        if result.rowcount == 0:
            raise AccountError(f"No account for {normalise_email(email)}.")


def verify_credentials(email: str, password: str) -> Optional[Account]:
    """Return the account if the credentials are right, else ``None``.

    Deliberately gives no hint about *which* half was wrong, and spends the
    same time either way.
    """
    engine = get_engine()
    if engine is None:
        verify_password(password, _DUMMY_HASH)
        return None

    email = normalise_email(email)
    with engine.connect() as connection:
        row = connection.execute(
            select(users.c.email, users.c.name, users.c.role,
                   users.c.password_hash, users.c.is_active)
            .where(users.c.email == email)).mappings().first()

    if row is None or not row["is_active"]:
        # Still hash, so an unregistered address is not detectable by timing.
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return Account(email=row["email"], name=row["name"], role=row["role"])


def default_name(email: str) -> str:
    """"a.serwah@st.knust.edu.gh" -> "A Serwah", for seeding a display name."""
    local = email.split("@", 1)[0]
    words = [w for w in local.replace(".", " ").replace("_", " ").split() if w]
    return " ".join(w.capitalize() for w in words) or email


# ---------------------------------------------------------------------------
# Bootstrap CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse
    import getpass

    parser = argparse.ArgumentParser(
        prog="python -m app.accounts",
        description="Manage dashboard accounts. The database is created on "
                    "first use (SQLite by default).")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="create an account")
    add.add_argument("email")
    add.add_argument("--name", default="")
    add.add_argument("--role", default="student_teacher",
                     choices=["student_teacher", "tutor", "researcher"])
    add.add_argument("--password", default=None,
                     help="omit to be prompted, or use --generate")
    add.add_argument("--generate", action="store_true",
                     help="generate a random password and print it once")

    sub.add_parser("list", help="list accounts")

    pw = sub.add_parser("passwd", help="change a password")
    pw.add_argument("email")
    pw.add_argument("--password", default=None)
    pw.add_argument("--generate", action="store_true")

    rm = sub.add_parser("remove", help="delete an account")
    rm.add_argument("email")

    args = parser.parse_args()

    def resolve_password() -> str:
        if getattr(args, "generate", False):
            generated = secrets.token_urlsafe(12)
            print(f"Generated password: {generated}")
            print("Copy it now — it is not stored anywhere in readable form.")
            return generated
        if args.password:
            return args.password
        first = getpass.getpass("Password: ")
        if first != getpass.getpass("Repeat password: "):
            raise AccountError("Passwords did not match.")
        return first

    try:
        if args.command == "add":
            account = create_account(args.email, resolve_password(),
                                     args.name, args.role)
            print(f"Created {account.email} ({account.name}, {account.role}).")
        elif args.command == "list":
            accounts = list_accounts()
            if not accounts:
                print("No accounts. Nobody can sign in until one is created.")
            for account in accounts:
                print(f"  {account.email:<40} {account.name:<24} {account.role}")
        elif args.command == "passwd":
            set_password(args.email, resolve_password())
            print(f"Password updated for {normalise_email(args.email)}.")
        elif args.command == "remove":
            delete_account(args.email)
            print(f"Removed {normalise_email(args.email)}.")
    except AccountError as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
