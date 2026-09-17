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

**Provisioning.** There is no self-registration. A cohort is created from its
class list in one step (``import``), which writes the generated passwords to a
sheet for distribution; a forgotten password is replaced with
``passwd <email> --generate``.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import os
import secrets
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy import delete, func, insert, select, update

from app.database.db import REPO_ROOT, get_engine, users

#: RFC 7914 interactive parameters. 16 MB of memory per verification.
SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
KEY_BYTES = 32

#: Rejected outright. Not a policy engine — just a floor that stops "1234".
MIN_PASSWORD_LENGTH = 10

ROLES = ("student_teacher", "tutor", "researcher")

#: Columns an import file may have. Only ``email`` is required.
IMPORT_COLUMNS = ("email", "name", "role")


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
    if role not in ROLES:
        raise AccountError(f"Unknown role {role!r}; use one of {', '.join(ROLES)}.")
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


def generate_password() -> str:
    """16 URL-safe characters, about 96 bits. Easy to read out, hard to guess."""
    return secrets.token_urlsafe(12)


# ---------------------------------------------------------------------------
# Bulk import
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImportResult:
    created: list[Account]
    #: Addresses that already had an account. Left exactly as they were.
    skipped: list[str]
    #: Where the passwords went, or ``None`` if nothing was created.
    sheet: Optional[Path]


def _read_import(csv_path: Path, default_role: str) -> list[Account]:
    """Parse and check a whole class list. Raises with *every* problem found."""
    if default_role not in ROLES:
        raise AccountError(f"Unknown role {default_role!r}; use one of "
                           f"{', '.join(ROLES)}.")
    try:
        # utf-8-sig: Excel's "CSV UTF-8" starts with a byte-order mark.
        text = Path(csv_path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise AccountError(f"No such file: {csv_path}") from None
    except UnicodeDecodeError:
        raise AccountError(f'{csv_path} is not UTF-8. In Excel, save it as '
                           f'"CSV UTF-8".') from None

    reader = csv.DictReader(io.StringIO(text))
    headers = [(h or "").strip().lower() for h in reader.fieldnames or []]
    reader.fieldnames = headers
    if "email" not in headers:
        raise AccountError(f"{csv_path} needs an 'email' column; found "
                           f"{', '.join(headers) or 'no header row'}.")
    # Strict on purpose: an unrecognised "full name" column would otherwise be
    # dropped silently and every account named after its email address.
    unknown = [h for h in headers if h not in IMPORT_COLUMNS]
    if unknown:
        raise AccountError(f"Unrecognised column(s): {', '.join(unknown)}. "
                           f"Use only: {', '.join(IMPORT_COLUMNS)}.")

    problems: list[str] = []
    seen: dict[str, int] = {}
    accounts: list[Account] = []
    for line, record in enumerate(reader, start=2):
        values = {key: (record.get(key) or "").strip() for key in headers}
        if not any(values.values()):
            continue
        email = normalise_email(values["email"])
        role = values.get("role", "").lower() or default_role
        if "@" not in email or any(ch.isspace() for ch in email):
            problems.append(f"line {line}: {values['email']!r} is not an email "
                            f"address")
        if role not in ROLES:
            problems.append(f"line {line}: unknown role {role!r}")
        if email in seen:
            problems.append(f"line {line}: {email} is also on line {seen[email]}")
        seen.setdefault(email, line)
        accounts.append(Account(email=email,
                                name=values.get("name") or default_name(email),
                                role=role))

    if problems:
        raise AccountError("Nothing was imported. Fix these and run it again:\n  "
                           + "\n  ".join(problems))
    if not accounts:
        raise AccountError(f"{csv_path} has no accounts in it.")
    return accounts


def _check_sheet_path(path: Path) -> None:
    """Refuse a password sheet that could be overwritten or committed."""
    resolved = Path(path).resolve()
    if resolved.exists():
        raise AccountError(f"{path} already exists. Choose a new file, so an "
                           f"earlier password sheet is not overwritten.")
    try:
        inside = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return
    if inside.parts[:1] != ("data",):
        raise AccountError(
            "That would put the password sheet inside the repository, where it "
            "could be committed. Write it outside the repository, or under "
            "data/ (which is gitignored).")


def import_accounts(csv_path: Path, sheet_path: Path,
                    default_role: str = "student_teacher") -> ImportResult:
    """Create an account for every row of a class list, with generated passwords.

    The file needs an ``email`` header; ``name`` and ``role`` columns are
    optional. It is all or nothing: one bad row and no account is created, so
    a class list is never left half-imported. Addresses that already have an
    account are skipped untouched, so re-running an import never resets
    anybody's password.

    The passwords are written to *sheet_path* and nowhere else. The sheet is
    written *before* the accounts are committed, so there is never an account
    whose password nobody has, and it is removed again if the commit fails.
    """
    engine = _require_engine()
    _check_sheet_path(sheet_path)
    wanted = _read_import(csv_path, default_role)

    with engine.connect() as connection:
        existing = set(connection.execute(select(users.c.email)).scalars())
    new = [account for account in wanted if account.email not in existing]
    skipped = [account.email for account in wanted if account.email in existing]
    if not new:
        return ImportResult([], skipped, None)

    issued = [(account, generate_password()) for account in new]
    # Hashed before the transaction opens: ~190ms each, and a class of 200
    # should not hold a write lock for 40 seconds.
    rows = [dict(email=account.email, name=account.name, role=account.role,
                 password_hash=hash_password(password))
            for account, password in issued]

    sheet = Path(sheet_path)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    with sheet.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("email", "name", "role", "password"))
        for account, password in issued:
            writer.writerow((account.email, account.name, account.role, password))
    try:
        with engine.begin() as connection:
            connection.execute(insert(users), rows)
    except Exception:
        sheet.unlink(missing_ok=True)
        raise
    return ImportResult([account for account, _ in issued], skipped, sheet)


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
    add.add_argument("--role", default="student_teacher", choices=ROLES)
    add.add_argument("--password", default=None,
                     help="omit to be prompted, or use --generate")
    add.add_argument("--generate", action="store_true",
                     help="generate a random password and print it once")

    bulk = sub.add_parser(
        "import", help="create accounts from a CSV class list",
        description="Columns: email (required), name, role. Every row is "
                    "checked first; one bad row and nothing is created. "
                    "Existing accounts are skipped, never reset.")
    bulk.add_argument("csv", type=Path)
    bulk.add_argument("--passwords", type=Path, required=True,
                      help="new file for the generated passwords, outside "
                           "the repository or under data/")
    bulk.add_argument("--role", default="student_teacher", choices=ROLES,
                      help="role for rows that leave the role column blank")

    sub.add_parser("list", help="list accounts")

    pw = sub.add_parser("passwd",
                        help="change a password, or reset a forgotten one "
                             "with --generate")
    pw.add_argument("email")
    pw.add_argument("--password", default=None)
    pw.add_argument("--generate", action="store_true")

    rm = sub.add_parser("remove", help="delete an account")
    rm.add_argument("email")

    args = parser.parse_args()

    def resolve_password() -> str:
        if getattr(args, "generate", False):
            generated = generate_password()
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
        elif args.command == "import":
            print("Checking the file and hashing passwords (about 0.2s per "
                  "account)...")
            result = import_accounts(args.csv, args.passwords, args.role)
            print(f"Created {len(result.created)} account(s).")
            if result.skipped:
                print(f"Skipped {len(result.skipped)} that already exist "
                      f"(passwords unchanged): {', '.join(result.skipped)}")
            if result.sheet:
                print(f"Passwords written to {result.sheet}")
                print("That file is the only copy. Hand the passwords out, "
                      "then delete it.")
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
