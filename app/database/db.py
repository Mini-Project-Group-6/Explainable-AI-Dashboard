# app/database/db.py
"""Submission storage — S3's layer. SQLite by default, PostgreSQL when configured.

Backend selection, in order:

    DATABASE_URL            whatever it points at, used verbatim
    DB_HOST + DB_NAME       PostgreSQL assembled from the discrete DB_* vars
    (neither)               SQLite at data/coteach.db

SQLite is the default because revision tracking is not a nice-to-have here: the
study measures lesson-plan quality *across drafts*, so a dashboard that forgets
every submission when the tab closes cannot produce the pre/post comparison the
research question depends on. Requiring Postgres before anything is saved put
that at the mercy of whether a docker container happened to be running.

The schema is declared with SQLAlchemy Core rather than raw DDL so one
definition serves both dialects — SERIAL vs AUTOINCREMENT, JSONB vs JSON, and
TIMESTAMPTZ vs DATETIME are the kind of difference that otherwise turns into
two drifting CREATE TABLE strings.

Nothing connects at import. The first draft built an engine and opened a
connection at module scope, which ran on every Streamlit re-run, printed to
stdout, and stalled the app for the TCP timeout whenever the database was
unreachable.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    func,
    insert,
    select,
    true,
)
from sqlalchemy.engine import Engine

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Overridable so a deployment can put the file on a mounted volume.
SQLITE_PATH = Path(os.getenv("COTEACH_DB_PATH", REPO_ROOT / "data" / "coteach.db"))


def database_url() -> str:
    """The URL to connect to. Always returns one — SQLite is the fallback."""
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit

    host, name = os.getenv("DB_HOST"), os.getenv("DB_NAME")
    if host and name:
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "")
        port = os.getenv("DB_PORT", "5432")
        credentials = f"{user}:{password}" if password else user
        return f"postgresql+psycopg2://{credentials}@{host}:{port}/{name}"

    return f"sqlite+pysqlite:///{SQLITE_PATH}"


def backend() -> str:
    """"sqlite" or "postgresql", from whichever URL is in force."""
    return "sqlite" if database_url().startswith("sqlite") else "postgresql"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

metadata = MetaData()

users = Table(
    "users", metadata,
    # Email is the natural key and is what submissions are filed under. Stored
    # lowercased by app.accounts so "A.Serwah@" and "a.serwah@" are one person.
    Column("email", String(320), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("role", String(32), nullable=False, server_default="student_teacher"),
    # Format: scrypt$n$r$p$<salt_b64>$<hash_b64>. Never a bare digest — the
    # parameters travel with the hash so they can be raised later without
    # invalidating existing accounts.
    Column("password_hash", String(256), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False,
           server_default=func.now()),
    Column("is_active", Boolean, nullable=False, server_default=true()),
)

submissions = Table(
    "submissions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_email", String(320), nullable=False),
    Column("source", String(512), nullable=False),
    Column("submitted_at", DateTime(timezone=True), nullable=False,
           server_default=func.now()),
    Column("overall_score", Numeric(5, 2)),
    Column("band", String(64)),
    # Generic JSON: JSONB on PostgreSQL, a TEXT column SQLAlchemy serialises on
    # SQLite. Either way the caller hands over a dict and gets a dict back.
    Column("rubric_scores", JSON, nullable=False),
    Column("contract_version", String(32)),
    Index("submissions_user_time", "user_email", "submitted_at"),
)


@lru_cache(maxsize=1)
def get_engine() -> Optional[Engine]:
    """Build the engine, create the schema, verify it once. ``None`` on failure.

    Cached, so a database that comes up *after* the app does needs a restart to
    be picked up. That trade is deliberate: the alternative is retrying a dead
    connection on every script re-run, which is every widget interaction.
    """
    url = database_url()

    from sqlalchemy import create_engine

    if url.startswith("sqlite"):
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because Streamlit serves each session from a
        # worker thread, and the pooled connection will not be the one that
        # created it.
        connect_args: dict[str, Any] = {"check_same_thread": False, "timeout": 10}
    else:
        connect_args = {"connect_timeout": 3}

    try:
        engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        metadata.create_all(engine)
    except Exception:                                 # noqa: BLE001
        # No server, bad credentials, missing driver, read-only filesystem —
        # all mean the same thing to the caller, and none should take the
        # dashboard down.
        return None
    return engine


def is_persistent() -> bool:
    """True when submissions are actually being written somewhere durable."""
    return get_engine() is not None


def status() -> str:
    """One line for the transparency panel."""
    engine = get_engine()
    if engine is None:
        return (f"Storage unavailable ({backend()} at {database_url()}). "
                f"Revision history is kept for this session only.")
    if backend() == "sqlite":
        return f"SQLite — revision history saved to {SQLITE_PATH}."
    return "PostgreSQL connected — revision history is saved."


# ---------------------------------------------------------------------------
# Reads and writes
# ---------------------------------------------------------------------------

#: Bumped on every write. Part of the read cache's key, so a new submission
#: invalidates the cached history immediately rather than on a timer.
_generation = 0


def record_submission(user_email: str, source: str, overall_score: Optional[float],
                      band: Optional[str], rubric_scores: dict[str, int],
                      contract_version: Optional[str] = None) -> bool:
    """Persist one scored submission. Returns False if it was not saved."""
    global _generation

    engine = get_engine()
    if engine is None:
        return False

    with engine.begin() as connection:
        connection.execute(insert(submissions).values(
            user_email=user_email,
            source=source,
            overall_score=overall_score,
            band=band,
            rubric_scores=rubric_scores,
            contract_version=contract_version,
        ))
    _generation += 1
    return True


@lru_cache(maxsize=64)
def _history(user_email: str, limit: int,
             generation: int) -> tuple[dict[str, Any], ...]:
    engine = get_engine()
    if engine is None:
        return ()
    query = (
        select(submissions.c.source, submissions.c.submitted_at,
               submissions.c.overall_score, submissions.c.band,
               submissions.c.rubric_scores)
        .where(submissions.c.user_email == user_email)
        .order_by(submissions.c.submitted_at.desc(), submissions.c.id.desc())
        .limit(limit)
    )
    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()
    return tuple(dict(row) for row in reversed(rows))


def submission_history(user_email: str, limit: int = 20) -> list[dict[str, Any]]:
    """This student teacher's submissions, oldest first.

    Cached on (user, generation). Streamlit re-executes the whole script on
    every widget interaction, so without this the history tab issued a query
    per click — including clicks on other tabs.
    """
    return list(_history(user_email, limit, _generation))
