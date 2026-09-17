# app/database/db.py
"""Submission storage — S3's layer. PostgreSQL.

Connection, in order:

    DATABASE_URL            whatever it points at, used verbatim
    DB_HOST + DB_NAME       PostgreSQL assembled from the discrete DB_* vars
    (neither)               not configured: nothing is stored, sign-in is closed

The project stores its data in PostgreSQL. There used to be a silent fallback
to a SQLite file when nothing was configured, which meant a machine missing its
.env quietly wrote study data somewhere other than the study database — two
stores, and nobody told. Now an unconfigured app fails closed and says why in
the server log. ``python -m app.database setup`` creates the database and the
.env in one step.

SQLite is still accepted when asked for explicitly
(``DATABASE_URL=sqlite+pysqlite:///...``). The test suite does that, so it runs
without a server; the app itself should not.

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

import logging
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
    UniqueConstraint,
    func,
    insert,
    select,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

NOT_CONFIGURED = ("No database is configured. Set DB_HOST, DB_NAME, DB_USER and "
                  "DB_PASSWORD in .env (python -m app.database setup writes "
                  "them), or DATABASE_URL.")


def database_url() -> Optional[str]:
    """The URL to connect to, or ``None`` when nothing is configured."""
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit

    host, name = os.getenv("DB_HOST"), os.getenv("DB_NAME")
    if host and name:
        # URL.create escapes each part. The f-string it replaced produced an
        # unparseable URL for any password containing @ : / or # — which is
        # most generated passwords.
        return URL.create(
            "postgresql+psycopg2",
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD") or None,
            host=host,
            port=int(os.getenv("DB_PORT", "5432")),
            database=name,
        ).render_as_string(hide_password=False)

    return None


def display_url() -> str:
    """``database_url()`` with the password masked — for anything a person reads.

    ``status()`` used to embed the raw URL, so an unreachable PostgreSQL put its
    password on the page of every signed-in user.
    """
    url = database_url()
    if url is None:
        return "(not configured)"
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:                                 # noqa: BLE001
        return f"{backend()} (DATABASE_URL could not be parsed)"


def backend() -> str:
    """"postgresql", "sqlite" or "unconfigured", from the URL in force."""
    url = database_url()
    if url is None:
        return "unconfigured"
    return "sqlite" if url.startswith("sqlite") else "postgresql"


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
    # JSONB on PostgreSQL (binary and indexable, so the per-criterion scores
    # can be queried directly for S5's analysis), plain JSON on SQLite, which
    # has no JSONB. Either way the caller hands over a dict and gets a dict
    # back — the variant only changes the storage type.
    Column("rubric_scores",
           JSON().with_variant(JSONB, "postgresql"), nullable=False),
    Column("contract_version", String(32)),
    Index("submissions_user_time", "user_email", "submitted_at"),
)

# S4's tables. Declared here so one ``create_all`` builds the whole schema;
# the reads and writes live in ``app.study.store``.

consents = Table(
    "consents", metadata,
    # One row per person holding the current decision, not a log. Someone who
    # declined and later opts in is updated in place.
    Column("user_email", String(320), primary_key=True),
    Column("decision", String(16), nullable=False),        # agreed | declined
    # Random, assigned on agreeing. The only identifier that leaves the
    # database — exports never carry an email, a name or a file name.
    Column("participant_code", String(16), unique=True),
    # Which wording of the items and participant documents was agreed to.
    Column("materials_fingerprint", String(64), nullable=False),
    Column("decided_at", DateTime(timezone=True), nullable=False,
           server_default=func.now()),
    Column("withdrawn_at", DateTime(timezone=True)),
)

survey_responses = Table(
    "survey_responses", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_email", String(320), nullable=False),
    Column("wave", String(8), nullable=False),             # pre | post
    Column("answers",
           JSON().with_variant(JSONB, "postgresql"), nullable=False),
    # Plans scored before this response. On a pre-survey anything above zero
    # means the baseline was taken after seeing AI feedback; on a post-survey
    # it measures exposure.
    Column("submissions_before", Integer, nullable=False),
    Column("duration_seconds", Integer),
    Column("instrument_version", String(32), nullable=False),
    Column("materials_fingerprint", String(64), nullable=False),
    # NULL while the instruments are unapproved, so pilot responses separate
    # from study data without anyone having to remember to tag them.
    Column("irb_protocol", String(64)),
    Column("submitted_at", DateTime(timezone=True), nullable=False,
           server_default=func.now()),
    UniqueConstraint("user_email", "wave", name="survey_one_response_per_wave"),
)


@lru_cache(maxsize=1)
def get_engine() -> Optional[Engine]:
    """Build the engine, create the schema, verify it once. ``None`` on failure.

    Cached, so a database that comes up *after* the app does needs a restart to
    be picked up. That trade is deliberate: the alternative is retrying a dead
    connection on every script re-run, which is every widget interaction.
    """
    url = database_url()
    if url is None:
        logger.error(NOT_CONFIGURED)
        return None

    from sqlalchemy import create_engine

    if url.startswith("sqlite"):
        database = make_url(url).database
        if database and database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because Streamlit serves each session from a
        # worker thread, and the pooled connection will not be the one that
        # created it.
        connect_args: dict[str, Any] = {"check_same_thread": False, "timeout": 10}
    else:
        connect_args = {"connect_timeout": 3}

    try:
        engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        metadata.create_all(engine)
    except Exception as error:                        # noqa: BLE001
        # No server, bad credentials, missing driver, read-only filesystem —
        # all mean the same thing to the caller, and none should take the
        # dashboard down. The log says which it was.
        first_line = (str(error).splitlines() or [type(error).__name__])[0]
        logger.error("Cannot use the database at %s: %s", display_url(), first_line)
        return None
    return engine


def is_persistent() -> bool:
    """True when submissions are actually being written somewhere durable."""
    return get_engine() is not None


def status() -> str:
    """One line for the transparency panel."""
    engine = get_engine()
    if database_url() is None:
        return ("Storage not configured — nothing is being saved. "
                "Run python -m app.database setup.")
    if engine is None:
        return (f"Storage unavailable ({backend()} at {display_url()}). "
                f"Revision history is kept for this session only.")
    if backend() == "sqlite":
        return (f"SQLite at {display_url()} — for tests only; the study "
                f"database is PostgreSQL.")
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

    try:
        with engine.begin() as connection:
            connection.execute(insert(submissions).values(
                user_email=user_email,
                source=source,
                overall_score=overall_score,
                band=band,
                rubric_scores=rubric_scores,
                contract_version=contract_version,
            ))
    except SQLAlchemyError:
        # The engine only proves the database was up at startup. A server that
        # goes away later lands here, and the caller already keeps the result
        # in session history when this returns False.
        return False
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
    try:
        return list(_history(user_email, limit, _generation))
    except SQLAlchemyError:
        return []


def submission_count(user_email: str) -> int:
    """How many plans this person has had scored. 0 when storage is down.

    Not capped like ``submission_history``: the study gates its follow-up
    survey on this number, so it has to be the true total.
    """
    engine = get_engine()
    if engine is None:
        return 0
    with engine.connect() as connection:
        return int(connection.execute(
            select(func.count())
            .select_from(submissions)
            .where(submissions.c.user_email == user_email)).scalar_one())
