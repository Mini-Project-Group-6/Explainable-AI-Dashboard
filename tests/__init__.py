"""S3 dashboard tests.

Run from the repository root:
    .venv\\Scripts\\python -m unittest discover -s tests -t . -v

This module points the app at a scratch database. ``app.database.db`` caches
its engine behind an lru_cache, so ``DATABASE_URL`` must be set before any
``app`` module is imported — which is why every test module starts with
``import tests``.

The app itself requires PostgreSQL and has no SQLite fallback. The suite asks
for a scratch SQLite file explicitly, so it runs without a server; set
``COTEACH_TEST_DATABASE_URL`` to run it on PostgreSQL instead.

That import is not decoration. ``unittest discover -s tests`` *without*
``-t .`` loads the modules as top-level ``test_storage`` rather than
``tests.test_storage``, so this package ``__init__`` never executes. The first
version relied on it executing, and the whole suite silently wrote 13 accounts
and 61 submissions into the real ``data/coteach.db`` instead. ``import tests``
makes the setup run under either invocation, and ``assert_scratch_database()``
turns any remaining slip into a loud failure rather than silent pollution.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: A scratch database in its own directory per run, deleted on exit.
#:
#: An earlier version named the file after ``os.getpid()`` and left it behind.
#: Windows recycles PIDs, so a later run could reopen a previous run's database
#: and inherit its rows — which showed up as "expected 1 submission, found 3".
#: The tests were right and the harness was wrong; a fresh directory removes
#: the possibility entirely.
_SCRATCH_DIR = Path(tempfile.mkdtemp(prefix="coteach_test_"))
SCRATCH_URL = f"sqlite+pysqlite:///{(_SCRATCH_DIR / 'coteach.db').as_posix()}"

#: Run the suite against PostgreSQL instead of the scratch SQLite file:
#:
#:     COTEACH_TEST_DATABASE_URL=postgresql+psycopg2://postgres@localhost:55432/coteach_test
#:
#: SQLite alone never exercised the JSONB, TIMESTAMPTZ and boolean paths the
#: schema declares for PostgreSQL. The database name must end in ``_test``,
#: because every table the app defines is dropped and recreated at the start of
#: the run.
TEST_DATABASE_URL = os.environ.get("COTEACH_TEST_DATABASE_URL") or None


@atexit.register
def _cleanup() -> None:
    shutil.rmtree(_SCRATCH_DIR, ignore_errors=True)


def assert_scratch_database() -> None:
    """Abort unless the app is pointed at this run's throwaway database.

    Called at import time by every test module. The tests create accounts and
    submissions; run against the real database they would bury a researcher's
    data in fixtures named ``dupe@`` and ``a.pdf``.
    """
    from app.database import db

    if TEST_DATABASE_URL:
        _reset_test_postgres(db)
        return

    if db.database_url() != SCRATCH_URL:
        raise RuntimeError(
            f"Tests would write to {db.display_url()!r}, not the scratch "
            f"database at "
            f"{_SCRATCH_DIR}. Something imported app.database.db before this "
            f"package. Run: python -m unittest discover -s tests -t .")


_postgres_reset = False


def _reset_test_postgres(db) -> None:
    """Point at the test database and empty it, once per run."""
    global _postgres_reset
    from sqlalchemy.engine import make_url

    if db.database_url() != TEST_DATABASE_URL:
        raise RuntimeError("Tests would not use COTEACH_TEST_DATABASE_URL. "
                           "Something imported app.database.db first.")
    name = make_url(TEST_DATABASE_URL).database or ""
    if not name.endswith("_test"):
        raise RuntimeError(f"Refusing to reset database {name!r}: the name "
                           f"must end in '_test'.")
    if _postgres_reset:
        return
    engine = db.get_engine()
    if engine is None:
        raise RuntimeError(f"Cannot connect to {db.display_url()}.")
    db.metadata.drop_all(engine)
    db.metadata.create_all(engine)
    _postgres_reset = True


# Never let a developer's real credentials leak into a test run.
#
# The database variables are blanked, not deleted. ``app.database.db`` calls
# ``load_dotenv()`` on import, which fills in any variable that is *absent* — so
# deleting them let a developer's .env (``cp .env.example .env``, as
# docker-compose.yml instructs) point the whole suite at their dev PostgreSQL.
# An empty value counts as present, and is falsy to ``database_url()``.
os.environ.pop("APP_EMAIL", None)
os.environ.pop("APP_PASSWORD", None)
for _name in ("DATABASE_URL", "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD",
              "DB_PORT"):
    os.environ[_name] = ""
os.environ["DB_PORT"] = "5432"
os.environ["DATABASE_URL"] = TEST_DATABASE_URL or SCRATCH_URL
