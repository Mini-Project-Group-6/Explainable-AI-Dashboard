# app/database/admin.py
"""Database administration: create the PostgreSQL database, check it, and copy
data across from an old SQLite file.

    python -m app.database setup            create role + database, write .env
    python -m app.database check            connect, create tables, count rows
    python -m app.database copy-sqlite data/coteach.db
    python -m app.database backup           pg_dump to data/backups, keep 14
    python -m app.database schedule-backup  run that daily (Windows)

``setup`` needs a PostgreSQL superuser once (``postgres`` on a Windows install).
Its password is prompted for, or taken from ``PGPASSWORD`` — never from a
command-line argument, which would land in shell history. The app itself then
connects as its own unprivileged role, which owns only its own database.

**Which server.** Without ``--port``, ``setup`` uses the newest PostgreSQL that
is installed *and running* on this machine. The Windows installer records each
server's port in the registry, and it does not always choose 5432: when an
older version already holds 5432, the new one gets 5433. Guessing 5432 would
then set the app up on the old server. The chosen port is written to .env, so
the app keeps using that server.
"""

from __future__ import annotations

import getpass
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import set_key
from sqlalchemy import DateTime, create_engine, func, insert, inspect, select, text
from sqlalchemy.engine import URL, Engine

from app.database import db

DEFAULT_ENV_FILE = db.REPO_ROOT / ".env"


class AdminError(Exception):
    """A problem an administrator can act on. Safe to print."""


# ---------------------------------------------------------------------------
# Finding the local server
# ---------------------------------------------------------------------------

DEFAULT_PORT = 5432
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class LocalServer:
    name: str                   # the Windows service, e.g. "postgresql-x64-18"
    version: tuple[int, ...]    # (18,) — from the service name
    port: int
    running: bool

    @property
    def label(self) -> str:
        return f"PostgreSQL {'.'.join(map(str, self.version))}"


def _version_from(name: str) -> tuple[int, ...]:
    """"postgresql-x64-18" -> (18,); "postgresql-x64-9.6" -> (9, 6)."""
    match = re.search(r"(\d+(?:\.\d+)*)$", name)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def _registered_servers() -> list[tuple[str, int]]:
    """``(service name, port)`` for every server the EDB installer registered.

    Windows only. Reads ``HKLM\\SOFTWARE\\PostgreSQL\\Services``, which ordinary
    users may read. Anywhere else, or with no installer entries, it is empty.
    """
    if sys.platform != "win32":
        return []
    import winreg

    found: list[tuple[str, int]] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                              r"SOFTWARE\PostgreSQL\Services")
    except OSError:
        return []
    with root:
        index = 0
        while True:
            try:
                name = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(root, name) as key:
                    port = int(winreg.QueryValueEx(key, "Port")[0])
            except (OSError, ValueError):
                continue
            found.append((name, port))
    return found


def _listening(port: int, host: str = "localhost") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def local_servers() -> list[LocalServer]:
    """Registered local servers, newest version first."""
    servers = [LocalServer(name, _version_from(name), port, _listening(port))
               for name, port in _registered_servers()]
    return sorted(servers, key=lambda s: s.version, reverse=True)


def detect_port() -> tuple[int, str]:
    """The port of the newest running local server, and how it was chosen."""
    servers = local_servers()
    running = [s for s in servers if s.running]
    if running:
        best = running[0]
        return best.port, f"{best.label}, the newest running server on this machine"
    if _listening(DEFAULT_PORT):
        return DEFAULT_PORT, "the default port, where a server is listening"
    if servers:
        stopped = ", ".join(f"{s.label} (service {s.name}, port {s.port})"
                            for s in servers)
        raise AdminError(f"PostgreSQL is installed but not running: {stopped}. "
                         f"Start the service (services.msc), then run setup "
                         f"again.")
    raise AdminError(f"No PostgreSQL server is running on this machine (nothing "
                     f"on port {DEFAULT_PORT}, and none registered by the "
                     f"installer). Start it, or pass --host/--port.")


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

@dataclass
class SetupResult:
    created_role: bool
    reset_password: bool
    created_database: bool
    env_file: Path
    port: int
    server_version: str
    notes: list[str] = field(default_factory=list)


def setup(host: str = "localhost", port: Optional[int] = None,
          admin_user: str = "postgres", name: str = "coteach",
          user: str = "coteach", env_file: Path = DEFAULT_ENV_FILE,
          reset_password: bool = False,
          admin_password: Optional[str] = None) -> SetupResult:
    """Create the app's role and database, and record the connection in .env.

    *port* ``None`` means: the newest running local server (see
    ``detect_port``), or 5432 for a remote *host*.
    """
    import psycopg2
    from psycopg2 import sql

    notes: list[str] = []
    if port is None:
        if host in _LOCAL_HOSTS:
            port, reason = detect_port()
            notes.append(f"Using port {port}: {reason}.")
        else:
            port = DEFAULT_PORT

    if admin_password is None:
        admin_password = os.getenv("PGPASSWORD") or getpass.getpass(
            f"Password for PostgreSQL user {admin_user!r} on {host}:{port}: ")

    try:
        admin = psycopg2.connect(host=host, port=port, user=admin_user,
                                 password=admin_password, dbname="postgres",
                                 connect_timeout=5)
    except psycopg2.OperationalError as error:
        raise AdminError(f"Could not connect as {admin_user} to {host}:{port} — "
                         f"{str(error).strip().splitlines()[-1]}") from None
    # CREATE DATABASE cannot run inside a transaction.
    admin.autocommit = True
    password = secrets.token_urlsafe(24)
    try:
        with admin.cursor() as cursor:
            cursor.execute("SHOW server_version")
            server_version = cursor.fetchone()[0].split()[0]
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,))
            role_exists = cursor.fetchone() is not None
            if role_exists and not reset_password:
                raise AdminError(
                    f"Role {user!r} already exists and its password is not known "
                    f"here. Re-run with --reset-password to issue a new one (the "
                    f"old password stops working).")
            # psycopg2 interpolates the literal client-side, which is what lets
            # a password be passed as a parameter to DDL at all.
            verb = "ALTER" if role_exists else "CREATE"
            cursor.execute(
                sql.SQL(verb + " ROLE {} LOGIN PASSWORD %s").format(
                    sql.Identifier(user)), (password,))

            cursor.execute("SELECT pg_get_userbyid(datdba) FROM pg_database "
                           "WHERE datname = %s", (name,))
            row = cursor.fetchone()
            if row is None:
                # Owning the database is what lets the role create its tables:
                # since PostgreSQL 15 ordinary roles cannot create in "public".
                cursor.execute(sql.SQL("CREATE DATABASE {} OWNER {} ENCODING 'UTF8'")
                               .format(sql.Identifier(name), sql.Identifier(user)))
            elif row[0] != user:
                cursor.execute(sql.SQL("ALTER DATABASE {} OWNER TO {}")
                               .format(sql.Identifier(name), sql.Identifier(user)))
                notes.append(f"Database {name!r} existed; its owner was changed "
                             f"from {row[0]!r} to {user!r}.")
    finally:
        admin.close()

    env_file = Path(env_file)
    env_file.touch(exist_ok=True)
    for key, value in (("DB_HOST", host), ("DB_PORT", str(port)),
                       ("DB_NAME", name), ("DB_USER", user),
                       ("DB_PASSWORD", password)):
        set_key(str(env_file), key, value, quote_mode="never")
    if os.getenv("DATABASE_URL"):
        notes.append("DATABASE_URL is set in the environment and takes "
                     "precedence over the DB_* values just written.")

    url = URL.create("postgresql+psycopg2", username=user, password=password,
                     host=host, port=port, database=name)
    engine = create_engine(url, connect_args={"connect_timeout": 5})
    try:
        db.metadata.create_all(engine)
    finally:
        engine.dispose()

    return SetupResult(created_role=not role_exists,
                       reset_password=role_exists,
                       created_database=row is None,
                       env_file=env_file, port=port,
                       server_version=server_version, notes=notes)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def server_version(engine: Engine) -> str:
    """"PostgreSQL 18.6", or the SQLite version for the test database."""
    with engine.connect() as connection:
        if engine.dialect.name == "postgresql":
            version = connection.execute(text("SHOW server_version")).scalar()
            return f"PostgreSQL {str(version).split()[0]}"
        return f"SQLite {connection.execute(text('select sqlite_version()')).scalar()}"


def row_counts(engine: Engine) -> dict[str, int]:
    with engine.connect() as connection:
        return {name: int(connection.execute(
                    select(func.count()).select_from(table)).scalar_one())
                for name, table in db.metadata.tables.items()}


# ---------------------------------------------------------------------------
# copy-sqlite
# ---------------------------------------------------------------------------

def _utc(value):
    """SQLite's CURRENT_TIMESTAMP is UTC but comes back without a zone.

    Inserted as-is into TIMESTAMPTZ it would be read in the *server's* zone, so
    every copied timestamp would shift by the UTC offset.
    """
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


#: Tables in dependency-free copy order, each with the columns that identify a
#: row already present in the target (so a second run copies nothing).
_COPY_PLAN = (
    ("users", ("email",)),
    ("submissions", ("user_email", "source", "submitted_at")),
    ("consents", ("user_email",)),
    ("survey_responses", ("user_email", "wave")),
)


def copy_from_sqlite(sqlite_path: Path, target: Engine) -> dict[str, tuple[int, int]]:
    """Copy every row the target does not already have. ``{table: (copied, skipped)}``.

    The source is opened read-only and never changed. Auto-increment ids are
    not copied — PostgreSQL assigns its own, in the source's order.
    """
    sqlite_path = Path(sqlite_path)
    if not sqlite_path.is_file():
        raise AdminError(f"No SQLite file at {sqlite_path}.")
    if target.dialect.name != "postgresql":
        raise AdminError("The configured database is not PostgreSQL; there is "
                         "nothing to copy into.")

    source = create_engine(
        f"sqlite+pysqlite:///file:{sqlite_path.resolve().as_posix()}"
        f"?mode=ro&uri=true")
    report: dict[str, tuple[int, int]] = {}
    try:
        present = set(inspect(source).get_table_names())
        with source.connect() as reader, target.begin() as writer:
            for name, key in _COPY_PLAN:
                if name not in present:
                    continue
                table = db.metadata.tables[name]
                columns = [c for c in table.columns
                           if not (c.primary_key and c.autoincrement is True)]
                order = [table.c.id] if "id" in table.c else []
                rows = [dict(row) for row in reader.execute(
                    select(*columns).order_by(*order)).mappings()]
                for row in rows:
                    for column in columns:
                        if isinstance(column.type, DateTime):
                            row[column.name] = _utc(row[column.name])

                key_cols = [table.c[k] for k in key]
                existing = {tuple(_utc(v) for v in found) for found in
                            writer.execute(select(*key_cols)).all()}
                fresh = [row for row in rows
                         if tuple(row[k] for k in key) not in existing]
                if fresh:
                    writer.execute(insert(table), fresh)
                report[name] = (len(fresh), len(rows) - len(fresh))
    finally:
        source.dispose()
    return report


# ---------------------------------------------------------------------------
# backup
# ---------------------------------------------------------------------------

DEFAULT_BACKUP_DIR = db.REPO_ROOT / "data" / "backups"
DEFAULT_KEEP = 14
TASK_NAME = "CoTeach database backup"
_EXE = ".exe" if sys.platform == "win32" else ""


def _registered_installations() -> list[tuple[str, Path]]:
    """``(installation name, base directory)`` from the EDB installer's registry."""
    if sys.platform != "win32":
        return []
    import winreg

    found: list[tuple[str, Path]] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                              r"SOFTWARE\PostgreSQL\Installations")
    except OSError:
        return []
    with root:
        index = 0
        while True:
            try:
                name = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(root, name) as key:
                    base = winreg.QueryValueEx(key, "Base Directory")[0]
            except OSError:
                continue
            found.append((name, Path(base)))
    return found


def _tool_major(path: Path) -> Optional[int]:
    """12 from "pg_dump (PostgreSQL) 12.4", or None if it will not say."""
    try:
        output = subprocess.run([str(path), "--version"], capture_output=True,
                                text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)(?:\.\d+)*", output)
    return int(match.group(1)) if match else None


def find_pg_tool(tool: str, server_major: int) -> Path:
    """A ``pg_dump``/``pg_restore`` at least as new as the server.

    pg_dump refuses to dump a server newer than itself, and on this kind of
    machine the one on PATH (if any) is often left over from an older install.
    Tried in order: ``COTEACH_PG_BIN``, the installer's own directories newest
    first, then PATH.
    """
    candidates: list[Path] = []
    override = os.getenv("COTEACH_PG_BIN")
    if override:
        candidates.append(Path(override) / f"{tool}{_EXE}")
    for name, base in sorted(_registered_installations(),
                             key=lambda item: _version_from(item[0]),
                             reverse=True):
        candidates.append(base / "bin" / f"{tool}{_EXE}")
    on_path = shutil.which(tool)
    if on_path:
        candidates.append(Path(on_path))

    for path in candidates:
        if path.is_file():
            major = _tool_major(path)
            if major is not None and major >= server_major:
                return path
    raise AdminError(
        f"No {tool} for PostgreSQL {server_major} or newer was found. Install "
        f"the PostgreSQL {server_major} command-line tools, or set "
        f"COTEACH_PG_BIN to the folder that contains {tool}{_EXE}.")


@dataclass(frozen=True)
class BackupResult:
    path: Path
    size_bytes: int
    removed: list[Path]
    tool: Path


def _dump_pattern(database: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(database)}-\d{{8}}-\d{{6}}\.dump$")


def prune(out_dir: Path, database: str, keep: int) -> list[Path]:
    """Delete all but the *keep* newest dumps of *database*. Nothing else is touched.

    Pruning is also what carries a deletion into the backups: data removed
    from the database is gone from every copy within *keep* days.
    """
    pattern = _dump_pattern(database)
    # The timestamp in the name sorts chronologically.
    dumps = sorted(p for p in Path(out_dir).iterdir()
                   if p.is_file() and pattern.match(p.name))
    old = dumps[:-keep] if keep > 0 else dumps
    for path in old:
        path.unlink()
    return old


def backup(out_dir: Path = DEFAULT_BACKUP_DIR, keep: int = DEFAULT_KEEP,
           engine: Optional[Engine] = None,
           now: Optional[datetime] = None) -> BackupResult:
    """Dump the study database to ``<out_dir>/<db>-<YYYYmmdd-HHMMSS>.dump``.

    Custom format, so ``pg_restore`` can restore all of it or one table. The
    dump is written under a temporary name and only renamed once
    ``pg_restore --list`` has read it back and found every app table, so a
    half-written or unreadable file never counts as a backup — and never
    pushes a good one out of the retention window.
    """
    if keep < 1:
        raise AdminError("--keep must be at least 1.")
    engine = engine if engine is not None else db.get_engine()
    if engine is None:
        raise AdminError("Cannot connect to the database, so nothing was backed "
                         "up. Run: python -m app.database check")
    if engine.dialect.name != "postgresql":
        raise AdminError("Backups are for the PostgreSQL study database.")

    url = engine.url
    with engine.connect() as connection:
        major = int(connection.execute(
            text("SHOW server_version_num")).scalar()) // 10000
    pg_dump = find_pg_tool("pg_dump", major)
    pg_restore = find_pg_tool("pg_restore", major)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    final = out_dir / f"{url.database}-{stamp}.dump"
    partial = final.with_name(final.name + ".partial")

    # The password goes to the child through its environment only — never on
    # the command line, where any process listing would show it.
    env = dict(os.environ, PGPASSWORD=url.password or "", PGCONNECT_TIMEOUT="10")
    command = [str(pg_dump), "--format=custom", "--no-owner", "--no-privileges",
               "--no-password", f"--host={url.host or 'localhost'}",
               f"--port={url.port or DEFAULT_PORT}",
               f"--username={url.username}", f"--file={partial}",
               str(url.database)]
    try:
        dumped = subprocess.run(command, env=env, capture_output=True,
                                text=True, timeout=900)
        if dumped.returncode != 0:
            detail = (dumped.stderr.strip().splitlines() or ["no output"])[-1]
            raise AdminError(f"pg_dump failed: {detail}")

        listing = subprocess.run([str(pg_restore), "--list", str(partial)],
                                 capture_output=True, text=True, timeout=120)
        missing = [name for name in db.metadata.tables
                   if not re.search(rf"\bTABLE public {re.escape(name)}\b",
                                    listing.stdout)]
        if listing.returncode != 0 or missing:
            raise AdminError(
                "The dump could not be verified"
                + (f" (tables missing: {', '.join(missing)})" if missing else "")
                + "; it was discarded.")
        partial.replace(final)
    finally:
        partial.unlink(missing_ok=True)

    return BackupResult(path=final, size_bytes=final.stat().st_size,
                        removed=prune(out_dir, str(url.database), keep),
                        tool=pg_dump)


def _ps_quote(value) -> str:
    """A PowerShell single-quoted literal: nothing inside is interpreted."""
    return "'" + str(value).replace("'", "''") + "'"


def _powershell(script: str) -> subprocess.CompletedProcess:
    # -Command, not a .ps1 file: an execution policy such as AllSigned blocks
    # unsigned script files but not an inline command.
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         "$ErrorActionPreference = 'Stop'; " + script],
        capture_output=True, text=True, timeout=120)


def schedule_backup(at: str = "18:00", keep: int = DEFAULT_KEEP,
                    out_dir: Path = DEFAULT_BACKUP_DIR) -> str:
    """Register (or update) a daily Windows scheduled task for ``backup``.

    Runs as the current user while they are signed in, so no password is
    stored with the task. ``-StartWhenAvailable`` catches up on a day the
    machine was off at *at*. It runs ``pythonw``, so no console window
    appears; results go to ``backup.log`` beside the dumps.
    """
    if sys.platform != "win32":
        raise AdminError("Scheduling is automated on Windows only. Elsewhere, "
                         "add `python -m app.database backup` to cron.")
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", at):
        raise AdminError(f"--at must be HH:MM (24-hour), not {at!r}.")
    if keep < 1:
        raise AdminError("--keep must be at least 1.")

    python = Path(sys.executable)
    windowless = python.with_name("pythonw.exe")
    runner = windowless if windowless.is_file() else python
    out_dir = Path(out_dir).resolve()
    arguments = (f'-m app.database backup --keep {keep} --out "{out_dir}" '
                 f'--log "{out_dir / "backup.log"}"')
    script = (
        f"$action = New-ScheduledTaskAction -Execute {_ps_quote(runner)} "
        f"-Argument {_ps_quote(arguments)} "
        f"-WorkingDirectory {_ps_quote(db.REPO_ROOT)}; "
        f"$trigger = New-ScheduledTaskTrigger -Daily -At {_ps_quote(at)}; "
        "$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable "
        "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries "
        "-ExecutionTimeLimit (New-TimeSpan -Minutes 30); "
        f"Register-ScheduledTask -TaskName {_ps_quote(TASK_NAME)} "
        "-Action $action -Trigger $trigger -Settings $settings "
        f"-Description {_ps_quote('Daily pg_dump of the co-teaching study database. Created by python -m app.database schedule-backup.')} "
        "-Force | Out-Null"
    )
    result = _powershell(script)
    if result.returncode != 0:
        detail = (result.stderr.strip().splitlines() or ["no output"])[0]
        raise AdminError(f"Could not register the scheduled task: {detail}")
    return TASK_NAME


def unschedule_backup() -> bool:
    """Remove the scheduled task. False if there was none."""
    if sys.platform != "win32":
        return False
    result = _powershell(
        f"$task = Get-ScheduledTask -TaskName {_ps_quote(TASK_NAME)} "
        "-ErrorAction SilentlyContinue; "
        "if ($task) { Unregister-ScheduledTask -TaskName $task.TaskName "
        "-Confirm:$false; 'removed' }")
    return "removed" in result.stdout
