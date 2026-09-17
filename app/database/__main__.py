"""``python -m app.database`` — create, check and fill the PostgreSQL database.

See ``app/database/admin.py``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from app.database import admin, db


def _suggest_running_server() -> None:
    """When .env points at the wrong port, name the server that is running."""
    if db.backend() != "postgresql":
        return
    running = [s for s in admin.local_servers() if s.running]
    if running:
        best = running[0]
        print(f"{best.label} is running on port {best.port}. If that is the "
              f"server you mean, set DB_PORT={best.port} in .env, or run "
              f"setup again.")


def _backup(args) -> int:
    """Run one backup. With --log, the outcome is appended there as well.

    The scheduled task runs under pythonw, which has no console, so the log is
    the only place a failure would ever be seen.
    """
    def log(line: str) -> None:
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with args.log.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now().isoformat(timespec='seconds')} "
                             f"{line}\n")

    try:
        result = admin.backup(out_dir=args.out, keep=args.keep)
    except Exception as error:                        # noqa: BLE001
        message = str(error) if isinstance(error, admin.AdminError) else \
            f"{type(error).__name__}: {error}"
        log(f"FAILED {message}")
        print(f"error: {message}")
        return 1
    summary = (f"OK {result.path} ({result.size_bytes / 1024:.0f} KB)"
               + (f", removed {len(result.removed)} older" if result.removed else ""))
    log(summary)
    print(f"Backed up to {result.path} ({result.size_bytes / 1024:.0f} KB) "
          f"with {result.tool}.")
    if result.removed:
        print(f"Removed {len(result.removed)} older dump(s); keeping "
              f"{args.keep}.")
    print("Restore with: pg_restore --clean --if-exists --no-owner "
          "-d <database> <file>")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="python -m app.database")
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser(
        "setup", help="create the app's PostgreSQL role and database, write .env",
        description="Connects once as a PostgreSQL superuser (password prompted, "
                    "or read from PGPASSWORD), creates a login role and a "
                    "database it owns, and writes the connection to .env with a "
                    "generated password.")
    make.add_argument("--host", default="localhost")
    make.add_argument("--port", type=int, default=None,
                      help="default: the newest running PostgreSQL on this "
                           "machine (from the Windows installer's registry), "
                           "else 5432")
    make.add_argument("--admin-user", default="postgres")
    make.add_argument("--name", default="coteach", help="database name")
    make.add_argument("--user", default="coteach", help="the app's login role")
    make.add_argument("--env-file", type=Path, default=admin.DEFAULT_ENV_FILE)
    make.add_argument("--reset-password", action="store_true",
                      help="the role exists: give it a new password")

    sub.add_parser("check", help="connect with the current settings and count rows")

    copy = sub.add_parser("copy-sqlite",
                          help="copy rows from an old SQLite file into PostgreSQL")
    copy.add_argument("path", type=Path)

    dump = sub.add_parser("backup", help="pg_dump the study database")
    dump.add_argument("--out", type=Path, default=admin.DEFAULT_BACKUP_DIR,
                      help="folder for the dumps (default: data/backups, "
                           "gitignored)")
    dump.add_argument("--keep", type=int, default=admin.DEFAULT_KEEP,
                      help="how many dumps to keep; older ones are deleted")
    dump.add_argument("--log", type=Path, default=None,
                      help="also append the outcome to this file")

    plan = sub.add_parser("schedule-backup",
                          help="run backup daily (Windows scheduled task)")
    plan.add_argument("--at", default="18:00", help="time of day, HH:MM")
    plan.add_argument("--keep", type=int, default=admin.DEFAULT_KEEP)
    plan.add_argument("--out", type=Path, default=admin.DEFAULT_BACKUP_DIR)
    plan.add_argument("--remove", action="store_true",
                      help="delete the scheduled task instead")

    args = parser.parse_args(argv)
    try:
        if args.command == "setup":
            result = admin.setup(host=args.host, port=args.port,
                                 admin_user=args.admin_user, name=args.name,
                                 user=args.user, env_file=args.env_file,
                                 reset_password=args.reset_password)
            print(f"Connected to PostgreSQL {result.server_version} on "
                  f"{args.host}:{result.port}.")
            print(("Created" if result.created_role else "Reset the password of")
                  + f" role {args.user!r}.")
            print(("Created" if result.created_database else "Using existing")
                  + f" database {args.name!r}.")
            ignored = (result.env_file.resolve()
                       == admin.DEFAULT_ENV_FILE.resolve())
            print(f"Connection saved to {result.env_file}"
                  + (" (gitignored)." if ignored else
                     ". Keep this file out of version control.")
                  + " Tables are ready.")
            for note in result.notes:
                print(f"Note: {note}")
            return 0

        if args.command == "check":
            print(f"Database: {db.display_url()}")
            engine = db.get_engine()
            if engine is None:
                print("Not usable — see the error above.")
                _suggest_running_server()
                return 1
            print(f"Server:   {admin.server_version(engine)}")
            if db.backend() != "postgresql":
                print("Warning: this is not PostgreSQL. The study database is "
                      "PostgreSQL; SQLite is for tests.")
            for table, count in admin.row_counts(engine).items():
                print(f"  {table:<18} {count}")
            return 0

        if args.command == "backup":
            return _backup(args)

        if args.command == "schedule-backup":
            if args.remove:
                print("Removed the scheduled backup." if admin.unschedule_backup()
                      else "There was no scheduled backup.")
                return 0
            name = admin.schedule_backup(at=args.at, keep=args.keep,
                                         out_dir=args.out)
            print(f"Scheduled task {name!r}: daily at {args.at}, keeping "
                  f"{args.keep} dumps in {args.out.resolve()}.")
            print("It runs while you are signed in to Windows, and catches up "
                  "after a day the machine was off. Results are logged to "
                  f"{(args.out / 'backup.log').resolve()}.")
            return 0

        engine = db.get_engine()
        if engine is None:
            print("error: cannot connect to the target database. Run "
                  "python -m app.database check.")
            return 1
        report = admin.copy_from_sqlite(args.path, engine)
        for table, (copied, skipped) in report.items():
            print(f"  {table:<18} copied {copied}, already present {skipped}")
        print(f"{args.path} was read, not changed. Once you have checked the "
              f"copy, it can be archived or deleted.")
        return 0
    except admin.AdminError as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
