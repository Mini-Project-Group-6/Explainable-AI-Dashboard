"""Submission storage tests.

Revision tracking is not incidental to this project — the research question is
whether explainable feedback improves lesson-plan quality *across drafts*, so a
lost or mis-attributed submission is lost data.

    python -m unittest tests.test_storage -v
"""

from __future__ import annotations

import tests  # noqa: F401  — sets DATABASE_URL before any app import

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from sqlalchemy import create_engine, insert
from sqlalchemy.engine import make_url

from app import accounts, session
from app.database import admin, db
from app.session import User
from app.study import flow, store

tests.assert_scratch_database()


ON_POSTGRES = bool(tests.TEST_DATABASE_URL)


class Backend(unittest.TestCase):

    def test_no_configuration_means_no_database_not_a_sqlite_file(self):
        """The study database is PostgreSQL.

        A silent SQLite fallback let a machine without its .env write study
        data somewhere other than the study database. Unconfigured now means
        unconfigured, and says so.
        """
        blank = {"DATABASE_URL": "", "DB_HOST": "", "DB_NAME": ""}
        with mock.patch.dict(os.environ, blank), \
                mock.patch.object(db, "get_engine", return_value=None):
            self.assertIsNone(db.database_url())
            self.assertEqual(db.backend(), "unconfigured")
            self.assertIn("not configured", db.status())
            self.assertEqual(db.display_url(), "(not configured)")

    def test_the_real_get_engine_refuses_when_unconfigured(self):
        blank = {"DATABASE_URL": "", "DB_HOST": "", "DB_NAME": ""}
        with mock.patch.dict(os.environ, blank):
            db.get_engine.cache_clear()
            try:
                with self.assertLogs(db.logger, "ERROR"):
                    self.assertIsNone(db.get_engine())
            finally:
                db.get_engine.cache_clear()
        self.assertIsNotNone(db.get_engine(), "scratch engine not restored")

    def test_the_suite_uses_the_database_it_asked_for(self):
        expected = tests.TEST_DATABASE_URL or tests.SCRATCH_URL
        self.assertEqual(db.database_url(), expected)

    def test_the_engine_builds_and_reports_persistent(self):
        self.assertIsNotNone(db.get_engine())
        self.assertTrue(db.is_persistent())

    def test_status_names_the_backend(self):
        self.assertIn("PostgreSQL" if ON_POSTGRES else "SQLite", db.status())

    @unittest.skipIf(ON_POSTGRES, "only SQLite is flagged")
    def test_status_flags_sqlite_as_test_only(self):
        self.assertIn("tests only", db.status())

    def test_schema_serves_both_dialects(self):
        # The tables are declared once in SQLAlchemy Core precisely so SERIAL /
        # TIMESTAMPTZ / JSONB do not have to be hand-written per dialect.
        for table in ("submissions", "users"):
            self.assertIn(table, db.metadata.tables)


SCORES = {"learning_outcomes": 3, "lesson_closure": 2, "lesson_sequencing": 2}


class Submissions(unittest.TestCase):

    def test_a_submission_round_trips(self):
        db.record_submission("rt@st.knust.edu.gh", "plan.pdf", 61.0, "Minimum",
                             SCORES, "2.0.0")
        rows = db.submission_history("rt@st.knust.edu.gh")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "plan.pdf")
        self.assertEqual(float(rows[0]["overall_score"]), 61.0)
        # JSON must come back as a dict, not a string, on either dialect.
        self.assertEqual(rows[0]["rubric_scores"], SCORES)

    def test_history_is_oldest_first(self):
        email = "order@st.knust.edu.gh"
        for index, source in enumerate(("v1.pdf", "v2.pdf", "v3.pdf")):
            db.record_submission(email, source, 40.0 + index * 10, "band",
                                 SCORES, "2.0.0")
        rows = db.submission_history(email)
        self.assertEqual([r["source"] for r in rows], ["v1.pdf", "v2.pdf", "v3.pdf"])

    def test_same_second_submissions_keep_their_order(self):
        """SQLite's CURRENT_TIMESTAMP is whole seconds.

        Three revisions saved inside one second all share a timestamp, so the
        ordering has to fall back to the insertion id or the trend chart draws
        them shuffled.
        """
        email = "tie@st.knust.edu.gh"
        for source in ("a.pdf", "b.pdf", "c.pdf"):
            db.record_submission(email, source, 50.0, "band", SCORES, "2.0.0")
        rows = db.submission_history(email)
        self.assertEqual([r["source"] for r in rows], ["a.pdf", "b.pdf", "c.pdf"])

    def test_one_student_teacher_cannot_see_another(self):
        db.record_submission("mine@st.knust.edu.gh", "mine.pdf", 70.0, "Good",
                             SCORES, "2.0.0")
        db.record_submission("yours@st.knust.edu.gh", "yours.pdf", 80.0, "Good",
                             SCORES, "2.0.0")
        mine = [r["source"] for r in db.submission_history("mine@st.knust.edu.gh")]
        self.assertEqual(mine, ["mine.pdf"])

    def test_a_new_submission_invalidates_the_cached_history(self):
        # The read is cached because Streamlit re-runs the script on every
        # interaction; a stale cache would hide the plan just uploaded.
        email = "cache@st.knust.edu.gh"
        db.record_submission(email, "one.pdf", 50.0, "band", SCORES, "2.0.0")
        self.assertEqual(len(db.submission_history(email)), 1)
        db.record_submission(email, "two.pdf", 60.0, "band", SCORES, "2.0.0")
        self.assertEqual(len(db.submission_history(email)), 2)

    def test_history_of_an_unknown_user_is_empty_not_an_error(self):
        self.assertEqual(db.submission_history("nobody@st.knust.edu.gh"), [])

    def test_a_missing_score_is_stored_rather_than_rejected(self):
        # An unscorable plan still belongs in the history.
        db.record_submission("null@st.knust.edu.gh", "bad.pdf", None, None,
                             SCORES, "2.0.0")
        rows = db.submission_history("null@st.knust.edu.gh")
        self.assertIsNone(rows[0]["overall_score"])



class Configuration(unittest.TestCase):
    """How the connection URL is built and shown. No database needed."""

    PASSWORD = "p@ss:w/rd#1?"

    def env(self, **values):
        base = {"DATABASE_URL": "", "DB_HOST": "db.example", "DB_NAME": "coteach",
                "DB_USER": "coteach", "DB_PASSWORD": self.PASSWORD,
                "DB_PORT": "5433"}
        base.update(values)
        return mock.patch.dict(os.environ, base)

    def test_a_password_with_url_characters_survives(self):
        """Regression: the URL was an f-string, so '@' or '/' broke parsing."""
        with self.env():
            url = make_url(db.database_url())
        self.assertEqual((url.password, url.host, url.port, url.database),
                         (self.PASSWORD, "db.example", 5433, "coteach"))

    def test_no_password_is_allowed(self):
        with self.env(DB_PASSWORD=""):
            self.assertIsNone(make_url(db.database_url()).password)

    def test_what_people_see_never_contains_the_password(self):
        """Regression: an unreachable server put its password in status()."""
        with self.env(), mock.patch.object(db, "get_engine", return_value=None):
            self.assertNotIn(self.PASSWORD, db.status())
            self.assertNotIn(self.PASSWORD, db.display_url())
            self.assertIn("***", db.display_url())

    def test_an_explicit_url_is_masked_too(self):
        with self.env(DATABASE_URL="postgresql+psycopg2://u:hunter2secret@h/d"):
            self.assertNotIn("hunter2secret", db.display_url())

    def test_an_unparseable_url_is_not_echoed(self):
        with self.env(DATABASE_URL="postgresql://u:hunter2secret@h:notaport/d"):
            self.assertNotIn("hunter2secret", db.display_url())


class DatabaseLostAfterStartup(unittest.TestCase):
    """The engine only proves the database was up at startup.

    A server that goes away later must degrade the page, not raise into it.
    Simulated with an engine whose SQLite file cannot be opened.
    """

    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="coteach_broken_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        broken = create_engine(
            f"sqlite+pysqlite:///{Path(tmp) / 'missing' / 'x.db'}")
        for target in (db, accounts):
            patcher = mock.patch.object(target, "get_engine", return_value=broken)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.user = User("lost@st.knust.edu.gh", "Lost", "student_teacher")

    def test_a_submission_is_reported_unsaved(self):
        self.assertFalse(db.record_submission(
            self.user.email, "p.pdf", 50.0, "band", SCORES, "2.0.0"))

    def test_history_is_empty(self):
        self.assertEqual(db.submission_history("lost-history@st.knust.edu.gh"), [])

    def test_the_study_reads_as_unavailable(self):
        state = store.current_state(self.user)
        self.assertEqual(state.stage, flow.UNAVAILABLE)
        self.assertFalse(state.upload_blocked)

    def test_study_writes_fail_softly(self):
        self.assertIsNone(store.record_consent(self.user.email, agreed=True))
        self.assertFalse(store.withdraw(self.user.email))
        saved, problems = store.record_response(self.user, "pre", {})
        self.assertFalse(saved)
        self.assertTrue(problems)

    def test_sign_in_says_unavailable_and_does_not_count_a_failure(self):
        session._failures.clear()
        ok, message = session.sign_in(self.user.email, "whatever password")
        self.assertFalse(ok)
        self.assertEqual(message, session.UNAVAILABLE_MESSAGE)
        self.assertNotIn(self.user.email, session._failures)

    def test_sign_in_does_not_claim_there_are_no_accounts(self):
        with mock.patch.object(accounts, "get_engine", return_value=None):
            _, message = session.sign_in(self.user.email, "whatever password")
        self.assertEqual(message, session.UNAVAILABLE_MESSAGE)



class LocalServerDetection(unittest.TestCase):
    """``setup`` without ``--port``: the newest *running* local PostgreSQL.

    The Windows installer gave PostgreSQL 18 port 5433 on the dev machine
    because 17 held 5432. Assuming 5432 would set the app up on the old server.
    """

    def servers(self, registered, listening):
        return (mock.patch.object(admin, "_registered_servers",
                                  return_value=registered),
                mock.patch.object(admin, "_listening",
                                  side_effect=lambda port, host="localhost":
                                  port in listening))

    def detect(self, registered, listening):
        first, second = self.servers(registered, listening)
        with first, second:
            return admin.detect_port()

    def test_versions_come_from_service_names(self):
        self.assertEqual(admin._version_from("postgresql-x64-18"), (18,))
        self.assertEqual(admin._version_from("postgresql-x64-9.6"), (9, 6))
        self.assertEqual(admin._version_from("postgresql"), ())

    def test_the_newest_running_server_wins(self):
        port, reason = self.detect(
            [("postgresql-x64-17", 5432), ("postgresql-x64-18", 5433)],
            listening={5432, 5433})
        self.assertEqual(port, 5433)
        self.assertIn("PostgreSQL 18", reason)

    def test_a_newer_server_that_is_stopped_is_passed_over(self):
        port, reason = self.detect(
            [("postgresql-x64-18", 5433), ("postgresql-x64-9.6", 5432)],
            listening={5432})
        self.assertEqual(port, 5432)
        self.assertIn("9.6", reason)

    def test_without_installer_entries_the_default_port_is_used(self):
        port, _ = self.detect([], listening={5432})
        self.assertEqual(port, 5432)

    def test_an_installed_but_stopped_server_is_named(self):
        with self.assertRaises(admin.AdminError) as raised:
            self.detect([("postgresql-x64-18", 5433)], listening=set())
        self.assertIn("postgresql-x64-18", str(raised.exception))
        self.assertIn("not running", str(raised.exception))

    def test_nothing_at_all_says_so(self):
        with self.assertRaises(admin.AdminError):
            self.detect([], listening=set())

    def test_setup_detects_only_for_a_local_host(self):
        # A remote host cannot be looked up in this machine's registry. The
        # admin connection is made to fail so nothing else runs.
        import psycopg2
        refused = psycopg2.OperationalError("refused")
        with mock.patch.object(admin, "detect_port") as detect, \
                mock.patch("psycopg2.connect", side_effect=refused) as connect:
            with self.assertRaises(admin.AdminError):
                admin.setup(host="db.example", admin_password="x")
            detect.assert_not_called()
            self.assertEqual(connect.call_args.kwargs["port"], admin.DEFAULT_PORT)

            detect.return_value = (5433, "PostgreSQL 18")
            with self.assertRaises(admin.AdminError) as raised:
                admin.setup(host="localhost", admin_password="x")
            self.assertEqual(connect.call_args.kwargs["port"], 5433)
            self.assertIn("localhost:5433", str(raised.exception))

    def test_reading_this_machines_registry_does_not_fail(self):
        for server in admin.local_servers():
            self.assertIsInstance(server.port, int)
            self.assertTrue(server.version, server.name)

    def test_the_server_version_is_reported(self):
        self.assertTrue(admin.server_version(db.get_engine()).startswith(
            "PostgreSQL" if ON_POSTGRES else "SQLite"))


class Backups(unittest.TestCase):
    """``python -m app.database backup``: the study data's only second copy."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="coteach_backup_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def touch(self, *names):
        for name in names:
            (self.tmp / name).write_bytes(b"x")

    def test_pruning_keeps_the_newest_and_nothing_else_is_touched(self):
        dumps = [f"coteach-2026091{n}-180000.dump" for n in range(5)]
        bystanders = ["backup.log", "coteach-20260101-000000.dump.partial",
                      "coteach_test-20260101-000000.dump", "notes.dump"]
        self.touch(*dumps, *bystanders)
        removed = admin.prune(self.tmp, "coteach", keep=2)
        self.assertEqual(sorted(p.name for p in removed), dumps[:3])
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()),
                         sorted(dumps[3:] + bystanders))

    def test_keep_must_be_at_least_one(self):
        with self.assertRaises(admin.AdminError):
            admin.backup(out_dir=self.tmp, keep=0)

    def test_only_postgresql_is_backed_up(self):
        with self.assertRaises(admin.AdminError):
            admin.backup(out_dir=self.tmp, engine=create_engine("sqlite+pysqlite://"))

    def test_no_database_means_no_backup_and_says_so(self):
        with mock.patch.object(db, "get_engine", return_value=None):
            with self.assertRaises(admin.AdminError) as raised:
                admin.backup(out_dir=self.tmp)
        self.assertIn("nothing was backed up", str(raised.exception))
        self.assertEqual(list(self.tmp.iterdir()), [])

    def fake_installs(self, versions):
        installs = []
        for version in versions:
            base = self.tmp / f"PostgreSQL{version}"
            (base / "bin").mkdir(parents=True)
            (base / "bin" / f"pg_dump{admin._EXE}").write_bytes(b"")
            installs.append((f"postgresql-x64-{version}", base))
        majors = {str(base / "bin" / f"pg_dump{admin._EXE}"): int(name.rsplit("-", 1)[1])
                  for name, base in installs}
        return (mock.patch.object(admin, "_registered_installations",
                                  return_value=installs),
                mock.patch.object(admin, "_tool_major",
                                  side_effect=lambda path: majors.get(str(path))),
                mock.patch.object(shutil, "which", return_value=None))

    def test_pg_dump_must_be_at_least_as_new_as_the_server(self):
        patches = self.fake_installs([17, 18])
        with patches[0], patches[1], patches[2]:
            self.assertIn("PostgreSQL18", str(admin.find_pg_tool("pg_dump", 18)))
            self.assertIn("PostgreSQL18", str(admin.find_pg_tool("pg_dump", 16)))
            with self.assertRaises(admin.AdminError) as raised:
                admin.find_pg_tool("pg_dump", 19)
        self.assertIn("COTEACH_PG_BIN", str(raised.exception))

    def test_an_explicit_bin_folder_is_tried_first(self):
        patches = self.fake_installs([17, 18])
        chosen = self.tmp / "PostgreSQL17" / "bin"
        with patches[0], patches[1], patches[2], \
                mock.patch.dict(os.environ, {"COTEACH_PG_BIN": str(chosen)}):
            self.assertEqual(admin.find_pg_tool("pg_dump", 17).parent, chosen)
            # ...but not when it is too old for the server.
            self.assertIn("PostgreSQL18", str(admin.find_pg_tool("pg_dump", 18)))

    def test_powershell_literals_cannot_break_out(self):
        self.assertEqual(admin._ps_quote("it's; Remove-Item x"),
                         "'it''s; Remove-Item x'")

    def test_a_bad_schedule_time_registers_nothing(self):
        with mock.patch.object(admin, "_powershell") as powershell:
            for bad in ("25:00", "6pm", "18:0"):
                with self.subTest(at=bad), self.assertRaises(admin.AdminError):
                    admin.schedule_backup(at=bad)
            powershell.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows scheduled task")
    def test_the_scheduled_task_runs_the_backup_windowless_from_the_repo(self):
        with mock.patch.object(admin, "_powershell") as powershell:
            powershell.return_value = subprocess.CompletedProcess([], 0, "", "")
            admin.schedule_backup(at="07:30", keep=5, out_dir=self.tmp)
        script = powershell.call_args.args[0]
        self.assertIn("pythonw.exe", script)
        self.assertIn("-m app.database backup --keep 5", script)
        self.assertIn(f"-WorkingDirectory '{db.REPO_ROOT}'", script)
        self.assertIn("-Daily -At '07:30'", script)
        self.assertIn("-StartWhenAvailable", script)

    @unittest.skipUnless(ON_POSTGRES, "needs a PostgreSQL server to dump")
    def test_a_backup_is_verified_and_old_ones_roll_off(self):
        from datetime import datetime as dt
        engine = db.get_engine()
        for day in (1, 2, 3):
            result = admin.backup(out_dir=self.tmp, keep=2, engine=engine,
                                  now=dt(2026, 9, day, 18, 0, 0))
        self.assertTrue(result.path.is_file())
        self.assertGreater(result.size_bytes, 0)
        self.assertEqual(len(result.removed), 1)
        names = sorted(p.name for p in self.tmp.iterdir())
        database = engine.url.database
        self.assertEqual(names, [f"{database}-20260902-180000.dump",
                                 f"{database}-20260903-180000.dump"])
        listing = subprocess.run(
            [str(admin.find_pg_tool("pg_restore", 0)), "--list", str(result.path)],
            capture_output=True, text=True).stdout
        for table in db.metadata.tables:
            self.assertIn(f"TABLE DATA public {table}", listing)

    @unittest.skipUnless(ON_POSTGRES, "needs a PostgreSQL server to dump")
    def test_a_failed_dump_leaves_no_file_behind(self):
        # Anything that exits non-zero will do as a broken pg_dump.
        with mock.patch.object(admin, "find_pg_tool", return_value=Path(sys.executable)):
            with self.assertRaises(admin.AdminError) as raised:
                admin.backup(out_dir=self.tmp, engine=db.get_engine())
        self.assertIn("pg_dump failed", str(raised.exception))
        self.assertEqual(list(self.tmp.iterdir()), [])


class CopyFromSqlite(unittest.TestCase):
    """``python -m app.database copy-sqlite``: moving off the old SQLite file."""

    def old_database(self, *, with_study_tables: bool) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="coteach_copy_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = tmp / "old.db"
        source = create_engine(f"sqlite+pysqlite:///{path.as_posix()}")
        # Databases created before S4 have only these two tables.
        tables = None if with_study_tables else [db.users, db.submissions]
        db.metadata.create_all(source, tables=tables)
        self.email = f"copied{id(self)}@st.knust.edu.gh"
        with source.begin() as connection:
            connection.execute(insert(db.users).values(
                email=self.email, name="Copied", role="student_teacher",
                password_hash=accounts.hash_password("correct horse battery staple"),
                created_at=datetime(2026, 8, 10, 14, 38, 8)))
            connection.execute(insert(db.submissions), [
                dict(user_email=self.email, source=f"v{n}.pdf",
                     submitted_at=datetime(2026, 8, 10, 15, n),
                     overall_score=50 + n, band="band",
                     rubric_scores={"learning_outcomes": n},
                     contract_version="2.0.0")
                for n in range(3)])
        source.dispose()
        return path

    def test_only_postgresql_is_a_target(self):
        path = self.old_database(with_study_tables=False)
        target = create_engine("sqlite+pysqlite://")
        with self.assertRaises(admin.AdminError):
            admin.copy_from_sqlite(path, target)

    def test_a_missing_file_is_reported(self):
        with self.assertRaises(admin.AdminError):
            admin.copy_from_sqlite(Path("no/such/file.db"), db.get_engine())

    @unittest.skipUnless(ON_POSTGRES, "copying needs a PostgreSQL target")
    def test_rows_arrive_once_with_their_times_and_passwords(self):
        path = self.old_database(with_study_tables=False)
        first = admin.copy_from_sqlite(path, db.get_engine())
        self.assertEqual(first["users"], (1, 0))
        self.assertEqual(first["submissions"], (3, 0))
        self.assertNotIn("consents", first)

        again = admin.copy_from_sqlite(path, db.get_engine())
        self.assertEqual(again["users"], (0, 1))
        self.assertEqual(again["submissions"], (0, 3))

        rows = db.submission_history(self.email)
        self.assertEqual([r["source"] for r in rows], ["v0.pdf", "v1.pdf", "v2.pdf"])
        # SQLite stored UTC without a zone; it must not shift on the way in.
        self.assertEqual(rows[0]["submitted_at"],
                         datetime(2026, 8, 10, 15, 0, tzinfo=timezone.utc))
        self.assertEqual(rows[0]["rubric_scores"], {"learning_outcomes": 0})
        self.assertIsNotNone(accounts.verify_credentials(
            self.email, "correct horse battery staple"))

    @unittest.skipUnless(ON_POSTGRES, "copying needs a PostgreSQL target")
    def test_study_tables_are_copied_when_present(self):
        path = self.old_database(with_study_tables=True)
        source = create_engine(f"sqlite+pysqlite:///{path.as_posix()}")
        with source.begin() as connection:
            connection.execute(insert(db.consents).values(
                user_email=self.email, decision="agreed",
                participant_code=f"P-{id(self) % 16**8:08X}",
                materials_fingerprint="abc"))
            connection.execute(insert(db.survey_responses).values(
                user_email=self.email, wave="pre", answers={"PU1": 4},
                submissions_before=0, instrument_version="0.1.0",
                materials_fingerprint="abc"))
        source.dispose()
        report = admin.copy_from_sqlite(path, db.get_engine())
        self.assertEqual(report["consents"], (1, 0))
        self.assertEqual(report["survey_responses"], (1, 0))
        self.assertEqual(store.completed_waves(self.email), {"pre": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
