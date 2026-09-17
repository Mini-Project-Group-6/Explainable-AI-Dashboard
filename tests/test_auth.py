"""Authentication tests.

The behaviour under test is what stops an open dashboard from collecting
identifiable data from student teachers. The first version of this app accepted
any email with any non-empty password and said nothing about it, so these are
regression tests for a defect that shipped, not hypotheticals.

    python -m unittest tests.test_auth -v
"""

from __future__ import annotations

import tests  # noqa: F401  — sets DATABASE_URL before any app import

import csv
import shutil
import tempfile
import time
import unittest
from itertools import count
from pathlib import Path

from app import accounts, session
from app.database import db

tests.assert_scratch_database()

PASSWORD = "correct horse battery staple"


def make_account(email: str, password: str = PASSWORD, **kwargs):
    try:
        return accounts.create_account(email, password, **kwargs)
    except accounts.AccountError:
        return accounts.Account(email=accounts.normalise_email(email), name="")


class PasswordHashing(unittest.TestCase):

    def test_hash_is_never_the_password(self):
        encoded = accounts.hash_password(PASSWORD)
        self.assertNotIn(PASSWORD, encoded)
        self.assertTrue(encoded.startswith("scrypt$"))

    def test_same_password_hashes_differently_each_time(self):
        # A shared salt would let one cracked hash unlock every account that
        # happened to choose the same password.
        self.assertNotEqual(accounts.hash_password(PASSWORD),
                            accounts.hash_password(PASSWORD))

    def test_verification_accepts_the_password_and_rejects_others(self):
        encoded = accounts.hash_password(PASSWORD)
        self.assertTrue(accounts.verify_password(PASSWORD, encoded))
        self.assertFalse(accounts.verify_password(PASSWORD + "x", encoded))
        self.assertFalse(accounts.verify_password("", encoded))

    def test_a_corrupt_hash_is_rejected_not_raised(self):
        # A truncated or hand-edited row must fail the login, not 500 the app.
        for broken in ("", "nonsense", "scrypt$notanumber$8$1$aaaa$bbbb",
                       "bcrypt$16384$8$1$aaaa$bbbb"):
            self.assertFalse(accounts.verify_password(PASSWORD, broken), broken)

    def test_unicode_normalisation(self):
        # The same password typed on two keyboards can differ byte-wise: one
        # composed accent, one combining. NFKC makes them the same password.
        composed, combining = "cafépass123", "cafépass123"
        self.assertTrue(
            accounts.verify_password(combining, accounts.hash_password(composed)))


class AccountStore(unittest.TestCase):

    def test_email_is_case_insensitive(self):
        make_account("Case.Test@st.knust.edu.gh")
        self.assertIsNotNone(
            accounts.verify_credentials("CASE.TEST@ST.KNUST.EDU.GH", PASSWORD))

    def test_duplicate_is_refused(self):
        make_account("dupe@st.knust.edu.gh")
        with self.assertRaises(accounts.AccountError):
            accounts.create_account("DUPE@st.knust.edu.gh", PASSWORD)

    def test_weak_password_is_refused(self):
        with self.assertRaises(accounts.AccountError):
            accounts.create_account("weak@st.knust.edu.gh", "1234")

    def test_wrong_password_and_unknown_account_both_return_none(self):
        make_account("known@st.knust.edu.gh")
        self.assertIsNone(
            accounts.verify_credentials("known@st.knust.edu.gh", "wrong"))
        self.assertIsNone(
            accounts.verify_credentials("nobody@st.knust.edu.gh", PASSWORD))

    def test_unknown_account_costs_the_same_time_as_a_wrong_password(self):
        """No enumeration by stopwatch.

        Returning early on an unknown email would make it measurably faster
        than a real account with a bad password, which leaks the enrolment
        list. The dummy-hash path exists to prevent that.
        """
        make_account("timed@st.knust.edu.gh")

        def median_ms(email):
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                accounts.verify_credentials(email, "definitely wrong")
                samples.append((time.perf_counter() - started) * 1000)
            return sorted(samples)[len(samples) // 2]

        known = median_ms("timed@st.knust.edu.gh")
        unknown = median_ms("ghost@st.knust.edu.gh")
        # Generous: catches "returns instantly", not microarchitectural noise.
        self.assertLess(abs(known - unknown), max(known, unknown) * 0.5 + 25,
                        f"known={known:.0f}ms unknown={unknown:.0f}ms")


class SignIn(unittest.TestCase):

    def setUp(self):
        session._failures.clear()
        import streamlit as st
        st.session_state.clear()

    def test_correct_credentials_are_accepted(self):
        make_account("signin@st.knust.edu.gh")
        ok, message = session.sign_in("signin@st.knust.edu.gh", PASSWORD)
        self.assertTrue(ok, message)
        self.assertIsNotNone(session.current_user())

    def test_wrong_password_is_refused_without_a_session(self):
        make_account("refuse@st.knust.edu.gh")
        ok, message = session.sign_in("refuse@st.knust.edu.gh", "nope")
        self.assertFalse(ok)
        self.assertIsNone(session.current_user())
        self.assertEqual(message, "Email or password is incorrect.")

    def test_the_message_does_not_reveal_whether_the_account_exists(self):
        make_account("exists@st.knust.edu.gh")
        _, known = session.sign_in("exists@st.knust.edu.gh", "wrong")
        session._failures.clear()
        _, unknown = session.sign_in("ghost2@st.knust.edu.gh", "wrong")
        self.assertEqual(known, unknown)

    def test_malformed_email_is_refused(self):
        ok, _ = session.sign_in("not-an-email", PASSWORD)
        self.assertFalse(ok)

    def test_lockout_after_repeated_failures(self):
        make_account("locked@st.knust.edu.gh")
        for _ in range(session.MAX_ATTEMPTS):
            session.sign_in("locked@st.knust.edu.gh", "wrong")
        ok, message = session.sign_in("locked@st.knust.edu.gh", PASSWORD)
        self.assertFalse(ok, "the correct password must not bypass a lockout")
        self.assertIn("Too many failed attempts", message)

    def test_a_successful_sign_in_clears_the_failure_count(self):
        make_account("reset@st.knust.edu.gh")
        for _ in range(session.MAX_ATTEMPTS - 1):
            session.sign_in("reset@st.knust.edu.gh", "wrong")
        self.assertTrue(session.sign_in("reset@st.knust.edu.gh", PASSWORD)[0])
        self.assertNotIn("reset@st.knust.edu.gh", session._failures)


class SessionLifetime(unittest.TestCase):

    def setUp(self):
        session._failures.clear()
        import streamlit as st
        st.session_state.clear()

    def test_an_idle_session_expires(self):
        make_account("idle@st.knust.edu.gh")
        session.sign_in("idle@st.knust.edu.gh", PASSWORD)
        import streamlit as st
        st.session_state["auth_last_seen"] = (
            time.time() - session.IDLE_TIMEOUT_SECONDS - 1)
        self.assertIsNone(session.current_user())
        self.assertTrue(st.session_state.get("auth_expired"))

    def test_an_active_session_does_not_expire(self):
        make_account("active@st.knust.edu.gh")
        session.sign_in("active@st.knust.edu.gh", PASSWORD)
        self.assertIsNotNone(session.current_user())

    def test_sign_out_clears_another_users_scored_plan(self):
        """A shared lab machine must not hand over the previous student's work."""
        import streamlit as st
        make_account("out@st.knust.edu.gh")
        session.sign_in("out@st.knust.edu.gh", PASSWORD)
        st.session_state["scored_submissions"] = {"abc": {"secret": True}}
        st.session_state["session_history"] = [{"source": "theirs.pdf"}]
        session.sign_out()
        self.assertIsNone(session.current_user())
        self.assertNotIn("scored_submissions", st.session_state)
        self.assertNotIn("session_history", st.session_state)


_batch = count()


class BulkImport(unittest.TestCase):
    """``python -m app.accounts import``: a whole class list in one step."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="coteach_import_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.tag = f"b{next(_batch)}"

    def email(self, who: str) -> str:
        return f"{who}.{self.tag}@st.knust.edu.gh"

    def class_list(self, text: str, encoding: str = "utf-8") -> Path:
        path = self.tmp / "class.csv"
        path.write_text(text.replace("{tag}", self.tag), encoding=encoding)
        return path

    def sheet(self) -> Path:
        return self.tmp / "out" / "passwords.csv"

    def read_sheet(self, path: Path) -> list[dict]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_a_class_list_becomes_working_accounts(self):
        source = self.class_list(
            "Email,Name,Role\n"
            "Ama.Mensah.{tag}@st.knust.edu.gh,Ama Mensah,\n"
            "kofi.{tag}@st.knust.edu.gh,,tutor\n"
            "\n")
        result = accounts.import_accounts(source, self.sheet())

        self.assertEqual([a.email for a in result.created],
                         [self.email("ama.mensah"), self.email("kofi")])
        rows = {r["email"]: r for r in self.read_sheet(result.sheet)}
        ama = rows[self.email("ama.mensah")]
        self.assertEqual((ama["name"], ama["role"]), ("Ama Mensah", "student_teacher"))
        # A blank name falls back to the address, as the single-account path does.
        self.assertEqual(rows[self.email("kofi")]["name"], f"Kofi {self.tag.capitalize()}")
        self.assertEqual(rows[self.email("kofi")]["role"], "tutor")
        for row in rows.values():
            self.assertGreaterEqual(len(row["password"]), accounts.MIN_PASSWORD_LENGTH)
            self.assertIsNotNone(
                accounts.verify_credentials(row["email"], row["password"]))

    def test_passwords_are_all_different(self):
        source = self.class_list("email\n" + "".join(
            f"s{n}.{{tag}}@st.knust.edu.gh\n" for n in range(5)))
        result = accounts.import_accounts(source, self.sheet())
        passwords = [r["password"] for r in self.read_sheet(result.sheet)]
        self.assertEqual(len(set(passwords)), 5)

    def test_an_excel_utf8_file_with_a_bom_reads(self):
        source = self.class_list("email,name\nefua.{tag}@st.knust.edu.gh,Efua Asantewaa Ɔdɔ\n",
                                 encoding="utf-8-sig")
        result = accounts.import_accounts(source, self.sheet())
        self.assertEqual(result.created[0].name, "Efua Asantewaa Ɔdɔ")

    def test_one_bad_row_and_nothing_is_created(self):
        source = self.class_list(
            "email,role\n"
            "good.{tag}@st.knust.edu.gh,\n"
            "not-an-address,\n"
            "dup.{tag}@st.knust.edu.gh,\n"
            "DUP.{tag}@st.knust.edu.gh,\n"
            "boss.{tag}@st.knust.edu.gh,admin\n")
        with self.assertRaises(accounts.AccountError) as raised:
            accounts.import_accounts(source, self.sheet())
        message = str(raised.exception)
        self.assertIn("line 3", message)
        self.assertIn("line 5", message)       # the duplicate, case-insensitively
        self.assertIn("line 6", message)       # the unknown role
        self.assertIsNone(accounts.verify_credentials(self.email("good"), PASSWORD))
        self.assertNotIn(self.email("good"),
                         [a.email for a in accounts.list_accounts()])
        self.assertFalse(self.sheet().exists())

    def test_an_unrecognised_column_is_refused(self):
        # Otherwise "full name" is dropped silently and names default to emails.
        source = self.class_list("email,full name\nx.{tag}@st.knust.edu.gh,X Y\n")
        with self.assertRaises(accounts.AccountError) as raised:
            accounts.import_accounts(source, self.sheet())
        self.assertIn("full name", str(raised.exception))

    def test_a_file_without_an_email_column_is_refused(self):
        source = self.class_list("name\nSomeone\n")
        with self.assertRaises(accounts.AccountError):
            accounts.import_accounts(source, self.sheet())

    def test_existing_accounts_are_skipped_not_reset(self):
        make_account(self.email("existing"))
        source = self.class_list("email\nexisting.{tag}@st.knust.edu.gh\n"
                                 "fresh.{tag}@st.knust.edu.gh\n")
        result = accounts.import_accounts(source, self.sheet())
        self.assertEqual(result.skipped, [self.email("existing")])
        self.assertEqual([a.email for a in result.created], [self.email("fresh")])
        self.assertIsNotNone(
            accounts.verify_credentials(self.email("existing"), PASSWORD))
        self.assertNotIn(self.email("existing"),
                         [r["email"] for r in self.read_sheet(result.sheet)])

    def test_rerunning_an_import_writes_no_sheet(self):
        source = self.class_list("email\nonce.{tag}@st.knust.edu.gh\n")
        accounts.import_accounts(source, self.sheet())
        again = accounts.import_accounts(source, self.tmp / "second.csv")
        self.assertEqual(again.created, [])
        self.assertIsNone(again.sheet)
        self.assertFalse((self.tmp / "second.csv").exists())

    def test_an_existing_sheet_is_never_overwritten(self):
        self.sheet().parent.mkdir(parents=True)
        self.sheet().write_text("earlier passwords", encoding="utf-8")
        source = self.class_list("email\nkeep.{tag}@st.knust.edu.gh\n")
        with self.assertRaises(accounts.AccountError):
            accounts.import_accounts(source, self.sheet())
        self.assertEqual(self.sheet().read_text(encoding="utf-8"), "earlier passwords")

    def test_the_sheet_may_not_land_where_git_would_commit_it(self):
        source = self.class_list("email\nrepo.{tag}@st.knust.edu.gh\n")
        for target in (db.REPO_ROOT / "passwords.csv",
                       db.REPO_ROOT / "docs" / "passwords.csv"):
            with self.subTest(target=target):
                with self.assertRaises(accounts.AccountError) as raised:
                    accounts.import_accounts(source, target)
                self.assertIn("repository", str(raised.exception))
                self.assertFalse(target.exists())
        self.assertIsNone(accounts.verify_credentials(self.email("repo"), PASSWORD))

    def test_the_gitignored_data_directory_is_allowed(self):
        target = db.REPO_ROOT / "data" / f"test-sheet-{self.tag}-{time.time_ns()}.csv"
        self.addCleanup(target.unlink, missing_ok=True)
        source = self.class_list("email\ndatadir.{tag}@st.knust.edu.gh\n")
        result = accounts.import_accounts(source, target)
        self.assertEqual(result.sheet, target)

    def test_single_accounts_reject_unknown_roles_too(self):
        with self.assertRaises(accounts.AccountError):
            accounts.create_account(self.email("root"), PASSWORD, role="admin")


if __name__ == "__main__":
    unittest.main(verbosity=2)
