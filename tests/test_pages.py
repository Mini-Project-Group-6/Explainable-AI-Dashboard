"""Page-level tests, driven through Streamlit's AppTest.

Both pages are entered through ``main.py``. Running ``pages/dashboard.py`` as
the entry point directly makes ``st.page_link`` raise ``KeyError:
'url_pathname'`` — the page is not registered in the multi-page router when it
is also the script being run. That is a harness artifact, not an app fault, but
it means the tests must navigate the way a browser does.

    python -m unittest tests.test_pages -v

Note: the first dashboard test imports the model layer (torch + spaCy), which
takes ~15s once per process.
"""

from __future__ import annotations

import tests  # noqa: F401  — sets DATABASE_URL before any app import

import time
import unittest
from datetime import date
from itertools import count
from unittest import mock

from streamlit.testing.v1 import AppTest

from app import accounts, session
from app.study import instruments, store

tests.assert_scratch_database()

PASSWORD = "correct horse battery staple"
USER = "pages@st.knust.edu.gh"
TIMEOUT = 180


def ensure_account(email: str = USER, role: str = "student_teacher") -> None:
    try:
        accounts.create_account(email, PASSWORD, name="Pages Tester", role=role)
    except accounts.AccountError:
        pass


def signed_in_app(email: str = USER, role: str = "student_teacher") -> AppTest:
    """The dashboard, entered with ``main.py`` still registered as the main script.

    ``switch_page`` rather than letting ``main.py`` redirect: AppTest starts
    every ``run()`` from its selected page, so without it each rerun went
    through ``main.py``'s ``st.switch_page`` — and a button click on the
    dashboard was consumed by that first hop and never reached the page.
    """
    app = AppTest.from_file("main.py", default_timeout=TIMEOUT)
    app.session_state["auth_user"] = {
        "email": email, "name": "Pages Tester", "role": role}
    app.session_state["auth_last_seen"] = time.time()
    return app.switch_page("pages/dashboard.py").run()


def has_session(app: AppTest) -> bool:
    return "auth_user" in list(app.session_state.filtered_state)


class SignInPage(unittest.TestCase):

    def setUp(self):
        session._failures.clear()
        ensure_account()

    def test_it_renders(self):
        app = AppTest.from_file("main.py", default_timeout=TIMEOUT).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual([i.label for i in app.text_input],
                         ["Email address", "Password"])

    def test_no_password_is_prefilled(self):
        """Regression: the first version shipped a working password in the box."""
        app = AppTest.from_file("main.py", default_timeout=TIMEOUT).run()
        for field in app.text_input:
            self.assertEqual(field.value, "", f"{field.label} is prefilled")

    def test_signing_in_establishes_a_session(self):
        app = AppTest.from_file("main.py", default_timeout=TIMEOUT).run()
        app.text_input[0].set_value(USER)
        app.text_input[1].set_value(PASSWORD)
        app.button[0].click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertTrue(has_session(app))

    def test_a_bad_password_shows_an_error_and_no_session(self):
        app = AppTest.from_file("main.py", default_timeout=TIMEOUT).run()
        app.text_input[0].set_value(USER)
        app.text_input[1].set_value("wrong")
        app.button[0].click().run()
        self.assertFalse(has_session(app))
        self.assertTrue(any("incorrect" in str(e.value) for e in app.error))


class ClosedSignIn(unittest.TestCase):

    def test_a_closed_sign_in_says_so_without_admin_detail(self):
        # No accounts, or no database: the button alone used to be the only
        # sign that anything was wrong.
        with mock.patch.object(session, "accounts_exist", return_value=False):
            app = AppTest.from_file("main.py", default_timeout=TIMEOUT).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        notice = " ".join(str(i.value) for i in app.info)
        self.assertIn("isn't available", notice)
        self.assertNotIn("python", notice)
        self.assertTrue(app.button[0].disabled)

    def test_an_open_sign_in_shows_no_notice(self):
        ensure_account()
        app = AppTest.from_file("main.py", default_timeout=TIMEOUT).run()
        self.assertFalse(any("isn't available" in str(i.value) for i in app.info))


class AuthGate(unittest.TestCase):

    def test_the_dashboard_does_not_render_without_a_session(self):
        """Regression: typing the dashboard URL used to render the whole page."""
        app = AppTest.from_file("pages/dashboard.py", default_timeout=TIMEOUT).run()
        self.assertEqual(len(app.tabs), 0, "dashboard rendered while signed out")

    def test_an_expired_session_is_dropped_on_the_next_load(self):
        ensure_account()
        app = AppTest.from_file("pages/dashboard.py", default_timeout=TIMEOUT)
        app.session_state["auth_user"] = {
            "email": USER, "name": "Pages Tester", "role": "student_teacher"}
        app.session_state["auth_last_seen"] = (
            time.time() - session.IDLE_TIMEOUT_SECONDS - 1)
        app.run()
        self.assertFalse(has_session(app))


class Dashboard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ensure_account()

    def test_it_renders_the_four_tabs(self):
        app = signed_in_app()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual([t.label for t in app.tabs],
                         ["Scoring", "Revision history", "Trust survey",
                          "Transparency"])

    def test_the_header_shows_the_current_academic_year(self):
        from app.components.title import academic_year
        app = signed_in_app()
        blob = " ".join(str(m.value) for m in app.markdown)
        self.assertIn(f"Academic Year {academic_year()}", blob)

    def test_the_upload_control_is_present(self):
        app = signed_in_app()
        self.assertTrue(app.get("file_uploader"))

    def test_researchers_see_access_control_and_storage(self):
        ensure_account("deploy.researcher@st.knust.edu.gh", role="researcher")
        app = signed_in_app("deploy.researcher@st.knust.edu.gh", role="researcher")
        self.assertFalse(app.exception, [e.message for e in app.exception])
        blob = " ".join(str(m.value) for m in app.markdown)
        self.assertIn("Access control", blob)
        self.assertIn("password required", blob)
        self.assertIn("Storage", blob)

    def test_student_teachers_do_not_see_deployment_details(self):
        # Server paths and account counts are operational detail.
        app = signed_in_app()
        blob = " ".join(str(m.value) for m in app.markdown)
        self.assertNotIn("Access control", blob)
        self.assertNotIn("Artifacts directory", blob)

    def test_the_sidebar_offers_sign_out(self):
        app = signed_in_app()
        self.assertTrue(any("Sign out" in str(b.label) for b in app.sidebar.button))

    def test_history_shows_the_signed_in_users_submissions(self):
        from app.database import db
        db.record_submission(USER, "history-probe.pdf", 64.0, "Minimum",
                             {"learning_outcomes": 3}, "2.0.0")
        app = signed_in_app()
        self.assertTrue(app.dataframe, "no history table rendered")
        frame = app.dataframe[0].value
        self.assertIn("history-probe.pdf", list(frame["Plan"]))

    def test_it_degrades_honestly_when_the_model_layer_is_absent(self):
        """The dashboard must say so, never substitute placeholder scores.

        Only asserts something when the model layer really is missing, so the
        test is meaningful before the S1/S2 merge and inert after it.
        """
        from app.model_bridge import model_layer
        app = signed_in_app()
        if model_layer().available:
            self.skipTest("model layer present; nothing to degrade")
        self.assertTrue(any("not loaded" in str(e.value) for e in app.error))


class AcademicYear(unittest.TestCase):

    def test_it_turns_over_in_september(self):
        from app.components.title import academic_year
        with mock.patch.dict("os.environ", {"COTEACH_ACADEMIC_YEAR": ""}):
            self.assertEqual(academic_year(date(2026, 8, 31)), "2025–2026")
            self.assertEqual(academic_year(date(2026, 9, 1)), "2026–2027")
            self.assertEqual(academic_year(date(2027, 1, 15)), "2026–2027")

    def test_it_can_be_set_outright(self):
        from app.components.title import academic_year
        with mock.patch.dict("os.environ", {"COTEACH_ACADEMIC_YEAR": "2026/27"}):
            self.assertEqual(academic_year(date(2026, 3, 1)), "2026/27")


_survey_ids = count()


def button(app: AppTest, label: str):
    matches = [b for b in app.button if b.label == label]
    if not matches:
        raise AssertionError(f"no button {label!r}; have "
                             f"{[b.label for b in app.button]}")
    return matches[0]


def uploader_disabled(app: AppTest) -> bool:
    return app.get("file_uploader")[0].proto.disabled


def page_text(app: AppTest) -> str:
    return " ".join(str(m.value) for m in (*app.markdown, *app.info,
                                          *app.success, *app.error))


class TrustSurvey(unittest.TestCase):
    """S4 in the real page: the choice, the gate, the form, withdrawal."""

    def fresh(self, role: str = "student_teacher") -> str:
        email = f"survey{next(_survey_ids)}.{role}@st.knust.edu.gh"
        ensure_account(email, role)
        return email

    def test_an_undecided_student_is_asked_and_cannot_upload_yet(self):
        app = signed_in_app(self.fresh())
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertIn("Research participation", page_text(app))
        button(app, "I agree to take part")
        button(app, "No thanks — just use the dashboard")
        self.assertTrue(uploader_disabled(app))

    def test_declining_unblocks_the_upload_and_is_not_asked_again(self):
        email = self.fresh()
        app = signed_in_app(email)
        button(app, "No thanks — just use the dashboard").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(store.consent_record(email).decision, "declined")
        self.assertFalse(uploader_disabled(app))
        self.assertNotIn("Research participation", page_text(app))

    def test_agreeing_opens_the_first_survey_with_nothing_preselected(self):
        email = self.fresh()
        app = signed_in_app(email)
        button(app, "I agree to take part").click().run()
        self.assertTrue(store.consent_record(email).agreed)
        self.assertTrue(uploader_disabled(app), "baseline would follow feedback")
        radios = [r for r in app.radio if str(r.key).startswith("study_pre_")]
        self.assertEqual(len(radios), len(instruments.items_for("pre"))
                         + len(instruments.background_for("pre")))
        self.assertTrue(all(r.value is None for r in radios))

    def test_an_incomplete_survey_is_refused_and_a_complete_one_saved(self):
        email = self.fresh()
        store.record_consent(email, agreed=True)
        app = signed_in_app(email)

        button(app, "Submit survey").click().run()
        self.assertTrue(any("no answer" in str(e.value) for e in app.error))
        self.assertEqual(store.completed_waves(email), {})

        for item in instruments.items_for("pre"):
            app.radio(key=f"study_pre_{item.id}").set_value(4)
        button(app, "Submit survey").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(store.completed_waves(email), {"pre": 0})
        self.assertFalse(uploader_disabled(app))
        self.assertIn("your answers are saved", page_text(app))

    def test_withdrawing_from_the_tab(self):
        email = self.fresh()
        store.record_consent(email, agreed=True)
        app = signed_in_app(email)
        self.assertTrue(button(app, "Withdraw").disabled, "needs confirming")
        app.checkbox(key="study_withdraw_confirm").check().run()
        button(app, "Withdraw").click().run()
        self.assertTrue(store.consent_record(email).withdrawn)
        self.assertFalse(uploader_disabled(app))

    def test_researchers_get_administration_not_a_survey(self):
        email = self.fresh("researcher")
        app = signed_in_app(email, role="researcher")
        self.assertFalse(app.exception, [e.message for e in app.exception])
        text = page_text(app)
        self.assertIn("Study administration", text)
        self.assertNotIn("Research participation", text)
        self.assertFalse(uploader_disabled(app))
        self.assertEqual(len(app.get("download_button")), 3)

    def test_tutors_are_not_asked(self):
        email = self.fresh("tutor")
        app = signed_in_app(email, role="tutor")
        self.assertNotIn("Research participation", page_text(app))
        self.assertFalse(uploader_disabled(app))
        self.assertIsNone(store.consent_record(email))

    def test_signing_out_clears_a_half_answered_survey(self):
        email = self.fresh()
        store.record_consent(email, agreed=True)
        app = signed_in_app(email)
        app.radio(key="study_pre_PU1").set_value(5).run()
        app.sidebar.button[0].click().run()
        self.assertFalse([k for k in app.session_state.filtered_state
                          if str(k).startswith("study_")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
