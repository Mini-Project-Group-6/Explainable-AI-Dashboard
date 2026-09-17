"""S4 study tests: instruments, consent, the survey gate, storage and export.

The failure modes worth guarding are the ones that would quietly spoil the
data rather than crash: a baseline taken after feedback, a withdrawn person's
answers surviving, an email reaching an export, approved wording edited
without anyone noticing.

    python -m unittest tests.test_study -v
"""

from __future__ import annotations

import tests  # noqa: F401  — sets DATABASE_URL before any app import

import csv
import io
import re
import unittest
from itertools import count
from unittest import mock

from app import accounts
from app.database import db
from app.session import User
from app.study import approval, documents, export, flow, instruments, store

tests.assert_scratch_database()

PASSWORD = "correct horse battery staple"
SCORES = {"learning_outcomes": 3, "lesson_closure": 2}
_ids = count()


def new_user(role: str = "student_teacher") -> User:
    """A fresh account per test, so tests sharing the scratch DB never collide."""
    email = f"study{next(_ids)}.{role}@st.knust.edu.gh"
    accounts.create_account(email, PASSWORD, name="Study Tester", role=role)
    return User(email=email, name="Study Tester", role=role)


def complete(wave: str, value: int = 4, **overrides) -> dict:
    answers = {item.id: value for item in instruments.items_for(wave)}
    answers.update(overrides)
    return answers


def score_plans(user: User, n: int) -> None:
    for index in range(n):
        db.record_submission(user.email, f"{user.name} draft {index}.pdf",
                             50.0 + index, "band", SCORES, "2.0.0")


def rows(table) -> list[dict]:
    return list(csv.DictReader(io.StringIO(export.to_csv(table))))


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

class Instruments(unittest.TestCase):

    def test_item_ids_are_unique_and_belong_to_a_scale(self):
        ids = [item.id for item in instruments.ITEMS]
        self.assertEqual(len(ids), len(set(ids)))
        keys = {s.key for s in instruments.SCALES}
        for item in instruments.ITEMS:
            self.assertIn(item.scale, keys, item.id)
            self.assertTrue(item.text and item.source_text, item.id)

    def test_every_scale_cites_a_known_source(self):
        for s in instruments.SCALES:
            self.assertIn(s.source, instruments.SOURCES, s.key)

    def test_the_pre_and_post_surveys_ask_what_the_protocol_says(self):
        pre = [i.id for i in instruments.items_for("pre")]
        post = [i.id for i in instruments.items_for("post")]
        self.assertEqual(len(pre), 22)
        self.assertEqual(len(post), 30)
        # Every pre item is repeated verbatim, so each pair is like for like.
        self.assertEqual(post[:len(pre)], pre)
        self.assertFalse(any(i.startswith("ES") for i in pre),
                         "explanation satisfaction asked before any explanation")

    def test_background_is_pre_only(self):
        self.assertTrue(instruments.background_for("pre"))
        self.assertEqual(instruments.background_for("post"), ())

    def test_headings_do_not_name_the_constructs(self):
        for heading, _ in instruments.sections_for("post"):
            for s in instruments.SCALES:
                self.assertNotIn(s.name.lower(), heading.lower())

    def test_an_unknown_wave_is_refused(self):
        with self.assertRaises(ValueError):
            instruments.items_for("mid")

    def test_a_complete_survey_has_no_problems(self):
        self.assertEqual(instruments.problems(complete("pre"), "pre"), [])
        self.assertEqual(instruments.problems(complete("post"), "post"), [])

    def test_unanswered_statements_are_named_by_position(self):
        answers = complete("pre")
        answers["PU3"] = None
        del answers["TR1"]
        found = " ".join(instruments.problems(answers, "pre"))
        self.assertIn("Statements 3 and 15", found)

    def test_out_of_range_and_wrongly_typed_answers_are_refused(self):
        for bad in (0, 6, "3", True, 2.5):
            with self.subTest(bad=bad):
                self.assertTrue(instruments.problems(
                    complete("pre", PU1=bad), "pre"))

    def test_answers_for_the_other_wave_are_refused(self):
        self.assertTrue(instruments.problems(complete("pre", ES1=3), "pre"))

    def test_background_is_optional_but_must_be_a_listed_option(self):
        self.assertEqual(instruments.problems(complete("pre"), "pre"), [])
        self.assertEqual(instruments.problems(
            complete("pre", BG1="Level 200"), "pre"), [])
        self.assertTrue(instruments.problems(
            complete("pre", BG1="Level 900"), "pre"))

    def test_reverse_items_are_keyed_before_averaging(self):
        # TR6 is "I am wary of the dashboard": answering 2 is keyed as 4.
        self.assertEqual(
            instruments.scale_means(complete("pre", TR6=2), "pre")["TR"], 4.0)
        self.assertEqual(
            instruments.scale_means(complete("pre"), "pre")["TR"], (7 * 4 + 2) / 8)
        # Answered 5 means *low* trust: keyed as 1.
        means = instruments.scale_means(complete("pre", TR6=5), "pre")
        self.assertEqual(means["TR"], (7 * 4 + 1) / 8)
        self.assertEqual(means["PU"], 4.0)

    def test_an_incomplete_scale_has_no_mean(self):
        answers = complete("pre")
        del answers["PEOU2"]
        self.assertIsNone(instruments.scale_means(answers, "pre")["PEOU"])

    def test_the_committed_appendix_matches_the_code(self):
        """docs/irb/survey_instruments.md is what the committee reads.

        Regenerate with ``python -m app.study instruments --write``.
        """
        committed = documents.read(documents.DOCS_DIR / "survey_instruments.md")
        self.assertEqual(committed, instruments.adaptation_markdown(),
                         "survey_instruments.md is stale; regenerate it")


# ---------------------------------------------------------------------------
# Documents and approval
# ---------------------------------------------------------------------------

class Approval(unittest.TestCase):

    def test_the_consent_form_yields_its_statements(self):
        statements = documents.consent_statements()
        self.assertEqual(len(statements), 6)
        self.assertTrue(any("voluntary" in s for s in statements))
        self.assertTrue(any("18" in s for s in statements))

    def test_the_information_sheet_loads_without_its_title(self):
        text = documents.body(documents.INFORMATION_SHEET)
        self.assertIn("Do I have to take part?", text)
        self.assertFalse(text.startswith("# "))

    def test_placeholders_are_found(self):
        found = documents.unresolved_placeholders()
        self.assertIn("ETHICS_COMMITTEE", found.get(documents.INFORMATION_SHEET.name, []))

    def test_draft_materials_are_not_approved(self):
        self.assertFalse(approval.is_approved())
        self.assertIsNone(approval.protocol_for_record())
        self.assertTrue(any("No ethics approval" in p
                            for p in approval.approval_problems()))

    def test_the_fingerprint_is_stable(self):
        self.assertEqual(approval.fingerprint(), approval.fingerprint())
        self.assertRegex(approval.fingerprint(), r"^[0-9a-f]{16}$")

    def _approved(self):
        """Patch every precondition so the current materials count as approved."""
        return [
            mock.patch.object(documents, "unresolved_placeholders", return_value={}),
            mock.patch.object(instruments, "_VERIFIED_SOURCES",
                              frozenset(instruments.SOURCES)),
            mock.patch.object(approval, "IRB_PROTOCOL", "TEST/001"),
            mock.patch.object(approval, "APPROVED_FINGERPRINT",
                              approval.fingerprint()),
        ]

    def test_recorded_approval_of_the_current_materials_is_approved(self):
        patches = self._approved()
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        self.assertEqual(approval.approval_problems(), [])
        self.assertEqual(approval.protocol_for_record(), "TEST/001")

    def test_editing_an_approved_item_voids_approval(self):
        patches = self._approved()
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        self.assertTrue(approval.is_approved())

        edited = list(instruments.ITEMS)
        edited[0] = instruments.Item("PU1", "PU", "Something else entirely.",
                                     edited[0].source_text)
        with mock.patch.object(instruments, "ITEMS", tuple(edited)):
            self.assertFalse(approval.is_approved())
            self.assertIsNone(approval.protocol_for_record())
            self.assertTrue(any("changed since approval" in p
                                for p in approval.approval_problems()))

    def test_changing_the_procedure_voids_approval(self):
        # The information sheet tells participants "after two more plans".
        patches = self._approved()
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        with mock.patch.object(flow, "POST_AFTER_SUBMISSIONS", 3):
            self.assertFalse(approval.is_approved())

    def test_unverified_wording_blocks_approval(self):
        patches = self._approved()
        for p in patches:
            p.start()
        self.addCleanup(mock.patch.stopall)
        with mock.patch.object(instruments, "_VERIFIED_SOURCES", frozenset()):
            self.assertFalse(approval.is_approved())


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

AGREED = flow.ConsentRecord("agreed", "P-00000001")


def state(consent=AGREED, completed=None, submissions=0,
          role="student_teacher", storage_ok=True) -> flow.StudyState:
    return flow.resolve(role, storage_ok, consent, completed or {}, submissions)


class Gate(unittest.TestCase):

    def test_tutors_and_researchers_are_never_asked_or_blocked(self):
        for role in ("tutor", "researcher"):
            s = state(consent=None, role=role)
            self.assertEqual(s.stage, flow.NOT_PARTICIPANT)
            self.assertFalse(s.upload_blocked)

    def test_no_storage_means_no_study_and_no_block(self):
        s = state(consent=None, storage_ok=False)
        self.assertEqual(s.stage, flow.UNAVAILABLE)
        self.assertFalse(s.upload_blocked)

    def test_an_undecided_student_cannot_score_yet(self):
        s = state(consent=None)
        self.assertEqual(s.stage, flow.UNDECIDED)
        self.assertTrue(s.upload_blocked)
        self.assertTrue(s.block_reason)

    def test_declining_unblocks_immediately(self):
        s = state(consent=flow.ConsentRecord("declined"))
        self.assertEqual(s.stage, flow.DECLINED)
        self.assertFalse(s.upload_blocked)
        self.assertIsNone(s.due_wave)

    def test_the_baseline_comes_before_any_feedback(self):
        s = state()
        self.assertEqual((s.stage, s.due_wave), (flow.PRE_DUE, "pre"))
        self.assertTrue(s.upload_blocked)

    def test_the_follow_up_waits_for_enough_plans(self):
        needed = flow.POST_AFTER_SUBMISSIONS
        for scored in range(needed):
            s = state(completed={"pre": 0}, submissions=scored)
            self.assertEqual(s.stage, flow.WAITING)
            self.assertFalse(s.upload_blocked)
            self.assertEqual(s.plans_until_post, needed - scored)
        s = state(completed={"pre": 0}, submissions=needed)
        self.assertEqual((s.stage, s.due_wave), (flow.POST_DUE, "post"))
        self.assertFalse(s.upload_blocked)

    def test_plans_are_counted_from_the_first_survey_not_from_zero(self):
        # Declined, used the tool for a while, then opted in.
        s = state(completed={"pre": 5}, submissions=6)
        self.assertEqual(s.stage, flow.WAITING)
        self.assertEqual(s.plans_since_pre, 1)

    def test_both_surveys_done_is_complete(self):
        s = state(completed={"pre": 0, "post": 2}, submissions=4)
        self.assertEqual(s.stage, flow.COMPLETE)
        self.assertIsNone(s.due_wave)

    def test_withdrawn_is_never_blocked_or_surveyed(self):
        s = state(consent=flow.ConsentRecord("agreed", "P-1", withdrawn=True))
        self.assertEqual(s.stage, flow.WITHDRAWN)
        self.assertFalse(s.upload_blocked)
        self.assertIsNone(s.due_wave)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class Consent(unittest.TestCase):

    def test_the_schema_includes_the_study_tables(self):
        for table in ("consents", "survey_responses"):
            self.assertIn(table, db.metadata.tables)

    def test_agreeing_assigns_a_random_code(self):
        user = new_user()
        record = store.record_consent(user.email, agreed=True)
        self.assertTrue(record.agreed)
        self.assertRegex(record.participant_code, r"^P-[0-9A-F]{8}$")
        self.assertNotIn(user.email.split("@")[0], record.participant_code)

    def test_codes_are_unique(self):
        codes = {store.record_consent(new_user().email, True).participant_code
                 for _ in range(5)}
        self.assertEqual(len(codes), 5)

    def test_declining_assigns_no_code(self):
        record = store.record_consent(new_user().email, agreed=False)
        self.assertEqual(record.decision, "declined")
        self.assertIsNone(record.participant_code)

    def test_a_decliner_can_opt_in_later(self):
        user = new_user()
        store.record_consent(user.email, agreed=False)
        record = store.record_consent(user.email, agreed=True)
        self.assertTrue(record.agreed)
        self.assertIsNotNone(record.participant_code)

    def test_an_agreement_is_not_turned_into_a_decline(self):
        user = new_user()
        first = store.record_consent(user.email, agreed=True)
        self.assertEqual(store.record_consent(user.email, agreed=False), first)

    def test_a_double_click_does_not_change_the_code(self):
        user = new_user()
        first = store.record_consent(user.email, agreed=True)
        self.assertEqual(store.record_consent(user.email, agreed=True), first)

    def test_withdrawal_is_final_here(self):
        user = new_user()
        store.record_consent(user.email, agreed=True)
        self.assertTrue(store.withdraw(user.email))
        self.assertTrue(store.record_consent(user.email, agreed=True).withdrawn)
        self.assertEqual(store.current_state(user).stage, flow.WITHDRAWN)

    def test_only_an_active_participant_can_withdraw(self):
        self.assertFalse(store.withdraw(new_user().email))
        decliner = new_user()
        store.record_consent(decliner.email, agreed=False)
        self.assertFalse(store.withdraw(decliner.email))

    def test_the_consent_records_which_materials_were_agreed_to(self):
        user = new_user()
        store.record_consent(user.email, agreed=True)
        with db.get_engine().connect() as connection:
            stored = connection.execute(
                db.consents.select().where(db.consents.c.user_email == user.email)
            ).mappings().one()["materials_fingerprint"]
        self.assertEqual(stored, approval.fingerprint())


class Responses(unittest.TestCase):

    def agreed_user(self) -> User:
        user = new_user()
        store.record_consent(user.email, agreed=True)
        return user

    def test_nobody_undecided_can_submit(self):
        saved, problems = store.record_response(new_user(), "pre", complete("pre"))
        self.assertFalse(saved)
        self.assertIn("not open", problems[0])

    def test_tutors_cannot_submit(self):
        tutor = new_user("tutor")
        saved, _ = store.record_response(tutor, "pre", complete("pre"))
        self.assertFalse(saved)

    def test_the_follow_up_cannot_be_answered_first(self):
        saved, _ = store.record_response(self.agreed_user(), "post", complete("post"))
        self.assertFalse(saved)

    def test_an_invalid_survey_stores_nothing(self):
        user = self.agreed_user()
        saved, problems = store.record_response(user, "pre", complete("pre", PU1=None))
        self.assertFalse(saved)
        self.assertTrue(problems)
        self.assertEqual(store.completed_waves(user.email), {})

    def test_the_full_journey(self):
        user = self.agreed_user()
        saved, problems = store.record_response(
            user, "pre", complete("pre", BG2="Often"), duration_seconds=312.4)
        self.assertTrue(saved, problems)
        self.assertEqual(store.completed_waves(user.email), {"pre": 0})
        self.assertEqual(store.current_state(user).stage, flow.WAITING)

        score_plans(user, flow.POST_AFTER_SUBMISSIONS)
        self.assertEqual(store.current_state(user).stage, flow.POST_DUE)

        saved, problems = store.record_response(user, "post", complete("post"))
        self.assertTrue(saved, problems)
        self.assertEqual(store.completed_waves(user.email),
                         {"pre": 0, "post": flow.POST_AFTER_SUBMISSIONS})
        self.assertEqual(store.current_state(user).stage, flow.COMPLETE)

    def test_a_wave_is_recorded_once(self):
        user = self.agreed_user()
        self.assertTrue(store.record_response(user, "pre", complete("pre"))[0])
        saved, _ = store.record_response(user, "pre", complete("pre", PU1=1))
        self.assertFalse(saved)

    def test_unapproved_responses_carry_no_protocol(self):
        user = self.agreed_user()
        store.record_response(user, "pre", complete("pre"))
        with db.get_engine().connect() as connection:
            row = connection.execute(
                db.survey_responses.select()
                .where(db.survey_responses.c.user_email == user.email)
            ).mappings().one()
        self.assertIsNone(row["irb_protocol"])
        self.assertEqual(row["instrument_version"], instruments.INSTRUMENT_VERSION)
        self.assertEqual(row["answers"]["PU1"], 4)

    def test_withdrawal_deletes_the_answers(self):
        user = self.agreed_user()
        store.record_response(user, "pre", complete("pre"))
        self.assertTrue(store.withdraw(user.email))
        self.assertEqual(store.completed_waves(user.email), {})

    def test_enrolment_counts_move(self):
        before = store.enrolment_counts()
        user = self.agreed_user()
        store.record_response(user, "pre", complete("pre"))
        store.record_consent(new_user().email, agreed=False)
        after = store.enrolment_counts()
        self.assertEqual(after["agreed"], before["agreed"] + 1)
        self.assertEqual(after["declined"], before["declined"] + 1)
        self.assertEqual(after["pre"], before["pre"] + 1)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class Export(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # A participant with the full journey.
        cls.done = new_user()
        cls.code = store.record_consent(cls.done.email, True).participant_code
        store.record_response(cls.done, "pre", complete("pre", TR6=5))
        score_plans(cls.done, flow.POST_AFTER_SUBMISSIONS)
        store.record_response(cls.done, "post", complete("post"))
        score_plans(cls.done, 1)

        cls.decliner = new_user()
        store.record_consent(cls.decliner.email, False)
        score_plans(cls.decliner, 1)

        cls.withdrawn = new_user()
        cls.withdrawn_code = store.record_consent(
            cls.withdrawn.email, True).participant_code
        store.record_response(cls.withdrawn, "pre", complete("pre"))
        store.withdraw(cls.withdrawn.email)

        cls.tables = export.build(include_pilot=True)
        cls.text = "".join(export.to_csv(t) for t in cls.tables.values())

    def mine(self, name: str) -> list[dict]:
        return [r for r in rows(self.tables[name])
                if r["participant_code"] == self.code]

    def test_pilot_data_is_left_out_by_default(self):
        self.assertFalse(approval.is_approved())
        tables = export.build()
        self.assertEqual([len(t[1]) for t in tables.values()], [0, 0, 0])

    def test_no_identity_reaches_the_export(self):
        self.assertNotIn("@", self.text)
        self.assertNotIn("Study Tester", self.text)
        self.assertNotIn(".pdf", self.text)

    def test_declined_and_withdrawn_people_are_absent(self):
        self.assertNotIn(self.withdrawn_code, self.text)
        codes = {r["participant_code"] for r in rows(self.tables["participants"])}
        self.assertIn(self.code, codes)
        self.assertNotIn("", codes)

    def test_participant_summary(self):
        (row,) = self.mine("participants")
        self.assertEqual(row["pre_completed"], "1")
        self.assertEqual(row["post_completed"], "1")
        self.assertEqual(row["pre_after_feedback"], "0")
        self.assertEqual(row["plans_between_surveys"],
                         str(flow.POST_AFTER_SUBMISSIONS))
        self.assertEqual(row["total_plans"], str(flow.POST_AFTER_SUBMISSIONS + 1))
        self.assertEqual(row["pilot_consent"], "1")

    def test_responses_hold_raw_items_and_keyed_means(self):
        by_wave = {r["wave"]: r for r in self.mine("responses")}
        pre = by_wave["pre"]
        self.assertEqual(pre["TR6"], "5")                      # raw
        self.assertEqual(float(pre["TR_mean"]), (7 * 4 + 1) / 8)  # keyed
        self.assertEqual(pre["ES1"], "")                       # not asked
        self.assertEqual(pre["ES_mean"], "")
        self.assertEqual(pre["pilot"], "1")
        self.assertEqual(float(by_wave["post"]["ES_mean"]), 4.0)
        self.assertEqual(by_wave["post"]["pre_after_feedback"], "")

    def test_submissions_are_placed_on_the_timeline(self):
        plans = self.mine("submissions")
        self.assertEqual([p["sequence"] for p in plans],
                         [str(n) for n in range(1, len(plans) + 1)])
        self.assertEqual([p["phase"] for p in plans],
                         ["between_surveys"] * flow.POST_AFTER_SUBMISSIONS
                         + ["after_post"])
        self.assertEqual(plans[0]["learning_outcomes"], "3")

    def test_the_columns_are_fixed_across_waves(self):
        header = next(csv.reader(io.StringIO(export.to_csv(self.tables["responses"]))))
        self.assertEqual(header, export.RESPONSE_COLUMNS)
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_]+", c) for c in header))

    def test_files_are_written_with_a_bom_for_excel(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            written = export.write(Path(tmp), tables=self.tables)
            self.assertEqual(sorted(p.name for p in written),
                             ["participants.csv", "responses.csv", "submissions.csv"])
            self.assertTrue(written[0].read_bytes().startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
