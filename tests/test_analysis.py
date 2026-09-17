"""S5 analysis tests.

The statistics are checked against hand calculations; the pipeline is checked
by recovering effects that were built into a synthetic export on purpose; and
one test goes all the way from the app's own tables through the export into
the report, so the S4 → S5 hand-off cannot drift apart unnoticed.

    python -m unittest tests.test_analysis -v
"""

from __future__ import annotations

import tests  # noqa: F401  — sets DATABASE_URL before any app import

import io
import math
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from analysis import __main__ as cli
from analysis import load as loader
from analysis import report, stats
from analysis.synthetic import SHIFTS, write_synthetic_export
from app import accounts
from app.database import db
from app.session import User
from app.study import export, flow, instruments, store

tests.assert_scratch_database()


class Statistics(unittest.TestCase):

    def test_alpha_of_identical_items_is_one(self):
        self.assertAlmostEqual(stats.cronbach_alpha(np.array([[1, 1], [2, 2], [3, 3]])), 1.0)

    def test_alpha_matches_the_formula_by_hand(self):
        # Item variances 1 and 1, total scores 4, 3, 5 (variance 1):
        # 2/1 × (1 − 2/1) = −2. Alpha can be negative; it is not clipped.
        self.assertAlmostEqual(stats.cronbach_alpha(np.array([[1, 3], [2, 1], [3, 2]])), -2.0)

    def test_alpha_is_undefined_rather_than_invented(self):
        self.assertTrue(math.isnan(stats.cronbach_alpha(np.array([[1], [2]]))))
        self.assertTrue(math.isnan(stats.cronbach_alpha(np.array([[1, 2]]))))
        self.assertTrue(math.isnan(stats.cronbach_alpha(np.array([[3, 3], [3, 3]]))))

    def test_alpha_uses_complete_rows_only(self):
        with_gap = np.array([[1, 1], [2, 2], [3, 3], [5, np.nan]])
        self.assertAlmostEqual(stats.cronbach_alpha(with_gap), 1.0)

    def test_rank_biserial(self):
        self.assertEqual(stats.rank_biserial([1, 2, 3]), 1.0)
        self.assertEqual(stats.rank_biserial([-1, -2, -3]), -1.0)
        # Ranks 1 (+) and 2 (−): (1 − 2) / 3. The zero is dropped.
        self.assertAlmostEqual(stats.rank_biserial([1, -2, 0]), -1 / 3)
        self.assertTrue(math.isnan(stats.rank_biserial([0, 0])))

    def test_paired_against_a_hand_calculation(self):
        # Differences 1, 1, 2, 1: mean 1.25, SD 0.5, n 4.
        result = stats.paired([1, 2, 3, 4], [2, 3, 5, 5])
        self.assertEqual(result.n, 4)
        self.assertAlmostEqual(result.mean_diff, 1.25)
        self.assertAlmostEqual(result.d_z, 2.5)
        self.assertAlmostEqual(result.t, 5.0)
        self.assertEqual(result.df, 3)
        self.assertAlmostEqual(result.p, 2 * sps.t.sf(5.0, 3))
        half = sps.t.ppf(0.975, 3) * 0.25
        self.assertAlmostEqual(result.ci_low, 1.25 - half)
        self.assertAlmostEqual(result.ci_high, 1.25 + half)
        self.assertEqual(result.r_rank_biserial, 1.0)

    def test_paired_drops_incomplete_pairs(self):
        result = stats.paired([1, 2, np.nan, 4], [2, np.nan, 5, 6])
        self.assertEqual(result.n, 2)

    def test_identical_changes_have_no_test_statistic(self):
        result = stats.paired([1, 2, 3], [2, 3, 4])
        self.assertTrue(math.isnan(result.t))
        self.assertEqual((result.ci_low, result.ci_high), (1.0, 1.0))

    def test_no_change_at_all(self):
        result = stats.paired([2, 3], [2, 3])
        self.assertEqual(result.mean_diff, 0.0)
        self.assertTrue(math.isnan(result.p_wilcoxon))

    def test_holm(self):
        # Sorted: .01×3 = .03; .03×2 = .06; .04×1 = .04 → raised to .06.
        self.assertEqual([round(p, 10) for p in stats.holm([0.01, 0.04, 0.03])],
                         [0.03, 0.06, 0.06])
        adjusted = stats.holm([0.01, math.nan])
        self.assertEqual(adjusted[0], 0.01)             # a family of one
        self.assertTrue(math.isnan(adjusted[1]))
        self.assertEqual(stats.holm([0.6, 0.7]), [1.0, 1.0])

    def test_spearman(self):
        self.assertAlmostEqual(stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]).rho, 1.0)
        self.assertTrue(math.isnan(stats.spearman([1, 2], [1, 2]).rho))
        self.assertTrue(math.isnan(stats.spearman([1, 1, 1], [1, 2, 3]).rho))

    def test_apa_formatting(self):
        self.assertEqual(stats.fmt_p(0.0004), "< .001")
        self.assertEqual(stats.fmt_p(0.0432), ".043")
        self.assertEqual(stats.fmt_p(1.0), "1.000")
        self.assertEqual(stats.fmt_p(math.nan), "—")
        self.assertEqual(stats.p_clause(0.0004), "p < .001")
        self.assertEqual(stats.p_clause(0.0432), "p = .043")
        self.assertEqual(stats.fmt_signed(-0.004), "-0.00")


class _Synthetic(unittest.TestCase):

    n = 120

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="coteach_analysis_"))
        cls.export_dir = write_synthetic_export(cls.tmp / "export", n=cls.n, seed=11)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def edited_copy(self, edit) -> Path:
        """A copy of the synthetic export with *edit(name, frame)* applied."""
        target = Path(tempfile.mkdtemp(prefix="coteach_edit_"))
        self.addCleanup(shutil.rmtree, target, ignore_errors=True)
        for name in ("participants", "responses", "submissions"):
            frame = pd.read_csv(self.export_dir / f"{name}.csv", encoding="utf-8-sig",
                                dtype={"participant_code": str})
            frame = edit(name, frame)
            frame.to_csv(target / f"{name}.csv", index=False, encoding="utf-8-sig")
        return target


class Screening(_Synthetic):

    def test_the_sample_flow_adds_up(self):
        data = loader.load(self.export_dir)
        flow_ = data.flow
        self.assertEqual(flow_["participants_in_export"], self.n)
        self.assertEqual(flow_["pre_completed"], self.n)
        self.assertLess(flow_["post_completed"], self.n)        # attrition built in
        self.assertGreater(flow_["late_baseline"], 0)          # late baselines built in
        self.assertEqual(flow_["paired_analysed"],
                         flow_["both_waves"] - flow_["late_baseline"])
        self.assertTrue(data.synthetic)
        self.assertEqual(data.notes, [])

    def test_late_baselines_can_be_kept_on_request(self):
        data = loader.load(self.export_dir, include_late_baseline=True)
        self.assertEqual(data.flow["paired_analysed"], data.flow["both_waves"])
        self.assertEqual(data.late_baseline, set())

    def test_pilot_data_is_left_out_unless_asked_for(self):
        first = "P-SYN00000"

        def mark_pilot(name, frame):
            if name == "participants":
                frame.loc[frame["participant_code"] == first, "pilot_consent"] = 1
            if name == "responses":
                frame.loc[frame["participant_code"] == "P-SYN00001", "pilot"] = 1
            return frame

        edited = self.edited_copy(mark_pilot)
        data = loader.load(edited)
        self.assertEqual(data.flow["excluded_pilot_consent"], 1)
        self.assertNotIn(first, set(data.participants["participant_code"]))
        self.assertNotIn(first, set(data.submissions["participant_code"]))
        self.assertNotIn("P-SYN00001", set(data.responses["participant_code"]))
        self.assertGreaterEqual(data.flow["excluded_pilot_responses"], 1)

        kept = loader.load(edited, include_pilot=True)
        self.assertIn(first, set(kept.participants["participant_code"]))

    def test_mixed_instrument_versions_are_refused(self):
        def mix(name, frame):
            if name == "responses":
                frame.loc[frame.index[0], "instrument_version"] = "0.0.9"
            return frame
        with self.assertRaises(loader.AnalysisError) as raised:
            loader.load(self.edited_copy(mix))
        self.assertIn("0.0.9", str(raised.exception))

    def test_a_different_instrument_version_is_refused(self):
        def other(name, frame):
            if name == "responses":
                frame["instrument_version"] = "9.9.9"
            return frame
        with self.assertRaises(loader.AnalysisError):
            loader.load(self.edited_copy(other))

    def test_missing_files_and_columns_are_reported(self):
        empty = Path(tempfile.mkdtemp(prefix="coteach_empty_"))
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        with self.assertRaises(loader.AnalysisError) as raised:
            loader.load(empty)
        self.assertIn("python -m app.study export", str(raised.exception))

        def drop(name, frame):
            return frame.drop(columns=["TR6"]) if name == "responses" else frame
        with self.assertRaises(loader.AnalysisError) as raised:
            loader.load(self.edited_copy(drop))
        self.assertIn("TR6", str(raised.exception))

    def test_an_empty_export_gives_an_empty_report_not_a_crash(self):
        empty = Path(tempfile.mkdtemp(prefix="coteach_none_"))
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        export.write(empty / "export", tables={
            "participants": (export.PARTICIPANT_COLUMNS, []),
            "responses": (export.RESPONSE_COLUMNS, []),
            "submissions": (export.SUBMISSION_BASE_COLUMNS, []),
        })
        data = loader.load(empty / "export")
        results = report.analyse(data)
        self.assertEqual(results.prepost.set_index("scale").loc["TR", "n"], 0)
        written = report.write(data, results, empty / "report")
        self.assertFalse([p for p in written if p.suffix == ".png"])
        self.assertIn("**0**", (empty / "report" / "summary.md").read_text(encoding="utf-8"))

    def test_a_disagreeing_exported_mean_is_noted(self):
        def tamper(name, frame):
            if name == "responses":
                frame.loc[frame.index[0], "TR_mean"] = 1.0
            return frame
        data = loader.load(self.edited_copy(tamper))
        self.assertTrue(any("differ" in note for note in data.notes))

    def test_reverse_keying_is_what_makes_trust_reliable(self):
        data = loader.load(self.export_dir)
        pre = data.responses[data.responses["wave"] == "pre"]
        keyed = loader.keyed_items(pre, "TR").dropna()
        raw = pre[[item.id for item in instruments.ITEMS if item.scale == "TR"]].dropna()
        self.assertGreater(stats.cronbach_alpha(keyed.to_numpy()),
                           stats.cronbach_alpha(raw.to_numpy()))


class RecoversBuiltInEffects(_Synthetic):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = loader.load(cls.export_dir)
        cls.results = report.analyse(cls.data)

    def test_trust_rises_most_and_significantly(self):
        prepost = self.results.prepost.set_index("scale")
        self.assertEqual(prepost["mean_diff"].idxmax(), "TR")
        self.assertLess(prepost.loc["TR", "p_holm"], 0.001)
        self.assertAlmostEqual(prepost.loc["TR", "mean_diff"], SHIFTS["TR"], delta=0.15)
        self.assertTrue(prepost.loc["TR", "primary"])
        for key in ("PU", "PEOU"):
            self.assertGreater(prepost.loc[key, "mean_diff"], 0)
            self.assertLess(prepost.loc[key, "p_holm"], 0.05)

    def test_holm_never_lowers_a_p_value(self):
        prepost = self.results.prepost
        self.assertTrue((prepost["p_holm"] >= prepost["p"] - 1e-12).all())

    def test_scales_are_reliable(self):
        reliability = self.results.reliability.set_index(["scale", "wave"])
        for key in ("PU", "PEOU", "TR"):
            self.assertGreater(reliability.loc[(key, "pre"), "alpha"], 0.8)
        self.assertEqual(reliability.loc[("ES", "post"), "items"], 8)
        self.assertNotIn(("ES", "pre"), reliability.index)

    def test_plans_improve(self):
        overall = self.results.quality_overall.iloc[0]
        self.assertGreater(overall["mean_diff"], 0)
        self.assertLess(overall["p"], 0.001)
        self.assertEqual(len(self.results.quality_criteria), 10)

    def test_only_trust_change_tracks_quality_change(self):
        link = self.results.link.set_index("scale")
        self.assertGreater(link.loc["TR", "rho"], 0.3)
        self.assertLess(link.loc["TR", "p_holm"], 0.05)
        for key in ("PU", "PEOU", "BI"):
            self.assertLess(abs(link.loc[key, "rho"]), 0.3)

    def test_change_scores_leave_out_late_baselines(self):
        self.assertFalse(set(self.results.change["participant_code"])
                         & self.data.late_baseline)

    def test_background_counts_cover_every_first_survey(self):
        background = self.results.background
        for item in ("BG1", "BG2"):
            self.assertEqual(background.loc[background["item"] == item, "n"].sum(),
                             self.data.flow["pre_completed"])

    def test_the_report_is_written_and_labelled_synthetic(self):
        out = self.tmp / "report"
        written = report.write(self.data, self.results, out)
        summary = (out / "summary.md").read_text(encoding="utf-8")
        self.assertIn("SYNTHETIC DATA", summary)
        self.assertIn("RQ1", summary)
        for name in ("prepost", "trust_vs_quality", "score_trajectories"):
            self.assertTrue((out / "figures" / f"{name}.png").stat().st_size > 1000)
        tables = {p.name for p in written if p.suffix == ".csv"}
        self.assertIn("prepost.csv", tables)
        self.assertNotIn("@", summary)


_ids = count()


class FromTheAppToTheReport(unittest.TestCase):
    """The real hand-off: app tables → S4 export → S5 report."""

    def test_participants_recorded_by_the_app_are_analysed(self):
        tmp = Path(tempfile.mkdtemp(prefix="coteach_handoff_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rng = np.random.default_rng(3)
        codes = set()
        for _ in range(6):
            email = f"handoff{next(_ids)}.{id(self)}@st.knust.edu.gh"
            accounts.create_account(email, "correct horse battery staple",
                                    name="Handoff", role="student_teacher")
            user = User(email, "Handoff", "student_teacher")
            codes.add(store.record_consent(email, agreed=True).participant_code)
            pre = {item.id: int(rng.integers(1, 4)) for item in instruments.items_for("pre")}
            self.assertTrue(store.record_response(user, "pre", pre)[0])
            for n in range(flow.POST_AFTER_SUBMISSIONS):
                db.record_submission(email, f"{email}-{n}.pdf", 40.0 + 10 * n + rng.normal(),
                                     "band", {"learning_outcomes": 2 + n % 2}, "2.0.0")
            post = {item.id: int(rng.integers(2, 6)) for item in instruments.items_for("post")}
            self.assertTrue(store.record_response(user, "post", post)[0])

        # The study is unapproved in tests, so everything here is pilot data.
        export.write(tmp / "export", include_pilot=True)
        unapproved = loader.load(tmp / "export")
        self.assertFalse(codes & set(unapproved.participants["participant_code"]))
        data = loader.load(tmp / "export", include_pilot=True)
        mine = data.responses[data.responses["participant_code"].isin(codes)]
        self.assertEqual(len(mine), 12)
        self.assertEqual(data.notes, [])        # export and analysis agree on means

        results = report.analyse(data)
        self.assertGreaterEqual(results.prepost.set_index("scale").loc["TR", "n"], 6)
        report.write(data, results, tmp / "report")
        summary = (tmp / "report" / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Includes pilot data", summary)
        self.assertNotIn("@", summary)


class CommandLine(unittest.TestCase):

    def test_synthetic_runs_end_to_end(self):
        tmp = Path(tempfile.mkdtemp(prefix="coteach_cli_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(["synthetic", "--out", str(tmp), "--n", "30", "--seed", "5"])
        self.assertEqual(code, 0)
        self.assertIn("SYNTHETIC", output.getvalue())
        self.assertTrue((tmp / "report" / "summary.md").is_file())
        self.assertTrue((tmp / "export" / loader.SYNTHETIC_MARKER).is_file())

    def test_a_missing_export_is_an_error_not_a_traceback(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = cli.main(["run", "--export", "no/such/folder"])
        self.assertEqual(code, 1)
        self.assertIn("error:", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
