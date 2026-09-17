"""Tests for the XAI UI components' framework-free layer.

The Streamlit ``st_*`` functions are thin wrappers over these; what is worth
testing is the view models and the HTML they produce — particularly escaping
and span placement, since both take untrusted uploaded text.

Run from the ``model/`` directory:
    python -m unittest tests.test_s2_render -v
"""

from __future__ import annotations

import unittest

import model_contract as mc
from explainability import render
from explainability.reconcile import SpanAttribution, reconcile_criterion, reconcile_plan
from tests.test_s2_explainability import coverage_gap

COVERED = "learning_outcomes"
UNCOVERED = "lesson_introduction_rpk"


def shap_for(**named: float) -> list[float]:
    values = [0.0] * mc.N_FEATURES
    for name, value in named.items():
        values[mc.FEATURE_ORDER.index(name)] = value
    return values


class TestBands(unittest.TestCase):
    def test_bands_follow_the_rubric(self):
        from rubric_schema import RUBRIC_BANDS
        self.assertEqual(render.band_for(92.0), "Outstanding")
        self.assertEqual(render.band_for(RUBRIC_BANDS["good"][0]), "Good")
        self.assertEqual(render.band_for(55.0), "Minimum level of practice")
        self.assertEqual(render.band_for(10.0), "Inadequate")
        self.assertEqual(render.band_for(None), "Not scored")


class TestViewModels(unittest.TestCase):
    def _explained(self):
        return reconcile_criterion(
            COVERED, structural_score=3.4,
            shap_values=shap_for(smart_objective_count=0.6, objective_count=-0.2),
            shap_base_value=3.0)

    def test_criterion_view_carries_the_nts_tag(self):
        view = render.criterion_view(self._explained())
        self.assertEqual(view["id"], "C01")
        self.assertTrue(view["nts_indicator"])
        self.assertTrue(view["nts_descriptor"])
        self.assertIn("nts_verified", view)

    def test_strengths_and_weaknesses_are_split_by_direction(self):
        view = render.criterion_view(self._explained())
        self.assertTrue(all(s["direction"] == "raises" for s in view["strengths"]))
        self.assertTrue(all(w["direction"] == "lowers" for w in view["weaknesses"]))

    def test_unexplainable_criterion_is_marked(self):
        # Reachable only by synthesising a gap now that v0.3 covers every
        # criterion; the "not explained" rendering still has to be right.
        with coverage_gap(UNCOVERED):
            explanation = reconcile_criterion(
                UNCOVERED, structural_score=2.0,
                shap_values=shap_for(objective_count=0.5), shap_base_value=1.5)
            view = render.criterion_view(explanation)
        self.assertFalse(view["is_explainable"])
        self.assertEqual(view["confidence_label"], "Not explained")
        self.assertEqual(view["evidence"], [])

    def test_plan_view_counts_explained_criteria(self):
        plan = reconcile_plan(
            structural_scores={k: 3.0 for k in mc.CRITERION_KEYS},
            shap_values={k: shap_for(smart_objective_count=0.5)
                         for k in mc.CRITERION_KEYS},
            shap_base_values={k: 2.5 for k in mc.CRITERION_KEYS})
        view = render.plan_view(plan)
        self.assertEqual(view["criteria_count"], mc.N_CRITERIA)
        # v0.3: every criterion has a feature, so all ten are explainable.
        self.assertEqual(view["explained_count"], mc.N_CRITERIA)
        self.assertEqual(len(view["criteria"]), mc.N_CRITERIA)


class TestContributionBars(unittest.TestCase):
    def test_renders_a_row_per_contribution(self):
        explanation = reconcile_criterion(
            COVERED, structural_score=3.0,
            shap_values=shap_for(objective_count=0.4, smart_objective_count=-0.4),
            shap_base_value=3.0, top_k=2)
        markup = render.contribution_bars_html(explanation.evidence)
        self.assertIn(render.POSITIVE_COLOUR, markup)
        self.assertIn(render.NEGATIVE_COLOUR, markup)
        self.assertIn("raises", markup)
        self.assertIn("lowers", markup)

    def test_empty_evidence_says_so(self):
        markup = render.contribution_bars_html([])
        self.assertIn("No evidence available", markup)

    def test_quoted_text_is_escaped(self):
        explanation = reconcile_criterion(
            COVERED, structural_score=3.0, text_score=3.0,
            text_spans=[SpanAttribution(text="<script>alert(1)</script>",
                                        value=0.9)])
        markup = render.contribution_bars_html(explanation.evidence)
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)


class TestHighlighting(unittest.TestCase):
    TEXT = "OBJECTIVES:\nLearners will list three parts.\nCLOSURE:\nNone.\n"

    def test_highlights_the_span_at_its_offsets(self):
        start = self.TEXT.index("Learners will list three parts.")
        span = SpanAttribution(text="Learners will list three parts.", value=0.8,
                               start=start, end=start + 31, section="objectives")
        markup = render.highlight_spans_html(self.TEXT, [span])
        self.assertIn("<mark", markup)
        self.assertIn("Learners will list three parts.", markup)
        self.assertIn(render.POSITIVE_COLOUR, markup)

    def test_negative_spans_use_the_negative_colour(self):
        start = self.TEXT.index("None.")
        span = SpanAttribution(text="None.", value=-0.6, start=start,
                               end=start + 5, section="closure")
        markup = render.highlight_spans_html(self.TEXT, [span])
        self.assertIn(render.NEGATIVE_COLOUR, markup)

    def test_unlocated_spans_are_not_guessed_at(self):
        span = SpanAttribution(text="somewhere", value=0.9, start=-1, end=-1)
        markup = render.highlight_spans_html(self.TEXT, [span])
        self.assertNotIn("<mark", markup)

    def test_overlapping_spans_resolve_to_the_stronger_one(self):
        strong = SpanAttribution(text="Learners will", value=0.9, start=12, end=25)
        weak = SpanAttribution(text="will list", value=0.1, start=21, end=30)
        markup = render.highlight_spans_html(self.TEXT, [weak, strong])
        self.assertEqual(markup.count("<mark"), 1)

    def test_the_full_text_survives_highlighting(self):
        start = self.TEXT.index("None.")
        span = SpanAttribution(text="None.", value=-0.6, start=start, end=start + 5)
        markup = render.highlight_spans_html(self.TEXT, [span])
        stripped = (markup.replace("&lt;", "<").replace("&gt;", ">")
                    .replace("&amp;", "&"))
        for fragment in ("OBJECTIVES:", "Learners will list three parts.", "CLOSURE:"):
            self.assertIn(fragment, stripped)

    def test_uploaded_text_is_escaped(self):
        hostile = "OBJECTIVES:\n<img src=x onerror=alert(1)>\n"
        span = SpanAttribution(text="<img src=x onerror=alert(1)>", value=0.5,
                               start=12, end=40)
        markup = render.highlight_spans_html(hostile, [span])
        self.assertNotIn("<img", markup)
        self.assertIn("&lt;img", markup)

    def test_no_spans_still_renders_the_plan(self):
        markup = render.highlight_spans_html(self.TEXT, [])
        self.assertIn("OBJECTIVES:", markup)
        self.assertNotIn("<mark", markup)


class TestLegend(unittest.TestCase):
    def test_legend_explains_both_directions_in_words(self):
        markup = render.legend_html()
        self.assertIn("raises the score", markup)
        self.assertIn("lowers the score", markup)
        self.assertIn(render.POSITIVE_COLOUR, markup)
        self.assertIn(render.NEGATIVE_COLOUR, markup)


if __name__ == "__main__":
    unittest.main(verbosity=2)
