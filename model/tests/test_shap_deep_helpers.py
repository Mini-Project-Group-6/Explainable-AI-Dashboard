"""Tests for the span-location helpers in ``explainability/shap_deep.py``.

``_locate`` and ``aggregate_by_section`` are pure functions and are where the
text channel is most likely to go quietly wrong: SHAP's Text masker returns
segments with separators attached and no offset mapping for regex maskers, so
offsets are recovered by a forward scan. If that scan drifts, the dashboard
highlights the wrong sentence and the explanation is worse than none.

Imported lazily so the module's shap/torch dependencies are not needed.

Run from the ``model/`` directory:
    python -m unittest tests.test_shap_deep_helpers -v
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from explainability.reconcile import SpanAttribution
from ingestion.extract_text import section_at, section_spans

MODEL_ROOT = Path(__file__).resolve().parent.parent


def _load_shap_deep_helpers():
    """Import shap_deep without executing its shap/torch imports.

    Those live inside functions, so a plain import works — but the module also
    imports ``explainability.reconcile`` and ``ingestion.extract_text``, both of
    which are stdlib-only. So a normal import is safe.
    """
    if str(MODEL_ROOT) not in sys.path:
        sys.path.insert(0, str(MODEL_ROOT))
    from explainability import shap_deep
    return shap_deep


shap_deep = _load_shap_deep_helpers()


PLAN = (
    "SUBJECT: Integrated Science\n"
    "OBJECTIVES:\n"
    "Learners will list three parts of a leaf.\n"
    "Learners will describe photosynthesis.\n"
    "EVALUATION:\n"
    "1. Name two parts of a leaf.\n"
)


class TestLocate(unittest.TestCase):
    def test_finds_each_segment_in_order(self):
        segments = ["OBJECTIVES:\n", "Learners will list three parts of a leaf.\n"]
        offsets = shap_deep._locate(PLAN, segments)
        for (start, end), segment in zip(offsets, segments):
            self.assertEqual(PLAN[start:end], segment.strip())

    def test_repeated_text_advances_rather_than_rematching(self):
        text = "alpha beta alpha beta"
        offsets = shap_deep._locate(text, ["alpha ", "beta ", "alpha ", "beta"])
        starts = [start for start, _ in offsets]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(set(starts)), 4)

    def test_whitespace_segments_are_unlocated(self):
        offsets = shap_deep._locate(PLAN, ["   ", "\n"])
        self.assertEqual(offsets, [(-1, -1), (-1, -1)])

    def test_missing_segment_is_unlocated_not_mislocated(self):
        offsets = shap_deep._locate(PLAN, ["this text is not in the plan"])
        self.assertEqual(offsets, [(-1, -1)])

    def test_a_missing_segment_does_not_derail_the_rest(self):
        segments = ["OBJECTIVES:", "not present at all", "EVALUATION:"]
        offsets = shap_deep._locate(PLAN, segments)
        self.assertEqual(PLAN[offsets[0][0]:offsets[0][1]], "OBJECTIVES:")
        self.assertEqual(offsets[1], (-1, -1))
        self.assertEqual(PLAN[offsets[2][0]:offsets[2][1]], "EVALUATION:")

    def test_empty_input(self):
        self.assertEqual(shap_deep._locate(PLAN, []), [])


class TestSectionResolution(unittest.TestCase):
    def test_spans_cover_the_document_in_order(self):
        spans = section_spans(PLAN)
        self.assertTrue(spans)
        for _, start, end in spans:
            self.assertLess(start, end)
        starts = [start for _, start, _ in spans]
        self.assertEqual(starts, sorted(starts))

    def test_offsets_resolve_to_the_right_section(self):
        spans = section_spans(PLAN)
        objectives_at = PLAN.index("Learners will list")
        evaluation_at = PLAN.index("1. Name two parts")
        self.assertEqual(section_at(spans, objectives_at), "objectives")
        self.assertEqual(section_at(spans, evaluation_at), "assessment")

    def test_offset_outside_every_section(self):
        self.assertEqual(section_at(section_spans(PLAN), 10_000), "")


class TestAggregateBySection(unittest.TestCase):
    def test_sums_signed_values_per_section(self):
        spans = [
            SpanAttribution(text="a", value=0.5, start=0, end=1, section="objectives"),
            SpanAttribution(text="b", value=-0.2, start=2, end=3, section="objectives"),
            SpanAttribution(text="c", value=-0.9, start=4, end=5, section="assessment"),
        ]
        totals = shap_deep.aggregate_by_section(spans)
        self.assertAlmostEqual(totals["objectives"], 0.3)
        self.assertAlmostEqual(totals["assessment"], -0.9)

    def test_ordered_by_absolute_influence(self):
        spans = [
            SpanAttribution(text="a", value=0.1, section="objectives"),
            SpanAttribution(text="b", value=-0.8, section="assessment"),
        ]
        self.assertEqual(list(shap_deep.aggregate_by_section(spans))[0], "assessment")

    def test_unlocated_spans_are_bucketed_not_dropped(self):
        spans = [SpanAttribution(text="a", value=0.4, section="")]
        self.assertEqual(shap_deep.aggregate_by_section(spans), {"unlocated": 0.4})


class TestGranularity(unittest.TestCase):
    def test_patterns_are_declared_for_every_supported_granularity(self):
        self.assertIn(shap_deep.DEFAULT_GRANULARITY, shap_deep.GRANULARITY_PATTERNS)
        self.assertEqual(set(shap_deep.GRANULARITY_PATTERNS),
                         {"sentence", "line", "token"})

    def test_patterns_compile(self):
        import re
        for pattern in shap_deep.GRANULARITY_PATTERNS.values():
            re.compile(pattern)


if __name__ == "__main__":
    unittest.main(verbosity=2)
