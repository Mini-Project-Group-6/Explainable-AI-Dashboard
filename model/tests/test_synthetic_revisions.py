"""Tests for the synthetic revision-pair generator.

The brief calls for synthetic lesson-plan *revisions* before real teacher data
arrives. What has to hold: a pair is recognisably the same lesson, the revision
is genuinely better on the criteria it claims to improve, and the manifest lets
S3 rebuild the revision history and S5 measure the delta.

Run from the ``model/`` directory:
    python -m unittest tests.test_synthetic_revisions -v
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from data.synthetic import (
    generate_revision_pair,
    generate_revisions,
    write_revisions,
)
from rubric_schema import RUBRIC_DIMENSIONS, overall_score, validate_scores


class TestRevisionPair(unittest.TestCase):
    def test_pair_is_the_same_lesson(self):
        pair = generate_revision_pair(seed=7, pair_id="rev_0000")
        self.assertEqual(pair.before.subject, pair.after.subject)
        self.assertEqual(pair.before.topic, pair.after.topic)

    def test_versions_are_distinctly_identified(self):
        pair = generate_revision_pair(seed=7, pair_id="rev_0000")
        self.assertEqual(pair.before.plan_id, "rev_0000_v1")
        self.assertEqual(pair.after.plan_id, "rev_0000_v2")

    def test_improved_criteria_actually_improve(self):
        for seed in range(20):
            pair = generate_revision_pair(seed=seed, pair_id=f"rev_{seed:04d}")
            self.assertTrue(pair.improved, "a revision must improve something")
            for dimension in pair.improved:
                self.assertGreater(pair.after.scores[dimension],
                                   pair.before.scores[dimension])

    def test_untouched_criteria_are_unchanged(self):
        for seed in range(20):
            pair = generate_revision_pair(seed=seed, pair_id=f"rev_{seed:04d}")
            for dimension in RUBRIC_DIMENSIONS:
                if dimension not in pair.improved:
                    self.assertEqual(pair.after.scores[dimension],
                                     pair.before.scores[dimension])

    def test_overall_score_never_goes_down(self):
        for seed in range(20):
            pair = generate_revision_pair(seed=seed, pair_id=f"rev_{seed:04d}")
            self.assertGreater(overall_score(pair.after.scores),
                               overall_score(pair.before.scores))

    def test_scores_stay_valid(self):
        for seed in range(20):
            pair = generate_revision_pair(seed=seed, pair_id=f"rev_{seed:04d}")
            validate_scores(pair.before.scores)
            validate_scores(pair.after.scores)

    def test_score_delta_is_reported(self):
        pair = generate_revision_pair(seed=3, pair_id="rev_0003")
        delta = pair.score_delta
        self.assertEqual(set(delta), set(RUBRIC_DIMENSIONS))
        self.assertTrue(all(value >= 0 for value in delta.values()))

    def test_text_actually_changes(self):
        pair = generate_revision_pair(seed=11, pair_id="rev_0011")
        self.assertNotEqual(pair.before.text, pair.after.text)

    def test_deterministic(self):
        first = generate_revision_pair(seed=5, pair_id="rev_0005")
        second = generate_revision_pair(seed=5, pair_id="rev_0005")
        self.assertEqual(first.before.text, second.before.text)
        self.assertEqual(first.after.text, second.after.text)
        self.assertEqual(first.improved, second.improved)


class TestWriteRevisions(unittest.TestCase):
    def test_writes_both_versions_and_a_pairing_manifest(self):
        pairs = generate_revisions(n=5, seed=42)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            manifest = write_revisions(pairs, out)

            plans = sorted(p.name for p in (out / "plans").glob("*.txt"))
            self.assertEqual(len(plans), 10)      # 5 pairs x 2 versions

            # Revision plans are also ordinary labelled training rows.
            self.assertTrue((out / "labels.csv").is_file())
            with open(out / "labels.csv", newline="", encoding="utf-8") as f:
                labels = list(csv.DictReader(f))
            self.assertEqual(len(labels), 10)
            for dimension in RUBRIC_DIMENSIONS:
                self.assertIn(dimension, labels[0])

            with open(manifest, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)
            for row in rows:
                self.assertTrue(row["improved_criteria"])
                self.assertGreater(float(row["overall_delta"]), 0)
                self.assertTrue((out / "plans" / f"{row['before_plan_id']}.txt").is_file())
                self.assertTrue((out / "plans" / f"{row['after_plan_id']}.txt").is_file())

    def test_generate_revisions_produces_distinct_pairs(self):
        pairs = generate_revisions(n=8, seed=1)
        self.assertEqual(len({p.pair_id for p in pairs}), 8)
        self.assertGreater(len({p.before.text for p in pairs}), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
