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
    RUN_ON_FILLER,
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

    def test_only_the_improved_criteria_change_text(self):
        """Blocks for unchanged criteria must be byte-identical.

        With one shared generator they were not: blocks consume a
        score-dependent number of draws, so the first changed criterion knocked
        every later block off-stream and rewrote it. A before/after pair whose
        unchanged sections move is useless for measuring a known delta.
        """
        block_of = {
            "resources_including_ict": "RESOURCES:",
            "attention_to_all_learners": "DIFFERENTIATION:",
            "lesson_closure": "CLOSURE:",
            "learning_outcomes": "OBJECTIVES:",
            "assessment_strategies_in_plan": "EVALUATION:",
        }

        def section(text: str, header: str) -> str:
            """The named block, with the run-on filler stripped.

            _apply_language_quality is a document-wide pass: it walks the joined
            text line by line, so once any block's length changes the filler
            lands on different lines everywhere. That is by design — writing
            quality is a property of the whole plan — and it is not the
            block-stream property under test here, so it is normalised away.
            """
            if header not in text:
                return ""
            after = text.split(header, 1)[1].split("\n\n", 1)[0]
            # The filler pass rewrites a line as rstrip(".") + filler + ".", so
            # removing the filler leaves punctuation that depends on how the
            # original line happened to end. Normalise both sides identically.
            return "\n".join(
                line.replace(RUN_ON_FILLER, "").rstrip(" .")
                for line in after.splitlines())

        checked = 0
        for seed in range(30):
            pair = generate_revision_pair(seed=seed, pair_id=f"rev_{seed:04d}")
            for dimension, header in block_of.items():
                if dimension in pair.improved:
                    continue
                self.assertEqual(
                    section(pair.before.text, header),
                    section(pair.after.text, header),
                    f"seed {seed}: {dimension} score is unchanged but its "
                    f"{header} block was rewritten")
                checked += 1
        self.assertGreater(checked, 20)

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
