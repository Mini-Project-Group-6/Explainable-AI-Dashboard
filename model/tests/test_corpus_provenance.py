"""Tests for the corpus-provenance guard.

The contract caught a feature-order mismatch but had no notion of *which data*
a checkpoint was fitted to. Regenerating the synthetic corpus while reusing a
text checkpoint left the two channels trained on different plans, and
``predict`` blended them into a score with no warning — every existing check
passed, because the feature names still matched.

Run from the ``model/`` directory:
    python -m unittest tests.test_corpus_provenance -v
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import model_contract as mc

ROWS = [
    {"plan_id": "p001", **{k: "3" for k in mc.CRITERION_KEYS}},
    {"plan_id": "p002", **{k: "2" for k in mc.CRITERION_KEYS}},
]


def write_labels(directory: Path, rows) -> Path:
    path = directory / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["plan_id", *mc.CRITERION_KEYS])
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestCorpusFingerprint(unittest.TestCase):
    def test_same_corpus_same_digest(self):
        with tempfile.TemporaryDirectory() as d:
            a = write_labels(Path(d), ROWS)
            first = mc.corpus_fingerprint(a)
            self.assertEqual(first, mc.corpus_fingerprint(a))

    def test_row_order_does_not_matter(self):
        # The digest identifies a *set* of labelled plans, so a manifest written
        # in a different order must not read as a different corpus.
        with tempfile.TemporaryDirectory() as d:
            digest = mc.corpus_fingerprint(write_labels(Path(d), ROWS))
        with tempfile.TemporaryDirectory() as d:
            shuffled = write_labels(Path(d), list(reversed(ROWS)))
            self.assertEqual(mc.corpus_fingerprint(shuffled), digest)

    def test_a_changed_score_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as d:
            digest = mc.corpus_fingerprint(write_labels(Path(d), ROWS))
        altered = [dict(ROWS[0]), dict(ROWS[1])]
        altered[0]["lesson_closure"] = "4"
        with tempfile.TemporaryDirectory() as d:
            self.assertNotEqual(
                mc.corpus_fingerprint(write_labels(Path(d), altered)), digest)

    def test_a_changed_plan_set_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as d:
            digest = mc.corpus_fingerprint(write_labels(Path(d), ROWS))
        with tempfile.TemporaryDirectory() as d:
            self.assertNotEqual(
                mc.corpus_fingerprint(write_labels(Path(d), ROWS[:1])), digest)


class TestConsistencyCheck(unittest.TestCase):
    @staticmethod
    def _artifacts(structural, text, tmpdir):
        bundle = {"models": {}, "feature_names": list(mc.FEATURE_ORDER),
                  "rubric_dimensions": list(mc.CRITERION_KEYS),
                  "f17_mean": 0.0, "f17_std": 1.0}
        if structural is not None:
            bundle["corpus_fingerprint"] = structural
        text_dir = None
        if text is not None:
            text_dir = Path(tmpdir) / "bert"
            text_dir.mkdir(parents=True, exist_ok=True)
            import json
            (text_dir / "text_bundle.json").write_text(
                json.dumps({"corpus_fingerprint": text,
                            "criterion_keys": list(mc.CRITERION_KEYS)}),
                encoding="utf-8")
        return mc.Artifacts(bundle=bundle, model_path=Path("x.joblib"),
                            text_model_dir=text_dir)

    def test_matching_fingerprints_are_consistent(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self._artifacts("abc", "abc", d).channels_consistent)

    def test_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            artifacts = self._artifacts("abc", "def", d)
            self.assertFalse(artifacts.channels_consistent)
            self.assertEqual(artifacts.corpus_fingerprints, ("abc", "def"))

    def test_unknown_provenance_is_not_treated_as_a_mismatch(self):
        # Artifacts predating the fingerprint have nothing to compare; refusing
        # to run on them would be worse than the risk it guards against.
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self._artifacts(None, "abc", d).channels_consistent)
            self.assertTrue(self._artifacts("abc", None, d).channels_consistent)
            self.assertTrue(self._artifacts(None, None, d).channels_consistent)

    def test_no_text_model_is_consistent(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self._artifacts("abc", None, d).channels_consistent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
