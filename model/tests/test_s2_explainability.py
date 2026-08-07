"""Tests for the S2 layer: contract, labels, reconciliation, suggestions.

Deliberately stdlib-only — no xgboost, shap, torch or spaCy. The whole point of
``model_contract.py`` is that the rules it enforces are checkable without the
model stack loaded, so these run on a bare checkout and belong in CI.

Run from the ``model/`` directory:
    python -m unittest tests.test_s2_explainability -v
"""

from __future__ import annotations

import contextlib
import dataclasses
import unittest

import model_contract as mc
from explainability import feature_labels as fl
from explainability.reconcile import (
    STRUCTURAL_EVIDENCE_FLOOR,
    Channel,
    SpanAttribution,
    channel_weights,
    reconcile_criterion,
    reconcile_plan,
)
from explainability.suggestions import (
    FEATURE_RULES,
    presence_suggestions,
    revision_suggestions,
)

COVERED = "learning_outcomes"          # C01, three primary features
UNCOVERED = "lesson_introduction_rpk"  # C06 — only inside coverage_gap(), see below


@contextlib.contextmanager
def coverage_gap(criterion_key: str):
    """Temporarily strip every feature that measures *criterion_key*.

    v0.3 gave all ten criteria a feature, so no criterion is uncovered any
    more. The honest-degradation path — withhold the structural channel, mark
    the criterion unexplained — is therefore unreachable in normal operation,
    which is exactly why it needs testing directly: it is the behaviour that
    protects the dashboard if a criterion is ever added without a feature.
    """
    reassign_to = next(k for k in mc.CRITERION_KEYS if k != criterion_key)
    originals = {
        name: label for name, label in fl.FEATURE_LABELS.items()
        if label.primary_criterion == criterion_key
        or criterion_key in label.secondary_criteria
    }
    for name, label in originals.items():
        fl.FEATURE_LABELS[name] = dataclasses.replace(
            label,
            primary_criterion=(reassign_to
                               if label.primary_criterion == criterion_key
                               else label.primary_criterion),
            secondary_criteria=tuple(c for c in label.secondary_criteria
                                     if c != criterion_key))
    try:
        yield
    finally:
        fl.FEATURE_LABELS.update(originals)


def zeros() -> list[float]:
    return [0.0] * mc.N_FEATURES


def shap_for(**named: float) -> list[float]:
    """SHAP vector in contract order, from feature-name keyword args."""
    values = zeros()
    for name, value in named.items():
        values[mc.FEATURE_ORDER.index(name)] = value
    return values


def healthy_features() -> dict[str, float]:
    """A feature dict where every value sits inside its healthy range."""
    features = {}
    for name in mc.FEATURE_ORDER:
        low, high = fl.FEATURE_LABELS[name].healthy_range
        if low is not None and high is not None:
            features[name] = (low + high) / 2
        elif low is not None:
            features[name] = low + abs(low) + 1.0
        elif high is not None:
            features[name] = high / 2
        else:
            features[name] = 0.0
    return features


class TestContract(unittest.TestCase):
    def test_feature_order_matches_the_extractor(self):
        from ingestion.feature_engineer import FEATURE_NAMES
        self.assertEqual(tuple(FEATURE_NAMES), mc.FEATURE_ORDER)

    def test_criteria_match_the_rubric_schema(self):
        from rubric_schema import RUBRIC_DIMENSIONS
        self.assertEqual(mc.CRITERION_KEYS, tuple(RUBRIC_DIMENSIONS))

    def test_shape(self):
        self.assertEqual(mc.N_FEATURES, 22)
        self.assertEqual(mc.N_CRITERIA, 10)

    def test_lookup_by_key_and_id(self):
        self.assertIs(mc.criterion("lesson_closure"), mc.criterion("C10"))
        with self.assertRaises(KeyError):
            mc.criterion("does_not_exist")

    def test_weights_come_from_the_rubric_schema(self):
        from rubric_schema import RUBRIC_WEIGHTS
        for item in mc.CRITERIA:
            self.assertAlmostEqual(item.weight, RUBRIC_WEIGHTS[item.key])

    def test_order_features_rejects_drift(self):
        good = {name: 1.0 for name in mc.FEATURE_ORDER}
        self.assertEqual(len(mc.order_features(good)), mc.N_FEATURES)

        missing = dict(good)
        missing.pop("readability_flesch")
        with self.assertRaises(mc.ArtifactContractError):
            mc.order_features(missing)

        extra = dict(good, bonus_feature=1.0)
        with self.assertRaises(mc.ArtifactContractError):
            mc.order_features(extra)

    def test_validate_bundle_accepts_a_conforming_bundle(self):
        mc.validate_bundle(self._bundle(), source="test")

    def test_validate_bundle_rejects_reordered_features(self):
        bundle = self._bundle()
        names = list(mc.FEATURE_ORDER)
        names[0], names[1] = names[1], names[0]
        bundle["feature_names"] = names
        with self.assertRaises(mc.ArtifactContractError) as caught:
            mc.validate_bundle(bundle, source="test")
        self.assertIn("REORDERED", str(caught.exception))

    def test_validate_bundle_rejects_renamed_feature(self):
        bundle = self._bundle()
        bundle["feature_names"] = list(mc.FEATURE_ORDER[:-1]) + ["flesch_readability"]
        with self.assertRaises(mc.ArtifactContractError):
            mc.validate_bundle(bundle, source="test")

    def test_validate_bundle_rejects_missing_model(self):
        bundle = self._bundle()
        del bundle["models"]["lesson_closure"]
        with self.assertRaises(mc.ArtifactContractError):
            mc.validate_bundle(bundle, source="test")

    def test_validate_bundle_rejects_nan_zscore_stats(self):
        bundle = self._bundle()
        bundle["f17_std"] = float("nan")
        with self.assertRaises(mc.ArtifactContractError):
            mc.validate_bundle(bundle, source="test")

    def test_load_artifacts_reports_how_to_train(self):
        with self.assertRaises(FileNotFoundError) as caught:
            mc.load_artifacts(model_path="definitely/not/here.joblib")
        self.assertIn("train_xgboost", str(caught.exception))

    @staticmethod
    def _bundle() -> dict:
        return {
            "models": {key: object() for key in mc.CRITERION_KEYS},
            "feature_names": list(mc.FEATURE_ORDER),
            "rubric_dimensions": list(mc.CRITERION_KEYS),
            "f17_mean": 40.0,
            "f17_std": 12.0,
        }


class TestFeatureLabels(unittest.TestCase):
    def test_every_feature_has_a_label(self):
        self.assertEqual(tuple(fl.FEATURE_LABELS), mc.FEATURE_ORDER)
        self.assertEqual(len(fl.axis_labels()), mc.N_FEATURES)

    def test_labels_are_not_just_the_column_names(self):
        for name, label in fl.FEATURE_LABELS.items():
            self.assertNotEqual(label.label, name)
            self.assertTrue(label.meaning.endswith("."), name)

    def test_every_label_points_at_a_real_criterion(self):
        for label in fl.FEATURE_LABELS.values():
            self.assertIn(label.primary_criterion, mc.CRITERION_KEYS)
            for key in label.secondary_criteria:
                self.assertIn(key, mc.CRITERION_KEYS)

    def test_every_criterion_now_has_a_feature(self):
        # v0.3 closed the coverage gap with F19-F22. The check stays because a
        # criterion added later without a feature must fail here rather than
        # quietly get an unexplainable SHAP panel.
        self.assertEqual(fl.uncovered_criteria(), ())
        report = fl.coverage_report()
        for key in mc.CRITERION_KEYS:
            self.assertTrue(report[key]["primary"],
                            f"{key} has no feature measuring it")

    def test_the_gap_machinery_still_works_if_a_gap_reappears(self):
        # Guard the degradation path itself: it is only exercised in production
        # when something regresses, so it must be tested directly.
        original = fl.FEATURE_LABELS["closure_quality_score"]
        patched = dataclasses.replace(
            original, primary_criterion="lesson_sequencing")
        fl.FEATURE_LABELS["closure_quality_score"] = patched
        try:
            self.assertIn("lesson_closure", fl.uncovered_criteria())
            self.assertLess(fl.tabular_confidence("lesson_closure"),
                            STRUCTURAL_EVIDENCE_FLOOR)
        finally:
            fl.FEATURE_LABELS["closure_quality_score"] = original
        self.assertEqual(fl.uncovered_criteria(), ())

    def test_tabular_confidence_separates_covered_from_uncovered(self):
        for key in mc.CRITERION_KEYS:
            confidence = fl.tabular_confidence(key)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            if key in fl.uncovered_criteria():
                self.assertLess(confidence, STRUCTURAL_EVIDENCE_FLOOR)
            else:
                self.assertGreaterEqual(confidence, STRUCTURAL_EVIDENCE_FLOOR)

    def test_is_weak(self):
        label = fl.FEATURE_LABELS["time_allocation_coverage"]   # (0.8, None)
        self.assertTrue(label.is_weak(0.2))
        self.assertFalse(label.is_weak(1.0))

        band = fl.FEATURE_LABELS["objective_count"]             # (2, 4)
        self.assertTrue(band.is_weak(1))
        self.assertFalse(band.is_weak(3))
        self.assertTrue(band.is_weak(9))


class TestChannelWeights(unittest.TestCase):
    def test_structural_only(self):
        w_s, w_t = channel_weights(COVERED, structural_available=True,
                                   text_available=False)
        self.assertAlmostEqual(w_s, 1.0)
        self.assertAlmostEqual(w_t, 0.0)

    def test_both_channels_sum_to_one(self):
        w_s, w_t = channel_weights(COVERED, structural_available=True,
                                   text_available=True)
        self.assertAlmostEqual(w_s + w_t, 1.0)
        self.assertGreater(w_s, w_t)   # structural is strong for this criterion

    def test_uncovered_criterion_hands_everything_to_text(self):
        w_s, w_t = channel_weights(UNCOVERED, structural_available=False,
                                   text_available=True)
        self.assertAlmostEqual(w_s, 0.0)
        self.assertAlmostEqual(w_t, 1.0)

    def test_no_channel(self):
        self.assertEqual(
            channel_weights(COVERED, structural_available=False,
                            text_available=False),
            (0.0, 0.0))

    def test_measured_reliability_favours_the_more_accurate_channel(self):
        w_s, w_t = channel_weights(COVERED, structural_available=True,
                                   text_available=True,
                                   structural_reliability=0.89,
                                   text_reliability=0.50)
        self.assertAlmostEqual(w_s + w_t, 1.0)
        self.assertGreater(w_s, w_t)
        self.assertAlmostEqual(w_s, 0.89 / 1.39)

    def test_a_weaker_text_model_gets_less_weight(self):
        # The regression this fix exists for: the text model measurably got
        # worse on every criterion, yet its weight rose, because a measured QWK
        # was being compared against a coverage heuristic.
        strong = channel_weights(COVERED, structural_available=True,
                                 text_available=True,
                                 structural_reliability=0.80,
                                 text_reliability=0.75)[1]
        weak = channel_weights(COVERED, structural_available=True,
                               text_available=True,
                               structural_reliability=0.80,
                               text_reliability=0.50)[1]
        self.assertLess(weak, strong)

    def test_fitted_blend_weight_overrides_every_heuristic(self):
        w_s, w_t = channel_weights(COVERED, structural_available=True,
                                   text_available=True,
                                   structural_blend_weight=0.8,
                                   structural_reliability=0.1,
                                   text_reliability=0.9)
        self.assertAlmostEqual(w_s, 0.8)
        self.assertAlmostEqual(w_t, 0.2)

    def test_a_fitted_weight_of_one_silences_the_text_channel(self):
        # blend_weights fits w=1.00 for three criteria; the text channel must
        # then contribute nothing to the score.
        result = reconcile_criterion(
            COVERED, structural_score=4.0,
            shap_values=shap_for(smart_objective_count=1.0),
            shap_base_value=3.0, text_score=1.0,
            text_spans=[SpanAttribution(text="x", value=0.5)],
            structural_blend_weight=1.0)
        self.assertAlmostEqual(result.score, 4.0)
        self.assertAlmostEqual(result.text_weight, 0.0)

    def test_fitted_weight_is_clamped(self):
        self.assertEqual(
            channel_weights(COVERED, structural_available=True,
                            text_available=True, structural_blend_weight=1.7),
            (1.0, 0.0))

    def test_never_compares_a_measurement_against_a_heuristic(self):
        # Only one side measured -> both fall back to heuristics, so the
        # comparison stays like-for-like.
        half = channel_weights(COVERED, structural_available=True,
                               text_available=True, text_reliability=0.50)
        none = channel_weights(COVERED, structural_available=True,
                               text_available=True)
        self.assertEqual(half, none)


class TestReconcile(unittest.TestCase):
    def test_structural_only_passes_the_score_through(self):
        result = reconcile_criterion(
            COVERED, structural_score=3.2,
            shap_values=shap_for(smart_objective_count=0.4,
                                 objective_count=-0.1),
            shap_base_value=2.9, feature_values=healthy_features())
        self.assertAlmostEqual(result.score, 3.2)
        self.assertEqual(result.rounded_score, 3)
        self.assertTrue(result.is_explainable)
        self.assertTrue(all(c.channel is Channel.STRUCTURAL for c in result.evidence))

    def test_blends_both_channels(self):
        spans = [SpanAttribution(text="Learners will list three uses.", value=0.5)]
        result = reconcile_criterion(
            COVERED, structural_score=4.0,
            shap_values=shap_for(smart_objective_count=1.0),
            shap_base_value=3.0, text_score=2.0, text_spans=spans)
        w_s, w_t = channel_weights(COVERED, structural_available=True,
                                   text_available=True)
        self.assertAlmostEqual(result.score, w_s * 4.0 + w_t * 2.0)
        self.assertEqual({c.channel for c in result.evidence},
                         {Channel.STRUCTURAL, Channel.TEXT})

    def test_both_channels_are_represented_even_when_one_dominates(self):
        # A single huge structural contribution would otherwise fill top_k.
        spans = [SpanAttribution(text="Review previous knowledge.", value=0.01)]
        result = reconcile_criterion(
            COVERED, structural_score=3.0,
            shap_values=shap_for(**{name: 1.0 for name in mc.FEATURE_ORDER[:6]}),
            shap_base_value=-3.0, text_score=3.0, text_spans=spans, top_k=3)
        self.assertIn(Channel.TEXT, {c.channel for c in result.evidence})

    def test_uncovered_criterion_without_text_is_not_explainable(self):
        with coverage_gap(UNCOVERED):
            result = reconcile_criterion(
                UNCOVERED, structural_score=2.4,
                shap_values=shap_for(objective_count=0.9),
                shap_base_value=1.5, feature_values=healthy_features())
        self.assertEqual(result.evidence, ())
        self.assertFalse(result.is_explainable)
        self.assertAlmostEqual(result.score, 2.4)   # score still passed through
        self.assertTrue(any("no structural feature measures" in c.lower()
                            for c in result.caveats))

    def test_uncovered_criterion_with_text_is_explained_by_text(self):
        spans = [SpanAttribution(text="Teacher reviews prior knowledge.",
                                 value=0.7, section="introduction")]
        with coverage_gap(UNCOVERED):
            result = reconcile_criterion(
                UNCOVERED, structural_score=2.4,
                shap_values=shap_for(objective_count=0.9), shap_base_value=1.5,
                text_score=3.6, text_spans=spans)
        self.assertAlmostEqual(result.score, 3.6)   # structural does not vote
        self.assertTrue(result.is_explainable)
        self.assertTrue(all(c.channel is Channel.TEXT for c in result.evidence))

    def test_covered_criterion_uses_the_structural_channel(self):
        # The v0.3 counterpart: with a feature measuring it, the same criterion
        # is now explained from structure rather than withheld.
        result = reconcile_criterion(
            UNCOVERED, structural_score=2.4,
            shap_values=shap_for(rpk_link_score=-0.6),
            shap_base_value=3.0, feature_values=healthy_features())
        self.assertTrue(result.is_explainable)
        self.assertTrue(any(c.feature == "rpk_link_score" for c in result.evidence))

    def test_influence_shares_sum_to_one_within_a_channel(self):
        result = reconcile_criterion(
            COVERED, structural_score=3.0,
            shap_values=shap_for(objective_count=1.0, smart_objective_count=-3.0),
            shap_base_value=5.0, top_k=mc.N_FEATURES)
        total = sum(c.influence for c in result.evidence
                    if c.channel is Channel.STRUCTURAL)
        self.assertAlmostEqual(total, 1.0)

    def test_additivity_violation_is_caveated(self):
        result = reconcile_criterion(
            COVERED, structural_score=3.0,
            shap_values=shap_for(objective_count=0.1),
            shap_base_value=0.0)          # 0.0 + 0.1 != 3.0
        self.assertTrue(any("additivity" in c.lower() for c in result.caveats))

    def test_wrong_shap_length_is_rejected(self):
        with self.assertRaises(ValueError):
            reconcile_criterion(COVERED, structural_score=3.0,
                                shap_values=[0.1, 0.2])

    def test_no_channel_yields_no_score(self):
        result = reconcile_criterion(COVERED)
        self.assertIsNone(result.score)
        self.assertFalse(result.is_explainable)

    def test_truncation_is_caveated(self):
        result = reconcile_criterion(
            COVERED, text_score=3.0,
            text_spans=[SpanAttribution(text="x", value=1.0)],
            text_token_count=mc.TEXT_MAX_TOKENS + 40)
        self.assertTrue(any(str(mc.TEXT_MAX_TOKENS) in c for c in result.caveats))

    def test_score_is_clipped_to_the_rubric_range(self):
        result = reconcile_criterion(
            COVERED, structural_score=9.9,
            shap_values=shap_for(objective_count=1.0), shap_base_value=8.9)
        self.assertLessEqual(result.score, mc.SCORE_MAX)


class TestReconcilePlan(unittest.TestCase):
    def setUp(self):
        self.scores = {key: 3.0 for key in mc.CRITERION_KEYS}
        self.shap = {key: shap_for(smart_objective_count=0.5)
                     for key in mc.CRITERION_KEYS}
        self.bases = {key: 2.5 for key in mc.CRITERION_KEYS}

    def test_all_ten_criteria_present_and_ordered(self):
        plan = reconcile_plan(structural_scores=self.scores,
                              shap_values=self.shap, shap_base_values=self.bases)
        self.assertEqual(tuple(c.criterion.key for c in plan.criteria),
                         mc.CRITERION_KEYS)

    def test_overall_score_uses_the_rubric_weights(self):
        from rubric_schema import overall_score
        plan = reconcile_plan(structural_scores=self.scores,
                              shap_values=self.shap, shap_base_values=self.bases)
        expected = overall_score({key: 3 for key in mc.CRITERION_KEYS})
        self.assertAlmostEqual(plan.overall_score_0_100, round(expected, 2))

    def test_all_criteria_are_explainable_now(self):
        plan = reconcile_plan(structural_scores=self.scores,
                              shap_values=self.shap, shap_base_values=self.bases)
        self.assertEqual(plan.caveats, ())
        self.assertTrue(all(item.is_explainable for item in plan.criteria))

    def test_a_reintroduced_gap_is_reported_at_plan_level(self):
        with coverage_gap("lesson_closure"):
            plan = reconcile_plan(structural_scores=self.scores,
                                  shap_values=self.shap,
                                  shap_base_values=self.bases)
        self.assertTrue(plan.caveats)
        self.assertIn("C10", plan.caveats[0])

    def test_serialises(self):
        plan = reconcile_plan(structural_scores=self.scores,
                              shap_values=self.shap, shap_base_values=self.bases)
        payload = plan.to_dict()
        self.assertEqual(len(payload["criteria"]), mc.N_CRITERIA)
        self.assertIn("nts_indicator", payload["criteria"][0])


class TestSuggestions(unittest.TestCase):
    def _plan_explanation(self, features, shap_values):
        return reconcile_plan(
            structural_scores={key: 2.0 for key in mc.CRITERION_KEYS},
            shap_values={key: shap_values for key in mc.CRITERION_KEYS},
            shap_base_values={key: 2.0 - sum(shap_values)
                              for key in mc.CRITERION_KEYS},
            feature_values=features)

    def test_fires_when_the_value_is_weak_and_the_contribution_negative(self):
        features = healthy_features()
        features["smart_objective_count"] = 0        # healthy_range is (2, None)
        explanation = self._plan_explanation(
            features, shap_for(smart_objective_count=-0.8))
        found = revision_suggestions(explanation, features,
                                     raw_text=_rich_plan_text())
        self.assertTrue(any(s.feature == "smart_objective_count" for s in found))

    def test_does_not_fire_when_the_plan_already_does_it_well(self):
        # The guard that matters: a negative Shapley value on a feature whose
        # measured value is fine must NOT produce advice to fix it.
        features = healthy_features()
        features["smart_objective_count"] = 6        # comfortably healthy
        explanation = self._plan_explanation(
            features, shap_for(smart_objective_count=-0.9))
        found = revision_suggestions(explanation, features,
                                     raw_text=_rich_plan_text())
        self.assertFalse(any(s.feature == "smart_objective_count" for s in found))

    def test_does_not_fire_on_a_positive_contribution(self):
        features = healthy_features()
        features["assessment_item_count"] = 0
        explanation = self._plan_explanation(
            features, shap_for(assessment_item_count=+0.9))
        found = revision_suggestions(explanation, features,
                                     raw_text=_rich_plan_text())
        self.assertFalse(any(s.feature == "assessment_item_count" for s in found))

    def test_band_features_get_the_right_side_of_the_advice(self):
        features = healthy_features()
        features["objective_count"] = 11             # healthy_range is (2, 4)
        explanation = self._plan_explanation(
            features, shap_for(objective_count=-0.7))
        found = revision_suggestions(explanation, features,
                                     raw_text=_rich_plan_text())
        matched = [s for s in found if s.feature == "objective_count"]
        self.assertTrue(matched)
        self.assertTrue(matched[0].id.endswith(".high"))
        self.assertIn("11", matched[0].message)

    def test_missing_sections_produce_suggestions(self):
        found = presence_suggestions(_rich_plan_text(),
                                     missing_sections=["assessment"])
        self.assertTrue(any(s.source == "missing_section"
                            and s.criterion_key == "assessment_strategies_in_plan"
                            for s in found))

    def test_absent_content_covers_the_criteria_features_cannot(self):
        found = presence_suggestions("OBJECTIVES:\nLearners will list three things.")
        keys = {s.criterion_key for s in found}
        # All four uncovered criteria are reachable through presence rules.
        self.assertTrue({"resources_including_ict", "attention_to_all_learners",
                         "lesson_introduction_rpk", "lesson_closure"} <= keys)

    def test_presence_rule_stays_quiet_when_the_plan_covers_it(self):
        found = presence_suggestions(_rich_plan_text())
        self.assertFalse(any(s.criterion_key == "resources_including_ict"
                             for s in found))

    def test_caps_are_respected(self):
        features = {name: 0.0 for name in mc.FEATURE_ORDER}   # everything weak
        explanation = self._plan_explanation(
            features, shap_for(**{name: -0.2 for name in mc.FEATURE_ORDER}))
        found = revision_suggestions(explanation, features, raw_text="",
                                     max_per_criterion=2, max_total=8)
        self.assertLessEqual(len(found), 8)
        for key in mc.CRITERION_KEYS:
            self.assertLessEqual(
                sum(1 for s in found if s.criterion_key == key), 2)

    def test_suggestions_are_traceable_and_serialise(self):
        features = healthy_features()
        features["time_allocation_coverage"] = 0.0
        explanation = self._plan_explanation(
            features, shap_for(time_allocation_coverage=-0.6))
        found = revision_suggestions(explanation, features,
                                     raw_text=_rich_plan_text())
        for suggestion in found:
            payload = suggestion.to_dict()
            self.assertIn(payload["criterion"], mc.CRITERION_KEYS)
            self.assertIn(payload["priority"], {"high", "medium", "low"})
            self.assertTrue(payload["message"])
            self.assertTrue(payload["evidence"])

    def test_every_shap_driven_item_carries_its_feature_importance(self):
        # Objective 22: a feature-importance explanation on every AI-generated
        # feedback item.
        features = healthy_features()
        features["time_allocation_coverage"] = 0.0
        explanation = self._plan_explanation(
            features, shap_for(time_allocation_coverage=-0.6))
        found = revision_suggestions(explanation, features,
                                     raw_text=_rich_plan_text())
        shap_driven = [s for s in found if s.source == "shap"]
        self.assertTrue(shap_driven)
        for suggestion in shap_driven:
            self.assertTrue(suggestion.has_shap_explanation)
            self.assertLess(suggestion.shap_value, 0)      # it lowered the score
            self.assertGreater(suggestion.influence, 0)
            self.assertEqual(suggestion.feature_label,
                             fl.FEATURE_LABELS[suggestion.feature].label)
            self.assertIn("contributed", suggestion.attribution_text)

    def test_presence_rules_do_not_claim_a_shap_basis(self):
        found = presence_suggestions("OBJECTIVES:\nLearners will list things.")
        self.assertTrue(found)
        for suggestion in found:
            self.assertFalse(suggestion.has_shap_explanation)
            self.assertIsNone(suggestion.shap_value)
            self.assertIn("not by a feature contribution",
                          suggestion.attribution_text)

    def test_the_same_fix_is_never_repeated_under_several_criteria(self):
        # One weak feature drags several criteria down at once; the student
        # teacher must still see the instruction once.
        features = healthy_features()
        features["time_allocation_coverage"] = 0.0
        explanation = self._plan_explanation(
            features, shap_for(time_allocation_coverage=-0.9))
        found = revision_suggestions(explanation, features,
                                     raw_text=_rich_plan_text())
        timing = [s for s in found if s.feature == "time_allocation_coverage"]
        self.assertEqual(len(timing), 1)
        self.assertEqual(len({s.message for s in found}), len(found))

    def test_a_fix_is_filed_under_the_criterion_it_hurt_most(self):
        features = healthy_features()
        features["assessment_item_count"] = 0
        # Same feature, much larger influence on C05 than on C03.
        explanation = reconcile_plan(
            structural_scores={key: 2.0 for key in mc.CRITERION_KEYS},
            shap_values={
                key: shap_for(assessment_item_count=-0.9 if key ==
                              "assessment_strategies_in_plan" else -0.1)
                for key in mc.CRITERION_KEYS},
            shap_base_values={key: 2.0 for key in mc.CRITERION_KEYS},
            feature_values=features)
        found = revision_suggestions(explanation, features,
                                     raw_text=_rich_plan_text())
        matched = [s for s in found if s.feature == "assessment_item_count"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].criterion_key, "assessment_strategies_in_plan")

    def test_every_rule_targets_a_real_feature(self):
        for name in FEATURE_RULES:
            self.assertIn(name, mc.FEATURE_ORDER)

    def test_band_features_have_advice_for_both_sides(self):
        for name, label in fl.FEATURE_LABELS.items():
            rule = FEATURE_RULES.get(name)
            self.assertIsNotNone(rule, f"{name} has no suggestion rule")
            low, high = label.healthy_range
            if low is not None:
                self.assertIsNotNone(rule.when_low, f"{name} can be too low")
            if high is not None and label.direction is not fl.Direction.BAND:
                self.assertIsNotNone(rule.when_high, f"{name} can be too high")


def _rich_plan_text() -> str:
    """A plan mentioning everything the presence rules look for."""
    return (
        "SUBJECT: Integrated Science\nDURATION: 60 minutes\n"
        "OBJECTIVES:\nBy the end of the lesson the learner will be able to list "
        "at least three parts of a leaf, correctly.\n"
        "INTRODUCTION:\nTeacher reviews relevant previous knowledge on plants.\n"
        "RESOURCES:\nFlashcards, a chart of the leaf, and real leaf samples.\n"
        "DIFFERENTIATION:\nA support task is given to learners who struggle and an "
        "extension task to those who finish early; learners with special "
        "educational needs receive targeted help.\n"
        "EVALUATION:\n1. List three parts of a leaf.\n"
        "CLOSURE:\nLearners summarise the key points and the teacher checks the "
        "objective was met.\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
