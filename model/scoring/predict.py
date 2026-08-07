"""Single-plan inference: file in -> scores, explanations and suggestions out.

This is the entry point S3's Streamlit dashboard calls per upload. It runs both
evidence channels, reconciles them into one explanation per rubric criterion,
and derives the rule-based revision suggestions:

    file -> ingestion -> 18 features ---- XGBoost -----> SHAP TreeExplainer --.
                      \\                                                       >-- reconcile -> suggestions
                       `-> raw text ---- DistilBERT --> SHAP Partition -------'

The text channel is optional. If the DistilBERT artifact is absent the result
is tabular-only and every affected criterion says so in its caveats, rather
than presenting a structural decomposition as if it explained a criterion no
structural feature measures.

Artifacts are loaded through ``model_contract.load_artifacts()`` — never by
reaching for a model file directly.

End-to-end latency budget is <=15s on the deployment laptop (proposal E3
criterion 5). XGBoost + TreeExplainer run well under 2s; the text channel is
what consumes the budget, so it is opt-in per call and meant to be cached per
submission by S3.

Run from the ``model/`` directory:
    python -m scoring.predict path\\to\\lesson_plan.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import pandas as pd

from model_contract import (
    CRITERION_KEYS,
    FEATURE_ORDER,
    Artifacts,
    load_artifacts,
    order_features,
)
from explainability.reconcile import PlanExplanation, SpanAttribution, reconcile_plan
from explainability.suggestions import Suggestion, revision_suggestions
from ingestion.extract_text import EXPECTED_SECTIONS, LessonPlanText, extract
from ingestion.feature_engineer import FeatureEngineer
from scoring.train_xgboost import plan_from_text, predict_scores

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Everything the dashboard needs for one submission."""

    source_path: str
    rubric_scores: dict[str, int]            # criterion key -> 1-4 (blended)
    overall_score_0_100: Optional[float]
    features: dict[str, float]               # the 18 raw feature values
    missing_sections: list[str]
    explanation: PlanExplanation
    suggestions: tuple[Suggestion, ...]
    latency_seconds: float
    used_text_channel: bool = False          # text score contributed to the blend
    used_text_attributions: bool = False     # text spans were computed (slow path)
    channel_latency: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path,
            "rubric_scores": self.rubric_scores,
            "overall_score_0_100": self.overall_score_0_100,
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "missing_sections": self.missing_sections,
            "used_text_channel": self.used_text_channel,
            "used_text_attributions": self.used_text_attributions,
            "latency_seconds": self.latency_seconds,
            "channel_latency": {k: round(v, 3) for k, v in self.channel_latency.items()},
            "explanation": self.explanation.to_dict(),
            "suggestions": [s.to_dict() for s in self.suggestions],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class RubricScorer:
    """Loads artifacts once; scores many plans (dashboard-resident)."""

    def __init__(self, model_path: str | Path | None = None,
                 with_shap: bool = True, with_text: bool = True,
                 with_text_attributions: bool = False,
                 text_model_dir: str | Path | None = None,
                 artifacts: Optional[Artifacts] = None):
        """
        Args:
            with_shap: compute tabular SHAP values. Off gives scores only.
            with_text: run the DistilBERT channel's *score*. One forward pass,
                ~1.2s, and it is part of the fitted blend, so it is on by default.
            with_text_attributions: also compute *where in the prose* the text
                model looked. Off by default — it is ~93s, and measurement says
                that is irreducible (see below).

        Why the split
        -------------
        Measured on the deployment-class CPU: one DistilBERT forward pass at 512
        tokens is ~1.2s. SHAP's Partition explainer needs roughly two coalitions
        per text segment, so a 40-segment plan costs ~80 passes ~= 93s. That is
        not a tuning problem: raising ``max_evals`` from 100 to 300 changed
        nothing, larger batches were *slower* per item (1159ms -> 1358ms), and
        doubling torch threads made no difference. The cost is the model.

        Scoring is one pass and fits the <=15s budget (E3 criterion 5) easily.
        Attribution does not, and cannot be made to. It is therefore opt-in, to
        be precomputed once per submission and cached by S3 rather than blocking
        an upload.

        Withholding it costs little now: v0.3 gave every criterion a structural
        feature, so the tabular SHAP channel explains all ten on its own. Text
        attribution was introduced to explain the four criteria that had no
        features, and that gap is closed.
        """
        self.artifacts = artifacts or load_artifacts(model_path, text_model_dir)
        self.bundle = self.artifacts.bundle
        self.engineer = FeatureEngineer()

        # Cross-validated agreement per criterion, written by
        # evaluation/cross_validate.py. Empty until CV has been run, in which
        # case reconcile falls back to the coverage heuristic for both channels
        # rather than comparing a heuristic against a measurement.
        from evaluation.blend_weights import load_blend_weights
        from evaluation.cross_validate import load_structural_reliability
        self.structural_reliability = load_structural_reliability() or None
        # Fitted weights win where they exist; reliability is the fallback.
        self.blend_weights = load_blend_weights() or None

        self.explainer = None
        if with_shap:
            from explainability.shap_tree import RubricExplainer
            self.explainer = RubricExplainer(self.bundle)

        self.text_scorer = None
        self.text_explainer = None
        if with_text:
            if not self.artifacts.has_text_model:
                logger.warning(
                    "with_text=True but no text model at %s — continuing "
                    "tabular-only. Train it with scoring/train_bert_lora.py.",
                    self.artifacts.text_model_dir)
            else:
                from scoring.train_bert_lora import load_text_scorer
                self.text_scorer = load_text_scorer(self.artifacts.text_model_dir)
                if with_text_attributions:
                    from explainability.shap_deep import TextRubricExplainer
                    self.text_explainer = TextRubricExplainer(
                        scorer=self.text_scorer)

    # -- channels -----------------------------------------------------------

    def _structural(self, features: dict[str, float]) -> tuple[dict, dict, dict]:
        """(scores, shap_values, base_values) keyed by criterion."""
        order_features(features)                 # contract check before it matters
        X = pd.DataFrame([features], columns=list(FEATURE_ORDER))
        raw = predict_scores(self.bundle, X, rounded=False).iloc[0]
        scores = {key: float(raw[key]) for key in CRITERION_KEYS}

        values: dict[str, list[float]] = {}
        bases: dict[str, float] = {}
        if self.explainer is not None:
            for key, explanation in self.explainer.explain_one(features).items():
                values[key] = [float(v) for v in explanation.values]
                bases[key] = float(explanation.base_values)
        return scores, values, bases

    def _text(self, raw_text: str) -> tuple[dict, dict, int]:
        """(scores, spans, token_count) keyed by criterion.

        Spans are empty unless attributions were requested — see __init__.
        """
        scores = self.text_scorer.predict_one(raw_text)
        token_count = len(
            self.text_scorer.tokenizer(raw_text, truncation=False)["input_ids"])
        spans: dict[str, list[SpanAttribution]] = {}
        if self.text_explainer is not None:
            spans = self.text_explainer.explain_one(raw_text)
        return scores, spans, token_count

    # -- public API ---------------------------------------------------------

    def score_plan(self, plan: LessonPlanText, top_k: int = 6,
                   max_suggestions: int = 8) -> ScoringResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}

        mark = time.perf_counter()
        features = self.engineer.extract_features(plan)
        timings["features"] = time.perf_counter() - mark

        mark = time.perf_counter()
        structural_scores, shap_values, shap_bases = self._structural(features)
        timings["structural"] = time.perf_counter() - mark

        text_scores: dict[str, float] = {}
        text_spans: dict[str, list[SpanAttribution]] = {}
        token_count = None
        text_reliability = None
        if self.text_scorer is not None:
            mark = time.perf_counter()
            text_scores, text_spans, token_count = self._text(plan.raw_text)
            timings["text"] = time.perf_counter() - mark
            # Measured held-out agreement per criterion, written beside the
            # checkpoint. Without it the reconciler falls back to a flat prior.
            text_reliability = self.text_scorer.reliability or None

        explanation = reconcile_plan(
            source=plan.source_path,
            structural_scores=structural_scores,
            shap_values=shap_values or None,
            shap_base_values=shap_bases or None,
            feature_values=features,
            text_scores=text_scores or None,
            text_spans=text_spans or None,
            text_reliability=text_reliability,
            structural_reliability=self.structural_reliability,
            structural_blend_weight=self.blend_weights,
            text_token_count=token_count,
            top_k=top_k,
        )

        # A plan the segmenter could not read produces near-zero features and
        # would otherwise be reported as a very poor lesson. Say so instead.
        if not plan.format_recognised:
            explanation = replace(
                explanation,
                caveats=(plan.format_warning,) + tuple(explanation.caveats))
            logger.warning("%s: only %d/%d sections recognised",
                           plan.source_path, plan.sections_found,
                           len(EXPECTED_SECTIONS))

        suggestions = revision_suggestions(
            explanation, features,
            raw_text=plan.raw_text,
            missing_sections=plan.missing_sections,
            max_total=max_suggestions)

        scores = {item.criterion.key: item.rounded_score
                  for item in explanation.criteria
                  if item.rounded_score is not None}

        return ScoringResult(
            source_path=plan.source_path,
            rubric_scores=scores,
            overall_score_0_100=explanation.overall_score_0_100,
            features=features,
            missing_sections=plan.missing_sections,
            explanation=explanation,
            suggestions=suggestions,
            latency_seconds=round(time.perf_counter() - started, 3),
            used_text_channel=self.text_scorer is not None,
            used_text_attributions=self.text_explainer is not None,
            channel_latency=timings,
        )

    def score_file(self, path: str | Path, **kwargs) -> ScoringResult:
        """PDF/DOCX upload path (dashboard) — extract, then score."""
        return self.score_plan(extract(path), **kwargs)

    def score_text(self, raw_text: str, source: str = "<text>",
                   **kwargs) -> ScoringResult:
        """Raw-text path (synthetic plans, tests)."""
        return self.score_plan(plan_from_text(raw_text, source), **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score and explain one lesson plan")
    parser.add_argument("path", help=".pdf/.docx lesson plan, or .txt raw text")
    parser.add_argument("--model", default=None,
                        help="joblib bundle (default: contract path)")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--no-text", action="store_true",
                        help="skip the DistilBERT score (structural only)")
    parser.add_argument("--text-attributions", action="store_true",
                        help="also attribute the prose — ~93s, see RubricScorer")
    parser.add_argument("--suggestions-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    scorer = RubricScorer(args.model, with_shap=not args.no_shap,
                          with_text=not args.no_text,
                          with_text_attributions=args.text_attributions)
    path = Path(args.path)
    if path.suffix.lower() == ".txt":
        result = scorer.score_text(path.read_text(encoding="utf-8"), str(path))
    else:
        result = scorer.score_file(path)

    if args.suggestions_only:
        for suggestion in result.suggestions:
            print(f"[{suggestion.priority:6}] {suggestion.criterion_id} "
                  f"{suggestion.criterion_label} (NTS {suggestion.nts_indicator})")
            print(f"          {suggestion.message}")
            print(f"          why: {suggestion.evidence}\n")
    else:
        print(result.to_json())


if __name__ == "__main__":
    main()
