"""Single-plan inference: file in -> rubric scores + SHAP explanations out.

This is the entry point S3's Streamlit dashboard calls per upload. End-to-end
latency budget is <=15s on the deployment laptop (proposal E3 criterion 5);
XGBoost + TreeExplainer alone run well under 2s — the budget exists for the
later BERT/DeepExplainer path.

Run from the ``model/`` directory:
    python -m scoring.predict path\to\lesson_plan.pdf --model artifacts/rubric_model.joblib
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from ingestion.extract_text import LessonPlanText, extract
from ingestion.feature_engineer import FEATURE_NAMES, FeatureEngineer
from scoring.train_xgboost import plan_from_text, predict_scores


@dataclass
class ScoringResult:
    source_path: str
    rubric_scores: dict[str, int]            # dimension -> 0-4
    features: dict[str, float]               # the 18 extracted feature values
    missing_sections: list[str]              # feeds S2's improvement suggestions
    latency_seconds: float
    # dimension -> [(feature, shap_value), ...] top contributors, when requested
    top_shap_features: dict[str, list] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


class RubricScorer:
    """Loads the model bundle once; scores many plans (dashboard-resident)."""

    def __init__(self, model_path: str | Path, with_shap: bool = True):
        self.bundle = joblib.load(model_path)
        self.engineer = FeatureEngineer()
        self.explainer = None
        if with_shap:
            from explainability.shap_tree import RubricExplainer
            self.explainer = RubricExplainer(self.bundle)

    def score_plan(self, plan: LessonPlanText, top_k: int = 5) -> ScoringResult:
        start = time.perf_counter()
        features = self.engineer.extract_features(plan)
        X = pd.DataFrame([features], columns=FEATURE_NAMES)
        preds = predict_scores(self.bundle, X).iloc[0]
        scores = {dim: int(preds[dim]) for dim in self.bundle["rubric_dimensions"]}

        top = {}
        if self.explainer is not None:
            top = {dim: self.explainer.top_features(features, dim, k=top_k)
                   for dim in scores}

        return ScoringResult(
            source_path=plan.source_path,
            rubric_scores=scores,
            features=features,
            missing_sections=plan.missing_sections,
            latency_seconds=round(time.perf_counter() - start, 3),
            top_shap_features=top,
        )

    def score_file(self, path: str | Path, top_k: int = 5) -> ScoringResult:
        """PDF/DOCX upload path (dashboard) — extract, then score."""
        return self.score_plan(extract(path), top_k=top_k)

    def score_text(self, raw_text: str, source: str = "<text>", top_k: int = 5
                   ) -> ScoringResult:
        """Raw-text path (synthetic plans, tests)."""
        return self.score_plan(plan_from_text(raw_text, source), top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one lesson plan")
    parser.add_argument("path", help=".pdf/.docx lesson plan, or .txt raw text")
    parser.add_argument("--model", default="artifacts/rubric_model.joblib")
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    scorer = RubricScorer(args.model, with_shap=not args.no_shap)
    path = Path(args.path)
    if path.suffix.lower() == ".txt":
        result = scorer.score_text(path.read_text(encoding="utf-8"), str(path))
    else:
        result = scorer.score_file(path)
    print(result.to_json())


if __name__ == "__main__":
    main()
