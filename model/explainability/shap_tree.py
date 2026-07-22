"""SHAP TreeExplainer pipeline for the XGBoost rubric scorer (proposal D3).

Produces the per-plan, per-dimension explanations that S2's dashboard
renders: exact Shapley values (<1s per plan), waterfall plots for a single
submission and beeswarm summary plots at corpus level (planned visuals E1
#2 and #5). Feature names come straight from FEATURE_NAMES so the plots
show the same tutor-readable columns as docs/FEATURES.md.

Run from the ``model/`` directory:
    python -m explainability.shap_tree --model artifacts/rubric_model.joblib \
        --labels data/synthetic_v1/labels.csv --plans data/synthetic_v1/plans \
        --out artifacts/shap
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scoring.train_xgboost import apply_f17_zscore

logger = logging.getLogger(__name__)


class RubricExplainer:
    """Wraps one shap.TreeExplainer per rubric dimension."""

    def __init__(self, bundle: dict):
        import shap

        self.bundle = bundle
        self.dimensions = bundle["rubric_dimensions"]
        self.feature_names = bundle["feature_names"]
        self.explainers = {
            dim: shap.TreeExplainer(bundle["models"][dim])
            for dim in self.dimensions
        }

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return apply_f17_zscore(X, self.bundle["f17_mean"], self.bundle["f17_std"])

    def explain(self, X: pd.DataFrame) -> dict[str, "object"]:
        """dimension -> shap.Explanation for the given feature matrix."""
        Xz = self._transform(X)
        return {dim: explainer(Xz) for dim, explainer in self.explainers.items()}

    def explain_one(self, features: dict[str, float]) -> dict[str, "object"]:
        """Explanations for a single plan's feature dict (dashboard entry point)."""
        X = pd.DataFrame([features], columns=self.feature_names)
        return {dim: exp[0] for dim, exp in self.explain(X).items()}

    def top_features(self, features: dict[str, float], dim: str, k: int = 5
                     ) -> list[tuple[str, float]]:
        """Top-k (feature, shap_value) for one dimension — G2/R3 fallback view."""
        exp = self.explain_one(features)[dim]
        order = np.argsort(-np.abs(exp.values))[:k]
        return [(self.feature_names[i], float(exp.values[i])) for i in order]


def save_waterfall(explanation, out_path: str | Path, title: str = "") -> Path:
    """Waterfall plot for one plan x one dimension (E1 visual #2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.plots.waterfall(explanation, max_display=12, show=False)
    fig = plt.gcf()
    if title:
        fig.suptitle(title)
    fig.set_size_inches(9, 6)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_beeswarm(explanation, out_path: str | Path, title: str = "") -> Path:
    """Corpus-level beeswarm summary for one dimension (E1 visual #5)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.plots.beeswarm(explanation, max_display=18, show=False)
    fig = plt.gcf()
    if title:
        fig.suptitle(title)
    fig.set_size_inches(9, 7)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    from scoring.train_xgboost import load_dataset

    parser = argparse.ArgumentParser(description="Generate SHAP plots for the rubric scorer")
    parser.add_argument("--model", required=True, help="joblib bundle from train_xgboost")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--out", default="artifacts/shap")
    parser.add_argument("--sample-plan", default=None,
                        help="plan_id for the single-plan waterfall (default: first plan)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bundle = joblib.load(args.model)
    X, _, ids = load_dataset(args.labels, args.plans)
    explainer = RubricExplainer(bundle)

    logger.info("Computing SHAP values for %d plans x %d dimensions",
                len(X), len(explainer.dimensions))
    explanations = explainer.explain(X)

    out_dir = Path(args.out)
    sample_idx = ids.index(args.sample_plan) if args.sample_plan else 0
    for dim, exp in explanations.items():
        save_beeswarm(exp, out_dir / f"beeswarm_{dim}.png",
                      title=f"Corpus SHAP summary — {dim}")
        save_waterfall(exp[sample_idx], out_dir / f"waterfall_{ids[sample_idx]}_{dim}.png",
                       title=f"{ids[sample_idx]} — {dim}")
    logger.info("Saved plots -> %s", out_dir)


if __name__ == "__main__":
    main()
