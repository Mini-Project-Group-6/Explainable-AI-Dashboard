"""SHAP TreeExplainer pipeline for the XGBoost rubric scorer (proposal D3).

Produces the per-plan, per-criterion explanations the dashboard renders — exact
Shapley values, under a second per plan. All four visual forms the project brief
names under Method & Tools are here:

    waterfall   one plan x one criterion, the per-submission view
    force       the same decomposition as a single additive bar, which reads
                better inline and is the one the brief calls out by name
    beeswarm    corpus-level summary: distribution of each feature's effect
    bar         corpus-level mean |SHAP| ranking

Axis labels come from ``explainability/feature_labels.py``, not the raw column
names, so every plot is readable by a teacher educator rather than by whoever
wrote the feature extractor.

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

# N_FEATURES rather than a literal in the plot calls below: with a hard-coded
# 18 against a 22-feature contract, shap collapses the remainder into a "Sum of
# N other features" row — hiding exactly the features v0.3 added to close the
# coverage gap, on the corpus plots a tutor uses to judge whether they measure
# anything at all.
from model_contract import N_FEATURES, criterion, validate_bundle
from explainability.feature_labels import axis_labels
from scoring.train_xgboost import apply_f17_zscore

logger = logging.getLogger(__name__)


class RubricExplainer:
    """Wraps one shap.TreeExplainer per rubric dimension."""

    def __init__(self, bundle: dict):
        import shap

        validate_bundle(bundle, source="RubricExplainer")
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


def _with_readable_labels(explanation, readable: bool = True):
    """A view of *explanation* with tutor-readable column names.

    A waterfall axis reading ``objective_measurability_ratio`` is not an
    explanation for a teacher educator; ``Share of objectives that are
    measurable`` is. Values are untouched — only the display names change.

    Builds a new Explanation rather than copying one. ``copy.copy`` looks right
    but is not: an Explanation keeps values, data and feature_names inside a
    shared Slicer, so a shallow copy hands back an object whose assignment
    writes through to the caller's explanation. Anything reading
    ``.feature_names`` afterwards to key SHAP columns by contract name would get
    display labels instead.
    """
    if not readable:
        return explanation

    import shap

    return shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=explanation.data,
        feature_names=axis_labels(),
    )


def save_waterfall(explanation, out_path: str | Path, title: str = "",
                   readable: bool = True) -> Path:
    """Waterfall plot for one plan x one dimension (E1 visual #2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.plots.waterfall(_with_readable_labels(explanation, readable),
                         max_display=12, show=False)
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


def save_beeswarm(explanation, out_path: str | Path, title: str = "",
                  readable: bool = True) -> Path:
    """Corpus-level beeswarm summary for one dimension (E1 visual #5)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.plots.beeswarm(_with_readable_labels(explanation, readable),
                        max_display=N_FEATURES, show=False)
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


def save_force_plot(explanation, out_path: str | Path, title: str = "",
                    readable: bool = True) -> Path:
    """Force plot for one plan x one criterion (brief: Method & Tools).

    Same Shapley decomposition as the waterfall, drawn as one additive bar:
    the base value pushed left and right to the final score. It reads better
    inline in the dashboard than a waterfall, which is why the brief names
    both.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    readable_explanation = _with_readable_labels(explanation, readable)
    shap.plots.force(
        readable_explanation.base_values,
        readable_explanation.values,
        features=readable_explanation.data,
        feature_names=readable_explanation.feature_names,
        matplotlib=True,
        show=False,
    )
    fig = plt.gcf()
    if title:
        fig.suptitle(title)
    fig.set_size_inches(14, 3.5)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def force_plot_html(explanation, out_path: str | Path | None = None,
                    readable: bool = True):
    """Interactive force plot. Returns the shap visualiser; writes HTML if asked.

    This is the object S3 embeds with ``streamlit.components.v1.html`` — see
    ``explainability/render.st_force_plot``.
    """
    import shap

    visualiser = shap.plots.force(_with_readable_labels(explanation, readable))
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shap.save_html(str(out_path), visualiser)
    return visualiser


def save_summary_bar(explanation, out_path: str | Path, title: str = "",
                     readable: bool = True) -> Path:
    """Corpus-level mean |SHAP| ranking — which features drive a criterion at all.

    The beeswarm shows distribution; this shows magnitude order. Together they
    are the "summary visualisations" the brief asks for.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    shap.plots.bar(_with_readable_labels(explanation, readable),
                   max_display=N_FEATURES, show=False)
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
    sample_id = ids[sample_idx]
    for dim, exp in explanations.items():
        label = criterion(dim).label
        save_beeswarm(exp, out_dir / f"beeswarm_{dim}.png",
                      title=f"Corpus SHAP summary — {label}")
        save_summary_bar(exp, out_dir / f"bar_{dim}.png",
                         title=f"Mean |SHAP| — {label}")
        save_waterfall(exp[sample_idx], out_dir / f"waterfall_{sample_id}_{dim}.png",
                       title=f"{sample_id} — {label}")
        save_force_plot(exp[sample_idx], out_dir / f"force_{sample_id}_{dim}.png",
                        title=f"{sample_id} — {label}")
        force_plot_html(exp[sample_idx], out_dir / f"force_{sample_id}_{dim}.html")
    logger.info("Saved plots -> %s", out_dir)


if __name__ == "__main__":
    main()
