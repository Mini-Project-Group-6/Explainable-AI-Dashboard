"""Train the XGBoost rubric scorer (rubric specification v2).

One XGBoost regressor per rubric criterion, trained on the feature vectors
from ``ingestion/feature_engineer.py``. Regression + rounding is used
(standard AES practice for ordinal 1-4 rubric scores and what quadratic-
weighted Kappa expects); predictions are clipped to [1, 4].

Hyperparameters are pinned by proposal D3:
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=20

The saved bundle also carries the F17 z-score statistics (mean/std of
``sentence_complexity_index`` over the training corpus — FEATURES.md says
the extractor returns the raw value and z-scoring happens here) so that
inference applies the identical transform.

Run from the ``model/`` directory:
    python -m scoring.train_xgboost --labels data/synthetic_v1/labels.csv \
        --plans data/synthetic_v1/plans --out artifacts/rubric_model.joblib
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ingestion.extract_text import LessonPlanText, parse_stated_duration, segment_sections
from ingestion.feature_engineer import FEATURE_NAMES, FeatureEngineer
from rubric_schema import RUBRIC_DIMENSIONS

logger = logging.getLogger(__name__)

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)
EARLY_STOPPING_ROUNDS = 20

Z_SCORED_FEATURE = "sentence_complexity_index"  # F17


def plan_from_text(raw_text: str, source: str = "<text>") -> LessonPlanText:
    """Build a LessonPlanText from already-extracted raw text (e.g. synthetic .txt)."""
    sections, missing = segment_sections(raw_text)
    return LessonPlanText(
        source_path=source,
        raw_text=raw_text,
        sections=sections,
        missing_sections=missing,
        stated_duration_minutes=parse_stated_duration(raw_text, sections.get("header", "")),
    )


def load_dataset(labels_csv: str | Path, plans_dir: str | Path,
                 engineer: FeatureEngineer | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Feature matrix X, label matrix Y and plan ids from a labels.csv manifest.

    The manifest format is produced by ``data/synthetic.py`` and will be
    reused for the real annotated CoE plans (same columns, is_synthetic=0).
    Plans are read as ``<plans_dir>/<plan_id>.txt``; PDF/DOCX plans should be
    converted through ``ingestion.extract_text.extract`` upstream.
    """
    labels_csv, plans_dir = Path(labels_csv), Path(plans_dir)
    engineer = engineer or FeatureEngineer()

    rows, labels, ids = [], [], []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            plan_id = record["plan_id"]
            text = (plans_dir / f"{plan_id}.txt").read_text(encoding="utf-8")
            plan = plan_from_text(text, source=plan_id)
            rows.append(engineer.extract_features(plan))
            labels.append({d: int(record[d]) for d in RUBRIC_DIMENSIONS})
            ids.append(plan_id)
            if len(ids) % 25 == 0:
                logger.info("Featurised %d plans", len(ids))

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    Y = pd.DataFrame(labels, columns=RUBRIC_DIMENSIONS)
    return X, Y, ids


def apply_f17_zscore(X: pd.DataFrame, mean: float, std: float) -> pd.DataFrame:
    X = X.copy()
    X[Z_SCORED_FEATURE] = (X[Z_SCORED_FEATURE] - mean) / (std if std > 1e-9 else 1.0)
    return X


def train(X: pd.DataFrame, Y: pd.DataFrame, validation_fraction: float = 0.2,
          seed: int = 42) -> dict:
    """Train one model per rubric dimension; returns the persistable bundle."""
    from xgboost import XGBRegressor

    rng = np.random.RandomState(seed)
    val_mask = rng.rand(len(X)) < validation_fraction
    if val_mask.all() or not val_mask.any():
        val_mask[:] = False
        val_mask[: max(1, len(X) // 5)] = True

    f17_mean = float(X.loc[~val_mask, Z_SCORED_FEATURE].mean())
    f17_std = float(X.loc[~val_mask, Z_SCORED_FEATURE].std(ddof=0))
    Xz = apply_f17_zscore(X, f17_mean, f17_std)

    X_train, X_val = Xz[~val_mask], Xz[val_mask]
    models: dict[str, object] = {}
    for dim in RUBRIC_DIMENSIONS:
        y_train, y_val = Y.loc[~val_mask, dim], Y.loc[val_mask, dim]
        model = XGBRegressor(**XGB_PARAMS,
                             early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models[dim] = model
        logger.info("Trained %s (best_iteration=%s)", dim, model.best_iteration)

    return {
        "models": models,
        "feature_names": FEATURE_NAMES,
        "rubric_dimensions": RUBRIC_DIMENSIONS,
        "f17_mean": f17_mean,
        "f17_std": f17_std,
        "xgb_params": XGB_PARAMS,
    }


def predict_scores(bundle: dict, X: pd.DataFrame, rounded: bool = True) -> pd.DataFrame:
    """6-dimension rubric predictions for a feature matrix (0-4 per dimension)."""
    Xz = apply_f17_zscore(X, bundle["f17_mean"], bundle["f17_std"])
    preds = {}
    for dim in bundle["rubric_dimensions"]:
        raw = bundle["models"][dim].predict(Xz)
        preds[dim] = np.clip(np.rint(raw) if rounded else raw, 1, 4)
    return pd.DataFrame(preds, index=X.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the XGBoost rubric scorer")
    parser.add_argument("--labels", required=True, help="labels.csv manifest")
    parser.add_argument("--plans", required=True, help="directory of <plan_id>.txt files")
    parser.add_argument("--out", default="artifacts/rubric_model.joblib")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logger.info("Extracting features...")
    X, Y, _ = load_dataset(args.labels, args.plans)
    logger.info("Training on %d plans x %d features", *X.shape)
    bundle = train(X, Y)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    logger.info("Saved model bundle -> %s", out_path)

    # In-sample sanity check only — real validation is evaluation/cross_validate.py
    preds = predict_scores(bundle, X)
    from evaluation.metrics import kappa_report
    for dim, kappa in kappa_report(Y, preds).items():
        logger.info("in-sample QWK %-16s %.3f", dim, kappa)


if __name__ == "__main__":
    main()
