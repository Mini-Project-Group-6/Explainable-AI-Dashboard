"""5-fold cross-validation of the rubric scorer (proposal E3, criterion 1).

Reports quadratic-weighted Cohen's Kappa per rubric dimension as
mean +/- SD across folds — the number that must reach >= 0.70 on the real
annotated CoE plans.

Run from the ``model/`` directory:
    python -m evaluation.cross_validate --labels data/synthetic_v1/labels.csv \
        --plans data/synthetic_v1/plans
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from evaluation.metrics import full_report
from scoring.train_xgboost import RUBRIC_DIMENSIONS, load_dataset, predict_scores, train

logger = logging.getLogger(__name__)


def cross_validate(X: pd.DataFrame, Y: pd.DataFrame, n_splits: int = 5,
                   seed: int = 42) -> pd.DataFrame:
    """Per-dimension QWK/Pearson/RMSE, mean +/- SD over K folds."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_reports = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        bundle = train(X.iloc[train_idx].reset_index(drop=True),
                       Y.iloc[train_idx].reset_index(drop=True), seed=seed)
        preds = predict_scores(bundle, X.iloc[test_idx].reset_index(drop=True))
        report = full_report(Y.iloc[test_idx].reset_index(drop=True), preds)
        fold_reports.append(report)
        logger.info("Fold %d/%d — mean QWK %.3f", fold, n_splits, report.loc["MEAN", "qwk"])

    stacked = pd.concat(fold_reports, keys=range(1, n_splits + 1))
    mean = stacked.groupby(level=1).mean()
    std = stacked.groupby(level=1).std()
    summary = mean.add_suffix("_mean").join(std.add_suffix("_sd"))
    # Preserve rubric ordering with the MEAN row last
    order = [d for d in RUBRIC_DIMENSIONS if d in summary.index] + ["MEAN"]
    return summary.loc[order]


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV of the rubric scorer")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    X, Y, _ = load_dataset(args.labels, args.plans)
    logger.info("Loaded %d plans", len(X))

    summary = cross_validate(X, Y, n_splits=args.folds)
    pd.set_option("display.width", 120)
    print("\n=== 5-fold cross-validation (mean +/- SD across folds) ===")
    print(summary.round(3).to_string())
    qwk = summary.loc["MEAN", "qwk_mean"]
    print(f"\nOverall mean QWK: {qwk:.3f}  (target >= 0.70)")


if __name__ == "__main__":
    main()
