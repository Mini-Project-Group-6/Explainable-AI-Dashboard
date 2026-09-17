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

from compat import preload_torch

preload_torch()  # must precede sklearn — see compat.py

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


#: Written beside the model so the reconciler can weight the structural channel
#: by measured agreement, symmetrically with the text channel's report. Without
#: it, reconcile falls back to the coverage heuristic for both sides.
RELIABILITY_REPORT_NAME = "tabular_holdout_report.json"


def write_reliability_report(summary: pd.DataFrame, out_dir,
                             n_plans: int, n_splits: int):
    """Persist per-criterion cross-validated QWK as channel weights.

    The text channel has had measured weights since it was trained; the
    structural channel was still being weighted by a *coverage* heuristic, which
    is not the same kind of quantity and produced a blend that under-weighted
    the more accurate channel. This closes that asymmetry.
    """
    import json
    from pathlib import Path

    from model_contract import CONTRACT_VERSION, CRITERION_KEYS

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "n_plans": n_plans,
        "n_splits": n_splits,
        "metric": "quadratic_weighted_kappa",
        "note": ("Cross-validated on synthetic plans. Re-run against the real "
                 "annotated CoE plans before relying on these weights."),
        "per_criterion": {
            key: {"qwk": round(float(summary.loc[key, "qwk_mean"]), 4),
                  "rmse": round(float(summary.loc[key, "rmse_mean"]), 4)}
            for key in CRITERION_KEYS if key in summary.index
        },
        "mean_qwk": round(float(summary.loc["MEAN", "qwk_mean"]), 4),
    }
    path = out_dir / RELIABILITY_REPORT_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_structural_reliability(artifacts_dir=None) -> dict[str, float]:
    """criterion -> cross-validated QWK, or {} if CV has not been run.

    Negative kappa is floored at 0: a criterion the model cannot do gets no
    weight rather than negative weight.
    """
    import json
    from pathlib import Path

    from model_contract import ARTIFACTS_DIR

    path = Path(artifacts_dir or ARTIFACTS_DIR) / RELIABILITY_REPORT_NAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: max(0.0, float(entry["qwk"]))
            for key, entry in payload.get("per_criterion", {}).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV of the rubric scorer")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--artifacts", default=None,
                        help="where to write the reliability report "
                             "(default: the contract's artifacts directory)")
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

    from model_contract import ARTIFACTS_DIR

    path = write_reliability_report(summary, args.artifacts or ARTIFACTS_DIR,
                                    len(X), args.folds)
    print(f"Wrote channel weights -> {path}")


if __name__ == "__main__":
    main()
