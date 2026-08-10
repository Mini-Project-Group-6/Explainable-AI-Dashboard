"""Evaluation metrics for the rubric scorer (proposal E3, success criterion 1).

Primary metric: quadratic-weighted Cohen's Kappa (QWK) per rubric dimension —
target >= 0.70 against human tutor scores. Secondary: Pearson r and RMSE per
dimension (proposal D3).
"""

from __future__ import annotations

from compat import preload_torch

preload_torch()  # must precede sklearn — see compat.py

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_squared_error

RUBRIC_LABELS = [0, 1, 2, 3, 4]


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    return cohen_kappa_score(
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        weights="quadratic",
        labels=RUBRIC_LABELS,
    )


def kappa_report(Y_true: pd.DataFrame, Y_pred: pd.DataFrame) -> dict[str, float]:
    """QWK per rubric dimension (column-aligned dataframes)."""
    return {dim: quadratic_weighted_kappa(Y_true[dim], Y_pred[dim])
            for dim in Y_true.columns}


def full_report(Y_true: pd.DataFrame, Y_pred: pd.DataFrame) -> pd.DataFrame:
    """QWK, Pearson r and RMSE per dimension, plus a mean row."""
    rows = {}
    for dim in Y_true.columns:
        t = np.asarray(Y_true[dim], dtype=float)
        p = np.asarray(Y_pred[dim], dtype=float)
        pearson = float(np.corrcoef(t, p)[0, 1]) if t.std() > 0 and p.std() > 0 else float("nan")
        rows[dim] = {
            "qwk": quadratic_weighted_kappa(t, p),
            "pearson_r": pearson,
            "rmse": float(np.sqrt(mean_squared_error(t, p))),
        }
    report = pd.DataFrame(rows).T
    report.loc["MEAN"] = report.mean()
    return report
