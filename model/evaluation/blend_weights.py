"""Fit the per-criterion blend between the structural and text channels.

Weighting each channel in proportion to its accuracy is intuitive and wrong.
Two channels scoring QWK 0.78 and 0.65 get 55/45 under that rule, so a channel
that is worse on *every* criterion still carries nearly half the score. The
proportion is not the quantity we care about: what we want is the weight that
makes the blended prediction most accurate, and that has to be measured, not
assumed.

So for each criterion this fits

    blended(w) = w * structural_prediction + (1 - w) * text_prediction

and picks the w minimising held-out squared error. If one channel is useless
for a criterion the fit drives its weight to zero on its own; if the two make
uncorrelated errors, the best w sits between and the blend beats both, which is
the only real justification for running two models at all.

Both channels are evaluated on the *same* held-out plans. That is not a
coincidence to rely on silently: ``scoring/train_xgboost.train`` and
``scoring/train_bert_lora.train`` both build their split with
``RandomState(42).rand(n) < 0.2``, so the two validation sets are identical by
construction. ``_assert_shared_holdout`` checks it rather than trusting it.

Run from the ``model/`` directory, after both models are trained:
    python -m evaluation.blend_weights --labels data/synthetic_v1/labels.csv \\
        --plans data/synthetic_v1/plans
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from compat import preload_torch

preload_torch()  # must precede xgboost/sklearn — see compat.py

import numpy as np
import pandas as pd

from model_contract import (
    CONTRACT_VERSION,
    CRITERION_KEYS,
    SCORE_MAX,
    SCORE_MIN,
    criterion,
    load_artifacts,
)

logger = logging.getLogger(__name__)

BLEND_REPORT_NAME = "blend_weights.json"

#: Resolution of the weight search. 0.01 is far finer than the data can
#: justify, but the search is free and it avoids an arbitrary-looking grid.
_GRID = np.linspace(0.0, 1.0, 101)


def _assert_shared_holdout(n: int) -> list[int]:
    from scoring.train_bert_lora import holdout_indices
    from scoring.train_xgboost import validation_mask

    text_idx = holdout_indices(n)
    tabular_idx = [int(i) for i in np.where(validation_mask(n))[0]]
    if text_idx != tabular_idx:
        raise ValueError(
            "The structural and text channels were validated on different "
            "held-out plans, so their predictions cannot be blended on a common "
            "set. Align the split scheme in scoring/train_xgboost.train and "
            "scoring/train_bert_lora.train before fitting blend weights.\n"
            f"  structural: {len(tabular_idx)} plans\n"
            f"  text:       {len(text_idx)} plans")
    return text_idx


def fit(labels_csv: str | Path, plans_dir: str | Path) -> dict:
    """Per-criterion optimal structural weight, measured on the shared holdout."""
    from scoring.train_bert_lora import load_text_dataset, load_text_scorer
    from scoring.train_xgboost import load_dataset, predict_scores

    artifacts = load_artifacts()
    X, Y, ids = load_dataset(labels_csv, plans_dir)
    dataset = load_text_dataset(labels_csv, plans_dir)

    holdout = _assert_shared_holdout(len(X))
    logger.info("Fitting on %d held-out plans", len(holdout))

    structural = predict_scores(artifacts.bundle, X, rounded=False).iloc[holdout]
    text = pd.DataFrame(
        load_text_scorer().predict([dataset.texts[i] for i in holdout]),
        columns=list(CRITERION_KEYS))
    truth = Y.iloc[holdout].reset_index(drop=True)
    structural = structural.reset_index(drop=True)

    results = {}
    for key in CRITERION_KEYS:
        s = structural[key].to_numpy(dtype=float)
        t = text[key].to_numpy(dtype=float)
        y = truth[key].to_numpy(dtype=float)

        errors = [float(np.mean(np.clip(w * s + (1 - w) * t,
                                        SCORE_MIN, SCORE_MAX) - y) ** 2
                        + np.var(np.clip(w * s + (1 - w) * t,
                                         SCORE_MIN, SCORE_MAX) - y))
                  for w in _GRID]
        best = int(np.argmin(errors))
        w = float(_GRID[best])
        results[key] = {
            "structural_weight": round(w, 3),
            "text_weight": round(1.0 - w, 3),
            "blended_rmse": round(float(np.sqrt(errors[best])), 4),
            "structural_only_rmse": round(float(np.sqrt(errors[-1])), 4),
            "text_only_rmse": round(float(np.sqrt(errors[0])), 4),
        }
    return {
        "contract_version": CONTRACT_VERSION,
        "n_holdout": len(holdout),
        "note": ("Fitted on synthetic plans. Refit against the real annotated "
                 "CoE plans before relying on these weights."),
        "per_criterion": results,
    }


def write(report: dict, artifacts_dir=None) -> Path:
    from model_contract import ARTIFACTS_DIR

    out_dir = Path(artifacts_dir or ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / BLEND_REPORT_NAME
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def load_blend_weights(artifacts_dir=None) -> dict[str, float]:
    """criterion -> fitted structural weight, or {} if never fitted."""
    from model_contract import ARTIFACTS_DIR

    path = Path(artifacts_dir or ARTIFACTS_DIR) / BLEND_REPORT_NAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(entry["structural_weight"])
            for key, entry in payload.get("per_criterion", {}).items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit per-criterion channel blend weights on the shared holdout")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--artifacts", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = fit(args.labels, args.plans)

    print(f"\n{'criterion':34} {'w_struct':>9} {'blend':>7} {'struct':>7} {'text':>7}")
    print("-" * 70)
    for key, entry in report["per_criterion"].items():
        target = criterion(key)
        gain = min(entry["structural_only_rmse"], entry["text_only_rmse"]) \
            - entry["blended_rmse"]
        flag = "  <- blend beats both" if gain > 1e-4 else ""
        print(f"{target.id + ' ' + target.label:34} "
              f"{entry['structural_weight']:9.2f} {entry['blended_rmse']:7.3f} "
              f"{entry['structural_only_rmse']:7.3f} {entry['text_only_rmse']:7.3f}{flag}")

    print(f"\nWrote {write(report, args.artifacts)}")


if __name__ == "__main__":
    main()
